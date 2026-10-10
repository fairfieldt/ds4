#!/usr/bin/env python3
"""Requantize a Qwen GGUF's BF16 n-gram table to 8-bit rows.

Each row becomes a little-endian F32 scale followed by 160 codes, stored as a
GGUF I8 tensor [164, rows]. `e4m3` writes E4M3FN codes scaled by row max/448;
`i8` writes symmetric int8 codes scaled by row max/127. Main and MTP tensors are
copied unchanged. The table shrinks from 95.37 GiB to 48.88 GiB.
"""

import argparse
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import shutil
import struct

import numpy as np

from qwen4_native_ngrams import NGRAM, copy_hash, ngram_geometry, size_of
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str

ENCODINGS = {'e4m3': ('e4m3_f32row', 448.0), 'i8': ('i8_f32row', 127.0)}
CHUNK_ROWS = 1 << 18


def e4m3_codes(y):
    """Round finite |y| <= 448 to nearest-even E4M3FN codes."""
    a = np.abs(y).astype(np.float32)
    bits = a.view(np.uint32)
    rounded = (bits + 0x7ffff + ((bits >> 20) & 1)) & 0xfff00000
    normal = ((rounded >> 23).astype(np.int32) - 120) << 3 | ((rounded >> 20) & 7).astype(np.int32)
    subnormal = np.rint(a * np.float32(512)).astype(np.int32)
    code = np.where(a >= np.float32(2.0 ** -6), normal, subnormal)
    return (code | np.where(y < 0, 128, 0)).astype(np.uint8)


def e4m3_values():
    b = np.arange(256)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e > 0, (8 + m) * 2.0 ** (e - 10), m * 2.0 ** -9)
    v = np.where(b & 128, -v, v)
    return np.where((b & 127) == 127, np.nan, v).astype(np.float32)


def quantize(x, encoding):
    """x: float32 [rows, width]. Returns uint8 [rows, width+4] and the decoded rows."""
    limit = ENCODINGS[encoding][1]
    amax = np.abs(x).max(1)
    scale = (amax / np.float32(limit)).astype(np.float32)
    safe = np.where(scale > 0, scale, np.float32(1))[:, None]
    y = np.clip(x / safe, -limit, limit)
    if encoding == 'e4m3':
        codes = e4m3_codes(y)
        decoded = e4m3_values()[codes] * scale[:, None]
    else:
        q = np.rint(y).astype(np.int8)
        codes = q.view(np.uint8)
        decoded = q.astype(np.float32) * scale[:, None]
    out = np.empty((x.shape[0], x.shape[1] + 4), dtype=np.uint8)
    out[:, :4] = scale.astype('<f4').view(np.uint8).reshape(-1, 4)
    out[:, 4:] = codes
    return out, decoded


def work(job):
    path, offset, rows, width, encoding = job
    fd = os.open(path, os.O_RDONLY)
    try:
        size = rows * width * 2
        raw = os.pread(fd, size, offset)
        if len(raw) != size:
            raise ValueError('Truncated BF16 n-gram table')
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    x = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(rows, width)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite BF16 n-gram value')
    out, decoded = quantize(x, encoding)
    err = decoded - x
    norm2 = (x * x).sum(1, dtype=np.float64)
    err2 = (err * err).sum(1, dtype=np.float64)
    rel = np.sqrt(err2 / np.maximum(norm2, 1e-300))
    return out.tobytes(), float(err2.sum()), float(norm2.sum()), float(rel.sum()), float(rel.max())


def convert(model_path, output, encoding, workers):
    pending = Path(str(output) + '.incomplete')
    if output.exists() or pending.exists():
        raise ValueError('Output already exists; choose a new path or inspect the incomplete file')
    model = Reader(str(model_path))
    try:
        if model.kv.get('general.architecture') != 'qwen4exp':
            raise ValueError('Expected an active qwen4exp GGUF')
        if 'qwen4exp.ple.ngram_encoding' in model.kv:
            raise ValueError('Expected original BF16 n-grams')
        ngram_geometry(model)
        kind, dims, table_offset = model.tensors[NGRAM]
        width, rows = dims
        if kind != 30 or width != model.kv['qwen4exp.ple.row_dimension'] or width > 160:
            raise ValueError('Expected original BF16 n-grams')
        source_table = model.data_start + table_offset
        if source_table + rows * width * 2 > model_path.stat().st_size:
            raise ValueError('Truncated BF16 n-gram table')

        metadata = {k: (model.kv_types[k], v) for k, v in model.kv.items()}
        metadata['qwen4exp.ple.ngram_encoding'] = (8, ENCODINGS[encoding][0])
        alignment = model.kv.get('general.alignment', 32)
        if alignment < 1 or alignment & (alignment - 1):
            raise ValueError('Invalid GGUF alignment')
        align = lambda n, a=alignment: (n + a - 1) // a * a
        plan, offset = [], 0
        for name, (k, shape, old_offset) in model.tensors.items():
            if name == NGRAM:
                continue
            size = size_of(k, shape)
            if model.data_start + old_offset + size > model_path.stat().st_size:
                raise ValueError('Main tensor exceeds input file')
            plan.append((name, k, shape, offset, size, old_offset))
            offset = align(offset + size)

        row_bytes = width + 4
        prefix = b'GGUF' + struct.pack('<IQQ', 3, len(plan)+1, len(metadata))
        prefix += b''.join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())

        def header(ngram_offset):
            data = bytearray(prefix)
            entries = [(n, k, s, o) for n, k, s, o, _, _ in plan] + [(NGRAM, 24, [row_bytes, rows], ngram_offset)]
            for name, k, shape, pos in entries:
                data += w_str(name) + struct.pack('<I', len(shape))
                data += struct.pack('<'+'Q'*len(shape), *shape) + struct.pack('<IQ', k, pos)
            return data

        data_start = align(len(header(0)))
        table_start = align(data_start + offset, max(65536, alignment))
        total = table_start + rows * row_bytes
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < total + (16 << 30):
            raise ValueError('Not enough disk space with a 16 GiB reserve')
        records = []
        with pending.open('xb') as dst:
            dst.write(header(table_start - data_start))
            for name, k, shape, pos, size, old in plan:
                dst.seek(data_start + pos)
                model.f.seek(model.data_start + old)
                records.append(dict(name=name, offset=data_start+pos, bytes=size,
                                    sha256=copy_hash(model.f, dst, size)))
            dst.seek(table_start)
            jobs = [(str(model_path), source_table + r * width * 2, min(CHUNK_ROWS, rows - r), width, encoding)
                    for r in range(0, rows, CHUNK_ROWS)]
            digest = hashlib.sha256()
            err2 = norm2 = rel_sum = rel_max = 0.0
            with Pool(workers) as pool:
                for i, (data, e2, n2, rs, rm) in enumerate(pool.imap(work, jobs)):
                    dst.write(data)
                    digest.update(data)
                    err2, norm2, rel_sum, rel_max = err2 + e2, norm2 + n2, rel_sum + rs, max(rel_max, rm)
                    if i % 64 == 0 or i == len(jobs) - 1:
                        print('Quantized %d/%d chunks' % (i + 1, len(jobs)), flush=True)
            records.append(dict(name=NGRAM, offset=table_start, bytes=rows * row_bytes,
                                sha256=digest.hexdigest()))
            if dst.tell() != total:
                raise ValueError('Incorrect assembled size')
            dst.flush()
            os.fsync(dst.fileno())
        print('Verifying every output tensor', flush=True)
        with pending.open('rb') as check:
            for rec in records:
                check.seek(rec['offset'])
                if copy_hash(check, None, rec['bytes']) != rec['sha256']:
                    raise ValueError('Output payload verification failed: ' + rec['name'])
        reread = Reader(str(pending))
        try:
            if reread.tensors[NGRAM] != (24, [row_bytes, rows], table_start - data_start) or \
               reread.kv.get('qwen4exp.ple.ngram_encoding') != ENCODINGS[encoding][0]:
                raise ValueError('Incorrect output n-gram header')
        finally:
            reread.f.close()
        error = dict(relative_rms=(err2 / norm2) ** 0.5, mean_row_relative_rms=rel_sum / rows,
                     max_row_relative_rms=rel_max)
        report = dict(source=str(model_path), encoding=ENCODINGS[encoding][0], bytes=total,
                      error=error, tensors=records)
        Path(str(output) + '.json').write_text(json.dumps(report, indent=2) + '\n')
        pending.rename(output)
        print('Verified', output, total, json.dumps(error), flush=True)
        return report
    finally:
        model.f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Qwen GGUF with original BF16 n-grams')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encoding', choices=sorted(ENCODINGS), required=True)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()
    try:
        convert(args.model, args.output, args.encoding, args.workers)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
