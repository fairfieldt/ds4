#!/usr/bin/env python3
"""Convert NVIDIA's QAD checkpoint nvidia/Qwen3.8-Flash-Next-FP4-Local
(ModelOpt MIXED_PRECISION) into a DS4 qwen4exp GGUF without requantizing.

The distillation trained the checkpoint's quantized values, so every tensor
keeps them:

  - Routed experts (trunk and MTP): NVFP4 E2M1 codes and E4M3 block scales,
    repacked into GGML NVFP4 blocks (type 40: four scales, then 32 code
    bytes, per 64 values).  Each expert's FP32 global scale goes into
    blk.N.ffn_{gate,up,down}_exps.scale (F32 [n_expert]), as llama.cpp
    stores ModelOpt NVFP4.  The trunk's compressed block scales
    (weight_scale.nvfp4_csf_fixed/_exceptions) are decoded exactly.
    input_scale is not used: activations stay unquantized (W4A16).
  - MXFP8 projections (attention, indexer, delta-net, shared experts): E4M3
    values and UE8M0 scales as DS4's MXFP8 type (type 100: four scales, then
    128 values), or exact BF16 with --dense bf16.
  - The n-gram table: NVFP4 rows of ten E4M3 scales and 80 code bytes (low
    nibble first), a GGUF I8 tensor [90, rows]; qwen4exp.ple.ngram_encoding
    is nvfp4_e4m3x16 and qwen4exp.ple.ngram_global_scale holds the table's
    FP32 global scale.
  - BF16 tensors stay exact: BF16 (embeddings, output head, MTP and PLE
    projections) and F32 (routers, norms, delta-net vectors).  The
    hyper-connection mixers become F16 for DS4's mixer kernels: about 0.01%
    of their values, all below 1e-5, fall in F16's subnormal range and round
    by at most 3e-8 (BF16's own step at a typical value is 1e-4).  The report
    records the count and the largest error; anything larger is refused.

An existing DS4 qwen4exp GGUF with MTP (--template) supplies the metadata,
tokenizer and tensor order; none of its weights are copied.  --check
compares every converted tensor with the template's values instead of
writing: F32 tensors should match exactly where QAD left them unchanged,
quantized ones closely.  The output is written to <out>.incomplete, every
payload is verified, and it is renamed when complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str  # noqa: E402

T_F32, T_F16, T_Q8_0, T_Q4_K, T_I8, T_BF16, T_MXFP4, T_NVFP4, T_MXFP8 = 0, 1, 8, 12, 24, 30, 39, 40, 100
BLOCKS = {T_F32: (1, 4), T_F16: (1, 2), T_Q8_0: (32, 34), T_Q4_K: (256, 144), T_I8: (1, 1),
          T_BF16: (1, 2), T_MXFP4: (32, 17), T_NVFP4: (64, 36), T_MXFP8: (128, 132)}
NGRAM = 'per_layer_token_embd.weight'
NGRAM_ENCODING = 'nvfp4_e4m3x16'
SOURCE_REPOSITORY = 'nvidia/Qwen3.8-Flash-Next-FP4-Local'
N_LAYER, N_EXPERT, N_K_HEAD, N_V_HEAD, HEAD_V = 48, 512, 16, 48, 128
TRUNK = 'model.language_model.'


def size_of(kind, dims):
    block, size = BLOCKS[kind]
    if not dims or any(d <= 0 for d in dims) or dims[0] % block:
        raise ValueError('Invalid tensor dimensions or block alignment')
    return math.prod(dims) // block * size


def e4m3_values():
    b = np.arange(256)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e > 0, (8 + m) * 2.0 ** (e - 10), m * 2.0 ** -9)
    v = np.where(b & 128, -v, v)
    return np.where((b & 127) == 127, np.nan, v).astype(np.float32)


E4M3 = e4m3_values()
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32)


# --- safetensors ----------------------------------------------------------

class Checkpoint:
    """Lazy pread access to the checkpoint's tensors through its index."""

    DTYPES = {'BF16': '<u2', 'F32': '<f4', 'U8': 'u1', 'F8_E4M3': 'u1', 'U32': '<u4', 'I64': '<i8'}

    def __init__(self, root, allow_missing=False):
        self.root = Path(root)
        index = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        self.entries = {}
        for filename in sorted(set(index.values())):
            path = self.root / filename
            if Path(filename).is_absolute() or '..' in Path(filename).parts:
                raise ValueError('Checkpoint shard names must stay inside the checkpoint')
            if allow_missing and not path.exists():
                index = {k: v for k, v in index.items() if v != filename}
                continue
            with path.open('rb') as fp:
                header_size, = struct.unpack('<Q', fp.read(8))
                if header_size > min(path.stat().st_size - 8, 64 << 20):
                    raise ValueError('Invalid safetensors header size in ' + filename)
                header = json.loads(fp.read(header_size))
            payload = path.stat().st_size - 8 - header_size
            for name, t in header.items():
                if name == '__metadata__':
                    continue
                start, end = t['data_offsets']
                if not 0 <= start <= end <= payload:
                    raise ValueError('Invalid tensor offsets for ' + name)
                self.entries[name] = (str(path), 8 + header_size + start, end - start, t['dtype'], t['shape'])
        missing = set(index) - set(self.entries)
        if missing:
            raise ValueError('Index names tensors absent from their shards: %s' % sorted(missing)[:4])

    def has(self, name):
        return name in self.entries

    def info(self, name):
        return self.entries[name]

    def read(self, name, dtype=None, shape=None):
        path, offset, size, kind, dims = self.entries[name]
        if dtype is not None and kind != dtype:
            raise ValueError('%s is %s, expected %s' % (name, kind, dtype))
        if shape is not None and list(dims) != list(shape):
            raise ValueError('%s has shape %s, expected %s' % (name, dims, list(shape)))
        return read_entry(self.entries[name])


def read_entry(entry):
    path, offset, size, kind, dims = entry
    fd = os.open(path, os.O_RDONLY)
    try:
        raw = os.pread(fd, size, offset)
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    if len(raw) != size:
        raise ValueError('Truncated tensor payload in ' + path)
    return np.frombuffer(raw, dtype=Checkpoint.DTYPES[kind]).reshape(dims if dims else ())


# --- NVFP4 experts -------------------------------------------------------

def csf_decode(fixed, exceptions, rows, cols):
    """Plain E4M3 block-scale bytes [rows, cols] from the CSF streams.

    Each 16-row slab stores its rows' base bytes, then a 4-bit offset per
    scale (even column in the low nibble).  Exceptions are u32 words: the
    position row * cols + col in the low 24 bits, the original byte above.
    """
    if fixed.dtype != np.uint8 or exceptions.dtype != np.uint32 or rows % 16 or cols % 2:
        raise ValueError('Invalid CSF stream types or geometry')
    stream = fixed.reshape(rows // 16, 16 * (1 + cols // 2))
    bases = stream[:, :16].reshape(rows).astype(np.int32)
    if bases.max() > 240:
        raise ValueError('CSF row bases must be in 0..240')
    packed = stream[:, 16:].reshape(rows, cols // 2)
    offsets = np.empty((rows, cols), dtype=np.int32)
    offsets[:, 0::2] = packed & 15
    offsets[:, 1::2] = packed >> 4
    out = (offsets + bases[:, None]).reshape(-1)
    words = exceptions.reshape(-1)
    pos = (words & 0xFFFFFF).astype(np.int64)
    if len(pos) and (pos[-1] >= rows * cols or np.any(pos[1:] <= pos[:-1])):
        raise ValueError('CSF exception positions must be sorted, unique and in range')
    if np.any(offsets.reshape(-1)[pos]):
        raise ValueError('CSF exception positions must hold zero offsets')
    values = (words >> 24).astype(np.int32)
    inside = (values >= bases[pos // cols]) & (values < bases[pos // cols] + 16)
    if np.any(inside):
        raise ValueError('CSF exception inside its row window: the decoding does not match')
    out[pos] = values
    return out.astype(np.uint8).reshape(rows, cols)


def nvfp4_ggml(weight, scale):
    """ModelOpt NVFP4 [rows, K/2] codes (element 2j low) and E4M3 scales
    [rows, K/16] into GGML NVFP4 blocks [rows, K/64, 36]: four UE4M3 scales
    (the E4M3 byte, sign clear), then byte j of each 16-value group holding
    elements j (low nibble) and j + 8."""
    rows, half = weight.shape
    k = half * 2
    if scale.shape != (rows, k // 16) or k % 64:
        raise ValueError('NVFP4 weight and scale geometry disagree')
    if np.any(scale & 128) or np.any(scale == 127):
        raise ValueError('NVFP4 block scales must be positive finite E4M3')
    codes = np.empty((rows, k), dtype=np.uint8)
    codes[:, 0::2] = weight & 15
    codes[:, 1::2] = weight >> 4
    codes = codes.reshape(rows, k // 16, 16)
    qs = (codes[:, :, :8] | (codes[:, :, 8:] << 4)).reshape(rows, k // 64, 32)
    return np.concatenate((scale.reshape(rows, k // 64, 4), qs), axis=2)


def nvfp4_dequant(weight, scale, global_scale):
    rows, half = weight.shape
    codes = np.empty((rows, half * 2), dtype=np.uint8)
    codes[:, 0::2] = weight & 15
    codes[:, 1::2] = weight >> 4
    blk = E4M3[scale].astype(np.float32) * np.float32(global_scale)
    return E2M1[codes] * np.repeat(blk, 16, axis=1)


def expert_job(job):
    """Repack one (layer, projection)'s experts and pwrite them in place."""
    entries, rows, k, out_path, offset, csf = job
    g = np.empty(len(entries), dtype=np.float32)
    exceptions = 0
    out = np.empty((len(entries), rows, k // 64 * 36), dtype=np.uint8)
    for e, ent in enumerate(entries):
        weight = read_entry(ent['weight'])
        if weight.shape != (rows, k // 2):
            raise ValueError('Unexpected NVFP4 weight shape %s' % (weight.shape,))
        if csf:
            fixed, exc = read_entry(ent['fixed']), read_entry(ent['exceptions'])
            scale = csf_decode(fixed, exc, rows, k // 16)
            exceptions += exc.size
        else:
            scale = read_entry(ent['scale'])
        gs = read_entry(ent['global'])
        if gs.size != 1 or not np.isfinite(gs).all() or float(gs.reshape(-1)[0]) <= 0:
            raise ValueError('NVFP4 global scale must be one positive finite FP32 value')
        g[e] = gs.reshape(-1)[0]
        out[e] = nvfp4_ggml(weight, scale).reshape(rows, -1)
    data = out.tobytes()
    write_region(out_path, offset, data)
    return hashlib.sha256(data).hexdigest(), g.tobytes(), exceptions


def write_region(path, offset, data):
    fd = os.open(path, os.O_WRONLY)
    try:
        done = 0
        while done < len(data):
            done += os.pwrite(fd, memoryview(data)[done:], offset + done)
        os.fdatasync(fd)
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(fd, offset, len(data), os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


# --- n-grams ---------------------------------------------------------------

def ngram_job(job):
    weight_entry, scale_entry, out_path, offset = job
    w = read_entry(weight_entry)
    s = read_entry(scale_entry)
    rows = w.shape[0]
    if w.shape != (rows, 80) or s.shape != (rows, 10):
        raise ValueError('Unexpected NVFP4 n-gram shard geometry')
    if np.any(s & 128) or np.any(s == 127):
        raise ValueError('NVFP4 n-gram scales must be positive finite E4M3')
    data = np.concatenate((s, w), axis=1).tobytes()
    write_region(out_path, offset, data)
    return hashlib.sha256(data).hexdigest(), rows


# --- dense ------------------------------------------------------------------

def mxfp8_x4(weight, scale):
    """MXFP8 [rows, K] E4M3 bytes and UE8M0 scales [rows, K/32] into DS4
    MXFP8 blocks [rows, K/128, 132]: four scales, then 128 values."""
    rows, k = weight.shape
    if scale.shape != (rows, k // 32) or k % 128:
        raise ValueError('MXFP8 weight and scale geometry disagree')
    if np.any(scale == 255) or np.any((weight & 127) == 127):
        raise ValueError('MXFP8 tensor holds NaN encodings')
    return np.concatenate((scale.reshape(rows, k // 128, 4), weight.reshape(rows, k // 128, 128)), axis=2)


def mxfp8_dequant(weight, scale):
    exps = np.repeat(scale.astype(np.int32) - 127, 32, axis=1)
    return np.ldexp(E4M3[weight], exps).astype(np.float32)


class Dense:
    """One projection's rows, still in the checkpoint's encoding."""

    def __init__(self, weight, scale=None, kind='bf16'):
        self.weight, self.scale, self.kind = weight, scale, kind

    @property
    def rows(self):
        return self.weight.shape[0]

    def take_rows(self, idx):
        return Dense(self.weight[idx], None if self.scale is None else self.scale[idx], self.kind)

    def take_cols(self, perm, unit):
        """Permute columns in whole units (multiples of the scale block)."""
        k = self.weight.shape[1]
        cols = np.arange(k).reshape(-1, unit)[perm].reshape(-1)
        if self.scale is None:
            return Dense(self.weight[:, cols], None, self.kind)
        block = k // self.scale.shape[1]
        if unit % block:
            raise ValueError('Column permutation splits scale blocks')
        scols = np.arange(self.scale.shape[1]).reshape(-1, unit // block)[perm].reshape(-1)
        return Dense(self.weight[:, cols], self.scale[:, scols], self.kind)

    def f32(self):
        if self.kind == 'mxfp8':
            return mxfp8_dequant(self.weight, self.scale)
        return bf16_to_f32(self.weight.tobytes()).reshape(self.weight.shape)

    def encode(self, kind):
        if kind == T_MXFP8:
            if self.kind != 'mxfp8':
                raise ValueError('Only MXFP8 sources encode as MXFP8')
            return mxfp8_x4(self.weight, self.scale).tobytes()
        if kind == T_BF16:
            if self.kind == 'bf16':
                return self.weight.astype('<u2').tobytes()
            v = self.f32()
            bits = v.view(np.uint32)
            if np.any(bits & 0xFFFF):
                raise ValueError('MXFP8 value not exact in BF16')
            return (bits >> 16).astype('<u2').tobytes()
        raise ValueError('Unsupported dense output type %d' % kind)


def v_order(head_dim):
    """HF grouped v-heads to qwen4exp tiled order: slot j takes HF row idx[j]."""
    return np.arange(N_V_HEAD * head_dim).reshape(N_K_HEAD, N_V_HEAD // N_K_HEAD, head_dim) \
        .transpose(1, 0, 2).reshape(-1)


# --- conversion plan ----------------------------------------------------------

class Converter:
    def __init__(self, ckpt, template, dense_type, output_type):
        self.ck = ckpt
        self.tp = template
        self.dense_type = dense_type
        self.output_type = output_type
        self.notes = {}

    def dense(self, prefix):
        """A projection as stored: BF16, or MXFP8 with its UE8M0 scales."""
        w = prefix + '.weight'
        path, offset, size, kind, dims = self.ck.info(w)
        if kind == 'BF16':
            return Dense(self.ck.read(w), None, 'bf16')
        if kind == 'F8_E4M3':
            return Dense(self.ck.read(w), self.ck.read(prefix + '.weight_scale', 'U8'), 'mxfp8')
        raise ValueError('Unexpected dense dtype %s for %s' % (kind, w))

    def vec(self, name, plus_one=False):
        path, offset, size, kind, dims = self.ck.info(name)
        if kind == 'BF16':
            v = bf16_to_f32(self.ck.read(name).tobytes())
        elif kind == 'F32':
            v = self.ck.read(name).astype(np.float32).reshape(-1)
        else:
            raise ValueError('Unexpected vector dtype %s for %s' % (kind, name))
        return (v + np.float32(1)).astype(np.float32) if plus_one else v

    def hc_bytes(self, name):
        """A BF16 hyper-connection mixer as F16 (see the module docstring)."""
        v = bf16_to_f32(self.ck.read(name, 'BF16').tobytes())
        h = v.astype(np.float16)
        err = np.abs(h.astype(np.float32) - v)
        if not np.isfinite(h).all() or err.max() > 1e-6:
            raise ValueError('Hyper-connection mixer %s does not fit F16' % name)
        n = self.notes.setdefault('hc_f16', {'values': 0, 'inexact': 0, 'max_abs_error': 0.0})
        n['values'] += int(v.size)
        n['inexact'] += int(np.count_nonzero(err))
        n['max_abs_error'] = max(n['max_abs_error'], float(err.max()))
        return T_F16, h.astype('<f2').tobytes()

    def plan(self):
        """(name, type, dims, producer) in template order, plus expert jobs."""
        tp = self.tp
        items = []
        for name, (kind, dims, _) in tp.tensors.items():
            if name == NGRAM:
                continue
            spec = self.spec(name, kind, dims)
            items.extend(spec)
        return items

    def spec(self, name, kind, dims):
        m = re.match(r'blk\.(\d+)\.(.+)$', name)
        if not m:
            return self.spec_global(name, kind, dims)
        il, rest = int(m.group(1)), m.group(2)
        mtp = il == N_LAYER
        P = 'mtp.layers.0.' if mtp else TRUNK + 'layers.%d.' % il
        D = self.dense_type if not mtp else T_BF16

        def dense(prefix, rows=None, cols=None, perm_rows=None):
            def make():
                d = self.dense(prefix)
                if rows is not None:
                    d = d.take_rows(np.arange(d.rows)[rows])
                if perm_rows is not None:
                    d = d.take_rows(perm_rows)
                if cols is not None:
                    d = d.take_cols(*cols)
                return d
            # MTP and PLE projections are BF16 in the checkpoint and stay BF16
            src = prefix + '.weight'
            t = D if self.ck.info(src)[3] == 'F8_E4M3' else T_BF16
            return [(name, t, dims, lambda: make().encode(t), ('dense', make))]

        def f32(src, plus_one=False, perm=None, fn=None):
            def make():
                v = self.vec(src, plus_one)
                if perm is not None:
                    v = v[perm]
                if fn is not None:
                    v = fn(v)
                return v.astype('<f4').tobytes()
            return [(name, T_F32, dims, make, ('f32', None))]

        def f32_matrix(src, perm=None):
            def make():
                d = self.dense(src)
                v = d.f32()
                if perm is not None:
                    v = v[perm]
                return v.astype('<f4').tobytes()
            return [(name, T_F32, dims, make, ('f32', None))]

        def hc(src):
            return [(name, None, dims, lambda: self.hc_bytes(src), ('hc', None))]

        hcmap = {'hc_attn': 'attn_hyper_connection.', 'hc_ffn': 'mlp_hyper_connection.'}
        for key, hp in hcmap.items():
            if rest == key + '_norm.weight':
                return f32(P + hp + 'hc_norm.weight', plus_one=True)
            if rest == key + '_down.weight':
                return hc(P + hp + 'input_mix_weight_down.weight')
            if rest == key + '_up.weight':
                return hc(P + hp + 'input_mix_weight_up.weight')
            if rest == key + '_inject.weight':
                return hc(P + hp + 'block_inject_weight.weight')
        la = P + 'linear_attn.'
        vq = v_order(HEAD_V)
        v1 = v_order(1)
        if rest == 'attn_qkv.weight':
            return dense(la + 'in_proj_qkv', perm_rows=np.concatenate((np.arange(4096), 4096 + vq)))
        if rest == 'attn_gate.weight':
            return dense(la + 'in_proj_z', perm_rows=vq)
        if rest == 'ssm_out.weight':
            return dense(la + 'out_proj', cols=(v_order(1), HEAD_V))
        if rest == 'ssm_alpha.weight':
            return f32_matrix(la + 'in_proj_a', perm=v1)
        if rest == 'ssm_beta.weight':
            return f32_matrix(la + 'in_proj_b', perm=v1)
        if rest == 'ssm_a':
            return f32(la + 'A_log', perm=v1, fn=lambda v: -np.exp(v.astype(np.float32)))
        if rest == 'ssm_dt.bias':
            return f32(la + 'dt_bias', perm=v1)
        if rest == 'ssm_norm.weight':
            return f32(la + 'norm.weight')
        if rest == 'ssm_conv1d.weight':
            def conv():
                v = self.vec(la + 'conv1d.weight').reshape(10240, 4)
                v = np.concatenate((v[:4096], v[4096:][vq]))
                return v.astype('<f4').tobytes()
            return [(name, T_F32, dims, conv, ('f32', None))]
        sa = P + 'self_attn.'
        simple = {'attn_q.weight': 'q_proj', 'attn_k.weight': 'k_proj', 'attn_v.weight': 'v_proj',
                  'attn_output.weight': 'o_proj'}
        if rest in simple:
            return dense(sa + simple[rest])
        if rest == 'attn_q_norm.weight':
            return f32(sa + 'q_norm.weight', plus_one=True)
        if rest == 'attn_k_norm.weight':
            return f32(sa + 'k_norm.weight', plus_one=True)
        if rest == 'indexer.q_proj.weight':
            return dense(sa + 'indexer.index_qk_proj', rows=slice(0, 512))
        if rest == 'indexer.k_proj.weight':
            return dense(sa + 'indexer.index_qk_proj', rows=slice(512, 640))
        if rest == 'indexer.q_norm.weight':
            return f32(sa + 'indexer.q_layernorm.weight', plus_one=True)
        if rest == 'indexer.k_norm.weight':
            return f32(sa + 'indexer.k_layernorm.weight', plus_one=True)
        mlp = P + 'mlp.'
        if rest == 'ffn_gate_inp.weight':
            return f32_matrix(mlp + 'gate')
        if rest == 'ffn_gate_inp_shexp.weight':
            return f32(mlp + 'shared_expert_gate.weight')
        for proj in ('gate', 'up', 'down'):
            if rest == 'ffn_%s_shexp.weight' % proj:
                return dense(mlp + 'shared_expert.%s_proj' % proj)
            if rest == 'ffn_%s_exps.weight' % proj:
                return self.spec_experts(name, il, proj, mtp)
        ple = P + 'ple.'
        if rest == 'ple_key.weight':
            return dense(ple + 'key_proj')
        if rest == 'ple_value.weight':
            return dense(ple + 'value_proj')
        for k in ('key', 'query', 'conv'):
            if rest == 'ple_norm_%s.weight' % k:
                return f32(ple + 'norm_%s.weight' % k, plus_one=True)
        if rest == 'ple_conv1d.weight':
            return f32(ple + 'conv1d.weight')
        if mtp:
            if rest == 'nextn.eh_proj.weight':
                def eh():
                    e = self.dense('mtp.fc_embedding')
                    h = self.dense('mtp.fc_hidden')
                    return Dense(np.concatenate((e.weight, h.weight), axis=1), None, 'bf16')
                return [(name, T_BF16, dims, lambda: eh().encode(T_BF16), ('dense', eh))]
            if rest == 'nextn.enorm.weight':
                return f32('mtp.pre_fc_norm_embedding.weight', plus_one=True)
            if rest == 'nextn.hnorm.weight':
                return f32('mtp.pre_fc_norm_hidden.weight', plus_one=True)
            if rest == 'nextn.hc_head_norm.weight':
                return f32('mtp.hyper_connection_mixer.hc_norm.weight', plus_one=True)
            if rest == 'nextn.hc_head_down.weight':
                return hc('mtp.hyper_connection_mixer.input_mix_weight_down.weight')
            if rest == 'nextn.hc_head_up.weight':
                return hc('mtp.hyper_connection_mixer.input_mix_weight_up.weight')
        raise ValueError('No checkpoint source for template tensor ' + name)

    def spec_global(self, name, kind, dims):
        mix = TRUNK + 'hyper_connection_mixer.'
        if name == 'token_embd.weight':
            return [(name, T_BF16, dims, lambda: self.ck.read(TRUNK + 'embed_tokens.weight', 'BF16').tobytes(),
                     ('dense', lambda: self.dense(TRUNK + 'embed_tokens')))]
        if name == 'output.weight':
            def head():
                d = self.dense('lm_head')
                if self.output_type == T_BF16:
                    return d.encode(T_BF16)
                return q8_0(d.f32())
            return [(name, self.output_type, dims, head, ('dense', lambda: self.dense('lm_head')))]
        if name == 'output_hc_norm.weight':
            return [(name, T_F32, dims, lambda: self.vec(mix + 'hc_norm.weight', True).astype('<f4').tobytes(),
                     ('f32', None))]
        if name == 'output_hc_down.weight':
            return [(name, None, dims, lambda: self.hc_bytes(mix + 'input_mix_weight_down.weight'), ('hc', None))]
        if name == 'output_hc_up.weight':
            return [(name, None, dims, lambda: self.hc_bytes(mix + 'input_mix_weight_up.weight'), ('hc', None))]
        raise ValueError('No checkpoint source for template tensor ' + name)

    def spec_experts(self, name, il, proj, mtp):
        """The NVFP4 tensor (an expert job) and its global scale tensor."""
        prefix = ('mtp.layers.0.' if mtp else TRUNK + 'layers.%d.' % il) + 'mlp.experts.'
        rows, k = (640, 2560) if proj != 'down' else (2560, 640)
        csf = not mtp
        entries = []
        for e in range(N_EXPERT):
            p = '%s%d.%s_proj.' % (prefix, e, proj)
            ent = {'weight': self.ck.info(p + 'weight'), 'global': self.ck.info(p + 'weight_scale_2')}
            if csf:
                ent['fixed'] = self.ck.info(p + 'weight_scale.nvfp4_csf_fixed')
                ent['exceptions'] = self.ck.info(p + 'weight_scale.nvfp4_csf_exceptions')
            else:
                ent['scale'] = self.ck.info(p + 'weight_scale')
            entries.append(ent)
        dims = [k, rows, N_EXPERT]
        scale_name = name[:-len('.weight')] + '.scale'
        return [(name, T_NVFP4, dims, None, ('experts', (entries, rows, k, csf))),
                (scale_name, T_F32, [N_EXPERT], None, ('scales', name))]


def q8_0(v):
    rows, k = v.shape
    b = v.reshape(rows, k // 32, 32)
    amax = np.abs(b).max(2)
    d = (amax / 127).astype(np.float16)
    df = d.astype(np.float32)
    q = np.where(df[:, :, None] > 0, np.rint(b / np.where(df > 0, df, 1)[:, :, None]), 0)
    q = np.clip(q, -127, 127).astype(np.int8)
    out = np.empty((rows, k // 32, 34), dtype=np.uint8)
    out[:, :, :2] = d.view(np.uint8).reshape(rows, k // 32, 2)
    out[:, :, 2:] = q.view(np.uint8)
    return out.tobytes()


# --- template dequantization (for --check) ----------------------------------------

def template_values(tp, name):
    kind, dims, offset = tp.tensors[name]
    tp.f.seek(tp.data_start + offset)
    raw = tp.f.read(size_of(kind, dims))
    return dequant(kind, raw, dims)


def dequant(kind, raw, dims):
    rows = math.prod(dims[1:]) if len(dims) > 1 else 1
    k = dims[0]
    if kind == T_F32:
        return np.frombuffer(raw, '<f4').reshape(rows, k)
    if kind == T_F16:
        return np.frombuffer(raw, '<f2').astype(np.float32).reshape(rows, k)
    if kind == T_BF16:
        return bf16_to_f32(raw).reshape(rows, k)
    if kind == T_Q8_0:
        b = np.frombuffer(raw, np.uint8).reshape(rows, k // 32, 34)
        d = b[:, :, :2].copy().view('<f2').astype(np.float32)
        return (b[:, :, 2:].view(np.int8).astype(np.float32) * d).reshape(rows, k)
    if kind == T_MXFP4:
        b = np.frombuffer(raw, np.uint8).reshape(rows, k // 32, 17)
        e = b[:, :, 0].astype(np.int32)
        scale = np.where(e == 0, np.float32(2.0 ** -127), np.ldexp(np.float32(1), e - 127)).astype(np.float32)
        q = np.concatenate((b[:, :, 1:] & 15, b[:, :, 1:] >> 4), axis=2)
        return (E2M1[q] * scale[:, :, None]).reshape(rows, k)
    if kind == T_Q4_K:
        b = np.frombuffer(raw, np.uint8).reshape(rows, k // 256, 144)
        d = b[:, :, 0:2].copy().view('<f2').astype(np.float32)[:, :, 0]
        dmin = b[:, :, 2:4].copy().view('<f2').astype(np.float32)[:, :, 0]
        s = b[:, :, 4:16].astype(np.int32)
        sc = np.empty(b.shape[:2] + (8,), np.int32)
        mn = np.empty_like(sc)
        for j in range(8):
            if j < 4:
                sc[:, :, j] = s[:, :, j] & 63
                mn[:, :, j] = s[:, :, j + 4] & 63
            else:
                sc[:, :, j] = (s[:, :, j + 4] & 15) | ((s[:, :, j - 4] >> 6) << 4)
                mn[:, :, j] = (s[:, :, j + 4] >> 4) | ((s[:, :, j] >> 6) << 4)
        qs = b[:, :, 16:].reshape(rows, k // 256, 4, 32)
        q = np.stack((qs & 15, qs >> 4), axis=3).reshape(rows, k // 256, 8, 32)
        v = d[:, :, None, None] * sc[:, :, :, None] * q - dmin[:, :, None, None] * mn[:, :, :, None]
        return v.reshape(rows, k).astype(np.float32)
    raise ValueError('No dequantizer for template type %d' % kind)


def compare(new, old):
    new = new.astype(np.float64).reshape(-1)
    old = old.astype(np.float64).reshape(-1)
    diff = new - old
    rel = math.sqrt((diff * diff).sum() / max((old * old).sum(), 1e-300))
    exact = float(np.mean(new == old))
    cos = float(new @ old / max(math.sqrt((new * new).sum() * (old * old).sum()), 1e-300))
    return rel, exact, cos


def check(conv, tp, ck, sample_experts):
    print('%-40s %-8s %-8s %12s %8s %10s' % ('tensor', 'old', 'new', 'rel_rms', 'exact', 'cos'), flush=True)
    worst = {}
    plan = []
    for tname, (tkind, tdims, _) in tp.tensors.items():
        if tname == NGRAM:
            continue
        try:
            plan.extend(conv.spec(tname, tkind, tdims))
        except KeyError as missing:
            print('%-40s source not available: %s' % (tname, missing), flush=True)
    for name, kind, dims, make, (role, extra) in plan:
        old_kind = tp.tensors[name][0] if name in tp.tensors else None
        if role == 'scales':
            continue
        if role == 'experts':
            m = re.match(r'blk\.(\d+)\.', name)
            if int(m.group(1)) not in sample_experts:
                continue
            entries, rows, k, csf = extra
            old_kind, old_dims, offset = tp.tensors[name]
            per = size_of(old_kind, [k, rows])
            stats = []
            for e in (0, 137, 511):
                ent = entries[e]
                weight = read_entry(ent['weight'])
                scale = csf_decode(read_entry(ent['fixed']), read_entry(ent['exceptions']), rows, k // 16) \
                    if csf else read_entry(ent['scale'])
                new = nvfp4_dequant(weight, scale, float(read_entry(ent['global']).reshape(-1)[0]))
                tp.f.seek(tp.data_start + offset + e * per)
                old = dequant(old_kind, tp.f.read(per), [k, rows])
                stats.append(compare(new, old))
            rel = max(s[0] for s in stats)
            cos = min(s[2] for s in stats)
            print('%-40s %-8d %-8d %12.5f %8s %10.6f' % (name, old_kind, kind, rel, '-', cos), flush=True)
            continue
        try:
            if role == 'hc':
                kind, data = make()
            else:
                data = make()
        except KeyError as missing:
            print('%-40s source not available: %s' % (name, missing), flush=True)
            continue
        new = mxfp8_dequant_blocks(data, dims) if kind == T_MXFP8 else dequant(kind, data, dims)
        old = template_values(tp, name)
        rel, exact, cos = compare(new, old)
        print('%-40s %-8d %-8d %12.5g %8.4f %10.6f' % (name, old_kind, kind, rel, exact, cos), flush=True)
        key = re.sub(r'blk\.\d+\.', 'blk.N.', name)
        worst[key] = max(worst.get(key, (0, 1, 1)), (rel, exact, cos))
    print('\nworst per tensor family:')
    for key, (rel, exact, cos) in sorted(worst.items()):
        print('%-40s %12.5g %8.4f %10.6f' % (key, rel, exact, cos))
    if conv.notes:
        print(json.dumps(conv.notes, indent=1))


def mxfp8_dequant_blocks(data, dims):
    k = dims[0]
    rows = math.prod(dims[1:])
    b = np.frombuffer(data, np.uint8).reshape(rows, k // 128, 132)
    scale = b[:, :, :4].reshape(rows, k // 32)
    weight = b[:, :, 4:].reshape(rows, k)
    return mxfp8_dequant(weight, scale)


# --- writer ----------------------------------------------------------------------

def convert(args):
    ck = Checkpoint(args.src, allow_missing=args.check is not None)
    tp = Reader(str(args.template))
    if tp.kv.get('general.architecture') != 'qwen4exp' or 'blk.%d.nextn.eh_proj.weight' % N_LAYER not in tp.tensors:
        raise ValueError('The template must be a qwen4exp GGUF with the MTP block')
    dense_type = {'mxfp8': T_MXFP8, 'bf16': T_BF16}[args.dense]
    output_type = {'bf16': T_BF16, 'q8_0': T_Q8_0}[args.output]
    conv = Converter(ck, tp, dense_type, output_type)
    if args.check is not None:
        check(conv, tp, ck, set(args.check))
        return

    out = Path(args.out)
    pending = Path(str(out) + '.incomplete')
    if out.exists() or pending.exists():
        raise ValueError('Output already exists; choose a new path or inspect the incomplete file')
    if not re.fullmatch('[0-9a-f]{40}', args.source_revision):
        raise ValueError('--source-revision must be an immutable 40-character commit SHA')

    # n-gram geometry and hash constants must match the template
    ple = TRUNK + 'layers.1.ple.ple_embedding.'
    for src, key in (('layer_multipliers', 'layer_multipliers'), ('ngram_heads_offsets', 'head_offsets'),
                     ('ngram_heads_vocab_sizes', 'head_vocab_sizes')):
        if [int(x) for x in ck.read(ple + src, 'I64')] != list(tp.kv['qwen4exp.ple.' + key].values):
            raise ValueError('Checkpoint n-gram hash constants differ from the template: ' + src)
    shards = sorted((int(n.split('.shard_')[1].split('.')[0]), n) for n in ck.entries
                    if n.startswith(ple + 'ngram_embedding.shard_') and n.endswith('.weight'))
    if [i for i, _ in shards] != list(range(len(shards))):
        raise ValueError('Missing or duplicate n-gram shards')
    rows = sum(ck.info(n)[4][0] for _, n in shards)
    if rows != tp.kv['qwen4exp.ple.row_count']:
        raise ValueError('N-gram row count %d differs from the template' % rows)
    gscale = ck.read(ple + 'ngram_embedding.weight_scale_2', 'F32').reshape(-1)
    if gscale.size != 1 or not (np.isfinite(gscale).all() and gscale[0] > 0):
        raise ValueError('Invalid n-gram global scale')

    items = conv.plan()
    # hyper-connection types depend on the values: resolve them up front
    resolved = []
    for name, kind, dims, make, role in items:
        if role[0] == 'hc':
            kind, data = make()
            resolved.append((name, kind, dims, (lambda d=data: d), role))
        else:
            resolved.append((name, kind, dims, make, role))
    items = resolved

    metadata = {k: (tp.kv_types[k], v) for k, v in tp.kv.items()}
    metadata['general.name'] = (8, 'Qwen3.8-Flash-Next NVFP4 QAD (%s)' % SOURCE_REPOSITORY)
    metadata['qwen4exp.ple.ngram_encoding'] = (8, NGRAM_ENCODING)
    metadata['qwen4exp.ple.ngram_global_scale'] = (6, float(gscale[0]))
    metadata['ds4.qwen4.ngram.source_revision'] = (8, args.source_revision)
    metadata['ds4.qwen4.ngram.source_repository'] = (8, SOURCE_REPOSITORY)
    metadata['ds4.qwen4.source_revision'] = (8, args.source_revision)
    metadata['ds4.qwen4.source_repository'] = (8, SOURCE_REPOSITORY)
    alignment = tp.kv.get('general.alignment', 32)
    align = lambda n, a=alignment: (n + a - 1) // a * a
    layout, offset = [], 0
    for name, kind, dims, make, role in items:
        size = size_of(kind, dims)
        layout.append((name, kind, dims, offset, size, make, role))
        offset = align(offset + size)
    row_bytes = 90
    prefix = b'GGUF' + struct.pack('<IQQ', 3, len(layout) + 1, len(metadata))
    prefix += b''.join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())

    def header(ngram_offset):
        data = bytearray(prefix)
        for name, kind, dims, pos in [(n, k, d, o) for n, k, d, o, _, _, _ in layout] + \
                [(NGRAM, T_I8, [row_bytes, rows], ngram_offset)]:
            data += w_str(name) + struct.pack('<I', len(dims))
            data += struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', kind, pos)
        return data

    data_start = align(len(header(0)))
    table_start = align(data_start + offset, max(65536, alignment))
    total = table_start + rows * row_bytes
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < total + (16 << 30):
        raise ValueError('Not enough disk space with a 16 GiB reserve')
    print('Writing %s: %.2f GB main weights, %.2f GB n-grams' % (pending, offset / 1e9, rows * row_bytes / 1e9),
          flush=True)
    with pending.open('xb') as dst:
        dst.write(header(table_start - data_start))
        dst.truncate(total)
    records = {}
    t0 = time.time()
    jobs, job_names = [], []
    for name, kind, dims, pos, size, make, role in layout:
        if role[0] == 'experts':
            entries, r, k, csf = role[1]
            jobs.append((entries, r, k, str(pending), data_start + pos, csf))
            job_names.append(name)
    scales = {}
    exceptions = 0
    with Pool(args.workers) as pool:
        expert_results = pool.imap(expert_job, jobs)
        # the small tensors run here while the pool repacks experts
        for name, kind, dims, pos, size, make, role in layout:
            if role[0] in ('experts', 'scales'):
                continue
            data = make()
            if len(data) != size:
                raise ValueError('%s produced %d bytes, expected %d' % (name, len(data), size))
            write_region(str(pending), data_start + pos, data)
            records[name] = dict(offset=data_start + pos, bytes=size, sha256=hashlib.sha256(data).hexdigest())
        print('Dense and control tensors written (%.0f s)' % (time.time() - t0), flush=True)
        for i, (digest, g, exc) in enumerate(expert_results):
            name = job_names[i]
            scales[name] = g
            exceptions += exc
            pos = [p for n, _, _, p, _, _, _ in layout if n == name][0]
            records[name] = dict(offset=data_start + pos, bytes=len(jobs[i][0]) * jobs[i][1] * jobs[i][2] // 64 * 36,
                                 sha256=digest)
            if i % 12 == 0 or i == len(jobs) - 1:
                print('Experts %d/%d (%.0f s)' % (i + 1, len(jobs), time.time() - t0), flush=True)
        for name, kind, dims, pos, size, make, role in layout:
            if role[0] != 'scales':
                continue
            data = scales[role[1]]
            write_region(str(pending), data_start + pos, data)
            records[name] = dict(offset=data_start + pos, bytes=size, sha256=hashlib.sha256(data).hexdigest())
        ngram_jobs, at = [], table_start
        for _, n in shards:
            base = n[:-len('.weight')]
            ngram_jobs.append((ck.info(n), ck.info(base + '.weight_scale'), str(pending), at))
            at += ck.info(n)[4][0] * row_bytes
        digest = []
        for i, (d, r) in enumerate(pool.imap(ngram_job, ngram_jobs)):
            digest.append(d)
            if i % 16 == 0 or i == len(ngram_jobs) - 1:
                print('N-gram shards %d/%d (%.0f s)' % (i + 1, len(ngram_jobs), time.time() - t0), flush=True)
    with pending.open('rb+') as dst:
        os.fsync(dst.fileno())
    print('Verifying every output tensor', flush=True)
    with pending.open('rb') as check_fp:
        for name, rec in records.items():
            check_fp.seek(rec['offset'])
            if hashlib.sha256(check_fp.read(rec['bytes'])).hexdigest() != rec['sha256']:
                raise ValueError('Output payload verification failed: ' + name)
        at = table_start
        for i, (_, n) in enumerate(shards):
            size = ck.info(n)[4][0] * row_bytes
            check_fp.seek(at)
            if hashlib.sha256(check_fp.read(size)).hexdigest() != digest[i]:
                raise ValueError('Output n-gram verification failed: shard %d' % i)
            at += size
    reread = Reader(str(pending))
    try:
        if reread.tensors[NGRAM] != (T_I8, [row_bytes, rows], table_start - data_start):
            raise ValueError('Incorrect output n-gram header')
    finally:
        reread.f.close()
    report = dict(source=SOURCE_REPOSITORY, source_revision=args.source_revision, template=str(args.template),
                  dense=args.dense, output=args.output, bytes=total, csf_exceptions=exceptions,
                  notes=conv.notes, tensors=records, ngram_shard_sha256=digest)
    Path(str(out) + '.json').write_text(json.dumps(report, indent=1) + '\n')
    pending.rename(out)
    print('Verified %s (%.2f GB, %.0f s)' % (out, total / 1e9, time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True, help='checkpoint directory (index and shards)')
    ap.add_argument('--template', type=Path, required=True, help='DS4 qwen4exp GGUF with MTP (metadata only)')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--source-revision', default='')
    ap.add_argument('--dense', choices=['mxfp8', 'bf16'], default='mxfp8',
                    help='MXFP8 projections: DS4 MXFP8 blocks, or exact BF16')
    ap.add_argument('--output', choices=['bf16', 'q8_0'], default='bf16',
                    help='output head type (BF16 keeps the checkpoint exactly)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--check', type=int, nargs='*', metavar='LAYER',
                    help='compare with the template instead of writing; experts of the given layers')
    args = ap.parse_args()
    if args.check is None and not args.out:
        ap.error('--out is required unless --check is given')
    try:
        convert(args)
    except (OSError, ValueError, KeyError) as error:
        ap.exit(1, '%s: %s\n' % (type(error).__name__, error))


if __name__ == '__main__':
    main()
