#!/usr/bin/env python3
"""Download a llama.cpp-split GGUF from Hugging Face and merge it into one DwarfStar GGUF.

DwarfStar maps one GGUF file. Some releases, such as antirez's DeepSeek V4.1
Flash 1017 MXFP4 experts, ship as llama.cpp split parts instead.

Usage:
  merge_split_gguf.py REPO PREFIX PARTS OUT TMPDIR [--hf HF_CLI] [--donor GGUF]

Example:
  merge_split_gguf.py antirez/deepseek-v4.1-flash-gguf \
      DeepSeek-V4.1-Flash-1017-MXFP4-experts 9 \
      gguf/DeepSeek-V4.1-Flash-1017-MXFP4.gguf gguf/mxfp4-parts \
      --donor gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf

PREFIX is the shard name without "-0000N-of-0000M.gguf". The merged file takes
part 1's metadata without the split.* keys, uses general.alignment = 16384 with
every tensor 16 KiB aligned (like the official V4.1 GGUFs), and places the
Engram tables last. Parts are downloaded one at a time into TMPDIR, hashed
against the Hub's SHA-256 while their tensors are copied, then deleted, so the
peak extra disk use is one part. Rerunning the command resumes after the last
merged part. The `hf` command comes from the huggingface_hub package.

--donor names a local GGUF that may already hold some parts' tensors with the
same names, types and shapes (the Engram tables of another quant of the same
checkpoint). A part is rebuilt from the donor when the rebuilt bytes have the
Hub's SHA-256; otherwise it is downloaded.

After the last part, every tensor of the merged file is re-read and compared
with the SHA-256 of its source bytes.
"""
import hashlib
import json
import os
import struct
import subprocess
import sys
import urllib.request

ALIGN = 16384
CHUNK = 64 << 20
# ggml type -> (block size, bytes per block)
TYPES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 24: (1, 1), 30: (1, 2), 39: (32, 17), 10: (256, 84),
         12: (256, 144), 16: (256, 66)}


class Reader:
    def __init__(self, b):
        self.b, self.o = b, 0

    def u(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.o)
        self.o += struct.calcsize(fmt)
        return v[0]

    def s(self):
        n = self.u("Q")
        v = self.b[self.o:self.o + n].decode("utf-8")
        self.o += n
        return v

    def skip_val(self, t):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in sizes:
            self.o += sizes[t]
        elif t == 8:
            self.s()
        elif t == 9:
            et, n = self.u("I"), self.u("Q")
            if et in sizes:
                self.o += sizes[et] * n
            else:
                for _ in range(n):
                    self.skip_val(et)
        else:
            raise ValueError(f"unknown GGUF value type {t}")


def parse_header(b):
    """Returns (kvs, tensors, data_start). Tensor "pos" is absolute in the file."""
    r = Reader(b)
    assert r.u("I") == 0x46554747, "not a GGUF"
    version, nt, nkv = r.u("I"), r.u("Q"), r.u("Q")
    assert version == 3
    kvs, align = [], 32
    for _ in range(nkv):
        start = r.o
        key = r.s()
        t = r.u("I")
        vstart = r.o
        r.skip_val(t)
        if key == "general.alignment":
            align = struct.unpack_from("<I", b, vstart)[0]
        kvs.append((key, b[start:r.o]))
    tensors = []
    for _ in range(nt):
        name = r.s()
        dims = [r.u("Q") for _ in range(r.u("I"))]
        typ, off = r.u("I"), r.u("Q")
        if typ not in TYPES:
            raise ValueError(f"{name}: ggml type {typ} is not supported")
        blk, bb = TYPES[typ]
        n = 1
        for d in dims:
            n *= d
        assert n % blk == 0, name
        tensors.append({"name": name, "dims": dims, "type": typ, "off": off, "bytes": n // blk * bb})
    data = (r.o + align - 1) // align * align
    for t in tensors:
        t["pos"] = data + t["off"]
    return kvs, tensors, data


def gguf_string(s):
    e = s.encode()
    return struct.pack("<Q", len(e)) + e


def fetch_head(url, nbytes=CHUNK):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    with urllib.request.urlopen(req) as f:
        return f.read()


def copy_stream(blocks, spans, fd, h, digests):
    """blocks yields (file_pos, bytes) in order. spans: sorted (src_pos, size, dst_pos, name).
    Writes each span's bytes to fd at dst_pos, hashes everything into h and each span
    into digests[name]."""
    si, sh = 0, None
    for lo, buf in blocks:
        h.update(buf)
        hi = lo + len(buf)
        while si < len(spans) and spans[si][0] < hi:
            src, size, dst, name = spans[si]
            a, b = max(src, lo), min(src + size, hi)
            if b > a:
                if sh is None:
                    sh = hashlib.sha256()
                piece = memoryview(buf)[a - lo:b - lo]
                sh.update(piece)
                os.pwrite(fd, piece, dst + (a - src))
            if src + size <= hi:
                digests[name] = sh.hexdigest() if sh else hashlib.sha256().hexdigest()
                sh = None
                si += 1
            else:
                break
    assert si == len(spans), "source ended before its last tensor"


def file_blocks(path):
    with open(path, "rb", buffering=0) as f:
        pos = 0
        while True:
            buf = f.read(CHUNK)
            if not buf:
                return
            yield pos, buf
            pos += len(buf)


def donor_blocks(head, tensors, size, donor_fd, donor_tensors):
    """The bytes a part file would have if its tensors came from the donor: the part's own
    header, zero padding, and the donor's tensor bytes."""
    yield 0, head
    pos = len(head)
    for t in sorted(tensors, key=lambda t: t["pos"]):
        if t["pos"] > pos:
            yield pos, bytes(t["pos"] - pos)
            pos = t["pos"]
        d = donor_tensors[t["name"]]
        done = 0
        while done < t["bytes"]:
            n = min(CHUNK, t["bytes"] - done)
            buf = os.pread(donor_fd, n, d["pos"] + done)
            assert len(buf) == n
            yield pos, buf
            pos += n
            done += n
    if size > pos:
        yield pos, bytes(size - pos)


def drop_cache(fd):
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def main():
    repo, prefix, parts, out, tmp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    hf = sys.argv[sys.argv.index("--hf") + 1] if "--hf" in sys.argv else "hf"
    donor = sys.argv[sys.argv.index("--donor") + 1] if "--donor" in sys.argv else None
    names = [f"{prefix}-{i:05d}-of-{parts:05d}.gguf" for i in range(1, parts + 1)]
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main"))
    meta = {f["path"]: f for f in tree}
    heads = []
    for n in names:
        b = fetch_head(f"https://huggingface.co/{repo}/resolve/main/{n}")
        kvs, tensors, data = parse_header(b)
        end = max(t["pos"] + t["bytes"] for t in tensors)
        assert end <= meta[n]["size"], f"{n}: tensors past the end of the file"
        heads.append((kvs, tensors, b[:data]))
        print(f"{n}: {len(tensors)} tensors, data at {data}, {meta[n]['size'] / 2**30:.2f} GiB", flush=True)
    donor_fd, donor_tensors = None, {}
    if donor:
        donor_fd = os.open(donor, os.O_RDONLY)
        _, dt, _ = parse_header(os.pread(donor_fd, CHUNK, 0))
        donor_tensors = {t["name"]: t for t in dt}
    kvs = [kv for kv in heads[0][0] if not kv[0].startswith("split.") and kv[0] != "general.alignment"]
    kvs.append(("general.alignment", gguf_string("general.alignment") + struct.pack("<II", 4, ALIGN)))
    order = [t for _, tensors, _ in heads for t in tensors]
    order = [t for t in order if "engram_embd" not in t["name"]] + [t for t in order if "engram_embd" in t["name"]]
    assert len({t["name"] for t in order}) == len(order), "duplicate tensor names"
    cur = 0
    for t in order:
        cur = (cur + ALIGN - 1) // ALIGN * ALIGN
        t["new"] = cur
        cur += t["bytes"]
    header = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(order), len(kvs)))
    for _, raw in kvs:
        header += raw
    for t in order:
        header += gguf_string(t["name"]) + struct.pack("<I", len(t["dims"]))
        header += b"".join(struct.pack("<Q", d) for d in t["dims"])
        header += struct.pack("<IQ", t["type"], t["new"])
    data_start = (len(header) + ALIGN - 1) // ALIGN * ALIGN
    total = data_start + cur
    print(f"merged: {len(order)} tensors, {len(kvs)} keys, data at {data_start}, {total / 2**30:.2f} GiB", flush=True)
    state_path = out + ".state"
    state = json.load(open(state_path)) if os.path.exists(state_path) else None
    if state is None and os.path.exists(out):
        sys.exit(f"{out} exists without {state_path}; remove it to merge again")
    fd = os.open(out, os.O_RDWR | os.O_CREAT, 0o644)
    if state is None:
        os.ftruncate(fd, total)
        os.pwrite(fd, bytes(header) + bytes(data_start - len(header)), 0)
        state = {"done": [], "digests": {}}
    os.makedirs(tmp, exist_ok=True)
    for pi, n in enumerate(names):
        if n in state["done"]:
            continue
        _, tensors, head = heads[pi]
        size, want = meta[n]["size"], meta[n]["lfs"]["oid"]
        spans = sorted((t["pos"], t["bytes"], data_start + t["new"], t["name"]) for t in tensors)
        digests = {}
        ok = False
        if donor_fd is not None and all(
                t["name"] in donor_tensors and donor_tensors[t["name"]]["type"] == t["type"]
                and donor_tensors[t["name"]]["dims"] == t["dims"] for t in tensors):
            h = hashlib.sha256()
            copy_stream(donor_blocks(head, tensors, size, donor_fd, donor_tensors), spans, fd, h, digests)
            ok = h.hexdigest() == want
            print(f"{n}: donor rebuild {'matches' if ok else 'differs from'} the Hub SHA-256", flush=True)
        if not ok:
            path = os.path.join(tmp, n)
            if not (os.path.exists(path) and os.path.getsize(path) == size):
                env = dict(os.environ, HF_XET_HIGH_PERFORMANCE="1")
                subprocess.run([hf, "download", repo, n, "--local-dir", tmp], check=True, env=env,
                               stdout=subprocess.DEVNULL)
            h = hashlib.sha256()
            digests = {}
            copy_stream(file_blocks(path), spans, fd, h, digests)
            if h.hexdigest() != want:
                sys.exit(f"{n}: SHA-256 {h.hexdigest()} != Hub {want}; keeping the part")
            os.remove(path)
        os.fsync(fd)
        drop_cache(fd)
        state["done"].append(n)
        state["digests"].update(digests)
        with open(state_path + ".new", "w") as f:
            json.dump(state, f)
        os.replace(state_path + ".new", state_path)
        print(f"{n}: verified and merged", flush=True)
    bad = 0
    for t in order:
        h = hashlib.sha256()
        done = 0
        while done < t["bytes"]:
            k = min(CHUNK, t["bytes"] - done)
            h.update(os.pread(fd, k, data_start + t["new"] + done))
            done += k
        if h.hexdigest() != state["digests"][t["name"]]:
            print(f"{t['name']}: merged bytes differ from the source", flush=True)
            bad += 1
    drop_cache(fd)
    os.close(fd)
    if bad:
        sys.exit(f"{bad} tensors differ")
    with open(out + ".sha256-tensors.json", "w") as f:
        json.dump(state["digests"], f, indent=0, sort_keys=True)
    os.remove(state_path)
    print(f"merged GGUF complete: {len(order)} tensors re-read and match their sources", flush=True)


if __name__ == "__main__":
    main()
