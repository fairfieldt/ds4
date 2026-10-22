#!/usr/bin/env python3
"""merge_split_gguf.py on synthetic split parts with a stubbed Hub."""

import hashlib
import io
import json
import os
import random
import shutil
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import merge_split_gguf as m


def kv_u32(k, v):
    return m.gguf_string(k) + struct.pack("<II", 4, v)


def kv_u16(k, v):
    return m.gguf_string(k) + struct.pack("<IH", 2, v)


def kv_str(k, v):
    return m.gguf_string(k) + struct.pack("<I", 8) + m.gguf_string(v)


def write_gguf(path, kvs, tensors, align):
    hdr = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kvs)))
    for kv in kvs:
        hdr += kv
    off, offs = 0, []
    for name, typ, dims, data in tensors:
        off = (off + align - 1) // align * align
        offs.append(off)
        hdr += m.gguf_string(name) + struct.pack("<I", len(dims))
        hdr += b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", typ, off)
        off += len(data)
    data_start = (len(hdr) + align - 1) // align * align
    with open(path, "wb") as f:
        f.write(hdr + bytes(data_start - len(hdr)))
        for (_, _, _, data), o in zip(tensors, offs):
            f.seek(data_start + o)
            f.write(data)


class BytesResponse(io.BytesIO):
    def __enter__(self):
        return self


class MergeSplitTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.hub = os.path.join(self.dir, "hub")
        os.makedirs(self.hub)
        self.rnd = random.Random(7)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def tensor(self, name, typ, dims):
        blk, bb = m.TYPES[typ]
        n = 1
        for d in dims:
            n *= d
        return (name, typ, dims, bytes(self.rnd.getrandbits(8) for _ in range(n // blk * bb)))

    def test_merge_with_donor(self):
        t1 = [self.tensor("token_embd.weight", 1, [64, 10]), self.tensor("blk.0.ffn_gate_exps.weight", 39, [64, 3, 4])]
        t2 = [self.tensor("blk.0.ffn_down_exps.weight", 39, [96, 5, 4]), self.tensor("blk.1.engram_kv.weight", 1, [8, 7])]
        t3 = [self.tensor("blk.1.engram_embd.weight", 24, [33, 1001])]
        t4 = [self.tensor("blk.14.engram_embd.weight", 24, [33, 999])]
        names = [f"Fake-{i:05d}-of-00004.gguf" for i in range(1, 5)]
        split = lambda i: [kv_u16("split.no", i), kv_u32("split.count", 4), kv_u32("split.tensors.count", 6)]
        write_gguf(os.path.join(self.hub, names[0]),
                   [kv_str("general.architecture", "deepseek41"), kv_u32("general.alignment", 64)] + split(0), t1, 64)
        write_gguf(os.path.join(self.hub, names[1]), split(1), t2, 32)
        write_gguf(os.path.join(self.hub, names[2]), split(2), t3, 32)
        write_gguf(os.path.join(self.hub, names[3]), split(3), t4, 32)
        # The donor holds layer 1's table and a different layer 14 table.
        donor = os.path.join(self.dir, "donor.gguf")
        write_gguf(donor, [kv_str("general.architecture", "deepseek41"), kv_u32("general.alignment", 16384)],
                   [self.tensor("x", 0, [4]), t3[0], self.tensor("blk.14.engram_embd.weight", 24, [33, 999])], 16384)
        tree = []
        for n in names:
            data = open(os.path.join(self.hub, n), "rb").read()
            tree.append({"type": "file", "path": n, "size": len(data),
                         "lfs": {"oid": hashlib.sha256(data).hexdigest()}})
        downloads = []

        def urlopen(req):
            url = req if isinstance(req, str) else req.full_url
            if "/api/models/" in url:
                return BytesResponse(json.dumps(tree).encode())
            lo, hi = map(int, req.headers["Range"].split("=")[1].split("-"))
            with open(os.path.join(self.hub, url.rsplit("/", 1)[1]), "rb") as f:
                f.seek(lo)
                return BytesResponse(f.read(hi - lo + 1))

        def run(cmd, check, env, stdout):
            downloads.append(cmd[3])
            shutil.copy(os.path.join(self.hub, cmd[3]), os.path.join(cmd[5], cmd[3]))
            return types.SimpleNamespace(returncode=0)

        out = os.path.join(self.dir, "merged.gguf")
        tmp = os.path.join(self.dir, "parts")
        argv = ["merge_split_gguf.py", "fake/repo", "Fake", "4", out, tmp, "--donor", donor]
        with mock.patch.object(m.urllib.request, "urlopen", urlopen), \
                mock.patch.object(m.subprocess, "run", run), mock.patch.object(sys, "argv", argv):
            m.main()
        self.assertEqual(downloads, [names[0], names[1], names[3]])
        self.assertFalse([f for f in os.listdir(tmp) if f.endswith(".gguf")])

        b = open(out, "rb").read()
        kvs, tensors, data = m.parse_header(b)
        self.assertEqual([k for k, _ in kvs], ["general.architecture", "general.alignment"])
        self.assertEqual(data % 16384, 0)
        self.assertEqual([t["name"] for t in tensors][-2:], ["blk.1.engram_embd.weight", "blk.14.engram_embd.weight"])
        want = {n: d for n, _, _, d in t1 + t2 + t3 + t4}
        for t in tensors:
            self.assertEqual(t["off"] % 16384, 0, t["name"])
            self.assertEqual(b[t["pos"]:t["pos"] + t["bytes"]], want[t["name"]], t["name"])
        self.assertEqual(len(b), tensors[-1]["pos"] + tensors[-1]["bytes"])

        # A finished merge is not overwritten by a rerun.
        with mock.patch.object(m.urllib.request, "urlopen", urlopen), \
                mock.patch.object(m.subprocess, "run", run), mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                m.main()
        self.assertEqual(open(out, "rb").read(), b)


if __name__ == "__main__":
    unittest.main()
