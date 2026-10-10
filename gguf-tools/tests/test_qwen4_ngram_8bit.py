"""8-bit Qwen n-grams: nearest-value rounding and an unchanged main payload."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen4_ngram_8bit as q8
import qwen4_native_ngrams as pack
from qwen4_pack_to_qwen4exp import Arr, Reader, kv_bytes, w_str


class E4M3(unittest.TestCase):
    def test_codes_round_trip(self):
        values = q8.e4m3_values()
        codes = [c for c in range(256) if c & 127 != 127 and c != 128]
        self.assertEqual(q8.e4m3_codes(values[codes]).tolist(), codes)
        self.assertEqual(values[0x7e], 448)
        self.assertEqual(values[1], 2.0 ** -9)

    def test_nearest_even(self):
        values = q8.e4m3_values()
        finite = np.sort(values[~np.isnan(values)])
        rng = np.random.default_rng(3)
        y = (rng.choice([-1, 1], 200000) * np.exp(rng.uniform(np.log(1e-4), np.log(448), 200000))).astype(np.float32)
        mid = ((finite[1:] + finite[:-1]) / 2).astype(np.float32)
        y = np.concatenate([y, mid, np.float32([0, 448, -448])])
        got = values[q8.e4m3_codes(y)]
        best = np.abs(finite[None, :] - y[:, None]).min(1)
        self.assertTrue((np.abs(got - y) == best).all())
        # Ties go to the even code.
        tie = q8.e4m3_codes(mid[mid > 0])
        self.assertTrue((tie % 2 == 0).all())


class Convert(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.model = self.root/'model.gguf'
        metadata = {
            'general.architecture': (8, 'qwen4exp'),
            'general.alignment': (4, 32),
            'qwen4exp.ple.row_dimension': (4, 160),
            'qwen4exp.ple.row_count': (10, 6),
            'qwen4exp.ple.head_offsets': (9, Arr(10, [0, 2])),
            'qwen4exp.ple.head_vocab_sizes': (9, Arr(10, [2, 4]))}
        rng = np.random.default_rng(7)
        self.x = (rng.standard_normal((6, 160)) * 0.006).astype(np.float32)
        self.x[1] = 0
        bf16 = (self.x.view(np.uint32) >> 16).astype('<u2')
        self.x = (bf16.astype(np.uint32) << 16).view(np.float32)
        self.payload = bytes(range(66))
        header = b'GGUF' + struct.pack('<IQQ', 3, 2, len(metadata))
        header += b''.join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())
        header += w_str('blk.0.ffn_gate_exps.weight') + struct.pack('<I3QIQ', 3, 256, 1, 1, 16, 0)
        header += w_str(pack.NGRAM) + struct.pack('<I2QIQ', 2, 160, 6, 30, 96)
        header += bytes((-len(header)) % 32)
        self.model.write_bytes(header + self.payload + bytes(30) + bf16.tobytes())

    def check(self, encoding, tolerance):
        out = self.root/(encoding + '.gguf')
        report = q8.convert(self.model, out, encoding, 2)
        reader = Reader(str(out))
        self.addCleanup(reader.f.close)
        self.assertEqual(reader.kv['qwen4exp.ple.ngram_encoding'], q8.ENCODINGS[encoding][0])
        kind, dims, offset = reader.tensors[pack.NGRAM]
        self.assertEqual((kind, dims), (24, [164, 6]))
        self.assertEqual((reader.data_start+offset) % 65536, 0)
        reader.f.seek(reader.data_start+offset)
        rows = np.frombuffer(reader.f.read(), dtype=np.uint8).reshape(6, 164)
        scale = rows[:, :4].copy().view('<f4')[:, 0]
        if encoding == 'e4m3':
            decoded = q8.e4m3_values()[rows[:, 4:]] * scale[:, None]
        else:
            decoded = rows[:, 4:].view(np.int8).astype(np.float32) * scale[:, None]
        self.assertTrue((decoded[1] == 0).all())
        rel = np.linalg.norm(decoded - self.x) / np.linalg.norm(self.x)
        self.assertLess(rel, tolerance)
        self.assertAlmostEqual(report['error']['relative_rms'], rel, places=6)
        kind, dims, offset = reader.tensors['blk.0.ffn_gate_exps.weight']
        reader.f.seek(reader.data_start+offset)
        self.assertEqual(reader.f.read(len(self.payload)), self.payload)
        with self.assertRaisesRegex(ValueError, 'BF16'):
            q8.convert(out, self.root/'again.gguf', encoding, 1)

    def test_e4m3(self):
        self.check('e4m3', 0.035)

    def test_i8(self):
        self.check('i8', 0.009)


if __name__ == '__main__':
    unittest.main()
