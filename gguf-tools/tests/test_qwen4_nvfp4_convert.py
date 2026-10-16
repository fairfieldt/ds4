"""NVFP4 checkpoint conversion: CSF scale decoding and the NVFP4/MXFP8 repacks keep every value."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen4_nvfp4_convert as nv


def csf_encode(scales):
    """Independent CSF encoder (the checkpoint's 16-row slab layout): each row's
    base is the lowest start of a 16-byte window holding the most of its bytes."""
    rows, cols = scales.shape
    bases = np.empty(rows, dtype=np.uint8)
    offsets = np.zeros((rows, cols), dtype=np.uint8)
    words = []
    for r in range(rows):
        row = scales[r].astype(np.int32)
        cover = [np.count_nonzero((row >= b) & (row < b + 16)) for b in range(241)]
        base = int(np.argmax(cover))
        bases[r] = base
        for c in range(cols):
            if base <= row[c] < base + 16:
                offsets[r, c] = row[c] - base
            else:
                words.append((r * cols + c) | (int(row[c]) << 24))
    packed = offsets[:, 0::2] | (offsets[:, 1::2] << 4)
    fixed = np.concatenate((bases.reshape(rows // 16, 16), packed.reshape(rows // 16, -1)), axis=1)
    return fixed.reshape(-1).astype(np.uint8), np.array(words, dtype=np.uint32)


class Csf(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(7)
        for rows, cols in ((640, 160), (2560, 40), (32, 4)):
            scales = (rng.integers(0, 13, (rows, cols)) + rng.integers(40, 60, (rows, 1))).astype(np.uint8)
            scales[rng.random((rows, cols)) < 0.02] = rng.integers(1, 127, 1)[0]
            scales[0, :2] = (0, 126)
            fixed, exc = csf_encode(scales)
            self.assertTrue(len(exc) > 0)
            np.testing.assert_array_equal(nv.csf_decode(fixed, exc, rows, cols), scales)

    def test_rejects_misread_streams(self):
        scales = np.full((16, 4), 50, dtype=np.uint8)
        scales[3, 1] = 120
        fixed, exc = csf_encode(scales)
        bad = exc.copy()
        bad[0] = (bad[0] & 0xFFFFFF) | (50 << 24)       # an exception inside its row window
        with self.assertRaises(ValueError):
            nv.csf_decode(fixed, bad, 16, 4)
        with self.assertRaises(ValueError):
            nv.csf_decode(fixed, np.concatenate((exc, exc)), 16, 4)   # unsorted positions


class Repack(unittest.TestCase):
    def test_nvfp4_blocks_keep_values(self):
        rng = np.random.default_rng(11)
        rows, k = 8, 256
        weight = rng.integers(0, 256, (rows, k // 2), dtype=np.uint8)
        scale = rng.integers(1, 0x70, (rows, k // 16)).astype(np.uint8)
        blocks = nv.nvfp4_ggml(weight, scale)
        # GGML NVFP4: four scales, then byte j of a 16-value group holds values j and j + 8
        got = np.empty((rows, k), dtype=np.float32)
        for r in range(rows):
            for b in range(k // 64):
                blk = blocks[r, b]
                for g in range(4):
                    d = nv.E4M3[blk[g]]
                    for j in range(8):
                        q = blk[4 + g * 8 + j]
                        got[r, b * 64 + g * 16 + j] = nv.E2M1[q & 15] * d
                        got[r, b * 64 + g * 16 + j + 8] = nv.E2M1[q >> 4] * d
        np.testing.assert_array_equal(got, nv.nvfp4_dequant(weight, scale, 1.0))

    def test_nvfp4_rejects_signed_scales(self):
        with self.assertRaises(ValueError):
            nv.nvfp4_ggml(np.zeros((1, 32), np.uint8), np.full((1, 4), 0x80, np.uint8))

    def test_mxfp8_blocks_keep_values(self):
        rng = np.random.default_rng(13)
        rows, k = 4, 384
        weight = rng.integers(0, 256, (rows, k), dtype=np.uint8)
        weight[(weight & 127) == 127] ^= 1
        scale = rng.integers(100, 140, (rows, k // 32)).astype(np.uint8)
        data = nv.Dense(weight, scale, 'mxfp8').encode(nv.T_MXFP8)
        np.testing.assert_array_equal(nv.mxfp8_dequant_blocks(data, [k, rows]), nv.mxfp8_dequant(weight, scale))
        # widened to BF16 the values stay exact
        bf16 = nv.bf16_to_f32(nv.Dense(weight, scale, 'mxfp8').encode(nv.T_BF16)).reshape(rows, k)
        np.testing.assert_array_equal(bf16, nv.mxfp8_dequant(weight, scale))

    def test_column_permutation_moves_scales(self):
        rng = np.random.default_rng(17)
        weight = rng.integers(0, 120, (2, 6 * 128), dtype=np.uint8)
        scale = rng.integers(110, 130, (2, 6 * 4)).astype(np.uint8)
        d = nv.Dense(weight, scale, 'mxfp8')
        perm = np.array([3, 0, 4, 1, 5, 2])
        moved = d.take_cols(perm, 128)
        want = d.f32().reshape(2, 6, 128)[:, perm].reshape(2, -1)
        np.testing.assert_array_equal(moved.f32(), want)

    def test_v_order_tiles_heads(self):
        order = nv.v_order(1)
        # slot j holds HF v-head (j % 16) * 3 + j // 16
        self.assertEqual(order.tolist(), [(j % 16) * 3 + j // 16 for j in range(48)])


if __name__ == '__main__':
    unittest.main()
