"""Check weight-free packet reconstruction and the omitted-zero ambiguity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experimental.derive_pr_encoder import convolution_scatter
from whisper.pr_encoder_kernel import rebuild_hwx, reconstruct, goc_affine


class EncoderPackingTests(unittest.TestCase):
    def setUp(self):
        self.template = bytearray(1024)
        self.template[:8] = b"HWX-test"
        self.matrix = np.arange(1, 13, dtype="<f2").reshape(4, 3)
        self.conv = np.arange(100, 112, dtype="<f2").reshape(4, 1, 3)
        self.tensors = {"matrix":self.matrix, "conv":self.conv,
                        "gamma":np.array([1, 2, 4, 8], "<f2"), "beta":np.array([1, 1, 1, 1], "<f2")}
        self.offsets = np.array([[128, 160], [144, 176], [152, 184]], "<i4")
        self.packing = dict(stripped_bytes=64, operations=[
            dict(kind="tile", tensor="matrix", first=0, count=2, offset=32, bytes=12),
            dict(kind="tile", tensor="matrix", first=2, count=2, offset=512, bytes=12),
            dict(kind="tensor", tensor="gamma", offset=600, bytes=8),
            dict(kind="ratio", tensor="beta", gamma="gamma", offset=608, bytes=8)],
            scatter=dict(tensor="conv", extra=[], offset_bytes=np.diff(self.offsets.ravel(), prepend=0).astype("<i4").tobytes()))

    def test_distant_banks_transposes_and_layer_norm_are_exact(self):
        expected = bytearray(self.template)
        expected[32:44] = self.matrix[:2].T.tobytes()
        expected[512:524] = self.matrix[2:].T.tobytes()
        expected[600:608] = self.tensors["gamma"].tobytes()
        expected[608:616] = np.array([1, .5, .25, .125], "<f2").tobytes()
        for column in range(3):
            for pair in range(2):
                offset = int(self.offsets[column, pair])
                expected[offset:offset + 4] = self.conv[pair * 2:pair * 2 + 2, 0, column].tobytes()
        self.assertEqual(rebuild_hwx(self.template, self.packing, self.tensors), expected)

    def test_overlap_retained_values_and_invalid_offsets_are_rejected(self):
        for mode in ("overlap", "retained", "negative", "scatter"):
            with self.subTest(mode=mode):
                template, packing = bytearray(self.template), copy.deepcopy(self.packing)
                if mode == "overlap":
                    packing["operations"][1]["offset"] = 32
                elif mode == "retained":
                    template[32] = 1
                elif mode == "negative":
                    packing["operations"][0]["offset"] = -1
                else:
                    offsets = self.offsets.copy()
                    offsets[0, 0] = 32
                    packing["scatter"]["offset_bytes"] = np.diff(offsets.ravel(), prepend=0).astype("<i4").tobytes()
                with self.assertRaises(ValueError):
                    rebuild_hwx(template, packing, self.tensors)

    def test_unmapped_nonzero_convolution_values_are_rejected(self):
        offsets = self.offsets.copy()
        offsets[0, 0] = -1
        self.packing["scatter"]["offset_bytes"] = np.diff(offsets.ravel(), prepend=0).astype("<i4").tobytes()
        with self.assertRaisesRegex(ValueError, "unmapped nonzero"):
            rebuild_hwx(self.template, self.packing, self.tensors)

    def test_wrong_checkpoint_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "wrong.bin"
            checkpoint.write_bytes(b"not the pinned model")
            (root / "meta.json").write_text(json.dumps(dict(format="whisper-pr3905-h13g/v1", dimensions={}, checkpoint=dict(sha256="0" * 64))))
            with self.assertRaisesRegex(ValueError, "checkpoint checksum"):
                reconstruct(checkpoint, root)

    def test_omitted_zero_does_not_match_an_unrelated_packet(self):
        vectors = np.arange(1, 769, dtype="<f2").reshape(192, 4)
        vectors[100, 2] = vectors[10, 3]
        vectors[100, 3] = 0
        weight = vectors.reshape(1, 64, 3, 4).transpose(0, 3, 1, 2).reshape(4, 64, 3)
        coefficients = bytearray(4096)
        for index, vector in enumerate(vectors):
            offset = 64 + index * 16
            coefficients[offset:offset + 8] = vector.tobytes()
        # The sparse scalar's actual neighbour is a packet word, not its omitted zero.
        coefficients[64 + 100 * 16 + 6:64 + 100 * 16 + 8] = b"\xfe\xca"
        positions, extra = convolution_scatter(bytes(coefficients), weight, 64, 64 + 192 * 16, np.zeros(4096, bool))
        self.assertEqual(positions[100, 1], -1)
        self.assertEqual(extra, [[100, 2, 64 + 100 * 16 + 4]])
        self.assertEqual(positions[10, 1], 64 + 10 * 16 + 4)

    def test_goc_affine_uses_engine_order_and_doubled_ratio(self):
        gamma = np.arange(1, 33, dtype="<f2")
        beta = gamma * np.float16(.25)
        packed = goc_affine(gamma, beta, 16)
        np.testing.assert_array_equal(packed[0], [[1, .5], [17, .5]])
        np.testing.assert_array_equal(packed[15], [[16, .5], [32, .5]])


if __name__ == "__main__":
    unittest.main()
