"""Independent small-matrix checks for packing, transforms and recurrent state."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .native import Native
from .weights import Matrix, SafeTensors, bf16_round, hadamard, unpack4


def write_tensors(path, values, metadata=None):
    header, chunks, offset = {"__metadata__": metadata or {}}, [], 0
    names = {np.dtype("u1"): "U8", np.dtype("<u2"): "BF16",
             np.dtype("<i4"): "I32", np.dtype("<f4"): "F32"}
    for name, value in values.items():
        data = value.tobytes()
        header[name] = dict(dtype=names[value.dtype], shape=list(value.shape), data_offsets=[offset, offset + len(data)])
        offset += len(data)
        chunks.append(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


class QuantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = Native(threads=1)
        h = np.array([[1]], dtype=np.float32)
        for _ in range(5):
            h = np.block([[h, h], [h, -h]])
        cls.h = h / np.sqrt(np.float32(32))

    def matrix(self, directory, embedding=False):
        rng = np.random.default_rng(903 if embedding else 2026)
        n, k = 64, 128
        codes = rng.integers(0, 16, (n, k), dtype=np.uint8)
        scales = rng.integers(1, 13, (n, k // 32)).astype(np.float32) / 16
        zeros = rng.integers(0, 16, (n, k // 32), dtype=np.uint8)
        output = rng.choice([-1, 1], k if embedding else n).astype(np.int32)
        input_ = rng.choice([-1, 1], k).astype(np.int32)
        weights = codes[:, 0::2] | (codes[:, 1::2] << 4)
        stored_scales = scales if embedding else scales.T
        z = zeros if embedding else zeros.T
        packed_z = z[:, 0::2] | (z[:, 1::2] << 4)
        prefix = "embedding" if embedding else "projection"
        values = {prefix + ".quantized.weights": weights,
                  prefix + ".quantized.scales": (stored_scales.view(np.uint32) >> 16).astype(np.uint16),
                  prefix + ".quantized.zero_points": packed_z,
                  prefix + ".incoherence_signs.output_signs": output}
        if not embedding:
            values[prefix + ".incoherence_signs.input_signs"] = input_
        spec = dict(type="HybridSpec", quantization_spec=dict(type="IntSpec", bits=4, group_size=32,
                    is_symmetric=False, layout="input_output" if embedding else "output_input"),
                    adapter_spec=None, incoherence_block_size=32,
                    incoherence_processing_mode="output" if embedding else "input_output")
        path = Path(directory) / "test.safetensors"
        write_tensors(path, values, {prefix + ".spec": json.dumps(spec)})
        matrix = Matrix(SafeTensors(path), prefix, n, k, embedding=embedding)
        dense = ((codes.reshape(n, -1, 32).astype(np.float32) - zeros[:, :, None])
                 * scales[:, :, None]).reshape(n, k)
        return matrix, dense, input_, output

    def test_hadamard_matches_dense_and_involution(self):
        x = np.random.default_rng(30).normal(size=(3, 128)).astype(np.float32)
        expected = (x.reshape(-1, 32) @ self.h).reshape(x.shape)
        np.testing.assert_allclose(hadamard(x), expected, atol=1e-6)
        np.testing.assert_allclose(hadamard(hadamard(x)), x, atol=1e-6)

    def test_adjacent_nibbles(self):
        np.testing.assert_array_equal(unpack4(np.array([[0x10, 0xfe]], dtype=np.uint8)), [[0, 1, 14, 15]])

    def test_body_group_output_layout_and_native(self):
        with tempfile.TemporaryDirectory() as directory:
            m, dense, inp, out = self.matrix(directory)
            np.testing.assert_array_equal(m.decode(), dense)
            x = np.random.default_rng(14).normal(size=m.cols).astype(np.float32)
            tx = ((x * inp).reshape(-1, 32) @ self.h).reshape(-1)
            expected = ((dense @ tx).reshape(-1, 32) @ self.h).reshape(-1) * out
            np.testing.assert_allclose(m.reference(x), expected, rtol=1e-5, atol=2e-5)
            np.testing.assert_allclose(self.native.linear(m, x), expected, rtol=1e-5, atol=2e-5)

    def test_embedding_output_group_layout_tied_head(self):
        with tempfile.TemporaryDirectory() as directory:
            m, dense, _, signs = self.matrix(directory, True)
            np.testing.assert_array_equal(m.decode(), dense)
            expected_embedding = (dense[7].reshape(-1, 32) @ self.h).reshape(-1) * signs
            np.testing.assert_allclose(m.lookup(7), expected_embedding, atol=1e-6)
            x = np.random.default_rng(13).normal(size=m.cols).astype(np.float32)
            tx = ((x * signs).reshape(-1, 32) @ self.h).reshape(-1)
            expected = dense @ tx
            np.testing.assert_allclose(self.native.linear(m, x), expected, rtol=1e-5, atol=2e-5)
            canonical = (dense.reshape(-1, 32) @ self.h).reshape(dense.shape) * signs
            np.testing.assert_allclose(expected, canonical @ x, rtol=1e-5, atol=2e-5)

    def test_reject_truncated_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            write_tensors(path, {"x": np.ones((4, 4), dtype=np.float32)})
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                SafeTensors(path)

    def test_dot_activation_precision_and_zero_correction(self):
        kernel = Native(threads=1, integer=True)
        with tempfile.TemporaryDirectory() as directory:
            m, dense, inp, out = self.matrix(directory)
            rng = np.random.default_rng(40)
            for x in [rng.normal(size=m.cols).astype(np.float32),
                      np.zeros(m.cols, dtype=np.float32),
                      np.full(m.cols, 1e-36, dtype=np.float32)]:
                expected = self.native.linear(m, x)
                actual = kernel.linear(m, x)
                difference = np.linalg.norm(actual - expected)
                self.assertLess(difference / max(np.linalg.norm(expected), 1e-20), 1e-4)

    def test_bf16_ties_to_even(self):
        u = np.array([0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000], dtype=np.uint32)
        np.testing.assert_array_equal(bf16_round(u.view(np.float32)).view(np.uint32),
                                      [0x3f800000, 0x3f820000, 0xbf800000, 0xbf820000])

    def test_recurrent_state_matches_explicit_delta_rule(self):
        rng = np.random.default_rng(808)
        state = rng.normal(0, .05, (16, 128, 128)).astype(np.float32)
        projected = rng.normal(0, .2, 8224).astype(np.float32)
        a_log = rng.normal(0, .1, 16).astype(np.float32)
        dt = rng.normal(0, .1, 16).astype(np.float32)
        norm = rng.normal(1, .1, 128).astype(np.float32)
        q, k, v = projected[:6144].reshape(3, 16, 128)
        q = q / np.sqrt((q * q).sum(1, keepdims=True) + 1e-6) / np.sqrt(np.float32(128))
        k = k / np.sqrt((k * k).sum(1, keepdims=True) + 1e-6)
        decay = np.exp(-np.exp(a_log) * np.logaddexp(0, projected[8208:] + dt))
        expected_state = state * decay[:, None, None]
        beta = 1 / (1 + np.exp(-projected[8192:8208]))
        correction = (v - np.einsum("hij,hj->hi", expected_state, k)) * beta[:, None]
        expected_state += correction[:, :, None] * k[:, None, :]
        o = np.einsum("hij,hj->hi", expected_state, q)
        z = projected[6144:8192].reshape(16, 128)
        expected = o / np.sqrt((o * o).mean(1, keepdims=True) + 1e-6) * norm * z / (1 + np.exp(-z))
        actual = self.native.gdn(state, projected, a_log, dt, norm)
        np.testing.assert_allclose(state, expected_state, rtol=1e-4, atol=2e-7)
        np.testing.assert_allclose(actual, expected.reshape(-1), rtol=2e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
