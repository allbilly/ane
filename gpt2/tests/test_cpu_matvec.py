"""Native packed kernels compared with independently decoded NumPy matrices."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gguf
import numpy as np
from cpu_matvec import CPUMatvec, PackedMatrix, native_library, performance_cpus
from external_weights import find_weights, load_weights
from model import CPUKernels, GPT2


class NativeMatvecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library, reason = native_library()
        if cls.library is None:
            raise unittest.SkipTest(reason)

    def test_quantized_row_tails_blocks_and_strided_activations(self):
        rng = np.random.default_rng(4290)
        for rows, cols in ((1, 32), (3, 768), (4, 3072), (5, 768), (2049, 32)):
            source = rng.normal(0, 0.3, (rows, cols)).astype(np.float32)
            x = rng.normal(size=cols * 2).astype(np.float32)[::2]
            for kind in (gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q8_0):
                raw = gguf.quantize(source, kind)
                expected = gguf.dequantize(raw, kind).astype(np.float16).astype(np.float32) @ x
                for threads in (1, 4):
                    with self.subTest(rows=rows, cols=cols, kind=kind.name, threads=threads):
                        matrix = PackedMatrix(kind.name, raw, source.shape, self.library, threads)
                        np.testing.assert_allclose(matrix(x), expected, rtol=2e-5, atol=2e-5)
                        # Repacking stays compressed, with padding for at most
                        # three rows; it never becomes a model-size FP16 array.
                        size = raw.nbytes * (((rows + 3) // 4) * 4) // rows
                        self.assertEqual(matrix.data.nbytes, size)

    def test_q4_signed_scales_nibble_order_and_subnormals(self):
        scales = np.array([0.5, -2, 2 ** -24, -(2 ** -14), 0, -0.0, 12.125], dtype=np.float16)
        q = np.arange(16, dtype=np.uint8) | ((15 - np.arange(16, dtype=np.uint8)) << 4)
        raw = np.stack([np.frombuffer(d.tobytes() + q.tobytes(), dtype=np.uint8) for d in scales])
        decoded = np.stack([np.concatenate((np.arange(16) - 8, 7 - np.arange(16))) * float(d) for d in scales])
        decoded = decoded.astype(np.float16).astype(np.float32)
        x = np.arange(32, dtype=np.float32) / 31
        actual = PackedMatrix("Q4_0", raw, (len(scales), 32), self.library, 1)(x)
        np.testing.assert_allclose(actual, decoded @ x, rtol=2e-6, atol=2e-5)

    def test_fp16_kernel_and_input_validation(self):
        rng = np.random.default_rng(710)
        w = rng.normal(size=(7, 768)).astype(np.float16)
        matrix = PackedMatrix("F16", w, w.shape, self.library, 1)
        x = rng.normal(size=768).astype(np.float32)
        np.testing.assert_allclose(matrix(x), w.astype(np.float32) @ x, rtol=2e-5, atol=2e-5)
        for bad in (x[:767], np.full(768, np.nan), np.full(768, np.inf)):
            with self.assertRaisesRegex(ValueError, "finite FP32 activation"):
                matrix(bad)
        with self.assertRaisesRegex(ValueError, "byte size"):
            PackedMatrix("Q4_0", np.zeros(1, np.uint8), (4, 32), self.library, 1)
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            PackedMatrix("F16", w, (7, 767), self.library, 1)

    def test_packed_dispatch_does_not_decode_tensor(self):
        rng = np.random.default_rng(784)
        raw = gguf.quantize(rng.normal(size=(5, 768)).astype(np.float32), gguf.GGMLQuantizationType.Q4_0)
        class Weights:
            def packed_matrix(self, name, shape):
                return "Q4_0", raw
            def get(self, name, shape):
                raise AssertionError("native packed matvec must not request decoded weights")
        matvec = CPUMatvec(Weights(), "exact", threads=1)
        x = rng.normal(size=768).astype(np.float32)
        expected = gguf.dequantize(raw, gguf.GGMLQuantizationType.Q4_0).astype(np.float16).astype(np.float32) @ x
        np.testing.assert_allclose(matvec("test", (5, 768), x), expected, rtol=2e-5, atol=2e-5)
        self.assertIs(matvec.matrix("test", (5, 768)), matvec.matrix("test", (5, 768)))

    def test_integer_dot_precision_and_extreme_activation_fallback(self):
        if not self.library.gpt2_matvec_dotprod():
            self.skipTest("integer dot instructions unavailable")
        rng = np.random.default_rng(4381)
        for kind in (gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q8_0):
            for rows, cols in ((1, 32), (5, 768), (9, 3072), (2049, 32)):
                source = rng.normal(0, 0.3, (rows, cols)).astype(np.float32)
                raw = gguf.quantize(source, kind)
                x = rng.normal(size=cols * 2).astype(np.float32)[::2]
                decoded = gguf.dequantize(raw, kind)
                expected = decoded @ x
                for threads in (1, 4):
                    with self.subTest(kind=kind.name, rows=rows, cols=cols, threads=threads):
                        matrix = PackedMatrix(kind.name, raw, source.shape, self.library, threads, integer=True)
                        actual = matrix(x)
                        relative = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                        self.assertLess(relative, 4e-5)
                        # Per-block rounding bounds each activation error;
                        # this also covers outputs near cancellation.
                        maxima = np.max(np.abs(x.reshape(-1, 32)), axis=1)
                        bound = (np.abs(decoded).reshape(rows, -1, 32).sum(axis=2)
                                 @ (maxima / (2 * 32639)))
                        np.testing.assert_array_less(np.abs(actual - expected), bound + 2e-5)
                        np.testing.assert_array_equal(matrix(np.zeros(cols, np.float32)), np.zeros(rows, np.float32))
                exact = PackedMatrix(kind.name, raw, source.shape, self.library, 1)
                integer = PackedMatrix(kind.name, raw, source.shape, self.library, 1, integer=True)
                for magnitude in (1e-35, 1e35):
                    extreme = rng.uniform(-magnitude, magnitude, cols).astype(np.float32)
                    extreme[0] = magnitude
                    np.testing.assert_array_equal(integer(extreme), exact(extreme))

    def test_native_integer_dispatch_retains_packed_weights(self):
        if not self.library.gpt2_matvec_dotprod():
            self.skipTest("integer dot instructions unavailable")
        rng = np.random.default_rng(4279)
        raw = gguf.quantize(rng.normal(size=(8, 768)).astype(np.float32), gguf.GGMLQuantizationType.Q4_0)
        class Weights:
            def packed_matrix(self, name, shape):
                return "Q4_0", raw
            def get(self, name, shape):
                raise AssertionError("integer matvec must keep compressed weights")
        matvec = CPUMatvec(Weights(), "native", threads=1)
        x = rng.normal(size=768).astype(np.float32)
        expected = gguf.dequantize(raw, gguf.GGMLQuantizationType.Q4_0) @ x
        actual = matvec("lm_head", (8, 768), x)
        self.assertTrue(matvec.matrix("lm_head", (8, 768)).integer)
        self.assertFalse(matvec.matrix("layer0/wo", (8, 768)).integer)
        self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 4e-5)

    def test_explicit_numpy_and_unavailable_native_modes(self):
        class Weights:
            def get(self, name, shape):
                return np.ones(shape, np.float32)
        weights = Weights()
        with patch("cpu_matvec.native_library", return_value=(None, "compiler unavailable")):
            fallback = CPUMatvec(weights)
            np.testing.assert_array_equal(fallback("w", (3, 16), np.ones(16)), np.full(3, 16))
            with self.assertRaisesRegex(ValueError, "compiler unavailable"):
                CPUMatvec(weights, "native")
        with self.assertRaisesRegex(ValueError, "between 1 and 256"):
            CPUMatvec(weights, "numpy", threads=0)


class NativeModelTests(unittest.TestCase):
    def test_reference_full_cpu_logits_and_greedy_trace(self):
        source = find_weights()
        if source is None or source.suffix != ".safetensors":
            self.skipTest("cached reference safetensors required")
        library, reason = native_library()
        if library is None:
            self.skipTest(reason)
        weights = load_weights(source)
        native_matvec = CPUMatvec(weights, "native")
        native = GPT2(weights, CPUKernels(weights, native_matvec), native_matvec)
        oracle = GPT2(weights, CPUKernels(weights, "numpy"), "numpy")
        for token in [15496, 995, 11, 314, 1101, 407, 1654, 644, 284, 910]:
            expected, actual = oracle.step(token), native.step(token)
            np.testing.assert_allclose(actual, expected, rtol=0.0002, atol=0.0005)
            self.assertEqual(int(actual.argmax()), int(expected.argmax()))


if __name__ == "__main__":
    unittest.main()
