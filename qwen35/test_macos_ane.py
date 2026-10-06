"""Independent arithmetic checks for the private-runtime compensation codec."""
import unittest
import numpy as np
from qwen35.macos_ane import precision_planes, combine_planes


class CompensationTests(unittest.TestCase):
    def test_matches_float64_projection_across_scales(self):
        rng = np.random.default_rng(78)
        matrix = rng.normal(0, .03, (7, 64)).astype(np.float32)
        for scale in (1., 1e-30, 1e30):
            x = (rng.normal(size=64) * scale).astype(np.float32)
            planes, shifts, active = precision_planes(x, 2, (1., 1.375), 64.)
            output = planes.astype(np.float32) @ matrix.T
            actual = combine_planes(output, shifts, active, 2, (1., 1.375)).astype(np.float64)
            expected = matrix.astype(np.float64) @ x.astype(np.float64)
            self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 4e-7)
            self.assertFalse(planes[8:].any())
            self.assertFalse(planes[:2, 32:].any())
            self.assertFalse(planes[2:4, :32].any())

    def test_zero_and_subnormal_inputs(self):
        for x in (np.zeros(64, np.float32), np.full(64, np.nextafter(np.float32(0), np.float32(1)), np.float32)):
            planes, shifts, active = precision_planes(x, 2, (1., 1.375), 64.)
            output = planes.astype(np.float32)
            actual = combine_planes(output, shifts, active, 2, (1., 1.375))
            np.testing.assert_array_equal(actual, x)

    def test_invalid_inputs_and_unwritten_output_fail(self):
        for x in (np.ones(63, np.float32), np.full(64, np.nan, np.float32), np.ones((2, 64), np.float32)):
            with self.assertRaises(ValueError):
                precision_planes(x, 2, (1., 1.375), 64.)
        with self.assertRaises(ValueError):
            precision_planes(np.ones(64), 2, (0.,), 64.)
        with self.assertRaises(RuntimeError):
            combine_planes(np.full((32, 1), np.nan), np.zeros(8, np.int32), np.ones(8, bool), 2, (1., 1.375))


if __name__ == "__main__":
    unittest.main()
