"""Exercise actual Qwen dimensions at multiple batches against FP32 matmul."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qwen35.ane import Ane
from qwen35.weights import bf16_round, hadamard


def main():
    rng = np.random.default_rng(531)
    results = []
    with Ane() as ane:
        for k, n in [(1024, 8224), (1024, 5120), (2048, 1024), (1024, 7168), (3584, 1024)]:
            w = rng.normal(0, 1 / np.sqrt(k), (n, k)).astype(np.float16)
            plan = ane.create(w)
            try:
                for rows in (1, 8, 32):
                    x = rng.normal(size=(rows, k)).astype(np.float16).astype(np.float32)
                    expected = x @ w.astype(np.float32).T
                    actual = ane.run(plan, x, n)
                    error = float(np.sqrt(np.mean((actual - expected) ** 2)) / np.sqrt(np.mean(expected ** 2)))
                    result = dict(k=k, n=n, rows=rows, normalized_rmse=error,
                                  max_abs=float(np.max(np.abs(actual - expected))))
                    print(json.dumps(result), flush=True)
                    results.append(result)
                    if not error < .005:
                        raise RuntimeError(f"ANE shape check failed: {result}")
            finally:
                ane.free(plan)
        # This stream used to lose small dynamic coefficients even when their
        # sum was a normal FP16 value. Exact powers of two make this a hardware
        # regression check rather than a tolerance comparison against itself.
        plan = ane.create(np.full((32, 1024), 2 ** -5, dtype=np.float16))
        try:
            amplitudes = np.array([2 ** -5, 2 ** -14, 2 ** -24, 2 ** -100,
                                   2 ** -149, 0., 2 ** 15], dtype=np.float32)
            x = np.repeat(amplitudes[:, None], 1024, axis=1)
            expected = np.repeat((32 * amplitudes)[:, None], 32, axis=1)
            actual = ane.run(plan, x, 32)
            np.testing.assert_array_equal(actual, expected)
            # A matrix with one row must retain its batch dimension.
            np.testing.assert_array_equal(ane.run(plan, x[:1], 32), expected[:1])
            np.testing.assert_array_equal(ane.run(plan, x[0], 32), expected[0])
            results.append(dict(case="power_of_two_dynamic_range", amplitudes=amplitudes.tolist(),
                                exact_match=True))
        finally:
            ane.free(plan)
        # Mirai's BF16 group scales times W4 codes are not always exactly
        # representable in FP16. Compare complete RHT projections against an
        # independent float64 matmul, including small and nonuniform inputs.
        for k, n in [(1024, 8224), (1024, 5120), (2048, 1024), (1024, 7168), (3584, 1024)]:
            codes = rng.integers(0, 16, (n, k), dtype=np.uint8).reshape(n, k // 32, 32)
            zeros = rng.integers(0, 16, (n, k // 32, 1), dtype=np.uint8)
            scales = bf16_round(rng.uniform(.002, .025, (n, k // 32, 1)).astype(np.float32))
            w = ((codes.astype(np.float32) - zeros) * scales).reshape(n, k)
            matrix = SimpleNamespace(name=f"mirai-fixture-{k}-{n}", rows=n, cols=k,
                                     input_signs=rng.choice([-1, 1], k).astype(np.int32),
                                     output_signs=rng.choice([-1, 1], n).astype(np.int32),
                                     decode=lambda: w)
            ane.prepare(matrix)
            x = rng.normal(size=k).astype(np.float32)
            x[::7] *= 2 ** -16
            for amplitude in (1., 2 ** -14):
                value = x * np.float32(amplitude)
                z = hadamard(value * matrix.input_signs)
                expected = hadamard((w.astype(np.float64) @ z.astype(np.float64)).astype(np.float32)) * matrix.output_signs
                before = ane.submissions
                actual = ane.linear(matrix, value)
                error = float(np.linalg.norm(actual.astype(np.float64) - expected) / np.linalg.norm(expected))
                record = dict(case="compensated_mirai_projection", k=k, n=n, amplitude=amplitude,
                              normalized_rmse=error, submissions=ane.submissions - before)
                print(json.dumps(record), flush=True)
                results.append(record)
                if error >= .0002 or record["submissions"] != 1:
                    raise RuntimeError(f"ANE compensated projection failed: {record}")
        result = dict(submissions=ane.submissions, results=results)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
