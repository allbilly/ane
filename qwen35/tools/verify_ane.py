"""Exercise actual Qwen dimensions at multiple batches against FP32 matmul."""
import json
import sys
from pathlib import Path

import numpy as np

from qwen35.ane import Ane


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
                ane.lib.ane_plan_free(plan)
        result = dict(submissions=ane.submissions, results=results)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
