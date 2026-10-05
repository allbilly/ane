"""Compare both ANE scale policies on identical full-model projection inputs."""
import argparse
import json
from pathlib import Path

import numpy as np

from qwen35.ane import Ane
from qwen35.model import Model
from qwen35.weights import DEFAULT_MODEL, REVISION, hadamard


def error(actual, expected):
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--token", type=int, default=248045)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    model = Model(a.model, kernels="native", context=1)
    records = []
    with Ane(scale_inputs=False) as ane:
        class Probe:
            def linear(self, matrix, x, precision):
                cpu = model.native.linear(matrix, x, precision)
                z = hadamard(np.asarray(x, dtype=np.float32) * matrix.input_signs)
                w = matrix.decode().astype(np.float16)
                plan = ane.create(w)
                try:
                    raw = ane.run(plan, z, matrix.rows)
                    peak = float(np.max(np.abs(z)))
                    shift = int(np.floor(np.log2(ane.input_limits[plan]) - np.log2(peak))) if peak else 0
                    scaled = np.ldexp(ane.run(plan, np.ldexp(z, shift), matrix.rows), -shift)
                finally:
                    ane.free(plan)
                fp16_math = (w.astype(np.float32) @ z.astype(np.float16).astype(np.float32)).astype(np.float16).astype(np.float32)
                record = dict(matrix=matrix.name, input_max=peak,
                              input_rms=float(np.sqrt(np.mean(z * z))), scale_exponent=shift,
                              unscaled_vs_fp16_math=error(raw, fp16_math),
                              unscaled_vs_cpu=error(hadamard(raw) * matrix.output_signs, cpu),
                              scaled_vs_cpu=error(hadamard(scaled) * matrix.output_signs, cpu))
                records.append(record)
                # Keep CPU residual/recurrent history so both device calls see
                # identical inputs; this is a projection diagnostic, not hybrid
                # inference or a fallback route.
                return cpu
        model.backend = Probe()
        model.step(a.token)
        result = dict(revision=REVISION, token=a.token, projections=len(records),
                      ane_submissions=ane.submissions, results=records,
                      max_unscaled_nrmse=max(r["unscaled_vs_cpu"] for r in records),
                      max_scaled_nrmse=max(r["scaled_vs_cpu"] for r in records))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "results"}))


if __name__ == "__main__":
    main()
