#!/usr/bin/env python3
"""Verify tiny.en projection shapes and the final 28-position tile on actual ANE."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen35.ane import Ane


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(61006)
    records = []
    with Ane() as ane:
        for k, n in ((384, 384), (384, 1536), (1536, 384)):
            weight = rng.normal(0, 1/np.sqrt(k), (n, k)).astype(np.float16)
            plan = ane.create(weight)
            try:
                for rows in (1, 8, 28, 32):
                    x = rng.normal(size=(rows, k)).astype(np.float32)
                    reference = x.astype(np.float64) @ weight.astype(np.float64).T
                    before = ane.submissions
                    actual = ane.run(plan, x, n).astype(np.float64)
                    nrmse = float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
                    record = dict(k=k, n=n, rows=rows, nrmse=nrmse,
                                  submissions=ane.submissions-before)
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    if nrmse >= .005 or record["submissions"] != 1:
                        raise RuntimeError("Whisper ANE matrix verification failed")
            finally:
                ane.free(plan)
        report = dict(status="PASS", gate_nrmse=.005, submissions=ane.submissions, records=records)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
