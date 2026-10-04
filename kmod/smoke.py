#!/usr/bin/env python3
"""Check existing ANE examples numerically, each in a separate process."""
import contextlib
import io
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CASES = [("elementwise", mode) for mode in ("add", "mul", "max", "min", "sq")]
CASES += [(name, "") for name in ("relu", "conv", "gemm", "concat", "sigmoid")]


def check_case(name, mode):
    os.chdir(ROOT)
    path = ROOT / "examples" / (name + ".py")
    sys.argv = [str(path)] + ([mode] if mode else [])
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        result = runpy.run_path(str(path), run_name="__main__")
    if result.get("ret") != 0:
        raise AssertionError(f"submit returned {result.get('ret')}")
    if name == "relu":
        actual = result["output"]
        expected = np.maximum(0, result["input_a"][:result["W"]])
    elif name == "concat":
        actual = result["out_ch"]
        expected = np.concatenate((np.full(result["C2"], 2.0),
                                   np.full(result["C1"], 3.0)))
    elif name == "sigmoid":
        actual, expected = result["out_arr"][:1], [result["expected"]]
    else:
        actual = result["output"] if "output" in result else result["out"]
        expected = result["expected"]
    np.testing.assert_allclose(actual, expected, rtol=0.005, atol=0.005)
    print(f"PASS {name}{':' + mode if mode else ''}: {np.size(actual)} outputs")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_case(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        for name, mode in CASES:
            subprocess.run([sys.executable, __file__, name, mode], check=True, timeout=30)
        print(f"PASS: all {len(CASES)} ANE operation checks")
