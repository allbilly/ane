"""Compile and invoke the portable/NEON packed CPU kernels."""
import ctypes
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from .weights import bf16_round, hadamard


def pointer(x):
    return ctypes.c_void_p(x.ctypes.data)


class Native:
    def __init__(self, threads=None):
        cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        capacity = {c: int(Path(f"/sys/devices/system/cpu/cpu{c}/cpu_capacity").read_text())
                    for c in cpus if Path(f"/sys/devices/system/cpu/cpu{c}/cpu_capacity").exists()}
        if capacity:
            cpus = [c for c in cpus if capacity.get(c) == max(capacity.values())]
        self.threads = threads or min(4, len(cpus) or 1)
        if self.threads < 1:
            raise ValueError("threads must be positive")
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        if cpus:
            os.environ.setdefault("OMP_PLACES", ",".join("{" + str(c) + "}" for c in cpus))
            os.environ.setdefault("OMP_PROC_BIND", "CLOSE")
        source = Path(__file__).with_name("cpu.c")
        compiler = os.environ.get("CC", "cc")
        flags = ["-O3", "-ffp-contract=off", "-std=c11", "-fPIC", "-shared", "-fopenmp"]
        if platform.machine() in ("aarch64", "arm64"):
            flags.append("-march=armv8.2-a+fp16+dotprod")
        version = subprocess.check_output([compiler, "--version"])
        identity = hashlib.sha256(source.read_bytes() + version + json.dumps(flags).encode()).hexdigest()
        cache = Path.home() / ".cache/ane-qwen35/native" / identity
        cache.mkdir(parents=True, exist_ok=True)
        library = cache / "cpu.so"
        if not library.exists():
            temporary = cache / f"cpu.{os.getpid()}.so"
            subprocess.run([compiler, *flags, str(source), "-lm", "-o", str(temporary)], check=True)
            temporary.replace(library)
        self.lib = ctypes.CDLL(str(library))
        if self.lib.qwen35_abi() != 1:
            raise RuntimeError("CPU kernel ABI mismatch")
        p, i = ctypes.c_void_p, ctypes.c_int
        self.lib.qwen35_w4.argtypes = [p, p, p, p, p, i, i, i]
        self.lib.qwen35_w4.restype = None
        self.lib.qwen35_gdn.argtypes = [p, p, p, p, p, p]
        self.lib.qwen35_gdn.restype = None

    def linear(self, matrix, x, precision="fp32"):
        signs = matrix.output_signs if matrix.embedding else matrix.input_signs
        transformed = hadamard(np.asarray(x, dtype=np.float32) * signs)
        if precision == "bf16":
            transformed = bf16_round(transformed)
        transformed = np.ascontiguousarray(transformed, dtype=np.float32)
        if transformed.ndim != 1:
            raise ValueError("native matvec accepts one token")
        y = np.empty(matrix.rows, dtype=np.float32)
        self.lib.qwen35_w4(pointer(matrix.weights), pointer(matrix.scales), pointer(matrix.zeros),
                           pointer(transformed), pointer(y), matrix.rows, matrix.cols, self.threads)
        if precision == "bf16":
            y = bf16_round(y)
        if not matrix.embedding:
            y = hadamard(y) * matrix.output_signs
            if precision == "bf16":
                y = bf16_round(y)
        return np.asarray(y, dtype=np.float32)

    def gdn(self, state, projected, a_log, dt_bias, norm):
        output = np.empty(2048, dtype=np.float32)
        self.lib.qwen35_gdn(*(pointer(v) for v in (state, projected, a_log, dt_bias, norm, output)))
        return output
