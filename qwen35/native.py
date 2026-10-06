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


def dot_supported():
    if platform.machine() not in ("aarch64", "arm64"):
        return False
    return ("asimddp" in Path("/proc/cpuinfo").read_text()
            if platform.system() == "Linux" else platform.system() == "Darwin")


def pointer(x):
    return ctypes.c_void_p(x.ctypes.data)


class Native:
    def __init__(self, threads=None, integer=False):
        self.integer = integer
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
            flags.append("-march=armv8.2-a+fp16" + ("+dotprod" if dot_supported() else ""))
        version = subprocess.check_output([compiler, "--version"])
        identity = hashlib.sha256(source.read_bytes() + version + json.dumps(flags).encode()).hexdigest()
        cache_root = Path(os.environ.get("QWEN35_CACHE_DIR", str(Path.home() / ".cache/ane-qwen35")))
        cache = cache_root / "native" / identity
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
        self.lib.qwen35_w4_dot.argtypes = self.lib.qwen35_w4.argtypes
        self.lib.qwen35_w4_dot.restype = None
        self.lib.qwen35_w4_serial.argtypes = self.lib.qwen35_w4.argtypes
        self.lib.qwen35_w4_serial.restype = None
        self.lib.qwen35_rms_bf16.argtypes = [p, p, p, i, i]
        self.lib.qwen35_rms_bf16.restype = None
        self.lib.qwen35_rope_bf16.argtypes = [p, i, i]
        self.lib.qwen35_rope_bf16.restype = None
        self.lib.qwen35_attention_bf16.argtypes = [p, p, p, p, p, i]
        self.lib.qwen35_attention_bf16.restype = None
        self.lib.qwen35_conv_bf16.argtypes = [p, p, p]
        self.lib.qwen35_conv_bf16.restype = None
        self.lib.qwen35_mlp_bf16.argtypes = [p, p]
        self.lib.qwen35_mlp_bf16.restype = None
        self.lib.qwen35_gdn.argtypes = [p, p, p, p, p, p]
        self.lib.qwen35_gdn.restype = None

    def linear(self, matrix, x, precision="fp32"):
        signs = matrix.output_signs if matrix.embedding else matrix.input_signs
        transformed = hadamard(np.asarray(x, dtype=np.float32) * signs)
        if precision == "bf16":
            transformed = bf16_round(transformed)
        transformed = np.ascontiguousarray(transformed, dtype=np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError(f"nonfinite projection input: {matrix.name}")
        if transformed.ndim != 1:
            raise ValueError("native matvec accepts one token")
        y = np.empty(matrix.rows, dtype=np.float32)
        function = (self.lib.qwen35_w4_dot if self.integer else
                    self.lib.qwen35_w4_serial if precision == "bf16" else self.lib.qwen35_w4)
        function(pointer(matrix.weights), pointer(matrix.scales), pointer(matrix.zeros),
                 pointer(transformed), pointer(y), matrix.rows, matrix.cols, self.threads)
        if precision == "bf16":
            y = bf16_round(y)
        if not matrix.embedding:
            y = hadamard(y) * matrix.output_signs
            if precision == "bf16":
                y = bf16_round(y)
        return np.asarray(y, dtype=np.float32)

    def rms_bf16(self, x, scale):
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.empty_like(x)
        self.lib.qwen35_rms_bf16(pointer(x), pointer(scale), pointer(y), x.size // x.shape[-1], x.shape[-1])
        return y

    def rope_bf16(self, x, position):
        self.lib.qwen35_rope_bf16(pointer(x), len(x), position)

    def conv_bf16(self, projected, state, weights):
        self.lib.qwen35_conv_bf16(pointer(projected), pointer(state), pointer(weights))

    def mlp_bf16(self, projected):
        y = np.empty(3584, dtype=np.float32)
        self.lib.qwen35_mlp_bf16(pointer(projected), pointer(y))
        return y

    def attention_bf16(self, q, keys, values, gate, length):
        y = np.empty(2048, dtype=np.float32)
        self.lib.qwen35_attention_bf16(*(pointer(v) for v in (q, keys, values, gate, y)), length)
        return y

    def gdn(self, state, projected, a_log, dt_bias, norm):
        output = np.empty(2048, dtype=np.float32)
        self.lib.qwen35_gdn(*(pointer(v) for v in (state, projected, a_log, dt_bias, norm, output)))
        return output
