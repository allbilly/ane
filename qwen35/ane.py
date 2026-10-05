"""Explicit Linux M1 ANE projection backend; submission failures are fatal."""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

from .native import pointer
from .weights import bf16_round, hadamard


class Ane:
    def __init__(self):
        source = Path(__file__).with_name("ane_matmul.c")
        header = source.with_suffix(".h")
        template = source.with_name("linear_template.h")
        compiler = os.environ.get("CC", "cc")
        flags = ["-O3", "-std=gnu11", "-fPIC", "-shared", "-fopenmp", "-march=armv8.2-a+fp16"]
        identity = hashlib.sha256(source.read_bytes() + header.read_bytes() + template.read_bytes()
                                  + subprocess.check_output([compiler, "--version"]) + repr(flags).encode()).hexdigest()
        cache = Path.home() / ".cache/ane-qwen35/native" / identity
        cache.mkdir(parents=True, exist_ok=True)
        library = cache / "ane.so"
        if not library.exists():
            temporary = cache / f"ane.{os.getpid()}.so"
            subprocess.run([compiler, *flags, str(source), "-lm", "-o", str(temporary)], check=True)
            temporary.replace(library)
        self.lib = ctypes.CDLL(str(library))
        p, i = ctypes.c_void_p, ctypes.c_int
        self.lib.ane_device_open.restype = p
        self.lib.ane_device_close.argtypes = [p]
        self.lib.ane_plan_create_f16.argtypes = [p, p, i, i]
        self.lib.ane_plan_create_f16.restype = p
        self.lib.ane_plan_run_batch.argtypes = [p, p, p, i]
        self.lib.ane_plan_run_batch.restype = i
        self.lib.ane_plan_free.argtypes = [p]
        self.lib.ane_device_submissions.argtypes = [p]
        self.lib.ane_device_submissions.restype = ctypes.c_ulonglong
        self.device = self.lib.ane_device_open()
        if not self.device:
            raise RuntimeError("M1 ANE device is unavailable")
        self.plans = {}

    def create(self, weights):
        w = np.ascontiguousarray(weights, dtype=np.float16)
        if w.ndim != 2 or not np.isfinite(w).all():
            raise ValueError("ANE weights must be a finite 2D FP16 matrix")
        n, k = w.shape
        plan = self.lib.ane_plan_create_f16(self.device, pointer(w), k, n)
        if not plan:
            raise RuntimeError(f"ANE could not prepare {k} x {n} matrix")
        return plan

    def run(self, plan, x, outputs):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim not in (1, 2):
            raise ValueError("ANE accepts a vector or matrix")
        rows = 1 if x.ndim == 1 else len(x)
        y = np.empty((rows, outputs), dtype=np.float32)
        if not self.lib.ane_plan_run_batch(plan, pointer(x), pointer(y), rows):
            raise RuntimeError("ANE submission returned invalid output")
        return y[0] if x.ndim == 1 else y

    def prepare(self, matrix):
        if matrix.name not in self.plans:
            self.plans[matrix.name] = self.create(matrix.decode())
        return self.plans[matrix.name]

    def linear(self, matrix, x, precision="fp32"):
        plan = self.prepare(matrix)
        transformed = hadamard(np.asarray(x, dtype=np.float32) * matrix.input_signs)
        if precision == "bf16":
            transformed = bf16_round(transformed)
        y = self.run(plan, transformed, matrix.rows)
        if precision == "bf16":
            y = bf16_round(y)
        y = hadamard(y) * matrix.output_signs
        return bf16_round(y) if precision == "bf16" else np.asarray(y, dtype=np.float32)

    @property
    def submissions(self):
        return self.lib.ane_device_submissions(self.device)

    def close(self):
        if self.device:
            for plan in self.plans.values():
                self.lib.ane_plan_free(plan)
            self.plans.clear()
            self.lib.ane_device_close(self.device)
            self.device = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
