"""Private E5RT projections with the Asahi high/residual compensation policy."""
import hashlib
import os
from pathlib import Path
import platform
import sys

import numpy as np

from .ane import Ane


def precision_planes(x, partitions, gains, limit):
    """Pack the same partition/replica rows as ane_plan_run_compensated."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    gains = np.asarray(gains, dtype=np.float32)
    if (x.ndim != 1 or not np.isfinite(x).all() or partitions < 1
            or x.size % (32 * partitions) or gains.ndim != 1 or not len(gains)
            or 2 * partitions * len(gains) > 32 or not np.isfinite(gains).all()
            or np.any(gains < 1) or np.any(gains >= 2)
            or not np.isfinite(limit) or not 0 < limit <= 65504):
        raise ValueError("invalid compensated projection configuration/input")
    planes = np.zeros((32, x.size), dtype=np.float16)
    shifts = np.zeros(2 * partitions * len(gains), dtype=np.int32)
    active = np.zeros(len(shifts), dtype=bool)
    width = x.size // partitions
    for replica, gain in enumerate(gains):
        for part in range(partitions):
            row = 2 * (replica * partitions + part)
            start = part * width
            values = x[start:start + width]
            peak = float(np.max(np.abs(values)))
            if peak == 0:
                continue
            shift = int(np.floor(np.log2(float(limit)) - np.log2(peak * float(gain))))
            scaled = np.ldexp(values, shift) * gain
            high = scaled.astype(np.float16)
            planes[row, start:start + width] = high
            residual = scaled - high.astype(np.float32)
            shifts[row], active[row] = -shift, True
            low_peak = float(np.max(np.abs(residual)))
            if low_peak:
                low_shift = int(np.floor(np.log2(float(limit)) - np.log2(low_peak)))
                planes[row + 1, start:start + width] = np.ldexp(residual, low_shift).astype(np.float16)
                shifts[row + 1], active[row + 1] = -shift - low_shift, True
    if not np.isfinite(planes).all():
        raise ValueError("compensated input exceeds FP16 range")
    return planes, shifts, active


def combine_planes(outputs, shifts, active, partitions, gains):
    outputs = np.asarray(outputs, dtype=np.float32)
    if not np.isfinite(outputs).all():
        raise RuntimeError("ANE projection returned nonfinite output")
    gains = np.asarray(gains, dtype=np.float32)
    result = np.zeros(outputs.shape[1], dtype=np.float32)
    with np.errstate(over="ignore", under="ignore"):
        factors = np.ldexp(np.ones(len(shifts), dtype=np.float32), shifts)
    slow = bool(np.any(active & ((factors == 0) | ~np.isfinite(factors))))
    for replica, gain in enumerate(gains):
        partial = np.zeros_like(result)
        for row in range(replica * 2 * partitions, (replica + 1) * 2 * partitions):
            if active[row]:
                term = np.ldexp(outputs[row], int(shifts[row])) if slow else outputs[row] * factors[row]
                partial += term
        result += partial / gain
    result /= np.float32(len(gains))
    if not np.isfinite(result).all():
        raise RuntimeError("restored ANE projection exceeds FP32 range")
    return result


class MacosAne(Ane):
    """One ANE-only compiled program execution per body matrix; CPU reductions."""
    def __init__(self, scale_inputs=True, split_k=2, gains=(1., 1.375), cache=None):
        if platform.system() != "Darwin":
            raise RuntimeError("private E5RT backend requires macOS")
        if not scale_inputs or split_k != 2 or tuple(gains) != (1., 1.375):
            raise ValueError("macOS full-model backend implements the accurate Asahi policy")
        sys.path.insert(0, str(Path.home() / "Desktop/ANEForge"))
        import aneforge
        from aneforge._runtime import _find_dylib
        self.af = aneforge
        self.runtime = Path(_find_dylib())
        self.runtime_sha256 = hashlib.sha256(self.runtime.read_bytes()).hexdigest()
        self.cache = Path(cache or os.environ.get("QWEN35_MACOS_ANE_CACHE_DIR",
                                                 str(Path.home() / ".cache/ane-qwen35/macos-ane")))
        self.cache.mkdir(parents=True, exist_ok=True)
        self.scale_inputs, self.split_k = True, split_k
        self.gains = np.asarray(gains, dtype=np.float32)
        self.plans, self.input_limits, self.dimensions, self.programs = {}, {}, {}, {}
        self.program_receipts, self._submissions = [], 0

    def create(self, weights):
        w = np.ascontiguousarray(weights, dtype=np.float16)
        if w.ndim != 2 or not np.isfinite(w).all() or w.shape[1] % 64:
            raise ValueError("ANE weights must be finite and K divisible by 64")
        n, k = w.shape
        identity = hashlib.sha256(w.tobytes() + repr(w.shape).encode() + self.runtime_sha256.encode()).hexdigest()
        directory = self.cache / identity
        x = self.af.input((1, k, 1, 32))
        y = self.af.conv(x, w.reshape(n, k, 1, 1))
        program = self.af.compile(y, build_dir=directory, opt=0)
        if program._prog._device_mask != 4:
            program.release()
            raise RuntimeError("projection program must use ANE-only mask 4")
        plan = len(self.programs) + 1
        self.programs[plan] = (program, program.input_view(), program.output_view())
        bound = float(np.max(np.sum(np.abs(w).astype(np.float32), axis=1)))
        self.input_limits[plan] = min(64., 32768. / max(bound, 1.))
        self.dimensions[plan] = (k, n)
        self.program_receipts.append(dict(plan=plan, inputs=k, outputs=n, identity=identity,
                                         weights_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                                         mil_sha256=hashlib.sha256((directory / "model.mil").read_bytes()).hexdigest(),
                                         directory=str(directory), device_mask=4))
        return plan

    def _project(self, plan, transformed):
        if np.shape(transformed) != (self.dimensions[plan][0],):
            raise ValueError("ANE input dimensions differ from its plan")
        planes, shifts, active = precision_planes(transformed, self.split_k, self.gains, self.input_limits[plan])
        program, input_view, output_view = self.programs[plan]
        input_view[0, :, 0, :] = planes.T
        program.execute()
        self._submissions += 1
        outputs = output_view[0, :, 0, :].T
        return combine_planes(outputs, shifts, active, self.split_k, self.gains)

    @property
    def submissions(self):
        return self._submissions

    def close(self):
        for program, _, _ in self.programs.values():
            program.release()
        self.programs.clear()
        self.plans.clear()
        self.dimensions.clear()
        self.input_limits.clear()
