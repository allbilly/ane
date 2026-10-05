"""FP32-accumulating NEON matvecs over packed Q4/Q8 or FP16 weights."""
from functools import lru_cache
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import numpy as np
from hwx import require

ROOT = Path(__file__).resolve().parent


def performance_cpus():
    """Respect the process affinity; prefer the fastest available CPU cluster."""
    available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    capacities = {}
    for cpu in available:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpu_capacity")
        if path.is_file():
            capacities[cpu] = int(path.read_text())
    if len(capacities) == len(available):
        available = [cpu for cpu in available if capacities[cpu] == max(capacities.values())]
    return available


def neon_supported():
    if platform.machine().lower() not in ("aarch64", "arm64"):
        return False
    if platform.system() == "Linux":
        return "asimdhp" in Path("/proc/cpuinfo").read_text()
    return platform.system() == "Darwin"


def build_library():
    """Compile into the external cache; verify cached library bytes before load."""
    from external_weights import cache_root
    compiler = shutil.which(os.environ.get("CC", "cc"))
    require(compiler is not None, "packed CPU kernels require a C compiler (cc)")
    version = subprocess.run([compiler, "--version"], capture_output=True, check=True).stdout
    source = ROOT / "cpu_matvec.c"
    base_flags = ["-O3", "-std=c11", "-fPIC", "-shared", "-march=armv8.2-a+fp16"]
    errors = []
    for openmp in (True, False):
        flags = base_flags + (["-fopenmp"] if openmp else [])
        identity = hashlib.sha256(source.read_bytes() + version + json.dumps([compiler, flags, platform.platform()]).encode()).hexdigest()
        cache = cache_root() / "cpu-neon-v1" / identity
        cache.mkdir(parents=True, exist_ok=True)
        library, record = cache / "matvec.so", cache / "matvec.json"
        try:
            if record.is_file() and library.is_file() and json.loads(record.read_text())["sha256"] == hashlib.sha256(library.read_bytes()).hexdigest():
                return library
        except (ValueError, KeyError, OSError):
            pass
        with tempfile.TemporaryDirectory(prefix=".build-", dir=cache) as directory:
            output = Path(directory) / "matvec.so"
            compiled = subprocess.run([compiler, *flags, str(source), "-o", str(output)], capture_output=True, text=True)
            if compiled.returncode:
                errors.append(compiled.stderr)
                continue
            metadata = Path(directory) / "matvec.json"
            metadata.write_text(json.dumps(dict(sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                                                compiler=compiler, flags=flags)) + "\n")
            output.replace(library)
            metadata.replace(record)
            return library
    raise RuntimeError("could not build packed CPU kernels: " + errors[-1][-1500:])


@lru_cache(maxsize=1)
def native_library():
    if not neon_supported():
        return None, "ARM NEON with FP16 instructions is unavailable"
    try:
        # libgomp may bind the calling thread when the library is loaded.
        # Remember the initial allowed cluster before that narrows its mask.
        default_threads = min(4, len(performance_cpus()))
        # Sleeping workers avoid stealing CPU time while the main thread uses
        # the ANE. Explicit user OpenMP settings take precedence.
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        if platform.system() == "Linux":
            cpus = performance_cpus()
            os.environ.setdefault("OMP_PLACES", ",".join("{" + str(cpu) + "}" for cpu in cpus))
            os.environ.setdefault("OMP_PROC_BIND", "CLOSE")
        library = ctypes.CDLL(str(build_library()))
        library.default_threads = default_threads
        library.gpt2_matvec_abi.restype = ctypes.c_int
        require(library.gpt2_matvec_abi() == 2, "packed CPU kernel ABI mismatch")
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        library.gpt2_q4.argtypes = library.gpt2_q8.argtypes = library.gpt2_f16.argtypes = [pointer, pointer, pointer, integer, integer, integer]
        for name in ("gpt2_q4", "gpt2_q8", "gpt2_f16"):
            getattr(library, name).restype = None
        library.gpt2_matvec_openmp.restype = ctypes.c_int
        return library, None
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        return None, str(error)


class PackedMatrix:
    def __init__(self, kind, data, shape, library, threads):
        self.kind, self.shape, self.library, self.threads = kind, tuple(shape), library, threads
        rows, cols = self.shape
        require(rows > 0 and cols > 0 and cols % 16 == 0, "native matrix columns must be a positive multiple of 16")
        if kind in ("Q4_0", "Q8_0"):
            require(cols % 32 == 0, "quantized matrix columns must be a multiple of 32")
            block_bytes = 18 if kind == "Q4_0" else 34
            require(data.nbytes == rows * (cols // 32) * block_bytes, "packed matrix byte size mismatch")
            raw = np.ascontiguousarray(data).view(np.uint8).reshape(rows, cols // 32, block_bytes)
            padded = ((rows + 3) // 4) * 4
            if padded != rows:
                raw = np.pad(raw, ((0, padded - rows), (0, 0), (0, 0)))
            self.data = raw.reshape(padded // 4, 4, cols // 32, block_bytes).transpose(0, 2, 1, 3).copy()
        else:
            require(kind == "F16" and data.nbytes == rows * cols * 2, "invalid native FP16 matrix")
            self.data = np.ascontiguousarray(data, dtype="<f2").reshape(self.shape)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        require(x.shape == (self.shape[1],) and bool(np.isfinite(x).all()), "expected finite FP32 activation vector")
        x = np.ascontiguousarray(x)
        output = np.empty(self.shape[0], dtype=np.float32)
        args = (self.data.ctypes.data, x.ctypes.data, output.ctypes.data, *self.shape)
        function = {"Q4_0": self.library.gpt2_q4, "Q8_0": self.library.gpt2_q8, "F16": self.library.gpt2_f16}[self.kind]
        function(*args, self.threads)
        require(bool(np.isfinite(output).all()), "nonfinite native CPU output")
        return output


class CPUMatvec:
    def __init__(self, weights, mode="auto", threads=None):
        require(mode in ("auto", "native", "numpy"), "CPU kernels must be auto, native, or numpy")
        self.weights, self.mode, self.cache = weights, mode, {}
        self.library, self.reason = (None, None) if mode == "numpy" else native_library()
        if mode == "native":
            require(self.library is not None, self.reason)
        default_threads = self.library.default_threads if self.library is not None else 1
        selected_threads = os.environ.get("GPT2_CPU_THREADS", os.environ.get("OMP_NUM_THREADS", str(default_threads)).split(",")[0])
        self.threads = int(threads if threads is not None else selected_threads)
        require(1 <= self.threads <= 256, "GPT2_CPU_THREADS must be between 1 and 256")
        if self.library is not None and not self.library.gpt2_matvec_openmp():
            self.threads = 1

    def matrix(self, name, shape):
        if self.library is None or len(shape) != 2 or shape[1] % 16:
            return None
        key = (name, tuple(shape))
        if key not in self.cache:
            raw = self.weights.packed_matrix(name, shape) if hasattr(self.weights, "packed_matrix") else None
            if raw is None:
                raw = ("F16", self.weights.get(name, shape).astype("<f2"))
            self.cache[key] = PackedMatrix(*raw, shape, self.library, self.threads)
        return self.cache[key]

    def __call__(self, name, shape, x):
        if np.ndim(x) == 1:
            matrix = self.matrix(name, shape)
            if matrix is not None:
                return matrix(x)
        return self.weights.get(name, shape) @ x

    @property
    def description(self):
        return f"NEON packed Q4/Q8/FP16, {self.threads} CPU threads" if self.library is not None else "NumPy"


def make_matvec(weights, selection):
    return selection if isinstance(selection, CPUMatvec) else CPUMatvec(weights, selection)
