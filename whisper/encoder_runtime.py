"""Common encoder inputs, outputs and timing boundaries; narrow runtime adapters."""
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from whisper.encoder_kernel import ROOT, encoder_ports, port_shape, reconstruct, require, write_bundle


class EncoderRuntime:
    def __init__(self, checkpoint, kernels=ROOT):
        self.meta, self.payloads = reconstruct(checkpoint, kernels)
        require(self.meta["target"] == "apple,t8103" and self.meta["td_count"] in (1779, 1783),
                "unsupported encoder target")
        self.ports = encoder_ports(self.meta)
        self.submissions, self.dispatch_seconds = 0, 0.
        self.last_timing_ms = None

    def __call__(self, mel):
        began = time.perf_counter()
        mel = np.asarray(mel, dtype="<f2")
        require(mel.shape == (80, 3000) and bool(np.isfinite(mel).all()), "expected finite FP16 mel [80,3000]")
        self.prepare(mel)
        start = time.perf_counter()
        self.execute()
        completed = time.perf_counter()
        self.dispatch_seconds += completed - start
        self.submissions += 1
        output = np.asarray(self.read_output(), dtype="<f2").reshape(1500, 384).copy()
        require(bool(np.isfinite(output).all()), "encoder produced nonfinite/unwritten output")
        finished = time.perf_counter()
        self.last_timing_ms = dict(prepare_ms=(start-began)*1000,
            dispatch_ms=(completed-start)*1000, readback_ms=(finished-completed)*1000,
            total_ms=(finished-began)*1000)
        return output


class MacEncoder(EncoderRuntime):
    """Execute the same reconstructed MIL and checkpoint through ANEForge E5RT."""
    backend = "macos"

    def __init__(self, checkpoint, kernels=ROOT, work_dir=None):
        require(platform.system() == "Darwin" and platform.machine() == "arm64", "requires native Apple Silicon macOS")
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        require(chip == "Apple M1", "captured encoder comparison requires the same base M1")
        super().__init__(checkpoint, kernels)
        from aneforge._runtime import E5RT
        work_dir = Path(work_dir or Path(__file__).resolve().parent / "build/encoder-runtime" / Path(kernels).name)
        write_bundle(work_dir, self.meta, self.payloads)
        self.program = None
        try:
            inputs = {p["name"]:port_shape(p) for p in self.ports.values() if p["role"] == "input"}
            output_name = self.ports["output"]["name"]
            self.program = E5RT.compile(work_dir / "model.mil", cache_dir=work_dir / "compiled",
                inputs=inputs, outputs={output_name:(1500, 384)}, device_mask=4)
            port = self.ports["positions"]
            self.program.set_input(port["name"], np.frombuffer(self.payloads["positions"], "<f2").reshape(port_shape(port)))
        except BaseException:
            self.close()
            raise

    def prepare(self, mel):
        port = self.ports["mel"]
        self.program.set_input(port["name"], mel.reshape(port_shape(port)))
        self.program.output_view(self.ports["output"]["name"])[...] = np.nan

    def execute(self):
        self.program.execute()

    def read_output(self):
        return self.program.read_output(self.ports["output"]["name"])

    def close(self):
        if self.program is not None:
            self.program.release()
            self.program = None


def open_encoder(checkpoint, kernels=ROOT, backend="auto", device=None, work_dir=None):
    if backend == "auto":
        backend = {"Linux":"asahi", "Darwin":"macos"}.get(platform.system())
    require(backend in ("asahi", "macos"), "supported backends are asahi and macos")
    if backend == "macos":
        require(device is None, "--device selects an Asahi DRM device")
        return MacEncoder(checkpoint, kernels, work_dir)
    from whisper.replay_encoder import Encoder
    return Encoder(checkpoint, device, kernels)
