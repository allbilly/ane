"""Replay checkpoint-reconstructed paired projections through the M1 DRM ABI."""
import os
import ctypes
from pathlib import Path
import sys
import time

import numpy as np

from whisper.encoder_kernel import pack_port, port_view, require
from whisper.paired_kernel import ROOT, pack_coefficients, reconstruct, sha256


class Projection:
    def __init__(self, program, fd, bindings, reader=None):
        self.meta, self.fd, self.bindings = program["meta"], fd, bindings
        self.reader, self.last_read_workers = reader, 1
        self.buffers, self.bootstrap = {}, None
        self.submissions, self.timings = 0, []
        self.ports = {p["role"]:p for p in self.meta["layout"]["ports"]}
        self.feed = np.zeros((1, self.meta["input_features"], 1, 6000), "<f2")
        self.output_bytes = next(p["size"] for p in self.meta["layout"]["buffers"] if p["replay_bank"] == 5)
        self.output_cache = bytearray(self.output_bytes)
        self.receipt = dict(task_count=self.meta["td_count"], target=self.meta["target"],
                            runtime="Asahi ANE DRM", payload_sha256={
                                name:sha256(program[name]) for name in ("commands", "constants", "coefficients", "bootstrap")})
        try:
            sizes = {0:len(program["commands"]) + len(program["coefficients"]) + 1,
                     2:len(program["constants"]),
                     **{p["replay_bank"]:p["size"] for p in self.meta["layout"]["buffers"]}}
            for bank, size in sizes.items():
                self.buffers[bank] = bindings.Buffer(fd, size)
            self.buffers[0].write(program["commands"])
            self.buffers[0].write(program["coefficients"], len(program["commands"]))
            self.buffers[2].write(program["constants"])
            self.bootstrap = bindings.Buffer(fd, self.meta["td_size"])
            self.bootstrap.write(program["bootstrap"])
            self.request = bindings.Submit(tsk_size=len(program["commands"]),
                                          td_count=self.meta["td_count"], td_size=self.meta["td_size"],
                                          btsp_handle=self.bootstrap.handle)
            for bank, buffer in self.buffers.items():
                self.request.handles[bank] = buffer.handle
            require(self.request.handles[1] == 0, "paired coefficient BAR must be synthesized")
            self.buffers[4].write(bytes(self.buffers[4].size))
        except BaseException:
            self.release()
            raise

    def input_view(self):
        return self.feed

    def output_view(self):
        return port_view(self.output_cache, self.ports["output"])

    def execute(self):
        require(bool(np.isfinite(self.input_view()).all()), "nonfinite paired input")
        self.buffers[4].write(pack_port(self.feed, self.ports["input"], self.buffers[4].size))
        self.output_cache[:] = b"\x00\x7e" * (self.output_bytes // 2)
        self.buffers[5].write(self.output_cache)
        began = time.perf_counter()
        self.bindings.ioctl(self.fd, self.bindings.SUBMIT, self.request)
        self.timings.append((time.perf_counter() - began) * 1000)
        self.submissions += 1
        # Copy bytes before dtype conversion/checks: scalar NumPy arithmetic
        # directly on uncached DRM mappings incurs a transaction per element.
        if self.reader:
            target = ctypes.c_ubyte.from_buffer(self.output_cache)
            source = ctypes.c_ubyte.from_buffer(self.buffers[5].map)
            self.last_read_workers = self.reader(ctypes.addressof(target), ctypes.addressof(source), self.output_bytes, 4)
            require(self.last_read_workers == 4, "paired native reader requires four workers")
        else:
            self.output_cache[:] = self.buffers[5].map[:self.output_bytes]
        require(bool(np.isfinite(self.output_view()).all()), "unwritten/nonfinite paired output")

    def release(self):
        buffers = list(self.buffers.values()) + ([self.bootstrap] if self.bootstrap else [])
        self.buffers, self.bootstrap = {}, None
        for buffer in reversed(buffers):
            buffer.close()
        self.feed, self.output_cache = None, bytearray()


class PairedReplay:
    def __init__(self, checkpoint, root=ROOT):
        # Validate all 24 executable/weight payloads before opening hardware.
        self.manifest, self.payloads = reconstruct(checkpoint, root)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpt2"))
        import replay
        self.bindings = replay
        self.reader, self.reader_receipt = None, {}
        if path := os.environ.get("WHISPER_ASAHI_READER"):
            self.reader_library = ctypes.CDLL(str(Path(path).resolve()))
            self.reader = self.reader_library.ane_copy_uncached
            self.reader.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            self.reader.restype = ctypes.c_int
            self.reader_receipt = dict(reader_binary_sha256=sha256(Path(path).read_bytes()),
                                       reader_source="qwen35/ane_matmul.c::ane_copy_uncached")
        self.fd = os.open(replay.device_path(), os.O_RDWR | os.O_CLOEXEC)
        self.programs = []

    def projection(self, weights, name):
        require(name in self.payloads, "unknown paired projection")
        program = self.payloads[name]
        meta = program["meta"]
        require(weights.shape == (meta["output_features"], meta["input_features"]), "paired model dimensions differ")
        require(pack_coefficients(weights, meta["coefficient_tiles"], meta["coefficient_bytes"])
                == program["coefficients"], "paired model weights differ from checkpoint")
        result = Projection(program, self.fd, self.bindings, self.reader)
        result.receipt.update(self.reader_receipt)
        self.programs.append(result)
        return result

    def close(self):
        try:
            for program in self.programs:
                program.release()
            self.programs.clear()
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
