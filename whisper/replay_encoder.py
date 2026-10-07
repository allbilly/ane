"""Replay the complete compact tiny.en encoder on base-M1 Asahi Linux."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys
import time

import numpy as np

from whisper.encoder_kernel import reconstruct, require, ROOT, pack_port, port_view, port_shape


class Encoder:
    """One driver submission runs all convolutions, attention and encoder layers."""
    def __init__(self, checkpoint, device=None, kernels=ROOT):
        # Resolve the existing guarded DRM ABI only on its supported host.
        require(platform.system() == "Linux", "encoder replay requires native M1 Asahi Linux")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpt2"))
        from replay import Buffer, Submit, SUBMIT, device_path, ioctl
        from hwx import parse_tasks
        path = device_path(device)
        self.meta, payloads = reconstruct(checkpoint, kernels)
        require(self.meta["target"] == "apple,t8103" and self.meta["td_count"] in (1779, 1783), "unsupported encoder target")
        ports = self.meta["layout"]["ports"]
        self.ports = {}
        for role, kind, count in (("mel", "input", 80 * 3000), ("positions", "input", 384 * 1500),
                                  ("output", "output", 1500 * 384)):
            matches = [p for p in ports if p["role"] == kind and int(np.prod(port_shape(p))) == count]
            require(len(matches) == 1, "ambiguous encoder port: " + role)
            self.ports[role] = matches[0]
        commands = payloads["commands"]
        parse_tasks(commands, self.meta["td_size"], self.meta["td_count"])
        self.fd, self.buffers, self.bootstrap = os.open(path, os.O_RDWR | os.O_CLOEXEC), {}, None
        self.ioctl, self.submit_opcode = ioctl, SUBMIT
        self.submissions, self.dispatch_seconds = 0, 0.
        try:
            self.buffers[0] = Buffer(self.fd, len(commands) + len(payloads["coefficients"]) + 1)
            self.buffers[0].write(commands + payloads["coefficients"])
            self.buffers[2] = Buffer(self.fd, len(payloads["constants"]))
            self.buffers[2].write(payloads["constants"])
            for item in self.meta["layout"]["buffers"]:
                bank = item["replay_bank"]
                require(3 <= bank < 32 and bank not in self.buffers, "invalid encoder buffer bank")
                self.buffers[bank] = Buffer(self.fd, item["size"])
            port = self.ports["positions"]
            buffer = self.buffers[port["replay_bank"]]
            buffer.write(pack_port(np.frombuffer(payloads["positions"], "<f2"), port, buffer.size))
            self.bootstrap = Buffer(self.fd, self.meta["td_size"])
            bootstrap = bytearray(commands[:self.meta["td_size"]])
            header, = struct.unpack_from("<I", bootstrap)
            struct.pack_into("<I", bootstrap, 0, (header & ~(0xFF << 16)) | (0x40 << 16))
            self.bootstrap.write(bootstrap)
            self.request = Submit(tsk_size=len(commands), td_count=self.meta["td_count"], td_size=self.meta["td_size"],
                                  btsp_handle=self.bootstrap.handle)
            for bank, buffer in self.buffers.items():
                self.request.handles[bank] = buffer.handle
        except BaseException:
            self.close()
            raise

    def __call__(self, mel):
        mel = np.asarray(mel, dtype="<f2")
        require(mel.shape == (80, 3000) and bool(np.isfinite(mel).all()), "expected finite FP16 mel [80,3000]")
        port = self.ports["mel"]
        buffer = self.buffers[port["replay_bank"]]
        buffer.write(pack_port(mel, port, buffer.size))
        for item in self.meta["layout"]["buffers"]:
            if item["role"] == "scratch/intermediate":
                buffer = self.buffers[item["replay_bank"]]
                buffer.write(bytes(buffer.size))
        output_port = self.ports["output"]
        output_buffer = self.buffers[output_port["replay_bank"]]
        port_view(output_buffer.map, output_port)[...] = np.nan
        start = time.perf_counter()
        self.ioctl(self.fd, self.submit_opcode, self.request)
        self.dispatch_seconds += time.perf_counter() - start
        self.submissions += 1
        output = port_view(output_buffer.map, output_port).reshape(1500, 384).copy()
        require(bool(np.isfinite(output).all()), "encoder produced nonfinite/unwritten output")
        return output

    def close(self):
        try:
            for buffer in reversed(list(self.buffers.values()) + ([self.bootstrap] if self.bootstrap else [])):
                buffer.close()
        finally:
            self.buffers, self.bootstrap = {}, None
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--kernels", type=Path, default=ROOT)
    p.add_argument("--fixtures", type=Path, required=True, help="Local recovered kit containing the three hashed fixtures")
    p.add_argument("--device")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(platform.system() == "Linux", "requires native Asahi Linux; no hardware submission attempted")
    proof = json.loads((a.kernels / "proof.json").read_text())
    record, = proof["fixtures"]["records"]
    require(len(record["fixtures"]) == 3, "requires all three captured real-audio fixtures")
    report = dict(status="running", kernel=platform.release(), records=[])
    encoder = Encoder(a.checkpoint, a.device, a.kernels)
    try:
        for fixture in record["fixtures"]:
            path = a.fixtures / fixture["path"]
            require(hashlib.sha256(path.read_bytes()).hexdigest() == fixture["sha256"], "fixture checksum mismatch")
            with np.load(path, allow_pickle=False) as data:
                expected = data["output"].reshape(1500, 384).astype(np.float32)
                hf = data["hf_output"].reshape(1500, 384).astype(np.float32)
                position_port = encoder.ports["positions"]
                require(np.array_equal(data["input01"].ravel(), port_view(encoder.buffers[position_port["replay_bank"]].map, position_port).ravel()),
                        "checkpoint position input differs from fixture")
                actual = encoder(data["input00"].reshape(80, 3000)).astype(np.float32)
            relative = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-40))
            cosine = float(np.dot(actual.ravel(), hf.ravel()) / (np.linalg.norm(actual) * np.linalg.norm(hf)))
            passed = relative < .005 and bool(np.allclose(actual, expected, rtol=.01, atol=.03)) and cosine >= .999
            report["records"].append(dict(fixture=fixture["path"], relative_l2=relative, hf_cosine=cosine, pass_gate=passed))
            require(passed, "encoder capture/HF gate failed")
        report.update(status="pass", submissions=encoder.submissions, dispatch_seconds=encoder.dispatch_seconds)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        encoder.close()
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
