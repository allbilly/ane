"""Check real task structures and a fake transport, without claiming ANE execution."""
import copy
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gpt2.hwx import parse_container, parse_tasks, relocate
from qwen35.weights import sha256
from whisper.encoder_kernel import unpack
from whisper.pr_encoder_replay import Encoder, GATE, benchmark, digest, prepare, validate_fixtures

ROOT = Path(__file__).resolve().parents[1] / "whisper/kernels/pr3905"


class ReplayPlanTests(unittest.TestCase):
    def test_all_compiled_task_structures_keep_packets_and_extra_bank(self):
        # These are stripped instruction fixtures; weighted exports are audited separately.
        for size, count, banks in (("tiny", 1779, 1), ("base", 3434, 1), ("small", 10015, 2)):
            with self.subTest(model=size):
                root = ROOT / size
                meta = json.loads((root / "meta.json").read_text())
                hwx = unpack(root, meta["template"]["file"], meta["template"])
                positions = bytes(meta["dimensions"]["state"] * 1500 * 2)
                meta["hwx_sha256"], meta["position_sha256"] = digest(hwx), digest(positions)
                plan, payloads = prepare(meta, hwx, positions)
                self.assertEqual(plan["td_count"], count)
                self.assertEqual(len(plan["coefficient_banks"]), banks)
                self.assertEqual(plan["bank_map"][1], 2)
                self.assertEqual(plan["bank_map"][7], 1)
                self.assertNotIn(1, [b["bank"] for b in plan["buffers"]])
                if banks == 2:
                    self.assertEqual(plan["bank_map"][8], 8)
                    self.assertEqual(len(payloads["coefficients-8"]), 40599552)
                container = parse_container(hwx)
                segment = next(s for s in container["segments"] if s["name"] == "__TEXT")
                original = hwx[segment["fileoff"]:segment["fileoff"] + segment["filesize"]]
                tasks = parse_tasks(payloads["commands"], plan["td_size"], count)
                self.assertEqual(relocate(payloads["commands"], tasks, {v:k for k,v in plan["bank_map"].items()}), original)
                self.assertEqual((struct.unpack_from("<I", payloads["bootstrap"])[0] >> 16) & 255, 0x40)

    def test_corrupt_container_identity_and_aliased_ports_are_rejected(self):
        root = ROOT / "tiny"
        base = json.loads((root / "meta.json").read_text())
        hwx = unpack(root, base["template"]["file"], base["template"])
        positions = bytes(384 * 1500 * 2)
        base["hwx_sha256"], base["position_sha256"] = digest(hwx), digest(positions)
        for mode in ("checksum", "thread", "ports", "positions"):
            with self.subTest(mode=mode):
                meta = copy.deepcopy(base)
                actual_positions = positions
                if mode == "checksum":
                    meta["hwx_sha256"] = "0" * 64
                elif mode == "thread":
                    meta["layout"]["thread"]["td_count"] -= 1
                elif mode == "ports":
                    meta["layout"]["ports"].append(copy.deepcopy(meta["layout"]["ports"][0]))
                else:
                    actual_positions = bytes(2)
                    meta["position_sha256"] = digest(actual_positions)
                with self.assertRaises(ValueError):
                    prepare(meta, hwx, actual_positions)


class MemoryBuffer:
    instances = []
    fail_at = None
    def __init__(self, fd, size):
        if self.fail_at is not None and len(self.instances) == self.fail_at:
            raise OSError("allocation failed")
        self.size, self.map = size, bytearray(size)
        self.handle, self.closed = len(self.instances) + 1, False
        self.instances.append(self)
    def write(self, data, offset=0):
        if offset < 0 or offset + len(data) > self.size:
            raise ValueError("fake buffer overflow")
        self.map[offset:offset + len(data)] = data
    def close(self):
        self.closed = True


class Submission:
    def __init__(self, **args):
        self.__dict__.update(args)
        self.handles = [0] * 32


def port(bank, channels, height, width, row):
    return dict(replay_bank=bank, byte_offset=0, compiler_layout=dict(Type="Float16", Depth=1, Interleave=1,
        Batches=1, Channels=channels, Height=height, Width=width, PlaneCount=channels,
        RowStride=row, PlaneStride=height * row, BatchStride=channels * height * row))


class ReplayTransportTests(unittest.TestCase):
    def setUp(self):
        MemoryBuffer.instances, MemoryBuffer.fail_at = [], None
        self.mel = np.array([[1, 2, 3, 4, 5], [101, 102, 103, 104, 105]], "<f2")
        self.positions = np.arange(1, 13, dtype="<f2").reshape(4, 3)
        self.expected = np.arange(1, 13, dtype="<f2").reshape(3, 4)
        self.plan = dict(model="toy", dimensions=dict(mels=2, frames=5, state=4, context=3),
            command_bytes=64, td_count=1, td_size=40, hwx_sha256="a" * 64, source_mil_sha256="b" * 64,
            buffers=[dict(bank=bank, size=size, role=role) for bank, size, role in
                     ((0, 97, "commands+coefficients"), (2, 16, "constants"), (3, 16, "scratch"),
                      (4, 32, "input"), (5, 24, "input"), (6, 24, "output"), (8, 32, "coefficients"))],
            coefficient_banks=[dict(replay_bank=1), dict(replay_bank=8, payload="coefficients-8")],
            ports=dict(mel=port(5, 2, 1, 5, 12), positions=port(4, 4, 1, 3, 8), output=port(6, 1, 3, 4, 8)),
            payloads=dict(positions=dict(sha256=digest(self.positions.tobytes()))))
        self.payloads = dict(commands=bytes(range(64)), constants=b"c" * 16, positions=self.positions.tobytes(),
                             bootstrap=b"b" * 40, **{"coefficients-1":b"1" * 32, "coefficients-8":b"8" * 32})
        self.encoder = Encoder.__new__(Encoder)
        self.encoder.plan, self.encoder.fd = self.plan, 99
        self.encoder.buffers, self.encoder.bootstrap = {}, None
        self.encoder.submissions, self.encoder.last_timing_ms = 0, None
        self.encoder.submit_opcode, self.encoder.ioctl = 123, self.submit
        self.encoder.allocate(self.payloads, MemoryBuffer, Submission)

    def submit(self, fd, opcode, request):
        self.assertEqual((fd, opcode), (99, 123))
        self.assertEqual(request.handles[1], 0)
        self.assertNotEqual(request.handles[8], 0)
        self.assertEqual(request.tsk_size, 64)
        self.assertEqual(self.encoder.buffers[0].map[:64], self.payloads["commands"])
        self.assertEqual(self.encoder.buffers[0].map[64:96], self.payloads["coefficients-1"])
        self.assertEqual(self.encoder.buffers[8].map, b"8" * 32)
        self.assertEqual(self.encoder.buffers[3].map, bytes(16))
        for row in range(2):
            start = row * 12
            self.assertEqual(self.encoder.buffers[5].map[start:start + 10], self.mel[row].tobytes())
            self.assertEqual(self.encoder.buffers[5].map[start + 10:start + 12], b"\0\0")
        self.assertTrue(np.isnan(np.frombuffer(self.encoder.buffers[6].map, "<f2")).all())
        self.encoder.buffers[6].map[:] = self.expected.tobytes()

    def test_one_submission_preserves_padding_extra_coefficients_and_output(self):
        self.encoder.buffers[3].map[:] = b"\xff" * 16
        output = self.encoder(self.mel)
        self.assertEqual(output.tobytes(), self.expected.tobytes())
        self.assertEqual(self.encoder.submissions, 1)
        for channel in range(4):
            self.assertEqual(self.encoder.buffers[4].map[channel * 8:channel * 8 + 6], self.positions[channel].tobytes())
            self.assertEqual(self.encoder.buffers[4].map[channel * 8 + 6:channel * 8 + 8], bytes(2))
        times = self.encoder.last_timing_ms
        self.assertAlmostEqual(times["total_ms"], sum(times[k] for k in ("prepare_ms", "dispatch_ms", "readback_ms")), places=8)

    def test_failed_ioctl_does_not_count_a_completed_submission(self):
        def fail(*_):
            raise OSError("ioctl failed")
        self.encoder.ioctl = fail
        with self.assertRaisesRegex(OSError, "ioctl failed"):
            self.encoder(self.mel)
        self.assertEqual(self.encoder.submissions, 0)

    def test_unwritten_output_is_rejected(self):
        self.encoder.ioctl = lambda *_: None
        with self.assertRaisesRegex(ValueError, "unwritten output"):
            self.encoder(self.mel)

    def test_every_repetition_is_checked_and_warmups_are_excluded(self):
        report = benchmark(self.encoder, [("case", dict(mel=self.mel, output=self.expected))], 2, 3)
        self.assertEqual(report["submissions"], 5)
        samples = report["records"][0]["samples"]
        self.assertEqual([s["phase"] for s in samples], ["warmup"] * 2 + ["measured"] * 3)
        self.assertEqual(report["gate"], GATE)
        self.assertIn("unverified", report["strict_decoder_accuracy"])
        with self.assertRaisesRegex(ValueError, "captured encoder output mismatch"):
            benchmark(self.encoder, [("wrong", dict(mel=self.mel, output=self.expected + 10))], 0, 1)

    def test_native_host_guard_precedes_device_access(self):
        with patch("whisper.pr_encoder_replay.platform.system", return_value="Darwin"), patch("whisper.pr_encoder_replay.os.open") as open_device:
            with self.assertRaisesRegex(ValueError, "native M1 Asahi Linux"):
                Encoder(Path("missing"), Path("missing"))
            open_device.assert_not_called()

    def test_allocation_failure_closes_partial_buffers_and_device(self):
        MemoryBuffer.instances, MemoryBuffer.fail_at = [], 3
        bindings = SimpleNamespace(Buffer=MemoryBuffer, Submit=Submission, SUBMIT=123,
                                   ioctl=lambda *_:None, device_path=lambda _:Path("/dev/fake"))
        with patch("whisper.pr_encoder_replay.platform.system", return_value="Linux"), \
             patch.dict("sys.modules", replay=bindings), \
             patch("whisper.pr_encoder_replay.reconstruct", return_value=({}, b"", b"")), \
             patch("whisper.pr_encoder_replay.prepare", return_value=(self.plan, self.payloads)), \
             patch("whisper.pr_encoder_replay.os.open", return_value=99), \
             patch("whisper.pr_encoder_replay.os.close") as close_device:
            with self.assertRaisesRegex(OSError, "allocation failed"):
                Encoder(Path("fake"), Path("fake"))
            close_device.assert_called_once_with(99)
        self.assertEqual(len(MemoryBuffer.instances), 3)
        self.assertTrue(all(b.closed for b in MemoryBuffer.instances))

    def test_fixture_hashes_positions_and_gate_checked_before_device_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "case.npz"
            np.savez(path, mel=self.mel, positions=self.positions, output=self.expected)
            case = dict(name="case", file=path.name, sha256=sha256(path),
                        arrays={n:dict(sha256=digest(v.tobytes())) for n,v in
                                (("mel", self.mel), ("positions", self.positions), ("output", self.expected))})
            manifest = dict(format="whisper-pr3905-replay-fixtures/v1", gate=GATE.copy(),
                            models=dict(toy=dict(hwx_sha256=self.plan["hwx_sha256"], mil_sha256=self.plan["source_mil_sha256"], cases=[case])))
            self.assertEqual(len(validate_fixtures(self.plan, manifest, root)), 1)
            for mode in ("gate", "hash", "positions"):
                bad = copy.deepcopy(manifest)
                if mode == "gate":
                    bad["gate"]["relative_l2"] = .1
                elif mode == "hash":
                    bad["models"]["toy"]["cases"][0]["sha256"] = "0" * 64
                else:
                    bad["models"]["toy"]["cases"][0]["arrays"]["positions"]["sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    validate_fixtures(self.plan, bad, root)


if __name__ == "__main__":
    unittest.main()
