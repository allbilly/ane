"""Check compact payload integrity and reject unsafe reconstruction recipes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zlib

import numpy as np

from qwen35.tests import write_tensors
from whisper.encoder_kernel import reconstruct, pack_port, port_view, unpack
from experimental.capture_macos_program import parse_tasks
from whisper.replay_encoder import Encoder


class CompactWhisperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "test.safetensors"
        self.weight = np.array([1., -2., 3.25], np.float32)
        positions = np.zeros((1500, 384), np.float32)
        write_tensors(self.checkpoint, {"matrix":self.weight, "model.encoder.embed_positions.weight":positions})
        self.template = bytes(6)
        value = self.weight.astype("<f2").tobytes()
        self.meta = dict(checkpoint=dict(sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()),
                         position_sha256=hashlib.sha256(positions.astype("<f2").tobytes()).hexdigest(),
                         payloads=dict(coefficients=dict(template="template.zlib", bytes=6,
                          sha256=hashlib.sha256(value).hexdigest(), operations=[dict(kind="tensor", tensor="matrix", offset=0, bytes=6)])))
        mil = b"test MIL source"
        (self.root / "model.mil.zlib").write_bytes(zlib.compress(mil))
        self.meta["mil"] = dict(file="model.mil.zlib", bytes=len(mil), sha256=hashlib.sha256(mil).hexdigest())

    def save(self):
        (self.root / "template.zlib").write_bytes(zlib.compress(self.template))
        self.meta["payloads"]["coefficients"]["template_metadata"] = dict(bytes=len(self.template), sha256=hashlib.sha256(self.template).hexdigest())
        (self.root / "meta.json").write_text(json.dumps(self.meta))

    def test_checkpoint_values_are_repacked(self):
        self.save()
        _, result = reconstruct(self.checkpoint, self.root)
        self.assertEqual(result["coefficients"], self.weight.astype("<f2").tobytes())

    def test_overlap_and_learned_template_bytes_are_rejected(self):
        for learned in (False, True):
            if learned:
                self.template = b"\1" + bytes(5)
            else:
                self.meta["payloads"]["coefficients"]["operations"] *= 2
            self.save()
            with self.assertRaisesRegex(ValueError, "overlapping packing"):
                reconstruct(self.checkpoint, self.root)

    def test_corrupt_asset_and_checkpoint_are_rejected(self):
        self.save()
        (self.root / "template.zlib").write_bytes(zlib.compress(b"\1" + bytes(5)))
        with self.assertRaisesRegex(ValueError, "asset checksum"):
            reconstruct(self.checkpoint, self.root)
        self.checkpoint.write_bytes(self.checkpoint.read_bytes()[:-1] + b"\1")
        with self.assertRaisesRegex(ValueError, "checkpoint checksum"):
            reconstruct(self.checkpoint, self.root)

    def test_wrong_offsets_and_payload_hash_are_rejected(self):
        self.meta["payloads"]["coefficients"]["operations"][0]["offset"] = -1
        self.save()
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            reconstruct(self.checkpoint, self.root)
        self.meta["payloads"]["coefficients"]["operations"][0]["offset"] = 0
        self.meta["payloads"]["coefficients"]["sha256"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "differs from captured"):
            reconstruct(self.checkpoint, self.root)


class NativePortTests(unittest.TestCase):
    def test_committed_metadata_describes_the_complete_task_chain(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("tiny-en-encoder-fast", "tiny-en-encoder"):
            kernels = root / "whisper/kernels" / name
            meta = json.loads((kernels / "meta.json").read_text())
            recipe = meta["payloads"]["commands"]
            commands = unpack(kernels, recipe["template"], recipe["template_metadata"])
            tasks = parse_tasks(commands, meta["td_size"], meta["td_count"])
            self.assertEqual(len(tasks), meta["td_count"])
            if name.endswith("-fast"):
                benchmark = json.loads((root / "whisper/results/benchmark-macos/summary.json").read_text())
                self.assertEqual(meta["mil"]["sha256"], benchmark["mil_sha256"])

    def port(self, width, channels, row):
        return dict(byte_offset=0, compiler_layout=dict(Type="Float16", Depth=1, Interleave=1,
            Batches=1, Channels=channels, Height=1, Width=width, PlaneCount=channels,
            RowStride=row, PlaneStride=row, BatchStride=channels * row))

    def test_native_mel_and_position_rows_do_not_shift_channels(self):
        for width, channels, row in ((3000, 80, 6016), (1500, 384, 3008)):
            port = self.port(width, channels, row)
            # Distinct channels reveal a tight-copy error at every row boundary.
            values = np.broadcast_to(np.arange(channels, dtype="<f2")[:, None] + 1, (channels, width))
            packed = pack_port(values, port)
            self.assertEqual(len(packed), channels * row)
            for channel in (0, 1, channels - 1):
                self.assertEqual(packed[channel * row:channel * row + width * 2], values[channel].tobytes())
                self.assertEqual(packed[channel * row + width * 2:(channel + 1) * row], bytes(row - width * 2))
            np.testing.assert_array_equal(port_view(packed, port).reshape(channels, width), values)

    def test_overlapping_strides_and_short_buffers_are_rejected(self):
        port = self.port(1500, 384, 3008)
        with self.assertRaisesRegex(ValueError, "exceeds buffer"):
            port_view(bytes(384 * 3000), port)
        port["compiler_layout"]["RowStride"] = 2998
        with self.assertRaisesRegex(ValueError, "overlapping.*strides"):
            pack_port(np.zeros((384, 1500), "<f2"), port)

    def test_dense_baseline_and_output_bits_are_preserved(self):
        port = self.port(384, 1, 768)
        port["compiler_layout"].update(Height=1500, PlaneStride=1152000, BatchStride=1152000)
        bits = np.arange(1500 * 384, dtype="<u2").reshape(1500, 384)
        buffer = bits.tobytes()
        # Readback must preserve FP16 bits, including subnormals and NaNs.
        np.testing.assert_array_equal(port_view(buffer, port).reshape(1500, 384).view("<u2"), bits)


class ReplayTimingTests(unittest.TestCase):
    def test_stage_accounting_preserves_one_submission_and_output_bits(self):
        root = Path(__file__).resolve().parents[1]
        meta = json.loads((root / "whisper/kernels/tiny-en-encoder-fast/meta.json").read_text())
        class MemoryBuffer:
            def __init__(self, size):
                self.size, self.map = size, bytearray(size)
            def write(self, data):
                self.map[:len(data)] = data
        encoder = Encoder.__new__(Encoder)
        encoder.meta = meta
        encoder.ports = {role:next(p for p in meta["layout"]["ports"] if p["role"] == kind and
            np.prod([p["compiler_layout"][k] for k in ("Batches", "Channels", "Height", "Width")]) == count)
            for role, kind, count in (("mel", "input", 240000), ("output", "output", 576000))}
        encoder.buffers = {b["replay_bank"]:MemoryBuffer(b["size"]) for b in meta["layout"]["buffers"]}
        encoder.fd, encoder.submit_opcode, encoder.request = 7, 123, object()
        encoder.submissions, encoder.dispatch_seconds = 0, 0.
        calls = []
        expected = np.full((1500, 384), .125, "<f2")
        def submit(fd, opcode, request):
            calls.append((fd, opcode, request))
            port = encoder.ports["output"]
            port_view(encoder.buffers[port["replay_bank"]].map, port)[...] = expected.reshape(1, 1, 1500, 384)
        encoder.ioctl = submit
        observed = encoder(np.full((80, 3000), -.5, "<f2"))
        self.assertEqual(calls, [(7, 123, encoder.request)])
        self.assertEqual(encoder.submissions, 1)
        self.assertEqual(observed.tobytes(), expected.tobytes())
        parts = encoder.last_timing_ms
        self.assertAlmostEqual(parts["total_ms"], sum(parts[k] for k in ("prepare_ms", "dispatch_ms", "readback_ms")), places=8)
        self.assertAlmostEqual(parts["dispatch_ms"], encoder.dispatch_seconds * 1000, places=8)
        self.assertTrue(all(t >= 0 for t in parts.values()))


if __name__ == "__main__":
    unittest.main()
