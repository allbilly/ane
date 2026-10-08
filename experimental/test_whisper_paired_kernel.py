"""Check checkpoint reconstruction, padded ports and strict paired export guards."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
import zlib

import numpy as np

from whisper.encoder_kernel import pack_port, port_view
from whisper.paired_kernel import ROOT, pack_coefficients, reconstruct, validate_layout, write_payloads, validate_payloads
from whisper.scripts.package_asahi_precision import accuracy_receipt
from whisper.paired_replay import Projection
from experimental.test_pr_encoder_replay import MemoryBuffer, Submission

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "whisper/models/hf-tiny.en/model.safetensors"


class PairedKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.programs = reconstruct(CHECKPOINT)

    def test_all_24_checkpoint_payloads_match_captured_bytes_and_task_counts(self):
        proof = json.loads((ROOT / "proof.json").read_text())
        self.assertEqual(proof["status"], "PASS_RECONSTRUCTION")
        self.assertEqual(proof["submissions_per_encode"], 24)
        self.assertEqual(proof["hardware_tasks_per_encode"], 32)
        self.assertEqual(proof["linux_hardware_accuracy_and_performance"], "pending")
        self.assertEqual(proof["learned_coefficient_bytes_reconstructed"], 14155776)
        expected = {row["name"]: row for row in self.manifest["programs"]}
        self.assertEqual(len(self.programs), 24)
        for name, program in self.programs.items():
            self.assertEqual(hashlib.sha256(program["coefficients"]).hexdigest(),
                             expected[name]["coefficients_sha256"])
            bootstrap = bytearray(program["commands"][:628])
            bootstrap[2] = 0x40
            self.assertEqual(bytes(bootstrap), program["bootstrap"])
        for name, record in proof["files"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), record["sha256"])

    def test_all_four_planes_round_trip_with_zero_hardware_padding(self):
        for name in ("layer0-q_proj", "layer0-fc1", "layer0-fc2"):
            meta = self.programs[name]["meta"]
            for port in meta["layout"]["ports"]:
                channels = port["compiler_layout"]["Channels"]
                logical = np.zeros((1, channels, 1, 6000), "<f2")
                # Distinguish each temporal plane, channel and final position.
                for plane in range(4):
                    logical[0, :, 0, plane*1500] = np.arange(channels) % 31 + plane/4
                    logical[0, :, 0, (plane+1)*1500-1] = -(plane+1)
                packed = pack_port(logical, port)
                self.assertTrue(np.array_equal(port_view(packed, port), logical))
                rows = np.frombuffer(packed, np.uint8).reshape(channels, 12032)
                self.assertFalse(rows[:, 12000:].any())
                self.assertEqual(len(packed), channels*12032)

    def test_layout_rejects_dense_ports_and_wrong_target_or_task_count(self):
        original = self.programs["layer0-q_proj"]["meta"]
        for key, value in (("target", "apple,t6000"), ("td_count", 2),
                           ("temporal_planes", 2), ("gains", [1., 1.])):
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_layout(changed)
        changed = copy.deepcopy(original)
        changed["layout"]["ports"][0]["compiler_layout"]["PlaneStride"] = 12000
        with self.assertRaises(ValueError):
            validate_layout(changed)

    def test_coefficient_map_rejects_missing_duplicate_and_nonexact_weights(self):
        meta = self.programs["layer0-q_proj"]["meta"]
        from whisper.scripts.prepare_macos_precision import checkpoint_weights
        weights = checkpoint_weights(CHECKPOINT)["layer0-q_proj"]
        tiles = meta["coefficient_tiles"]
        for altered in (tiles[:-1], tiles + [tiles[0]],
                        [dict(tiles[0], offset=-1)] + tiles[1:]):
            with self.assertRaises(ValueError):
                pack_coefficients(weights, altered, meta["coefficient_bytes"])
        changed = weights.copy()
        changed[0, 0] += np.float32(1e-7)
        with self.assertRaises(ValueError):
            pack_coefficients(changed, tiles, meta["coefficient_bytes"])

    def test_changed_assets_and_partial_program_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"kernels"
            shutil.copytree(ROOT, path)
            asset = path/"384-384/commands.zlib"
            original = asset.read_bytes()
            data = bytearray(zlib.decompress(original))
            data[-1] ^= 1
            asset.write_bytes(zlib.compress(data))
            with self.assertRaisesRegex(ValueError, "checksum"):
                reconstruct(CHECKPOINT, path)
            asset.write_bytes(original)
            manifest = copy.deepcopy(self.manifest)
            manifest["programs"].pop()
            (path/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "24 projections"):
                reconstruct(CHECKPOINT, path)

    def test_native_receipt_rejects_a_single_failed_full_vector(self):
        native = REPO/"whisper/results/macos-native-precision-20261007.json"
        programs = json.loads(native.read_text())["precision_programs"]
        receipt = accuracy_receipt(native, programs)
        receipt["correctness"][-1]["hf_logit_checks"][-1]["ane"]["nrmse"] = .005
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"receipt.json"
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "full-vector gate"):
                accuracy_receipt(path, programs)

    def test_native_payloads_bind_templates_weights_and_descriptors(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"payloads"
            write_payloads(path, self.manifest, self.programs)
            self.assertEqual(validate_payloads(path, CHECKPOINT), self.manifest)
            descriptor = path/"layer0-fc1/native-layout.txt"
            original = descriptor.read_text()
            descriptor.write_text(original.replace(" 12032 ", " 12000 ", 1))
            with self.assertRaisesRegex(ValueError, "descriptor changed"):
                validate_payloads(path, CHECKPOINT)
            descriptor.write_text(original)
            coefficients = path/"layer0-fc1/coefficients.bin"
            changed = bytearray(coefficients.read_bytes())
            changed[-1] ^= 1
            coefficients.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "differs from checkpoint/template"):
                validate_payloads(path, CHECKPOINT)

    def test_native_runtime_gate_rejects_fallback_and_partial_execution(self):
        from whisper.scripts.benchmark_native import check_runtime
        base = "use gpu = 0\nfallbacks = 0 p / 0 h\nASAHI_PRECISION ready: shared arithmetic\n"
        complete = "ASAHI_PRECISION encoder: projections=24 submissions=24 tasks=32 read_workers=4\n"
        check_runtime(base+complete, 3, 1)
        for changed in (complete.replace("tasks=32", "tasks=24"),
                        complete.replace("read_workers=4", "read_workers=1"), ""):
            with self.assertRaises(ValueError):
                check_runtime(base+changed, 3, 1)
        with self.assertRaisesRegex(ValueError, "fallback"):
            check_runtime((base+complete).replace("0 p / 0 h", "1 p / 0 h"), 3, 1)
        with self.assertRaisesRegex(ValueError, "unrequested"):
            check_runtime(base+complete, 0, 1)


class PairedReplayTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.programs = reconstruct(CHECKPOINT)

    def setUp(self):
        MemoryBuffer.instances, MemoryBuffer.fail_at = [], None

    def test_each_shape_submits_complete_tasks_and_preserves_four_padded_planes(self):
        for name in ("layer0-q_proj", "layer0-fc1", "layer0-fc2"):
            program = self.programs[name]
            def submit(fd, opcode, request):
                self.assertEqual((fd, opcode), (99, 123))
                self.assertEqual((request.td_count, request.td_size),
                                 (program["meta"]["td_count"], 628))
                self.assertEqual(request.handles[1], 0)
                self.assertEqual(request.tsk_size, len(program["commands"]))
                self.assertEqual(runtime.buffers[0].map[:request.tsk_size], program["commands"])
                self.assertEqual(runtime.buffers[0].map[request.tsk_size:-1], program["coefficients"])
                self.assertEqual(runtime.bootstrap.map, program["bootstrap"])
                rows = np.frombuffer(runtime.buffers[4].map, np.uint8).reshape(-1, 12032)
                self.assertFalse(rows[:, 12000:].any())
                view = runtime.input_view()
                for plane in range(4):
                    self.assertEqual(view[0, 0, 0, plane*1500], plane+1)
                self.assertTrue(np.isnan(runtime.output_view()).all())
                port_view(runtime.buffers[5].map, runtime.ports["output"])[...] = np.float16(2)
            bindings = SimpleNamespace(Buffer=MemoryBuffer, Submit=Submission, SUBMIT=123, ioctl=submit)
            runtime = Projection(program, 99, bindings)
            runtime.input_view()[...] = 0
            for plane in range(4):
                runtime.input_view()[0, 0, 0, plane*1500] = plane+1
            runtime.execute()
            self.assertEqual(runtime.submissions, 1)
            self.assertTrue((runtime.output_view() == 2).all())
            runtime.release()
            self.assertTrue(all(b.closed for b in MemoryBuffer.instances))
            MemoryBuffer.instances = []

    def test_unwritten_output_and_nonfinite_input_are_rejected(self):
        calls = []
        bindings = SimpleNamespace(Buffer=MemoryBuffer, Submit=Submission, SUBMIT=123,
                                   ioctl=lambda *_:calls.append(1))
        runtime = Projection(self.programs["layer0-q_proj"], 99, bindings)
        try:
            with self.assertRaisesRegex(ValueError, "unwritten/nonfinite"):
                runtime.execute()
            runtime.input_view()[0, 0, 0, 0] = np.inf
            with self.assertRaisesRegex(ValueError, "nonfinite paired input"):
                runtime.execute()
            self.assertEqual(len(calls), 1)
        finally:
            runtime.release()

    def test_partial_allocation_failure_releases_every_created_buffer(self):
        MemoryBuffer.fail_at = 3
        bindings = SimpleNamespace(Buffer=MemoryBuffer, Submit=Submission, SUBMIT=123)
        with self.assertRaisesRegex(OSError, "allocation failed"):
            Projection(self.programs["layer0-q_proj"], 99, bindings)
        self.assertEqual(len(MemoryBuffer.instances), 3)
        self.assertTrue(all(b.closed for b in MemoryBuffer.instances))


if __name__ == "__main__":
    unittest.main()
