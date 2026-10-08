"""Check checkpoint reconstruction, padded ports and strict paired export guards."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zlib

import numpy as np

from whisper.encoder_kernel import pack_port, port_view
from whisper.paired_kernel import ROOT, pack_coefficients, reconstruct, validate_layout
from whisper.scripts.package_asahi_precision import accuracy_receipt

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


if __name__ == "__main__":
    unittest.main()
