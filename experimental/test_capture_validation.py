"""Regression checks for accepting complete, intact capture comparisons."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experimental.replay_capture import GATE, validate_manifest
from experimental.verify_macos_capture import verify


class CaptureValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.kit = Path(self.directory.name)
        hwx = self.kit / "model.hwx"
        hwx.write_bytes(b"checksum validation stub")
        fixture = self.kit / "fixture.npz"
        np.savez(fixture, input00=np.ones(2, np.float16),
                 output00=np.ones(2, np.float16), output01=np.zeros(2, np.float16))
        record = dict(name="program/output0", hwx=hwx.name,
                      hwx_sha256=hashlib.sha256(hwx.read_bytes()).hexdigest(),
                      input_port_names=["input"], inputs=[[1, 1, 1, 2]],
                      output_port_name="output0", output=[1, 1, 1, 2], output_key="output00",
                      fixtures=[dict(path=fixture.name, sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())])
        second = copy.deepcopy(record)
        second.update(name="program/output1", output_port_name="output1", output_key="output01")
        self.manifest = dict(unsupported=[], gate=GATE.copy(), records=[record, second])

    def test_complete_two_output_fixture(self):
        self.assertEqual(validate_manifest(self.manifest, self.kit), 2)

    def test_empty_fixtures_rejected_before_runtime_import_or_output(self):
        for record in self.manifest["records"]:
            record["fixtures"] = []
        (self.kit / "asahi-fixtures.json").write_text(json.dumps(self.manifest))
        with patch("experimental.verify_macos_capture.platform.system", return_value="Darwin"):
            with self.assertRaisesRegex(ValueError, "at least one fixture"):
                verify(self.kit, self.kit / "result")
        self.assertFalse((self.kit / "result").exists())

    def test_weakened_gate_rejected(self):
        self.manifest["gate"]["relative_l2"] = .5
        with self.assertRaisesRegex(ValueError, "unchanged replay gates"):
            validate_manifest(self.manifest, self.kit)

    def test_corrupt_hwx_rejected(self):
        (self.kit / "model.hwx").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_manifest(self.manifest, self.kit)

    def test_missing_input_port_rejected(self):
        self.manifest["records"][0]["input_port_names"] = []
        with self.assertRaisesRegex(ValueError, "input port count mismatch"):
            validate_manifest(self.manifest, self.kit)

    def test_duplicate_output_or_fixture_rejected(self):
        for mutation, expected in (("output", "duplicate output port"), ("fixture", "duplicate output fixture")):
            manifest = copy.deepcopy(self.manifest)
            if mutation == "output":
                manifest["records"][1]["output_port_name"] = "output0"
            else:
                manifest["records"][0]["fixtures"] *= 2
            with self.assertRaisesRegex(ValueError, expected):
                validate_manifest(manifest, self.kit)

    def test_nonfinite_reference_rejected(self):
        fixture = self.kit / "fixture.npz"
        np.savez(fixture, input00=np.ones(2, np.float16),
                 output00=np.full(2, np.nan, np.float16), output01=np.zeros(2, np.float16))
        for record in self.manifest["records"]:
            record["fixtures"][0]["sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "finite output ports"):
            validate_manifest(self.manifest, self.kit)


if __name__ == "__main__":
    unittest.main()
