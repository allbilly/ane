"""Reject stale or incomplete recurrence inputs before importing the device runtime."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class QwenCaptureInputTests(unittest.TestCase):
    def test_wrong_checkpoint_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"test checkpoint")
            (root / "report.json").write_text(json.dumps(dict(model_sha256="0" * 64)))
            result = subprocess.run([sys.executable, str(ROOT / "experimental/capture_qwen_recurrence.py"),
                                     "--model", str(root), "--inputs", str(root / "native-inputs.npz"),
                                     "--output", str(root / "output")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different checkpoint", result.stderr)
            self.assertFalse((root / "output").exists())

    def test_duplicate_prefixes_rejected_before_compile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"test checkpoint")
            (root / "report.json").write_text(json.dumps(dict(model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
            manifest = json.loads((ROOT / "qwen35/provenance/vendor-validation.json").read_text())
            tokens = next(r["tokens"] for r in manifest["checks"] if r["label"] == "arithmetic")
            np.savez_compressed(root / "native-inputs.npz", tokens=tokens, positions=[0, 0, 7, 22],
                                a_log=np.zeros(16), dt=np.zeros(16), norm=np.ones(128),
                                state=np.zeros((4, 16, 128, 128), np.float32),
                                next_state=np.zeros((4, 16, 128, 128), np.float32),
                                projected=np.zeros((4, 8224), np.float32), output=np.zeros((4, 2048), np.float32))
            result = subprocess.run([sys.executable, str(ROOT / "experimental/capture_qwen_recurrence.py"),
                                     "--model", str(root), "--inputs", str(root / "native-inputs.npz"),
                                     "--output", str(root / "output")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("distinct real prefix captures", result.stderr)
            self.assertFalse((root / "output/heads-0-3").exists())


if __name__ == "__main__":
    unittest.main()
