"""Check compact payload integrity and reject unsafe reconstruction recipes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zlib

import numpy as np

from qwen35.tests import write_tensors
from whisper.encoder_kernel import reconstruct


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


if __name__ == "__main__":
    unittest.main()
