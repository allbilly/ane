"""Validate the independent checkpoint import and full-vocabulary capture guards."""
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from whisper.ggml_encoder_weights import read_model
from whisper.validation import logits_records

ROOT = Path(__file__).resolve().parents[1]
GGML = ROOT / "whisper/models/hf-ggml/ggml-model.bin"
HF = ROOT / "whisper/models/hf-tiny.en/model.safetensors"


class ReferenceImportTests(unittest.TestCase):
    @unittest.skipUnless(GGML.is_file() and HF.is_file(), "external tiny.en checkpoints unavailable")
    def test_all_stored_encoder_decoder_values_match_original_checkpoint(self):
        from safetensors.numpy import load_file
        dimensions, actual = read_model(GGML)
        original = load_file(HF)
        original.pop("proj_out.weight", None)  # Shared with the token embedding.
        self.assertEqual(set(actual), set(original))
        self.assertEqual(dimensions["vocabulary"], 51864)
        self.assertEqual(dimensions["audio_layers"], 4)
        self.assertEqual(dimensions["text_layers"], 4)
        for name, value in actual.items():
            with self.subTest(tensor=name):
                np.testing.assert_array_equal(value, original[name].astype(value.dtype))


class FullLogitCaptureTests(unittest.TestCase):
    def record(self, vocabulary):
        tokens = np.array([50258, 50259, 50359, 50363], "<i4")
        logits = np.linspace(-1, 1, vocabulary, dtype="<f4")
        return struct.pack("<2i", len(tokens), vocabulary) + tokens.tobytes() + logits.tobytes()

    def test_both_vocabularies_and_explicit_multilingual_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "logits.bin"
            for vocabulary in (51864, 51865):
                path.write_bytes(self.record(vocabulary))
                tokens, values = logits_records(path, vocabulary)[0]
                self.assertEqual(tokens.tolist(), [50258, 50259, 50359, 50363])
                self.assertEqual(values.shape, (vocabulary,))
                self.assertEqual((values[0], values[-1]), (-1, 1))
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                logits_records(path)

    def test_incomplete_and_corrupt_full_vectors_are_rejected(self):
        complete = self.record(51865)
        wrong_token = bytearray(complete)
        struct.pack_into("<i", wrong_token, 8, 51865)
        nonfinite = bytearray(complete)
        struct.pack_into("<f", nonfinite, len(nonfinite) - 4, float("nan"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "logits.bin"
            for data in (b"", complete[:7], complete[:-1], complete + b"x", wrong_token, nonfinite):
                with self.subTest(bytes=len(data)):
                    path.write_bytes(data)
                    with self.assertRaises(ValueError):
                        logits_records(path, 51865)


if __name__ == "__main__":
    unittest.main()
