"""GGUF reader validation, independent block decoding, and real GPT-2 loading."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import gguf
from bpe import Tokenizer
from checks import compare, cpu_parity
from external_weights import find_weights, load_weights, verify_weights
from gguf_weights import GGUFWeights
from model import CPUKernels, GPT2
from packing import PackedAssets, reconstruct


def metadata(writer, **overrides):
    fields = {
        "gpt2.block_count": 12, "gpt2.context_length": 1024,
        "gpt2.embedding_length": 768, "gpt2.feed_forward_length": 3072,
        "gpt2.attention.head_count": 12, "gpt2.attention.layer_norm_epsilon": 1e-5,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "gpt-2",
        "tokenizer.ggml.bos_token_id": 50256, "tokenizer.ggml.eos_token_id": 50256,
    }
    vocab = json.loads((ROOT / "tokenizer/vocab.json").read_text())
    fields["tokenizer.ggml.tokens"] = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    fields["tokenizer.ggml.merges"] = (ROOT / "tokenizer/merges.txt").read_text().splitlines()[1:]
    fields.update(overrides)
    for name, value in fields.items():
        if isinstance(value, str):
            writer.add_string(name, value)
        elif isinstance(value, list):
            writer.add_array(name, value)
        elif isinstance(value, bool):
            writer.add_bool(name, value)
        elif isinstance(value, float):
            writer.add_float32(name, value)
        else:
            writer.add_uint32(name, value)


def finish(writer):
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def export_checkpoint(weights, target, qtype=gguf.GGMLQuantizationType.F16):
    """Use upstream's HF name mapping rather than the loader's mapping."""
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GPT2, 12)
    writer = gguf.GGUFWriter(target, "gpt2", use_temp_file=True)
    metadata(writer)
    for name, value in weights.state.items():
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            continue
        target_name = mapping.get_name(name, try_suffixes=(".weight", ".bias"))
        if target_name is None:
            raise ValueError(f"unmapped HF test tensor: {name}")
        if name.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
            value = value.T
        value = np.ascontiguousarray(value, dtype=np.float32)
        tensor_type = qtype if value.ndim == 2 else gguf.GGMLQuantizationType.F32
        writer.add_tensor(target_name, gguf.quantize(value, tensor_type), raw_dtype=tensor_type)
    finish(writer)


class GGUFValidationTests(unittest.TestCase):
    def checkpoint(self, arch="gpt2", dtype=np.float32, endian=gguf.GGUFEndian.LITTLE, **overrides):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        target = Path(directory.name) / "model.gguf"
        writer = gguf.GGUFWriter(target, arch, endianess=endian)
        metadata(writer, **overrides)
        writer.add_tensor("token_embd.weight", np.zeros((2, 768), dtype=dtype))
        finish(writer)
        return target

    def test_wrong_architecture_and_dimensions(self):
        with self.assertRaisesRegex(ValueError, "architecture must be gpt2"):
            load_weights(self.checkpoint(arch="llama"))
        with self.assertRaisesRegex(ValueError, "block_count must be 12"):
            load_weights(self.checkpoint(**{"gpt2.block_count": 24}))
        with self.assertRaisesRegex(ValueError, "tensor shape mismatch"):
            load_weights(self.checkpoint())

    def test_tokenizer_mismatch_and_special_token_policy(self):
        for field, value, message in (
            ("tokenizer.ggml.tokens", ["bad"], "token IDs differ"),
            ("tokenizer.ggml.merges", ["bad merge"], "BPE merges differ"),
            ("tokenizer.ggml.eos_token_id", 1, "EOS token ID"),
            ("tokenizer.ggml.pre", "llama-bpe", "pre-tokenization"),
            ("tokenizer.ggml.add_bos_token", True, "automatic BOS"),
            ("split.count", 2, "split GGUF"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                load_weights(self.checkpoint(**{field: value}))

    def test_unsupported_encoding_and_byte_order(self):
        with self.assertRaisesRegex(ValueError, "unsupported GGUF tensor type.*I8"):
            load_weights(self.checkpoint(dtype=np.int8))
        with self.assertRaisesRegex(ValueError, "little-endian"):
            load_weights(self.checkpoint(endian=gguf.GGUFEndian.BIG))

    def test_truncated_or_invalid_header(self):
        target = self.checkpoint()
        with target.open("r+b") as stream:
            stream.truncate(target.stat().st_size - 100)
        with self.assertRaisesRegex(ValueError, "invalid GGUF"):
            load_weights(target)
        target.write_bytes(b"invalid header")
        with self.assertRaisesRegex(ValueError, "invalid GGUF"):
            load_weights(target)

    def test_explicit_directory_discovery_and_ambiguous_files(self):
        target = self.checkpoint()
        self.assertEqual(find_weights(target.parent), target)
        (target.parent / "other.gguf").touch()
        with self.assertRaisesRegex(ValueError, "explicitly"):
            find_weights(target.parent)

    def decoded(self, data, qtype, shape):
        # Exercise the adapter with hand-authored blocks; no matching encoder.
        weights = GGUFWeights.__new__(GGUFWeights)
        weights.entries = {"test": ("raw", shape, None)}
        weights.records = {"raw": SimpleNamespace(data=data, tensor_type=qtype)}
        weights.cache = {}
        return weights.get("test", shape)

    def test_q4_0_nibble_order_signed_scale_and_block_order(self):
        scales = [np.float16(0.5), np.float16(-2)]
        packed = np.arange(16, dtype=np.uint8) | ((15 - np.arange(16, dtype=np.uint8)) << 4)
        blocks = np.stack([np.frombuffer(scale.tobytes() + packed.tobytes(), dtype=np.uint8) for scale in scales])
        expected = np.stack([np.concatenate((np.arange(16) - 8, 7 - np.arange(16))) * float(s) for s in scales])
        np.testing.assert_array_equal(self.decoded(blocks, gguf.GGMLQuantizationType.Q4_0, (2, 32)), expected)

    def test_q8_0_signed_values_and_per_block_scale(self):
        values = np.arange(-16, 16, dtype=np.int8)
        blocks = np.stack([np.frombuffer(np.float16(s).tobytes() + values.tobytes(), dtype=np.uint8) for s in (0.5, -2)])
        expected = np.stack((values * 0.5, values * -2))
        np.testing.assert_array_equal(self.decoded(blocks, gguf.GGMLQuantizationType.Q8_0, (2, 32)), expected)

    def test_nonfinite_and_fp16_overflow_rejected(self):
        for value in (float("nan"), float("inf"), 1e8):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonfinite/fp16-overflow"):
                self.decoded(np.array([value], np.float32), gguf.GGMLQuantizationType.F32, (1,))

    def test_float_matrix_shape_and_fp16_rounding(self):
        values = np.array([[0.00001, -1.234567, 3], [4, 5.678901, -6]], dtype=np.float32)
        for qtype, dtype in ((gguf.GGMLQuantizationType.F32, np.float32), (gguf.GGMLQuantizationType.F16, np.float16)):
            with self.subTest(qtype=qtype.name):
                np.testing.assert_array_equal(self.decoded(values.astype(dtype), qtype, (2, 3)),
                                              values.astype(np.float16).astype(np.float32))


class CheckpointPackingTests(unittest.TestCase):
    def test_fp16_attention_tolerance_keeps_global_error_gate(self):
        expected = np.ones(2000, dtype=np.float32)
        expected[0] = 0
        actual = expected.copy()
        actual[0] = 0.06
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            compare(actual, expected, "strict reference", rtol=0.02, atol=0.05)
        result = compare(actual, expected, "fp16 attention", rtol=0.02, atol=0.05, min_close_fraction=0.999)
        self.assertGreater(result["close_fraction"], 0.999)
        actual[0] = 1
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            compare(actual, expected, "large isolated error", rtol=0.02, atol=0.05, min_close_fraction=0.999)

    def test_checkpoint_cache_isolation_corruption_and_reference_protection(self):
        class Weights:
            reference = False
            def __init__(self, value):
                self.value = value
                self.fingerprint = hashlib.sha256(str(value).encode()).hexdigest()
            def get(self, name, shape):
                return np.full(shape, self.value if name == "gamma" else 0, np.float32)
        recipe = dict(size=3072, sha256="0" * 64, operations=[dict(kind="affine", gamma="gamma", beta="beta",
                      layout="linear", offset=0, size=3072)])
        meta = {"packing": {"program": recipe}}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            first, second = (PackedAssets(ROOT, Weights(value), cache) for value in (1, 2))
            one, two = first.payload(meta, "program"), second.payload(meta, "program")
            self.assertNotEqual(one, two)
            self.assertEqual(first.payload(meta, "program"), one)
            paths = list(cache.rglob("*.packed"))
            self.assertEqual(len(paths), 2)
            path = next(p for p in paths if first.fingerprint in p.parts)
            path.write_bytes(bytes(path.stat().st_size))
            self.assertEqual(first.payload(meta, "program"), one)
            self.assertEqual(path.read_bytes()[32:], one)
            # The default reference path continues to reject changed weights.
            with self.assertRaisesRegex(ValueError, "captured reference"):
                reconstruct(ROOT, Weights(1), recipe)
            # Static programs still require their pinned hash in custom mode.
            with self.assertRaisesRegex(ValueError, "captured reference"):
                reconstruct(ROOT, Weights(1), dict(size=8, operations=[], sha256="0" * 64), reference=False)


class GGUFIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selected = find_weights()
        if selected is None or selected.suffix.lower() != ".safetensors":
            raise unittest.SkipTest("cached reference safetensors required for real GGUF integration")
        verify_weights(selected, ROOT)
        cls.hf = load_weights(selected)
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.source = Path(cls.directory.name) / "gpt2-f16.gguf"
        export_checkpoint(cls.hf, cls.source)
        cls.weights = load_weights(cls.source)
        verify_weights(cls.source, ROOT, weights=cls.weights)

    def test_reference_conversion_preserves_all_tensors_and_cpu_logits(self):
        self.assertTrue(self.weights.reference)
        for name, (source, shape, section) in self.weights.entries.items():
            if name == "lm_head":
                continue
            shape = (768,) + shape[1:] if section is not None else shape
            np.testing.assert_array_equal(self.weights.get(name, shape), self.hf.get(name, shape), err_msg=name)
        tokenizer = Tokenizer(ROOT / "tokenizer")
        self.assertEqual(len(cpu_parity(ROOT, GPT2(self.weights, CPUKernels(self.weights)), tokenizer)), 3)

    def test_quantized_qkv_mapping_and_dynamic_packing(self):
        for qtype in (gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0):
            with self.subTest(qtype=qtype.name):
                source = Path(self.directory.name) / (qtype.name + ".gguf")
                # Keep other tensors F16 and independently encode an asymmetric
                # fused QKV tensor through upstream's actual GGUF writer.
                writer = gguf.GGUFWriter(source, "gpt2", use_temp_file=True)
                for field in self.weights.reader.fields.values():
                    if field.name not in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"):
                        writer.add_key_value(field.name, field.contents(), field.types[0],
                                             field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
                expected = None
                for tensor in self.weights.reader.tensors:
                    if tensor.name == "blk.0.attn_qkv.weight":
                        raw = gguf.quantize(gguf.dequantize(tensor.data, tensor.tensor_type), qtype)
                        expected = gguf.dequantize(raw, qtype).astype(np.float16).astype(np.float32)
                        writer.add_tensor(tensor.name, raw, raw_dtype=qtype)
                    else:
                        writer.add_tensor(tensor.name, tensor.data)
                finish(writer)
                weights = load_weights(source)
                self.assertEqual(verify_weights(source, ROOT, weights=weights), 196)
                self.assertFalse(weights.reference)
                for part, name in enumerate("qkv"):
                    np.testing.assert_array_equal(weights.layer(0, "w" + name), expected[part * 768:(part + 1) * 768])
                with self.assertRaisesRegex(ValueError, "shape mismatch"):
                    weights.get("layer0/wq", (1,))
                meta = json.loads((ROOT / "kernels/decode_proj_L0/meta.json").read_text())
                assets = PackedAssets(ROOT, weights, Path(self.directory.name) / "cache")
                self.assertNotEqual(hashlib.sha256(assets.payload(meta, "weights")).hexdigest(), meta["packing"]["weights"]["sha256"])
                model = GPT2(weights, CPUKernels(weights))
                self.assertTrue(np.isfinite(model.step(15496)).all())
                del weights, model, assets

    def test_explicit_output_head_is_used(self):
        source = Path(self.directory.name) / "untied-head.gguf"
        writer = gguf.GGUFWriter(source, "gpt2", use_temp_file=True)
        metadata(writer)
        for tensor in self.weights.reader.tensors:
            writer.add_tensor(tensor.name, tensor.data)
        writer.add_tensor("output.weight", np.zeros((50257, 768), dtype=np.float16))
        finish(writer)
        weights = load_weights(source)
        verify_weights(source, ROOT, weights=weights)
        self.assertFalse(weights.reference)
        self.assertTrue(np.any(weights.get("wte", (50257, 768)) != 0))
        logits = GPT2(weights, CPUKernels(weights)).step(15496)
        np.testing.assert_array_equal(logits, np.zeros(50257, np.float32))


if __name__ == "__main__":
    unittest.main()
