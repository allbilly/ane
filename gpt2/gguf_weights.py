"""Memory-mapped GPT-2 GGUF weights, decoded to the existing fp16 contract."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from checks import digest
from hwx import require


def tensor_map():
    """GGUF matrices already have [out,in] order, unlike HF Conv1D weights."""
    entries = {
        "wte": ("token_embd.weight", (50257, 768), None),
        "wpe": ("position_embd.weight", (1024, 768), None),
        "ln_f_g": ("output_norm.weight", (768,), None),
        "ln_f_b": ("output_norm.bias", (768,), None),
    }
    for i in range(12):
        for target, source, shape in (
            ("ln1_g", "attn_norm.weight", (768,)), ("ln1_b", "attn_norm.bias", (768,)),
            ("ln2_g", "ffn_norm.weight", (768,)), ("ln2_b", "ffn_norm.bias", (768,)),
            ("wo", "attn_output.weight", (768, 768)), ("bo", "attn_output.bias", (768,)),
            ("wfc", "ffn_up.weight", (3072, 768)), ("bfc", "ffn_up.bias", (3072,)),
            ("wproj", "ffn_down.weight", (768, 3072)), ("bproj", "ffn_down.bias", (768,)),
        ):
            entries[f"layer{i}/{target}"] = (f"blk.{i}.{source}", shape, None)
        for part, name in enumerate("qkv"):
            section = slice(part * 768, (part + 1) * 768)
            entries[f"layer{i}/w{name}"] = (f"blk.{i}.attn_qkv.weight", (2304, 768), section)
            entries[f"layer{i}/b{name}"] = (f"blk.{i}.attn_qkv.bias", (2304,), section)
    return entries


class GGUFWeights:
    def __init__(self, source, tokenizer_root=None):
        from gguf import GGUFReader, GGMLQuantizationType
        self.source = Path(source)
        try:
            self.reader = GGUFReader(str(self.source), "r")
        except (ValueError, IndexError, KeyError, OverflowError) as error:
            raise ValueError(f"invalid GGUF checkpoint: {error}") from error
        require(sys.byteorder == "little" and self.reader.byte_order == "I",
                "only little-endian GGUF checkpoints are supported")
        self.reference = False
        self.fingerprint = digest(self.source)
        self.cache = {}
        self.entries = tensor_map()
        self.records = {tensor.name: tensor for tensor in self.reader.tensors}
        require(len(self.records) == len(self.reader.tensors), "duplicate GGUF tensor names")
        self.supported_types = {GGMLQuantizationType.F32, GGMLQuantizationType.F16,
                                GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q4_0}
        self._validate_metadata(tokenizer_root or Path(__file__).parent / "tokenizer")
        self.entries["lm_head"] = ("output.weight" if "output.weight" in self.records else "token_embd.weight",
                                   (50257, 768), None)
        for name, shape, _ in self.entries.values():
            require(name in self.records, f"missing GPT-2 GGUF tensor: {name}")
            tensor = self.records[name]
            require(tensor.tensor_type in self.supported_types,
                    f"unsupported GGUF tensor type: {name}: {tensor.tensor_type.name}; supported: F32, F16, Q8_0, Q4_0")
            actual = tuple(int(n) for n in reversed(tensor.shape))
            require(actual == shape, f"GGUF tensor shape mismatch: {name}; expected {shape}, got {actual}")
            require(tensor.data.size * tensor.data.dtype.itemsize == tensor.n_bytes
                    and tensor.data_offset + tensor.n_bytes <= self.source.stat().st_size,
                    f"truncated GGUF tensor: {name}")

    def metadata(self, key, default=None):
        field = self.reader.get_field(key)
        return field.contents() if field is not None else default

    def _validate_metadata(self, tokenizer_root):
        require(self.metadata("general.architecture") == "gpt2", "GGUF architecture must be gpt2")
        require(self.metadata("split.count", 1) == 1, "split GGUF checkpoints are not supported; use a single file")
        for key, expected in (("block_count", 12), ("context_length", 1024),
                              ("embedding_length", 768), ("feed_forward_length", 3072),
                              ("attention.head_count", 12)):
            require(self.metadata("gpt2." + key) == expected,
                    f"GGUF gpt2.{key} must be {expected}; kernels target GPT-2 124M")
        epsilon = self.metadata("gpt2.attention.layer_norm_epsilon")
        require(isinstance(epsilon, (int, float)) and np.isclose(epsilon, 1e-5, rtol=1e-5, atol=0),
                "GGUF layer-norm epsilon must be 1e-5")
        require(self.metadata("tokenizer.ggml.model") == "gpt2", "GGUF tokenizer must be GPT-2 BPE")
        require(self.metadata("tokenizer.ggml.pre", "gpt-2") == "gpt-2",
                "GGUF tokenizer pre-tokenization must be gpt-2")
        vocab = json.loads((tokenizer_root / "vocab.json").read_text())
        tokens = [None] * len(vocab)
        for token, index in vocab.items():
            tokens[index] = token
        require(self.metadata("tokenizer.ggml.tokens") == tokens,
                "GGUF token IDs differ from the bundled GPT-2 vocabulary")
        merges = (tokenizer_root / "merges.txt").read_text().splitlines()[1:]
        require(self.metadata("tokenizer.ggml.merges") == [line for line in merges if line],
                "GGUF BPE merges differ from the bundled GPT-2 tokenizer")
        for kind in ("bos", "eos"):
            require(self.metadata(f"tokenizer.ggml.{kind}_token_id", 50256) == 50256,
                    f"GGUF {kind.upper()} token ID must be 50256")
            require(not self.metadata(f"tokenizer.ggml.add_{kind}_token", False),
                    f"GGUF automatic {kind.upper()} insertion is unsupported")

    def get(self, name, shape):
        from gguf import dequantize
        require(name in self.entries, f"unknown GPT-2 tensor: {name}")
        source, full_shape, section = self.entries[name]
        expected = (768,) + full_shape[1:] if section is not None else full_shape
        require(tuple(shape) == expected, f"GGUF tensor shape mismatch: {name}")
        if source not in self.cache:
            tensor = self.records[source]
            value = dequantize(tensor.data, tensor.tensor_type).reshape(full_shape)
            with np.errstate(over="ignore", invalid="ignore"):
                value = value.astype("<f2").astype(np.float32)
            require(bool(np.isfinite(value).all()), f"nonfinite/fp16-overflow GGUF tensor: {source}")
            self.cache[source] = value
        value = self.cache[source]
        return value[section] if section is not None else value

    def layer(self, index, name):
        from model import LAYER_SHAPES
        return self.get(f"layer{index}/{name}", LAYER_SHAPES[name])

    def validate(self, package):
        """Check every decoded tensor and detect equivalent reference weights."""
        from external_weights import blob_bytes
        records = json.loads((package / "model-checksums.json").read_text())
        matches = True
        for name, (_, shape, section) in self.entries.items():
            if name == "lm_head":
                continue
            target_shape = (768,) + shape[1:] if section is not None else shape
            value = self.get(name, target_shape)
            matches &= hashlib.sha256(blob_bytes(value)).hexdigest() == records.get(name + ".bin")
        head = self.get("lm_head", (50257, 768))
        self.reference = bool(matches and np.array_equal(head, self.get("wte", (50257, 768))))
        return len(records)
