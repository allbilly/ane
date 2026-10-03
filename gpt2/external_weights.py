"""Use cached HF weights directly, outside the replay package."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import urllib.request
import numpy as np
from checks import digest
from hwx import require

HF_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
HF_URL = f"https://huggingface.co/openai-community/gpt2/resolve/{HF_REVISION}/model.safetensors"
HF_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"


def cache_root():
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "orion-gpt2"


def find_weights(requested=None):
    if requested or os.environ.get("GPT2_WEIGHTS"):
        path = Path(requested or os.environ["GPT2_WEIGHTS"]).expanduser().resolve()
        require(path.exists(), f"weight path missing: {path}")
        if path.is_dir() and (path / "model.safetensors").is_file():
            return path / "model.safetensors"
        return path
    hf_home = Path(os.environ.get("HF_HOME", str(cache_root().parent / "huggingface")))
    hub = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub")))
    candidates = [cache_root() / "model.safetensors"]
    for repo in ("models--openai-community--gpt2", "models--gpt2"):
        candidates.extend(sorted((hub / repo).glob("snapshots/*/model.safetensors"),
                                 key=lambda p: p.stat().st_mtime, reverse=True))
    candidates.extend([Path.home() / "Desktop/Orion/model/blobs/gpt2_124m", cache_root() / "blobs"])
    return next((p for p in candidates if p.is_file() or (p / "wte.bin").is_file()), None)


def verify_weights(root, package):
    records = json.loads((package / "model-checksums.json").read_text())
    require(len(records) == 196, "invalid external model manifest")
    if root.is_file():
        require(digest(root) == HF_SHA256, "cached HF checkpoint differs from the model used for the ANE dump")
        return len(records)
    for name, expected in records.items():
        path = root / name
        require(path.is_file() and digest(path) == expected,
                f"external weights differ from the captured GPT-2 model: {name}")
    return len(records)


def model_tensors(state):
    """HF Conv1D [in,out] -> Orion [out,in], then standard fp16 blobs.

    This is CPU/MIL preparation, not the ANE's per-NE coefficient encoding.
    """
    def get(key):
        return np.asarray(state[key if key in state else "transformer." + key])
    yield "wte", get("wte.weight")
    yield "wpe", get("wpe.weight")
    for i in range(12):
        prefix, target = f"h.{i}.", f"layer{i}/"
        for name, key, transpose in (
            ("ln1_g", "ln_1.weight", False), ("ln1_b", "ln_1.bias", False),
            ("wo", "attn.c_proj.weight", True), ("bo", "attn.c_proj.bias", False),
            ("ln2_g", "ln_2.weight", False), ("ln2_b", "ln_2.bias", False),
            ("wfc", "mlp.c_fc.weight", True), ("bfc", "mlp.c_fc.bias", False),
            ("wproj", "mlp.c_proj.weight", True), ("bproj", "mlp.c_proj.bias", False)):
            value = get(prefix + key)
            yield target + name, value.T if transpose else value
        qkv, bias = get(prefix + "attn.c_attn.weight"), get(prefix + "attn.c_attn.bias")
        require(qkv.shape == (768, 2304) and bias.shape == (2304,), "expected GPT-2 124M fused QKV")
        for part, name in enumerate("qkv"):
            yield target + "w" + name, qkv[:, part * 768:(part + 1) * 768].T
            yield target + "b" + name, bias[part * 768:(part + 1) * 768]
    yield "ln_f_g", get("ln_f.weight")
    yield "ln_f_b", get("ln_f.bias")


def blob_bytes(value):
    payload = np.asarray(value, dtype="<f2").tobytes(order="C")
    header = bytearray(128)
    struct.pack_into("<II", header, 0, 1, 2)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, len(payload), 128)
    return bytes(header) + payload


def load_weights(source):
    from model import Weights
    if source.is_dir():
        return Weights(source)
    return HFWeights(source)


class HFWeights:
    """Read cached safetensors directly; match Orion's fp16 conversion in RAM."""
    def __init__(self, source):
        from safetensors.numpy import load_file
        self.source = source
        self.state = load_file(str(source))
        self.tensors = dict(model_tensors(self.state))
        self.cache = {}

    def get(self, name, shape):
        if name not in self.cache:
            value = self.tensors[name]
            require(value.shape == shape, f"HF tensor shape mismatch: {name}")
            self.cache[name] = value.astype("<f2").astype(np.float32)
        return self.cache[name]

    def layer(self, index, name):
        from model import LAYER_SHAPES
        return self.get(f"layer{index}/{name}", LAYER_SHAPES[name])


def setup(package, output=None, checkpoint=None, progress=print):
    if not output and not checkpoint:
        existing = find_weights()
        if existing is not None:
            verify_weights(existing, package)
            progress(f"Using verified existing weights: {existing}")
            return existing
    if checkpoint:
        source = Path(checkpoint).expanduser().resolve()
        require(source.is_file(), f"checkpoint missing: {source}")
    else:
        source = (Path(output).expanduser().resolve() if output else cache_root()) / "model.safetensors"
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file() or digest(source) != HF_SHA256:
            progress("Downloading the standard GPT-2 checkpoint from Hugging Face to external cache...")
            fd, name = tempfile.mkstemp(prefix=".gpt2-", suffix=".download", dir=source.parent)
            temporary = Path(name)
            try:
                request = urllib.request.Request(HF_URL, headers={"User-Agent": "orion-gpt2-port"})
                with os.fdopen(fd, "wb") as stream, urllib.request.urlopen(request, timeout=90) as response:
                    h = hashlib.sha256()
                    while chunk := response.read(1024 * 1024):
                        h.update(chunk)
                        stream.write(chunk)
                require(h.hexdigest() == HF_SHA256, "HF checkpoint SHA256 mismatch")
                temporary.replace(source)
            finally:
                temporary.unlink(missing_ok=True)
    verify_weights(source, package)
    progress(f"Verified external checkpoint (read directly, no blob copies): {source}")
    return source
