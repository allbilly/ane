"""Orion GPT-2 124M, with incremental KV cache and explicit CPU/ANE backends."""
import numpy as np

D, HEADS, HEAD_DIM, LAYERS, CONTEXT = 768, 12, 64, 12, 1024
LAYER_SHAPES = {**{n: (D,) for n in ("ln1_g", "ln1_b", "ln2_g", "ln2_b", "bq", "bk", "bv", "bo", "bproj")},
                **{n: (D, D) for n in ("wq", "wk", "wv", "wo")},
                "wfc": (3072, D), "bfc": (3072,), "wproj": (D, 3072)}


class Weights:
    def __init__(self, root):
        self.root, self.cache = root, {}

    def get(self, name, shape):
        if name not in self.cache:
            path = self.root / (name + ".bin")
            expected = 128 + 2 * int(np.prod(shape))
            if path.stat().st_size != expected:
                raise ValueError(f"weight size mismatch: {name}")
            self.cache[name] = np.fromfile(path, dtype="<f2", offset=128).astype(np.float32).reshape(shape)
        return self.cache[name]

    def layer(self, index, name):
        return self.get(f"layer{index}/{name}", LAYER_SHAPES[name])


def layernorm(x, gamma, beta):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5)) * gamma + beta


class CPUKernels:
    def __init__(self, weights):
        self.weights = weights

    def project(self, layer, x):
        w = lambda n: self.weights.layer(layer, n)
        normalized = layernorm(x, w("ln1_g"), w("ln1_b"))
        return tuple(w("w" + n) @ normalized + w("b" + n) for n in "qkv")

    def ffn(self, layer, x):
        w = lambda n: self.weights.layer(layer, n)
        normalized = layernorm(x, w("ln2_g"), w("ln2_b"))
        fc = w("wfc") @ normalized + w("bfc")
        activated = np.float32(0.5) * fc * (np.float32(1) + np.tanh(
            np.float32(np.sqrt(2 / np.pi)) * (fc + np.float32(0.044715) * fc * fc * fc)))
        return x + w("wproj") @ activated + w("bproj")


class ANEKernels:
    def __init__(self, device):
        self.device = device

    def project(self, layer, x):
        result = self.device.kernel(f"decode_proj_L{layer}").vector(x)
        return tuple(result[n][:, 0].astype(np.float32) for n in ("q16", "k16", "v16"))

    def ffn(self, layer, x):
        return self.device.kernel(f"decode_ffn_L{layer}").vector(x)["hidden"][:, 0].astype(np.float32)


class GPT2:
    def __init__(self, weights, kernels):
        self.weights, self.kernels = weights, kernels
        self.reset()

    def reset(self):
        self.position = 0
        self.keys = np.zeros((LAYERS, HEADS, CONTEXT, HEAD_DIM), dtype=np.float32)
        self.values = np.zeros_like(self.keys)

    def step(self, token):
        if not 0 <= token < 50257:
            raise ValueError("invalid token")
        if self.position >= CONTEXT:
            raise ValueError("GPT-2 context is limited to 1024 tokens")
        pos = self.position
        wte = self.weights.get("wte", (50257, D))
        x = wte[token] + self.weights.get("wpe", (CONTEXT, D))[pos]
        for layer in range(LAYERS):
            q, key, value = self.kernels.project(layer, x)
            self.keys[layer, :, pos] = key.reshape(HEADS, HEAD_DIM)
            self.values[layer, :, pos] = value.reshape(HEADS, HEAD_DIM)
            scores = np.einsum("htd,hd->ht", self.keys[layer, :, :pos + 1], q.reshape(HEADS, HEAD_DIM)) * np.float32(0.125)
            scores -= scores.max(axis=-1, keepdims=True)
            attention = np.exp(scores)
            attention /= attention.sum(axis=-1, keepdims=True)
            attended = np.einsum("ht,htd->hd", attention, self.values[layer, :, :pos + 1]).reshape(D)
            x = x + self.weights.layer(layer, "wo") @ attended + self.weights.layer(layer, "bo")
            x = self.kernels.ffn(layer, x)
        x = layernorm(x, self.weights.get("ln_f_g", (D,)), self.weights.get("ln_f_b", (D,)))
        logits = wte @ x
        if not np.isfinite(logits).all():
            raise RuntimeError("nonfinite logits")
        self.position += 1
        return logits

    def generate(self, prompt, count, temperature=0, top_k=40, seed=0):
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if count < 0 or len(prompt) + max(count - 1, 0) > CONTEXT:
            raise ValueError("prompt plus generation exceeds the 1024-token context")
        self.reset()
        if count == 0:
            return
        for token in prompt:
            logits = self.step(token)
        rng = np.random.default_rng(seed)
        for index in range(count):
            if temperature == 0:
                token = int(logits.argmax())
            else:
                candidates = np.argpartition(logits, -top_k)[-top_k:]
                scores = logits[candidates].astype(np.float64) / temperature
                probability = np.exp(scores - scores.max())
                probability /= probability.sum()
                token = int(rng.choice(candidates, p=probability))
            yield token
            if token == 50256 or index + 1 == count:
                break
            logits = self.step(token)
