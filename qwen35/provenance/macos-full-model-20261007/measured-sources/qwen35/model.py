"""Single-token Qwen3.5: packed Mirai projections, FP32 recurrent state."""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .native import Native, dot_supported
from .weights import Matrix, SafeTensors, bf16_round


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(-x))


def silu(x):
    with np.errstate(over="ignore"):
        return x / (1 + np.exp(-x))


class Model:
    def __init__(self, directory, precision="fp32", kernels="auto", threads=None, backend=None, context=4096):
        if kernels == "auto":
            kernels = "dot" if precision == "fp32" and dot_supported() else "native"
        self.kernels = kernels
        if precision not in ("fp32", "bf16") or kernels not in ("native", "numpy", "dot"):
            raise ValueError("unsupported precision or CPU kernels")
        self.precision, self.context = precision, context
        if not 1 <= context <= 262144:
            raise ValueError("context must be in 1..262144")
        self.config = json.loads((Path(directory) / "config.json").read_text())
        decoder = self.config["decoder_config"]
        transformer = decoder["transformer_config"]
        configs = transformer["layer_configs"]
        if (decoder["vocab_size"] != 248320 or transformer["model_dim"] != 1024
                or transformer["hidden_dim"] != 3584 or len(configs) != 24
                or decoder["embedding_config"]["type"] != "TiedEmbeddingConfig"):
            raise ValueError("this runtime requires Mirai Qwen3.5-0.8B")
        self.tensors = SafeTensors(Path(directory) / "model.safetensors")
        t = self.tensors
        self.embedding = Matrix(t, "decoder.embedding.embedding", 248320, 1024, embedding=True)
        self.norm = t.tensor("decoder.transformer.output_norm.scales", (1024,), "F32")
        self.layers = []
        for i, config in enumerate(configs):
            p = f"decoder.transformer.layers.{i}"
            mixer = config["mixer_config"]
            attention = mixer["type"] == "AttentionConfig"
            if attention != (i % 4 == 3):
                raise ValueError(f"unexpected mixer schedule: layer {i}")
            layer = dict(attention=attention,
                         n1=t.tensor(p + ".pre_mixer_norm.scales", (1024,), "F32"),
                         n2=t.tensor(p + ".pre_mlp_norm.scales", (1024,), "F32"),
                         up=Matrix(t, p + ".mlp.up_projection.weights", 7168, 1024),
                         down=Matrix(t, p + ".mlp.down_projection.weights", 1024, 3584))
            if attention:
                if (mixer["num_heads"], mixer["num_groups"], mixer["head_dim"]) != (8, 2, 256):
                    raise ValueError("unexpected attention dimensions")
                layer.update(proj=Matrix(t, p + ".mixer.qkvg_projection.weights", 5120, 1024),
                             out=Matrix(t, p + ".mixer.out_projection.weights", 1024, 2048),
                             qn=t.tensor(p + ".mixer.query_norm.scales", (256,), "F32"),
                             kn=t.tensor(p + ".mixer.key_norm.scales", (256,), "F32"))
            else:
                if (mixer["num_heads"], mixer["num_groups"], mixer["head_dim"], mixer["value_head_dim"], mixer["kernel_size"]) != (16, 16, 128, 128, 4):
                    raise ValueError("unexpected DeltaNet dimensions")
                layer.update(proj=Matrix(t, p + ".mixer.in_proj.weights", 8224, 1024),
                             out=Matrix(t, p + ".mixer.out_proj.weights", 1024, 2048),
                             conv=t.tensor(p + ".mixer.conv.weights", (6144, 4), "F32"),
                             a_log=t.tensor(p + ".mixer.a_log", (16,), "F32"),
                             dt=t.tensor(p + ".mixer.dt_bias", (16,), "F32"),
                             norm=t.tensor(p + ".mixer.norm.scales", (128,), "F32"))
            self.layers.append(layer)
        self.native = Native(threads, integer=kernels == "dot") if kernels != "numpy" else None
        self.backend = backend
        self.timings = defaultdict(float)
        self.reset()

    def reset(self):
        self.position = 0
        self.states = []
        for layer in self.layers:
            if layer["attention"]:
                self.states.append(dict(k=np.zeros((self.context, 2, 256), dtype=np.float32),
                                        v=np.zeros((self.context, 2, 256), dtype=np.float32)))
            else:
                self.states.append(dict(conv=np.zeros((6144, 3), dtype=np.float32),
                                        ssm=np.zeros((16, 128, 128), dtype=np.float32)))

    def round(self, x):
        return bf16_round(x) if self.precision == "bf16" else np.asarray(x, dtype=np.float32)

    def rms(self, x, scale, offset=1):
        if self.native and self.precision == "bf16" and offset == 1:
            return self.native.rms_bf16(x, scale)
        normalized = x * (1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-6)))
        # Mirai uses 'only_normalization': round before scale multiplication.
        return self.round(self.round(normalized) * self.round(scale + np.float32(offset)))

    def linear(self, matrix, x, label):
        start = time.perf_counter()
        if self.backend is not None and not matrix.embedding:
            y = self.backend.linear(matrix, x, self.precision)
        elif self.native:
            y = self.native.linear(matrix, x, self.precision)
        else:
            y = matrix.reference(x, self.precision)
        self.timings[label] += time.perf_counter() - start
        return y

    def delta(self, layer, state, projected):
        start = time.perf_counter()
        if self.native and self.precision == "bf16":
            self.native.conv_bf16(projected, state["conv"], layer["conv"])
        else:
            raw = projected[:6144].copy()
            convolved = np.sum(state["conv"] * layer["conv"][:, :3], axis=1) + raw * layer["conv"][:, 3]
            state["conv"][:, :2] = state["conv"][:, 1:]
            state["conv"][:, 2] = raw
            projected[:6144] = self.round(silu(convolved))
        if self.native:
            result = self.native.gdn(state["ssm"], projected, layer["a_log"], layer["dt"], layer["norm"])
        else:
            q, k, v = projected[:6144].reshape(3, 16, 128)
            q = q / np.sqrt(np.sum(q * q, axis=1, keepdims=True) + np.float32(1e-6)) / np.sqrt(np.float32(128))
            k = k / np.sqrt(np.sum(k * k, axis=1, keepdims=True) + np.float32(1e-6))
            beta = sigmoid(projected[8192:8208])
            decay = np.exp(-np.exp(layer["a_log"]) * np.logaddexp(np.float32(0), projected[8208:] + layer["dt"]))
            s = state["ssm"]
            s *= decay[:, None, None]
            retrieved = np.einsum("hij,hj->hi", s, k)
            change = beta[:, None] * (v - retrieved)
            s += change[:, :, None] * k[:, None, :]
            o = np.einsum("hij,hj->hi", s, q)
            result = (o / np.sqrt(np.mean(o * o, axis=1, keepdims=True) + np.float32(1e-6))
                      * layer["norm"] * silu(projected[6144:8192].reshape(16, 128))).reshape(-1)
        self.timings["recurrent"] += time.perf_counter() - start
        return self.round(result)

    def attention(self, layer, state, projected):
        start = time.perf_counter()
        q = self.rms(projected[:2048].reshape(8, 256), layer["qn"])
        k = self.rms(projected[2048:2560].reshape(2, 256), layer["kn"])
        v = projected[2560:3072].reshape(2, 256)
        if self.native and self.precision == "bf16":
            self.native.rope_bf16(q, self.position)
            self.native.rope_bf16(k, self.position)
            state["k"][self.position], state["v"][self.position] = k, v
            result = self.native.attention_bf16(q, state["k"], state["v"], projected[3072:], self.position + 1)
            self.timings["attention"] += time.perf_counter() - start
            return result
        angle = self.position / (np.float32(1e7) ** (np.arange(32, dtype=np.float32) / 32))
        cos, sin = np.cos(angle), np.sin(angle)
        for a in (q, k):
            left, right = a[:, :32].copy(), a[:, 32:64].copy()
            a[:, :32] = self.round(left * cos - right * sin)
            a[:, 32:64] = self.round(right * cos + left * sin)
        state["k"][self.position], state["v"][self.position] = k, v
        keys = state["k"][:self.position + 1]
        values = state["v"][:self.position + 1]
        output = np.empty((8, 256), dtype=np.float32)
        for h in range(8):
            scores = keys[:, h // 4] @ (q[h] / 16)
            probability = np.exp(scores - np.max(scores))
            probability /= np.sum(probability)
            output[h] = probability @ values[:, h // 4]
        result = self.round(self.round(output).reshape(-1) * sigmoid(projected[3072:5120]))
        self.timings["attention"] += time.perf_counter() - start
        return result

    def step(self, token, logits=True, trace=None):
        if self.position >= self.context:
            raise ValueError("context capacity exceeded")
        x = self.embedding.lookup(int(token), self.precision)
        for i, (layer, state) in enumerate(zip(self.layers, self.states)):
            y = self.linear(layer["proj"], self.rms(x, layer["n1"]), "mixer_projection")
            y = self.attention(layer, state, y) if layer["attention"] else self.delta(layer, state, y)
            x = self.round(x + self.linear(layer["out"], y, "mixer_output"))
            y = self.linear(layer["up"], self.rms(x, layer["n2"]), "mlp_up")
            y = (self.native.mlp_bf16(y) if self.native and self.precision == "bf16" else
                 self.round(y[:3584] * self.round(silu(y[3584:]))))
            x = self.round(x + self.linear(layer["down"], y, "mlp_down"))
            if trace is not None:
                trace.append(x.copy())
        self.position += 1
        if not logits:
            return None
        y = self.linear(self.embedding, self.rms(x, self.norm), "head")
        if not np.isfinite(y).all():
            raise RuntimeError("nonfinite model logits")
        return y
