"""ANE paired-plane projections with FP32 host encoder arithmetic on macOS."""
import copy
import hashlib
from pathlib import Path
import platform

import numpy as np

from whisper.validation import digest

CHECKPOINT_SHA256 = "db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad"


class PairedMacEncoder:
    """24 actual ANE submissions; HF encoder operations retain FP32 precision.

    Every projection splits its contraction in two, uses two rounding grids,
    and represents each grid with high/residual FP16 planes. Products return
    separately and combine in FP32; biases are added in FP32. All checkpoint
    encoder weights fit FP16 exactly. Convolutions and attention stay on CPU.
    """

    def __init__(self, hf_encoder, checkpoint, directory):
        if platform.system() != "Darwin" or digest(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("requires macOS and the pinned tiny.en checkpoint")
        import aneforge as af
        import torch

        self.encoder = copy.deepcopy(hf_encoder).eval()
        self.programs, self.receipts = [], []
        self.submissions = 0
        owner = self
        self.gains = (1., 1.375)

        class Projection(torch.nn.Module):
            def __init__(self, linear, name):
                super().__init__()
                weights = linear.weight.detach().numpy()
                if not np.array_equal(weights, weights.astype(np.float16).astype(np.float32)):
                    raise ValueError("projection weights require nonzero residual planes: " + name)
                self.bias = linear.bias.detach().numpy().copy() if linear.bias is not None else None
                self.n, self.k = weights.shape
                self.partitions = 2
                half = self.k // self.partitions
                grouped = np.concatenate([weights[:, :half], weights[:, half:]], axis=0)
                node = af.input((1, self.k, 1, 6000))
                output = af.conv(node, grouped.reshape(2*self.n, half, 1, 1), groups=2)
                build = Path(directory) / name
                self.program = af.compile(output, build_dir=build, opt=0)
                if self.program._prog._device_mask != 4:
                    self.program.release()
                    raise RuntimeError("paired projection must execute on ANE only")
                owner.programs.append(self.program)
                self.feed, self.output = self.program.input_view(), self.program.output_view()
                owner.receipts.append(dict(name=name, input_features=self.k, output_features=self.n,
                    contraction_partitions=2, gains=list(owner.gains), temporal_planes=4,
                    input_shape=[1,self.k,1,6000], output_shape=[1,2*self.n,1,6000],
                    device_mask=4, mil_sha256=digest(build / "model.mil"),
                    weight_sha256=hashlib.sha256(weights.tobytes()).hexdigest()))

            def forward(self, tensor):
                x = tensor.detach().numpy()
                if x.shape != (1,1500,self.k) or not np.isfinite(x).all():
                    raise ValueError("paired encoder requires one complete finite 1500-position context")
                channels = x[0].T
                for index, gain in enumerate(owner.gains):
                    scaled = channels * np.float32(gain)
                    high = scaled.astype(np.float16)
                    low = ((scaled - high.astype(np.float32))*4096).astype(np.float16)
                    begin = index*3000
                    self.feed[0,:,0,begin:begin+1500] = high
                    self.feed[0,:,0,begin+1500:begin+3000] = low
                self.program._prog.execute()
                owner.submissions += 1
                values = self.output.astype(np.float32).reshape(2,self.n,6000)
                result = np.zeros((self.n,1500), np.float32)
                for index, gain in enumerate(owner.gains):
                    begin = index*3000
                    for part in range(2):
                        result += (values[part,:,begin:begin+1500] +
                            values[part,:,begin+1500:begin+3000]/4096)/np.float32(gain*2)
                if self.bias is not None:
                    result += self.bias[:,None]
                if not np.isfinite(result).all():
                    raise RuntimeError("paired ANE projection returned nonfinite values")
                return torch.from_numpy(result.T.copy()).reshape(1,1500,self.n)

        try:
            for index, layer in enumerate(self.encoder.layers):
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    setattr(layer.self_attn, name, Projection(getattr(layer.self_attn, name), f"layer{index}-{name}"))
                for name in ("fc1", "fc2"):
                    setattr(layer, name, Projection(getattr(layer, name), f"layer{index}-{name}"))
        except BaseException:
            self.close()
            raise

    def __call__(self, mel):
        import torch
        before = self.submissions
        with torch.inference_mode():
            output = self.encoder(torch.from_numpy(np.asarray(mel, np.float32).reshape(1,80,3000))).last_hidden_state.numpy()
        if self.submissions-before != 24:
            raise RuntimeError("precision encoder did not execute all 24 ANE projections")
        return output

    def close(self):
        for program in self.programs:
            program.release()
        self.programs.clear()
