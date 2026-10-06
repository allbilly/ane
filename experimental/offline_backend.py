"""Emit ANEForge MIL without loading an ANE runtime; evaluate CPU reference fixtures."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path.home() / "Desktop/ANEForge"))


def emit_multi(outputs, directory):
    # The same lowering used by compile_multi, stopping before E5RT.compile.
    from aneforge._compile import _topo_multi, _Emitter, _EMIT, _emit_program_dir, NETPLIST_OPS
    order = _topo_multi(*outputs)
    if any(t.op in NETPLIST_OPS for t in order):
        raise NotImplementedError("offline export does not support segmented graphs")
    unsupported = {t.op for t in order if t.op != "input" and t.op not in _EMIT}
    if unsupported:
        raise NotImplementedError(f"unsupported ANEForge operations: {sorted(unsupported)}")
    inputs = sorted((t for t in order if t.op == "input"), key=lambda t: t.attrs.get("idx", 0))
    for i, tensor in enumerate(order):
        tensor._name = f"t{i}"
    emitter = _Emitter(False)
    for tensor in order:
        if tensor.op != "input":
            _EMIT[tensor.op](emitter, tensor, tensor._name, [source._name for source in tensor.srcs])
    _emit_program_dir(emitter, inputs, ", ".join(t._name for t in outputs), None, directory)
    return inputs


class CPUTensor:
    """FP32 Torch operations, with FP16 rounding only at program boundaries."""
    def __init__(self, value): self.value = value
    def _binary(self, other, operation):
        return CPUTensor(operation(self.value, other.value if isinstance(other, CPUTensor) else other))
    def __add__(self, other): return self._binary(other, lambda x, y: x + y)
    def __sub__(self, other): return self._binary(other, lambda x, y: x - y)
    def __mul__(self, other): return self._binary(other, lambda x, y: x * y)
    def __matmul__(self, other): return self._binary(other, lambda x, y: x @ y)
    def transpose(self, axes): return CPUTensor(self.value.permute(*axes))
    def mean(self, axes): return CPUTensor(self.value.mean(dim=tuple(axes), keepdim=True))
    def sum(self, axes): return CPUTensor(self.value.sum(dim=tuple(axes), keepdim=True))
    def tanh(self): return CPUTensor(self.value.tanh())
    def rsqrt(self): return CPUTensor(self.value.rsqrt())
    def softmax(self, axis): return CPUTensor(self.value.softmax(dim=axis))


class OfflineBackend:
    def __init__(self, directory):
        import torch
        import aneforge as af
        torch.set_num_threads(4)
        self.torch, self.af, self.dump = torch, af, Path(directory)
        self.cache, self.receipts, self.dispatches = {}, [], 0
        self.dispatch_seconds = 0.

    def program(self, name, shapes, builder):
        key = name, tuple(tuple(s) for s in shapes)
        if key in self.cache: return self.cache[key]
        inputs = [self.af.input(s) for s in shapes]
        output = builder(*inputs)
        directory = self.dump / "kernels" / name
        live = emit_multi([output], directory)
        ports = {id(t): t._name for t in live}
        self.receipts.append(dict(name=name, inputs=shapes, output=output.shape,
                                  mil=str((directory / "model.mil").relative_to(self.dump)),
                                  input_port_names=[ports.get(id(t)) for t in inputs],
                                  output_port_name=output._name, reference_kind="CPU FP32 with FP16 program boundaries"))
        import json
        (self.dump / "kernel-index.json").write_text(json.dumps(self.receipts, indent=2) + "\n")
        captured = False
        def run(*arrays):
            nonlocal captured
            rounded = [np.asarray(array, np.float16) for array in arrays]
            tensors = [CPUTensor(self.torch.from_numpy(array.astype(np.float32))) for array in rounded]
            with self.torch.no_grad(): value = builder(*tensors).value.numpy()
            result = value.astype(np.float16)
            if not np.isfinite(result).all():
                raise RuntimeError(f"non-finite CPU reference output: {name}")
            if not captured:
                np.savez_compressed(directory / "fixture.npz",
                                    **{f"input{i:02d}": array for i, array in enumerate(rounded)},
                                    output=result, fp32_output=value)
                captured = True
            self.dispatches += 1  # CPU program calls, never ANE submissions.
            return result.astype(np.float32)
        self.cache[key] = run
        return run

    def run(self, name, arrays, builder):
        return self.program(name, [array.shape for array in arrays], builder)(*arrays)

    def close(self): pass
