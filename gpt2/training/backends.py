"""Independent ANEForge and Orion graph/codegen/runtime adapters for GPT-2 training."""
import ctypes as C
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
I4 = C.c_int * 4


class OrionTensor:
  def __init__(self, graph, node, shape): self.graph, self.node, self.shape = graph, node, tuple(shape)
  def binary(self, other, op):
    if not isinstance(other, OrionTensor): other = self.graph.scalar(float(other))
    shape = np.broadcast_shapes(self.shape, other.shape)
    a, b = (self, other) if self.shape else (other, self)
    node = self.graph.call("orion_gb_" + op, a.node, b.node, self.graph.name())
    return OrionTensor(self.graph, node, shape)
  def __add__(self, other): return self.binary(other, "add")
  def __sub__(self, other): return self.binary(other, "sub")
  def __mul__(self, other): return self.binary(other, "mul")
  def __matmul__(self, other):
    shape = (*np.broadcast_shapes(self.shape[:-2], other.shape[:-2]), self.shape[-2], other.shape[-1])
    node = self.graph.call("orion_gb_matmul", self.node, other.node, False, False, self.graph.name(), I4(*shape))
    return OrionTensor(self.graph, node, shape)
  def transpose(self, perm):
    shape = tuple(self.shape[i] for i in perm)
    p = self.graph.ints(perm)
    node = self.graph.call("orion_gb_transpose", self.node, p, self.graph.name(), I4(*perm), I4(*shape))
    return OrionTensor(self.graph, node, shape)
  def unary(self, op):
    return OrionTensor(self.graph, self.graph.call("orion_gb_" + op, self.node, self.graph.name()), self.shape)
  def tanh(self): return self.unary("tanh")
  def rsqrt(self): return self.unary("rsqrt")
  def softmax(self, axis):
    return OrionTensor(self.graph, self.graph.call("orion_gb_softmax", self.node, axis, self.graph.name()), self.shape)
  def reduce(self, axes, op):
    shape = tuple(1 if i in axes else n for i, n in enumerate(self.shape))
    node = self.graph.call("orion_gb_reduce_" + op, self.node, self.graph.ints(axes), True,
                           self.graph.name(), I4(*shape))
    return OrionTensor(self.graph, node, shape)
  def mean(self, axes): return self.reduce(axes, "mean")
  def sum(self, axes): return self.reduce(axes, "sum")


class OrionGraph:
  def __init__(self, lib):
    self.lib, self.counter, self.inputs = lib, 0, []
    self.handle = lib.orion_graph_create()
  def name(self):
    self.counter += 1
    return f"n{self.counter:04d}".encode()
  def call(self, op, *args):
    node = getattr(self.lib, op)(self.handle, *args)
    if node < 0: raise RuntimeError(f"Orion graph construction failed: {op}")
    return node
  def input(self, shape):
    name = f"input{len(self.inputs):02d}".encode()
    node = self.call("orion_gb_input", name, 0, I4(*shape))
    value = OrionTensor(self, node, shape)
    self.inputs.append(value)
    return value
  def scalar(self, x):
    return OrionTensor(self, self.call("orion_gb_const_scalar", self.name(), 0, x), ())
  def ints(self, values):
    values = tuple(values)
    return self.call("orion_gb_const_int32", self.name(), I4(len(values), 0, 0, 0), I4(*values), len(values))


class OrionProgram:
  def __init__(self, backend, name, shapes, graph, output):
    self.backend, self.shapes, self.out_shape = backend, shapes, output.shape
    directory = backend.dump / "kernels" / name
    directory.mkdir(parents=True, exist_ok=True)
    graph.lib.orion_gb_output(graph.handle, output.node, b"output")
    mil = directory / "model.mil"
    if graph.lib.training_codegen(graph.handle, str(mil).encode()): raise RuntimeError("Orion codegen failed")
    self.handle = graph.lib.training_open(str(mil).encode(), str(directory / "compiled").encode())
    if not self.handle: raise RuntimeError(f"Orion ANE compilation failed: {name}")
    graph.lib.orion_graph_free(graph.handle)
    self.inputs = [graph.lib.orion_tensor_create(int(np.prod(s[:-1])), s[-1]) for s in shapes]
    self.outputs = [graph.lib.orion_tensor_create(int(np.prod(output.shape[:-1])), output.shape[-1])]
    if not all([*self.inputs, *self.outputs]): raise RuntimeError("IOSurface allocation failed")
    self.in_ptr = (C.c_void_p * len(self.inputs))(*self.inputs)
    self.out_ptr = (C.c_void_p * 1)(*self.outputs)
    self.name = name
    backend.receipts.append({"name": name, "inputs": shapes, "output": output.shape,
                             "mil": str(mil.relative_to(backend.dump)),
                             "input_port_names": [f"input{i:02d}" for i in range(len(shapes))],
                             "output_port_name": f"n{graph.counter:04d}"})
  def __call__(self, *arrays):
    lib = self.backend.lib
    for shape, surface, a in zip(self.shapes, self.inputs, arrays):
      if tuple(a.shape) != shape: raise ValueError(f"input shape {a.shape} != {shape}")
      a = np.ascontiguousarray(a, np.float16)
      lib.orion_tensor_write(surface, a.ctypes.data, a.nbytes)
    start = time.perf_counter()
    if not lib.orion_eval(self.handle, self.in_ptr, len(self.inputs), self.out_ptr, 1):
      raise RuntimeError(f"Orion ANE evaluation failed: {self.name}")
    self.backend.dispatch_seconds += time.perf_counter() - start
    self.backend.dispatches += 1
    out = np.empty(self.out_shape, np.float16)
    lib.orion_tensor_read(self.outputs[0], out.ctypes.data, out.nbytes)
    return out.astype(np.float32)
  def close(self):
    for surface in [*self.inputs, *self.outputs]: self.backend.lib.orion_tensor_release(surface)
    self.backend.lib.orion_release_program(self.handle)


class Backend:
  def __init__(self, name, dump):
    self.name, self.dump = name, Path(dump)
    self.dump.mkdir(parents=True, exist_ok=True)
    self.cache, self.receipts, self.dispatch_seconds, self.dispatches = {}, [], 0.0, 0
    self.compile_seconds = 0.0
    if name == "aneforge":
      sys.path.insert(0, str(Path.home() / "Desktop/ANEForge"))
      import aneforge as af
      self.af = af
    else:
      self.lib = C.CDLL(str(ROOT / "build/liborion_training.dylib"))
      self.bind_orion()
  def bind_orion(self):
    lib = self.lib
    signatures = {
      "orion_graph_create": (C.c_void_p, []), "orion_graph_free": (None, [C.c_void_p]),
      "orion_gb_input": (C.c_int, [C.c_void_p, C.c_char_p, C.c_int, I4]),
      "orion_gb_const_scalar": (C.c_int, [C.c_void_p, C.c_char_p, C.c_int, C.c_float]),
      "orion_gb_const_int32": (C.c_int, [C.c_void_p, C.c_char_p, I4, I4, C.c_int]),
      "orion_gb_matmul": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_bool, C.c_bool, C.c_char_p, I4]),
      "orion_gb_transpose": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_char_p, I4, I4]),
      "orion_gb_softmax": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_char_p]),
      "orion_gb_output": (None, [C.c_void_p, C.c_int, C.c_char_p]),
      "training_codegen": (C.c_int, [C.c_void_p, C.c_char_p]),
      "training_open": (C.c_void_p, [C.c_char_p, C.c_char_p]),
      "orion_tensor_create": (C.c_void_p, [C.c_int, C.c_int]),
      "orion_tensor_write": (None, [C.c_void_p, C.c_void_p, C.c_size_t]),
      "orion_tensor_read": (None, [C.c_void_p, C.c_void_p, C.c_size_t]),
      "orion_tensor_release": (None, [C.c_void_p]),
      "orion_eval": (C.c_bool, [C.c_void_p, C.POINTER(C.c_void_p), C.c_int, C.POINTER(C.c_void_p), C.c_int]),
      "orion_release_program": (None, [C.c_void_p]),
    }
    for op in ("add", "sub", "mul"):
      signatures["orion_gb_" + op] = (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_char_p])
    for op in ("tanh", "rsqrt"):
      signatures["orion_gb_" + op] = (C.c_int, [C.c_void_p, C.c_int, C.c_char_p])
    for op in ("mean", "sum"):
      signatures["orion_gb_reduce_" + op] = (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_bool, C.c_char_p, I4])
    for name, (result, args) in signatures.items():
      getattr(lib, name).restype, getattr(lib, name).argtypes = result, args
  def program(self, name, shapes, builder):
    shapes = [tuple(s) for s in shapes]
    key = name, tuple(shapes)
    if key in self.cache: return self.cache[key]
    start = time.perf_counter()
    if self.name == "aneforge":
      from aneforge._compile import compile_multi
      inputs = [self.af.input(s) for s in shapes]
      output = builder(*inputs)
      directory = self.dump / "kernels" / name
      model = compile_multi([output], build_dir=directory)
      ports = {id(t): n for t, n in model.input_ports}
      output_name = model.output_ports[0][1]
      def run(*arrays):
        for t, array in zip(inputs, arrays):
          if id(t) in ports: model.prog.set_input(ports[id(t)], np.asarray(array, np.float16))
        t0 = time.perf_counter()
        model.prog.execute()
        self.dispatch_seconds += time.perf_counter() - t0
        self.dispatches += 1
        return model.prog.read_output(output_name).astype(np.float32)
      run.close = model.release
      self.receipts.append({"name": name, "inputs": shapes, "output": output.shape,
                            "mil": str((directory / "model.mil").relative_to(self.dump)),
                            "input_port_names": [ports.get(id(t)) for t in inputs], "output_port_name": output_name})
      prog = run
    else:
      graph = OrionGraph(self.lib)
      inputs = [graph.input(s) for s in shapes]
      prog = OrionProgram(self, name, shapes, graph, builder(*inputs))
    self.compile_seconds += time.perf_counter() - start
    self.cache[key] = prog
    print(f"compiled {self.name}: {name} ({time.perf_counter() - start:.2f}s)", flush=True)
    (self.dump / "kernel-index.json").write_text(json.dumps(self.receipts, indent=2) + "\n")
    return prog
  def run(self, name, arrays, builder):
    return self.program(name, [a.shape for a in arrays], builder)(*arrays)
  def close(self):
    for program in self.cache.values(): program.close()


def transpose(x): return x.transpose([0, 1, 3, 2])


def norm_parts(x):
  centered = x - x.mean((3,))
  inv = ((centered * centered).mean((3,)) + 1e-5).rsqrt()
  return centered * inv, inv


def gelu(x):
  return x * 0.5 * ((x + x * x * x * 0.044715) * float(np.sqrt(2 / np.pi))).tanh().__add__(1.0)


class Primitives:
  def __init__(self, backend): self.backend = backend
  def add(self, a, b): return self.backend.run(f"add_{a.shape[-2]}_{a.shape[-1]}", [a, b], lambda x, y: x + y)
  def linear(self, x, w, b):
    return self.backend.run(f"linear_{x.shape[-2]}_{w.shape[-2]}_{w.shape[-1]}_forward", [x, w, b], lambda x, w, b: x @ w + b)
  def linear_backward(self, x, w, g):
    tag = f"linear_{x.shape[-2]}_{w.shape[-2]}_{w.shape[-1]}"
    dx = self.backend.run(tag + "_backward_dx", [g, w], lambda g, w: g @ transpose(w))
    dw = self.backend.run(tag + "_backward_dw", [x, g], lambda x, g: transpose(x) @ g)
    db = self.backend.run(tag + "_backward_db", [g], lambda g: g.sum((2,)))
    return dx, dw, db
  def norm(self, x, gamma, beta):
    return self.backend.run(f"layernorm_{x.shape[-2]}_{x.shape[-1]}_forward", [x, gamma, beta],
                            lambda x, gamma, beta: norm_parts(x)[0] * gamma + beta)
  def norm_backward(self, x, gamma, g):
    tag = f"layernorm_{x.shape[-2]}_{x.shape[-1]}"
    def dx(x, gamma, g):
      z, inv = norm_parts(x)
      u = g * gamma
      return (u - u.mean((3,)) - z * (u * z).mean((3,))) * inv
    gx = self.backend.run(tag + "_backward_dx", [x, gamma, g], dx)
    gg = self.backend.run(tag + "_backward_dgamma", [x, g], lambda x, g: (norm_parts(x)[0] * g).sum((2,)))
    gb = self.backend.run(tag + "_backward_dbeta", [g], lambda g: g.sum((2,)))
    return gx, gg, gb
  def gelu(self, x): return self.backend.run(f"gelu_{x.shape[-2]}_{x.shape[-1]}_forward", [x], gelu)
  def gelu_backward(self, x, g):
    def derivative(x, g):
      a = float(np.sqrt(2 / np.pi))
      t = ((x + x * x * x * 0.044715) * a).tanh()
      dt = ((x * x * 0.134145 + 1.0) * a) * (t * t * -1.0 + 1.0)
      return g * ((t + 1.0) * 0.5 + x * dt * 0.5)
    return self.backend.run(f"gelu_{x.shape[-2]}_{x.shape[-1]}_backward", [x, g], derivative)
  def mm(self, a, b, label):
    return self.backend.run(label + "_forward", [a, b], lambda a, b: a @ b)
  def mm_backward(self, a, b, g, label):
    ga = self.backend.run(label + "_backward_da", [g, b], lambda g, b: g @ transpose(b))
    gb = self.backend.run(label + "_backward_db", [a, g], lambda a, g: transpose(a) @ g)
    return ga, gb
  def softmax(self, scores, mask, scale):
    return self.backend.run(f"softmax_{scores.shape[1]}_{scores.shape[2]}_forward", [scores, mask],
                            lambda x, m: (x * scale + m).softmax(3))
  def softmax_backward(self, p, g, scale):
    return self.backend.run(f"softmax_{p.shape[1]}_{p.shape[2]}_backward", [p, g],
                            lambda p, g: p * (g - (g * p).sum((3,))) * scale)
