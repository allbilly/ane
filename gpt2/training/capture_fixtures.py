"""Capture first-use inputs/outputs for replay of all training kernel templates."""
import json
import os
from pathlib import Path
import numpy as np
from backends import Backend, Primitives
from train import GPT2, load_weights

ROOT = Path(__file__).resolve().parent


def main():
  for name in ("aneforge", "orion"):
    protocol = json.loads((ROOT / name / "results.json").read_text())
    weights = load_weights(Path(protocol["initial_weights"]))
    os.environ["ANEFORGE_CACHE_DIR"] = str(ROOT / name / "cache")
    backend = Backend(name, ROOT / name)
    original, captured = backend.program, set()
    def program(label, shapes, builder):
      native = original(label, shapes, builder)
      if label in captured: return native
      def capture(*arrays):
        out = native(*arrays)
        np.savez(ROOT / name / "kernels" / label / "fixture.npz",
                 **{f"input{i:02d}": np.asarray(a, np.float16) for i, a in enumerate(arrays)},
                 output=np.asarray(out, np.float16))
        captured.add(label)
        return out
      return capture
    backend.program = program
    try:
      model = GPT2(weights, Primitives(backend), np.array(protocol["tokens"], np.int64))
      loss, cache = model.forward()
      grads = model.backward(cache)
      print(f"{name}: loss {loss:.8f}; {len(captured)} kernel fixtures", flush=True)
      del weights, grads, cache, model
    finally:
      backend.close()


if __name__ == "__main__": main()
