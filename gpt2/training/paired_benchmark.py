"""Alternate isolated full GPT-2 forward/backward passes; no further weight updates."""
import json
import os
from pathlib import Path
import time
import numpy as np
from backends import Backend, Primitives
from train import GPT2, load_weights

ROOT = Path(__file__).resolve().parent


def main():
  protocols = {name: json.loads((ROOT / name / "results.json").read_text()) for name in ("aneforge", "orion")}
  weights = load_weights(Path(protocols["aneforge"]["initial_weights"]))
  tokens = np.array(protocols["aneforge"]["tokens"], np.int64)
  models, backends, rows = {}, {}, []
  for name in protocols:
    os.environ["ANEFORGE_CACHE_DIR"] = str(ROOT / name / "cache")
    backend = Backend(name, ROOT / name)
    model = GPT2(weights, Primitives(backend), tokens)
    loss, cache = model.forward()
    gradients = model.backward(cache)
    print(f"warm {name}: loss={loss:.8f}", flush=True)
    del gradients, cache
    backends[name], models[name] = backend, model
  try:
    for pair in range(10):
      order = ("aneforge", "orion") if pair % 2 == 0 else ("orion", "aneforge")
      for name in order:
        backend, model = backends[name], models[name]
        d0, n0 = backend.dispatch_seconds, backend.dispatches
        t0 = time.perf_counter()
        loss, cache = model.forward()
        t1 = time.perf_counter()
        gradients = model.backward(cache)
        t2 = time.perf_counter()
        row = {"pair": pair, "backend": name, "loss": loss, "forward_ms": (t1 - t0) * 1000,
               "backward_ms": (t2 - t1) * 1000, "total_ms": (t2 - t0) * 1000,
               "ane_execute_ms": (backend.dispatch_seconds - d0) * 1000,
               "dispatches": backend.dispatches - n0}
        rows.append(row)
        del gradients, cache
        print(f"pair {pair}: {name} {row['total_ms']:.1f} ms, ANE {row['ane_execute_ms']:.1f} ms", flush=True)
    summary = {name: {key: float(np.median([r[key] for r in rows if r["backend"] == name]))
                     for key in ("forward_ms", "backward_ms", "total_ms", "ane_execute_ms")}
               for name in protocols}
    (ROOT / "paired-benchmark.json").write_text(json.dumps({"protocol": "10 alternating pairs, same initial weights, no optimizer, no compile inside timing",
                                                           "rows": rows, "median": summary}, indent=2) + "\n")
  finally:
    for backend in backends.values(): backend.close()


if __name__ == "__main__": main()
