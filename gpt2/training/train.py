"""Matched full GPT-2 124M training POC using either repository's ANE compiler/runtime."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np

from backends import Backend, Primitives

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from bpe import Tokenizer
from external_weights import verify_weights

D, FF, H, DH, L, V = 768, 3072, 12, 64, 12, 50257
SCALE = 128.0
TEXT = ("The Apple Neural Engine can learn from examples. We train a language model on a small batch, "
        "compare its predictions with the next tokens, and update the weights to reduce the loss. "
        "This experiment checks that forward and backward computation work correctly.")


def hash_file(path):
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
  return digest.hexdigest()


def load_weights(source):
  weights = {}
  shapes = {"ln1_g": (D,), "ln1_b": (D,), "ln2_g": (D,), "ln2_b": (D,),
            **{name: (D, D) for name in ("wq", "wk", "wv", "wo")},
            **{name: (D,) for name in ("bq", "bk", "bv", "bo", "bproj")},
            "wfc": (FF, D), "bfc": (FF,), "wproj": (D, FF)}
  for layer in range(L):
    for name, shape in shapes.items():
      array = np.fromfile(source / f"layer{layer}/{name}.bin", dtype="<f2", offset=128).astype(np.float32).reshape(shape)
      if name.startswith("w"): array = array.T.copy().reshape(1, 1, shape[1], shape[0])
      else: array = array.reshape(1, 1, 1, -1)
      weights[f"layer{layer}/{name}"] = array
  for name, shape in {"wte": (V, D), "wpe": (1024, D), "ln_f_g": (1, 1, 1, D), "ln_f_b": (1, 1, 1, D)}.items():
    weights[name] = np.fromfile(source / f"{name}.bin", dtype="<f2", offset=128).astype(np.float32).reshape(shape)
  return weights


def heads(x):
  return np.ascontiguousarray(x.reshape(-1, H, DH).transpose(1, 0, 2)[None])


def unheads(x): return np.ascontiguousarray(x[0].transpose(1, 0, 2).reshape(1, 1, -1, D))


class GPT2:
  def __init__(self, weights, primitives, tokens):
    self.w, self.p, self.tokens = weights, primitives, tokens
    seq = len(tokens) - 1
    self.mask = np.triu(np.full((1, 1, seq, seq), -1e4, np.float32), 1)
  def forward(self):
    p, w, seq = self.p, self.w, len(self.tokens) - 1
    x = (w["wte"][self.tokens[:-1]] + w["wpe"][:seq]).reshape(1, 1, seq, D)
    caches = []
    for i in range(L):
      get = lambda n: w[f"layer{i}/{n}"]
      z = p.norm(x, get("ln1_g"), get("ln1_b"))
      q, k, v = (heads(p.linear(z, get("w" + n), get("b" + n))) for n in "qkv")
      kt = np.ascontiguousarray(k.transpose(0, 1, 3, 2))
      scores = p.mm(q, kt, "attention_qk")
      prob = p.softmax(scores, self.mask, 0.125)
      attended = p.mm(prob, v, "attention_av")
      a = unheads(attended)
      h = p.add(x, p.linear(a, get("wo"), get("bo")))
      zn = p.norm(h, get("ln2_g"), get("ln2_b"))
      fc = p.linear(zn, get("wfc"), get("bfc"))
      act = p.gelu(fc)
      out = p.add(h, p.linear(act, get("wproj"), get("bproj")))
      caches.append((x, z, q, kt, v, prob, a, h, zn, fc, act))
      x = out
    normalized = p.norm(x, w["ln_f_g"], w["ln_f_b"])
    hidden = normalized.reshape(seq, D)
    logits = hidden @ w["wte"].T
    shifted = logits - logits.max(-1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp / exp.sum(-1, keepdims=True)
    loss = float((np.log(exp.sum(-1)) - shifted[np.arange(seq), self.tokens[1:]]).mean())
    return loss, (caches, x, hidden, prob)
  def backward(self, cache):
    p, w, seq = self.p, self.w, len(self.tokens) - 1
    caches, x, hidden, prob = cache
    dp = prob.copy()
    dp[np.arange(seq), self.tokens[1:]] -= 1.0
    dp *= np.float32(SCALE / seq)
    grads = {"wte": dp.T @ hidden}
    g = (dp @ w["wte"]).reshape(1, 1, seq, D)
    g, grads["ln_f_g"], grads["ln_f_b"] = p.norm_backward(x, w["ln_f_g"], g)
    for i in range(L - 1, -1, -1):
      get = lambda n: w[f"layer{i}/{n}"]
      put = lambda n: f"layer{i}/{n}"
      x, z, q, kt, v, prob, a, h, zn, fc, act = caches[i]
      ga, grads[put("wproj")], grads[put("bproj")] = p.linear_backward(act, get("wproj"), g)
      gf = p.gelu_backward(fc, ga)
      gzn, grads[put("wfc")], grads[put("bfc")] = p.linear_backward(zn, get("wfc"), gf)
      gh, grads[put("ln2_g")], grads[put("ln2_b")] = p.norm_backward(h, get("ln2_g"), gzn)
      gh = p.add(g, gh)
      gat, grads[put("wo")], grads[put("bo")] = p.linear_backward(a, get("wo"), gh)
      gp, gv = p.mm_backward(prob, v, heads(gat), "attention_av")
      gs = p.softmax_backward(prob, gp, 0.125)
      gq, gkt = p.mm_backward(q, kt, gs, "attention_qk")
      gk = np.ascontiguousarray(gkt.transpose(0, 1, 3, 2))
      gz = None
      for name, upstream in zip("qkv", (gq, gk, gv)):
        gi, grads[put("w" + name)], grads[put("b" + name)] = p.linear_backward(z, get("w" + name), unheads(upstream))
        gz = gi if gz is None else p.add(gz, gi)
      gx, grads[put("ln1_g")], grads[put("ln1_b")] = p.norm_backward(x, get("ln1_g"), gz)
      g = p.add(gh, gx)
    np.add.at(grads["wte"], self.tokens[:-1], g.reshape(seq, D))
    grads["wpe"] = np.zeros_like(w["wpe"])
    grads["wpe"][:seq] = g.reshape(seq, D)
    for name in grads: grads[name] *= np.float32(1.0 / SCALE)
    if set(grads) != set(w): raise AssertionError("Not every GPT-2 parameter has a gradient")
    if not all(np.isfinite(g).all() for g in grads.values()): raise FloatingPointError("Nonfinite gradient")
    return grads


class Adam:
  def __init__(self, lr): self.lr, self.t, self.m, self.v = lr, 0, {}, {}
  def update(self, weights, grads):
    self.t += 1
    norm = float(np.sqrt(sum(np.sum(g * g, dtype=np.float64) for g in grads.values())))
    factor = min(1.0, 1.0 / (norm + 1e-12))
    for key in sorted(weights):
      g = grads[key]
      g *= np.float32(factor)
      if key not in self.m:
        self.m[key], self.v[key] = np.zeros_like(g), np.zeros_like(g)
      m, v = self.m[key], self.v[key]
      m *= 0.9
      m += g * 0.1
      v *= 0.999
      v += (g * g) * 0.001
      weights[key] -= np.float32(self.lr * np.sqrt(1.0 - 0.999 ** self.t) / (1.0 - 0.9 ** self.t)) * m / (np.sqrt(v) + np.float32(1e-8 * np.sqrt(1.0 - 0.999 ** self.t)))
    return norm


def torch_oracle(weights, tokens, compute_gradients=True):
  import torch
  import torch.nn.functional as F
  # CPU only. This oracle is outside every timed training step.
  torch.set_num_threads(4)
  w = {k: torch.from_numpy(a.copy()).requires_grad_(compute_gradients) for k, a in weights.items()}
  seq = len(tokens) - 1
  ids = torch.tensor(tokens, dtype=torch.long)
  x = (w["wte"][ids[:-1]] + w["wpe"][:seq]).reshape(1, 1, seq, D)
  def norm(x, gamma, beta): return F.layer_norm(x, (D,), gamma.reshape(D), beta.reshape(D), 1e-5)
  def hs(x): return x.reshape(seq, H, DH).permute(1, 0, 2)[None]
  mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
  for i in range(L):
    get = lambda n: w[f"layer{i}/{n}"]
    z = norm(x, get("ln1_g"), get("ln1_b"))
    q, k, v = (hs(z @ get("w" + n) + get("b" + n)) for n in "qkv")
    a = (torch.softmax(q @ k.transpose(-1, -2) * 0.125 + mask, -1) @ v)[0].permute(1, 0, 2).reshape(1, 1, seq, D)
    x = x + a @ get("wo") + get("bo")
    z = norm(x, get("ln2_g"), get("ln2_b"))
    x = x + F.gelu(z @ get("wfc") + get("bfc"), approximate="tanh") @ get("wproj") + get("bproj")
  logits = norm(x, w["ln_f_g"], w["ln_f_b"]).reshape(seq, D) @ w["wte"].T
  loss = F.cross_entropy(logits, ids[1:])
  probabilities = torch.softmax(logits.detach(), -1)
  target_probabilities = probabilities[torch.arange(seq), ids[1:]]
  report = {"loss": float(loss.detach()), "accuracy": float((logits.detach().argmax(-1) == ids[1:]).float().mean()),
            "target_probability_mean": float(target_probabilities.mean()),
            "target_probability_min": float(target_probabilities.min()),
            "predicted_tokens": logits.detach().argmax(-1).tolist()}
  if not compute_gradients: return report
  loss.backward()
  selected = ["layer0/wq", "layer0/wfc", "layer11/wproj", "ln_f_g"]
  report["gradient_samples"] = {k: w[k].grad.numpy().copy() for k in selected}
  return report


def metrics(actual, reference):
  a, b = actual.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
  return {"relative_l2": float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)),
          "cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
          "max_abs": float(np.max(np.abs(a - b)))}


def main():
  cli = argparse.ArgumentParser(description=__doc__)
  cli.add_argument("backend", choices=["aneforge", "orion", "oracle"])
  cli.add_argument("--steps", type=int, default=10)
  cli.add_argument("--seq", type=int, default=32)
  cli.add_argument("--lr", type=float, default=1e-4)
  cli.add_argument("--output", type=Path)
  cli.add_argument("--no-checkpoint", action="store_true")
  args = cli.parse_args()
  os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
  source = Path.home() / "Desktop/Orion/model/blobs/gpt2_124m"
  verified = verify_weights(source, ROOT.parent)
  tokenizer = Tokenizer(ROOT.parent / "tokenizer")
  tokens = np.array(tokenizer.encode(TEXT)[:args.seq + 1], np.int64)
  if len(tokens) != args.seq + 1: raise ValueError("Text is too short")
  weights = load_weights(source)
  if args.backend == "oracle":
    ref = torch_oracle(weights, tokens)
    loss = ref.pop("loss")
    np.savez(ROOT / "oracle.npz", loss=loss, **ref["gradient_samples"])
    print(f"Full GPT-2 CPU fp32 oracle loss: {loss:.8f}", flush=True)
    return
  destination = args.output or ROOT / args.backend
  destination.mkdir(parents=True, exist_ok=True)
  os.environ["ANEFORGE_CACHE_DIR"] = str(destination / "cache")
  backend = Backend(args.backend, destination)
  model = GPT2(weights, Primitives(backend), tokens)
  optimizer = Adam(args.lr)
  metadata = {"backend": args.backend, "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
              "macos": platform.mac_ver()[0], "model": "GPT-2 124M", "layers": L,
              "parameter_count": sum(a.size for a in weights.values()), "sequence_length": args.seq,
              "batch_size": 1, "steps": args.steps, "optimizer": "CPU fp32 Adam", "lr": args.lr,
              "betas": [0.9, 0.999], "eps": 1e-8, "gradient_clip_norm": 1.0, "loss_scale": SCALE,
              "dropout": 0.0, "fixed_batch": True, "tokens": tokens.tolist(), "text": tokenizer.decode(tokens.tolist()),
              "verified_initial_weight_files": verified, "weight_manifest_sha256": hash_file(ROOT.parent / "model-checksums.json"),
              "initial_weights": str(source), "repo_commit": subprocess.check_output(["git", "-C", str(Path.home() / "Desktop" / ("ANEForge" if args.backend == "aneforge" else "Orion")), "rev-parse", "HEAD"], text=True).strip(),
              "scope": "External GPT-2 adapter; transformer forward/backward and parameter gradients on ANE; embedding lookup/scatter, vocabulary projection/loss and optimizer on CPU. Neither stock training CLI.",
              "phase_times": []}
  (destination / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
  try:
    start = time.perf_counter()
    loss, caches = model.forward()
    grads = model.backward(caches)
    warm_seconds = time.perf_counter() - start
    metadata["initial_loss"] = loss
    metadata["warmup_including_compile_seconds"] = warm_seconds
    metadata["compile_seconds"] = backend.compile_seconds
    np.savez(destination / "initial-gradients.npz", **{k: grads[k] for k in ("layer0/wq", "layer0/wfc", "layer11/wproj", "ln_f_g")})
    oracle_path = ROOT / "oracle.npz"
    if oracle_path.exists():
      with np.load(oracle_path) as oracle:
        metadata["oracle_loss"] = float(oracle["loss"])
        metadata["oracle_loss_abs_error"] = abs(loss - metadata["oracle_loss"])
        metadata["gradient_oracle"] = {k: metrics(grads[k], oracle[k]) for k in oracle.files if k != "loss"}
      if metadata["oracle_loss_abs_error"] > 0.1 or any(m["cosine"] < 0.98 for m in metadata["gradient_oracle"].values()):
        raise AssertionError("Full-model oracle check failed")
    del grads, caches
    print(f"Initial loss: {loss:.8f}; setup {warm_seconds:.2f}s, compiler {backend.compile_seconds:.2f}s", flush=True)
    with (destination / "loss.csv").open("w") as stream:
      columns = ["step", "loss", "forward_ms", "backward_ms", "optimizer_ms", "total_ms", "ane_execute_ms", "ane_dispatches", "gradient_norm"]
      writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
      writer.writeheader()
      for step in range(1, args.steps + 1):
        before_dispatch, before_seconds = backend.dispatches, backend.dispatch_seconds
        t0 = time.perf_counter()
        loss, caches = model.forward()
        t1 = time.perf_counter()
        grads = model.backward(caches)
        t2 = time.perf_counter()
        norm = optimizer.update(weights, grads)
        t3 = time.perf_counter()
        row = {"step": step, "loss": loss, "forward_ms": (t1 - t0) * 1000, "backward_ms": (t2 - t1) * 1000,
               "optimizer_ms": (t3 - t2) * 1000, "total_ms": (t3 - t0) * 1000,
               "ane_execute_ms": (backend.dispatch_seconds - before_seconds) * 1000,
               "ane_dispatches": backend.dispatches - before_dispatch, "gradient_norm": norm}
        writer.writerow(row)
        stream.flush()
        metadata["phase_times"].append(row)
        print(f"{args.backend} step {step:2d}: loss={loss:.6f}, {row['total_ms']:.1f} ms, ANE={row['ane_execute_ms']:.1f} ms", flush=True)
        del grads, caches
    final_loss, caches = model.forward()
    del caches
    metadata["final_loss_after_updates"] = final_loss
    metadata["loss_decreased"] = final_loss < metadata["initial_loss"]
    metadata["median_step_ms"] = float(np.median([row["total_ms"] for row in metadata["phase_times"]]))
    metadata["median_ane_execute_ms"] = float(np.median([row["ane_execute_ms"] for row in metadata["phase_times"]]))
    metadata["compiled_programs"] = len(backend.cache)
    metadata["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata["status"] = "passed" if metadata["loss_decreased"] else "loss_did_not_decrease"
    if not args.no_checkpoint:
      np.savez(destination / "checkpoint-step10.npz", **weights)
      metadata["checkpoint"] = "checkpoint-step10.npz"
    print(f"Final loss after {args.steps} updates: {final_loss:.8f}", flush=True)
  except BaseException as error:
    metadata["status"], metadata["error"] = "failed", repr(error)
    raise
  finally:
    metadata["total_dispatches"] = backend.dispatches
    (destination / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    backend.close()


if __name__ == "__main__": main()
