"""Independent CPU checks of train/unseen loss before and after the ten updates."""
import json
from pathlib import Path
import numpy as np
from train import load_weights, torch_oracle, Tokenizer

ROOT = Path(__file__).resolve().parent
UNSEEN = ("In a quiet village near the mountains, a young musician practiced the violin every morning. "
          "One winter evening, a visitor arrived carrying an old wooden box and a letter from a distant friend.")


def main():
  protocol = json.loads((ROOT / "aneforge/results.json").read_text())
  tokenizer = Tokenizer(ROOT.parent / "tokenizer")
  batches = {"train": np.array(protocol["tokens"], np.int64),
             "unseen": np.array(tokenizer.encode(UNSEEN)[:33], np.int64)}
  report = {"method": "Independent PyTorch CPU fp32 forward, no ANE calls, no training. All checkpoint tensors rounded to fp16 before CPU evaluation to match ANE weight inputs.",
            "unseen_text": tokenizer.decode(batches["unseen"].tolist()), "unseen_tokens": batches["unseen"].tolist(), "results": {}}
  weights = load_weights(Path(protocol["initial_weights"]))
  report["results"]["initial"] = {name: torch_oracle(weights, tokens, False) for name, tokens in batches.items()}
  print(json.dumps(report["results"]["initial"], indent=2), flush=True)
  del weights
  for backend in ("aneforge", "orion"):
    with np.load(ROOT / backend / "checkpoint-step10.npz") as checkpoint:
      weights = {k: checkpoint[k].astype(np.float16).astype(np.float32) for k in checkpoint.files}
    report["results"][backend] = {name: torch_oracle(weights, tokens, False) for name, tokens in batches.items()}
    print(backend, json.dumps(report["results"][backend], indent=2), flush=True)
    del weights
  (ROOT / "overfitting-check.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__": main()
