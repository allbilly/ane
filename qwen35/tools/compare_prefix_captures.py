"""Locate the first differing prefix/layer in two existing BF16 oracle captures."""
import argparse
import json
from pathlib import Path
import numpy as np
from qwen35.tools.capture_vendor_macos import compare
from qwen35.weights import sha256


def comparison(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        tokens = a["tokens"]
        if (tokens.ndim != 1 or not len(tokens) or not np.array_equal(tokens, b["tokens"])
                or a["logits"].shape != (len(tokens), 248320) or b["logits"].shape != a["logits"].shape
                or a["layers"].shape != (len(tokens), 24, 1024) or b["layers"].shape != a["layers"].shape):
            raise ValueError("oracle token histories or array shapes differ")
        rows = [dict(prefix_tokens=i + 1, token=int(token), logits=compare(a["logits"][i], b["logits"][i]),
                     layers=[compare(a["layers"][i, j], b["layers"][i, j]) for j in range(24)])
                for i, token in enumerate(tokens)]
    first = next((r for r in rows if not r["logits"]["exact"] or any(not c["exact"] for c in r["layers"])), None)
    return dict(left=dict(path=str(left), sha256=sha256(left)), right=dict(path=str(right), sha256=sha256(right)),
                first_unequal_prefix=first["prefix_tokens"] if first else None,
                first_unequal_layer=next((i for i, c in enumerate(first["layers"]) if not c["exact"]), None) if first else None,
                same_arrays=first is None, token_checks=rows,
                scope="Array comparison; verify checkpoint and runtime identities in the accompanying host reports.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = comparison(a.left, a.right)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:report[k] for k in ("same_arrays", "first_unequal_prefix", "first_unequal_layer")}))


if __name__ == "__main__":
    main()
