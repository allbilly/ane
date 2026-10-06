"""Capture the pinned CPU oracle at every token and locate BF16 divergence."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

from qwen35.model import Model
from qwen35.weights import REVISION, sha256


def compare(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("invalid reference array")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return dict(exact=bool(np.array_equal(actual, expected)),
                unequal=int(np.count_nonzero(actual != expected)),
                max_abs=float(np.abs(delta).max()),
                normalized_rmse=float(np.linalg.norm(delta) / max(np.linalg.norm(expected.astype(np.float64)), 1e-40)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tag", default="macos" if platform.system() == "Darwin" else "asahi")
    p.add_argument("--compare-capture", type=Path, help="Independent all-token oracle NPZ from the other host")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    provenance = Path(__file__).resolve().parents[1] / "provenance"
    manifest = json.loads((provenance / "vendor-validation.json").read_text())
    report = dict(model_revision=REVISION, uzu_revision=manifest["uzu_revision"],
                  model_sha256=sha256(a.model / "model.safetensors"), oracle_sha256=sha256(a.oracle),
                  host=platform.platform(), capture_tag=a.tag,
                  commands=[], fixtures=[], token_checks=[])
    for item in manifest["checks"]:
        target = a.output / (item["label"] + ".bin")
        cmd = [str(a.oracle), str(a.model), ",".join(map(str, item["tokens"])), str(target)]
        with (a.output / (item["label"] + ".log")).open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        report["commands"].append(dict(argv=cmd, all_tokens=False))
        with np.load(provenance / item["fixture"]) as old:
            row = dict(label=item["label"], saved_asahi_logits=compare(np.fromfile(target, "<f4"), old["logits"]))
            if "layers" in old:
                row["saved_asahi_layers"] = compare(np.fromfile(target.with_suffix(".layers.bin"), "<f4").reshape(24, 1024), old["layers"])
        report["fixtures"].append(row)
        print(json.dumps(row), flush=True)
    item = next(item for item in manifest["checks"] if item["label"] == "arithmetic")
    target = a.output / "arithmetic-all.bin"
    cmd = [str(a.oracle), str(a.model), ",".join(map(str, item["tokens"])), str(target)]
    env = dict(os.environ, ANE_REFERENCE_ALL_TOKENS="1")
    with (a.output / "arithmetic-all.log").open("w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report["commands"].append(dict(argv=cmd, all_tokens=True))
    expected = np.fromfile(target, "<f4").reshape(len(item["tokens"]), 248320)
    layers = np.fromfile(target.with_suffix(".layers.bin"), "<f4").reshape(len(item["tokens"]), 24, 1024)
    last = np.fromfile(a.output / "arithmetic.bin", "<f4")
    report["instrumentation_preserves_final_logits"] = compare(expected[-1], last)
    if not report["instrumentation_preserves_final_logits"]["exact"]:
        raise RuntimeError("all-token instrumentation changed final logits")
    model = Model(a.model, precision="bf16", kernels="native", threads=a.threads, context=128)
    actual_logits, actual_layers = [], []
    for pos, token in enumerate(item["tokens"]):
        trace = []
        actual = model.step(token, trace=trace)
        actual_logits.append(actual)
        actual_layers.append(trace)
        row = dict(prefix_tokens=pos + 1, token=token, logits=compare(actual, expected[pos]),
                   argmax_matches=int(actual.argmax()) == int(expected[pos].argmax()),
                   layers=[compare(np.asarray(trace)[i], layers[pos, i]) for i in range(24)])
        report["token_checks"].append(row)
        first = next((i for i, check in enumerate(row["layers"]) if not check["exact"]), None)
        print(json.dumps(dict(prefix=pos + 1, logits_exact=row["logits"]["exact"], first_unequal_layer=first)), flush=True)
    report["first_unequal_prefix"] = next((row["prefix_tokens"] for row in report["token_checks"] if
                                           not row["logits"]["exact"] or any(not c["exact"] for c in row["layers"])), None)
    report["same_host_oracle_exact"] = report["first_unequal_prefix"] is None
    report["status"] = "same_host_pass" if report["same_host_oracle_exact"] else "same_host_fail"
    if a.tag == "macos": report["macos_oracle_exact"] = report["same_host_oracle_exact"]
    np.savez_compressed(a.output / f"uzu-{a.tag}-all.npz", tokens=item["tokens"], logits=expected, layers=layers)
    np.savez_compressed(a.output / f"native-{a.tag}-all.npz", tokens=item["tokens"], logits=actual_logits, layers=actual_layers)
    if a.compare_capture:
        with np.load(a.compare_capture) as other:
            if other["tokens"].tolist() != item["tokens"]: raise ValueError("cross-host token histories differ")
            checks = [dict(prefix_tokens=pos + 1, logits=compare(expected[pos], other["logits"][pos]),
                           layers=[compare(layers[pos, i], other["layers"][pos, i]) for i in range(24)])
                      for pos in range(len(item["tokens"]))]
        first = next((row for row in checks if not row["logits"]["exact"] or any(not c["exact"] for c in row["layers"])), None)
        report["cross_host"] = dict(capture=str(a.compare_capture), sha256=sha256(a.compare_capture),
                                   first_unequal_prefix=first["prefix_tokens"] if first else None,
                                   first_unequal_layer=next((i for i,c in enumerate(first["layers"]) if not c["exact"]), None) if first else None,
                                   token_checks=checks)
    # Keep compact captures; the NPZ arrays contain these same raw floats.
    for path in a.output.glob("*.bin"):
        report.setdefault("raw_export_sha256", {})[path.name] = sha256(path)
        path.unlink()
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("first_unequal_prefix", "same_host_oracle_exact")}), flush=True)
    if not report["same_host_oracle_exact"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
