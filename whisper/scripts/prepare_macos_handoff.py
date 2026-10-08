"""Verify and copy existing Mac fixtures/oracles without regenerating captures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from whisper.validation import digest

ROOT = Path(__file__).resolve().parents[2]


def prepare(fixtures, oracle, output):
    reference = json.loads((ROOT / "whisper/results/fast-recapture-20261007/encoder-reference.json").read_text())
    rows = []
    for row in reference["records"]:
        path = fixtures / (row["audio"] + ".npz")
        with np.load(path, allow_pickle=False) as capture:
            arrays = {}
            for name, shape, expected in (
                ("input00", (1, 80, 1, 3000), row["mel_sha256"]),
                ("input01", (1, 384, 1, 1500), reference["position_sha256"]),
                ("output", (1500, 384), row["output_sha256"]),
            ):
                array = capture[name]
                if array.shape != shape or array.dtype != np.dtype("<f2") or not np.isfinite(array).all():
                    raise ValueError(f"invalid captured array: {path}:{name}")
                actual = hashlib.sha256(array.tobytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"captured byte hash mismatch: {path}:{name}")
                arrays[name] = dict(shape=list(array.shape), strides=list(array.strides),
                    dtype=array.dtype.str, sha256=actual)
        rows.append(dict(audio=row["audio"], source=str(path), file=path.name,
            bytes=path.stat().st_size, sha256=digest(path), arrays=arrays))

    from qwen35.tools.compare_prefix_captures import comparison
    from qwen35.weights import MODEL_SHA256, REVISION
    oracle_report = json.loads((oracle / "report.json").read_text())
    if (oracle_report.get("status") != "same_host_pass" or not oracle_report.get("same_host_oracle_exact")
            or oracle_report["model_revision"] != REVISION or oracle_report["model_sha256"] != MODEL_SHA256):
        raise ValueError("oracle report does not establish the pinned same-host comparison")
    check = comparison(oracle / "uzu-macos-all.npz", oracle / "native-macos-all.npz")
    if not check["same_arrays"] or len(check["token_checks"]) != 23:
        raise ValueError("all 23 oracle prefixes must match on Mac")
    packaged = json.loads((ROOT / "qwen35/provenance/macos-bf16-oracle-package-20261007.json").read_text())
    oracle_files = []
    for name, expected in packaged["files"].items():
        path = oracle / name
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("existing oracle package hash mismatch: " + str(path))
        oracle_files.append(dict(source=str(path), file=name, **expected))

    # Validate all sources before creating a handoff. Copies retain NPZ bytes.
    output.mkdir(parents=True, exist_ok=False)
    for directory, files in (("whisper-fixtures", rows), ("qwen-oracles", oracle_files)):
        target = output / directory
        target.mkdir()
        for row in files:
            destination = target / row["file"]
            shutil.copy2(row["source"], destination)
            if digest(destination) != row["sha256"]:
                raise ValueError("handoff copy changed bytes: " + str(destination))
    report = dict(status="verified_existing_captures", whisper=dict(
        checkpoint_sha256=reference["checkpoint_sha256"], fixtures=rows,
        packing="NPZ arrays contain contiguous logical FP16 values. Native mel channels use 6016-byte strides; position channels use 3008-byte strides; output rows use 768-byte strides. The checkpoint repacker supplies zero padding.",
        replay_command="python -m whisper.scripts.benchmark_encoder --hf-model whisper/models/hf-tiny.en --backend asahi --fixtures HANDOFF/whisper-fixtures --compare-baseline --output whisper/build/exact-mac-fixture-replay.json"),
        qwen=dict(model_revision=REVISION, model_sha256=MODEL_SHA256,
            uzu_revision=oracle_report["uzu_revision"], oracle_sha256=oracle_report["oracle_sha256"],
            prefixes=23, layers=24, vocabulary=248320, same_host_bitwise_equal=True, files=oracle_files,
            comparison_command="python -m qwen35.tools.compare_prefix_captures HANDOFF/qwen-oracles/uzu-macos-all.npz qwen35/local-results/asahi-todo-20261007/vendor-all-prefix/uzu-asahi-all.npz --output qwen35/local-results/cross-host.json"),
        cross_host_verification="pending Asahi hardware and its all-prefix arrays; not accessible from this Mac")
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.fixtures, args.oracle, args.output)
    print("Verified handoff:", args.output / "manifest.json")


if __name__ == "__main__":
    main()
