"""Package existing macOS BF16 oracle arrays outside Git with a small manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from qwen35.tools.compare_prefix_captures import comparison
from qwen35.weights import MODEL_SHA256, REVISION, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    if a.archive.exists() or a.manifest.exists():
        raise FileExistsError("choose new archive and manifest destinations")
    source = json.loads((a.capture / "report.json").read_text())
    if (source["status"] != "same_host_pass" or not source["same_host_oracle_exact"]
            or source["model_revision"] != REVISION or source["model_sha256"] != MODEL_SHA256):
        raise ValueError("requires validated pinned macOS oracle captures")
    checks = comparison(a.capture / "uzu-macos-all.npz", a.capture / "native-macos-all.npz")
    if not checks["same_arrays"] or len(checks["token_checks"]) != 23:
        raise ValueError("macOS all-prefix oracle validation failed")
    names = ("uzu-macos-all.npz", "native-macos-all.npz", "report.json")
    files = {n:dict(bytes=(a.capture / n).stat().st_size, sha256=sha256(a.capture / n)) for n in names}
    a.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.archive, "w:gz") as archive:
        for n in names:
            archive.add(a.capture / n, arcname=n)
    with tarfile.open(a.archive, "r:gz") as archive:
        for n, entry in files.items():
            with archive.extractfile(n) as f:
                if hashlib.sha256(f.read()).hexdigest() != entry["sha256"]:
                    raise ValueError("packaged oracle checksum mismatch")
    manifest = dict(status="prepared", model_revision=REVISION, model_sha256=MODEL_SHA256,
                    uzu_revision=source["uzu_revision"], oracle_sha256=source["oracle_sha256"],
                    prefixes=23, layers=24, vocabulary=248320, same_host_bitwise_equal=True, files=files,
                    archive=dict(path=str(a.archive), bytes=a.archive.stat().st_size, sha256=sha256(a.archive)),
                    cross_host_comparison="pending new Asahi all-prefix NPZ arrays; no transfer destination provided",
                    command="python -m qwen35.tools.compare_prefix_captures uzu-macos-all.npz uzu-asahi-all.npz --output cross-host.json")
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
