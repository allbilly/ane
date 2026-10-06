"""Execute a prepared dense capture on macOS ANE and retain every output comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experimental.replay_capture import validate_manifest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(kit, output):
    if platform.system() != "Darwin":
        raise SystemExit("requires macOS; no hardware submission attempted")
    manifest = json.loads((kit / "asahi-fixtures.json").read_text())
    expected_comparisons = validate_manifest(manifest, kit)
    output.mkdir(parents=True, exist_ok=False)
    from aneforge._runtime import E5RT, _find_dylib
    library = _find_dylib()
    report = dict(status="running", kit=str(kit), device_mask=4,
                  runtime=str(library), runtime_sha256=digest(library),
                  reference_kind=manifest["reference_kind"], gate=manifest["gate"],
                  programs=[], comparisons=[], dispatches=0,
                  scope="ANE-only execution of the captured MIL; Linux HWX replay remains separate")
    groups = {}
    for record in manifest["records"]:
        groups.setdefault(record["hwx"], []).append(record)
    for group_index, (hwx_name, records) in enumerate(groups.items()):
        program = None
        row = dict(hwx=hwx_name, status="running")
        report["programs"].append(row)
        try:
            hwx = kit / hwx_name
            receipt = json.loads((hwx.parent / "receipt.json").read_text())
            source = hwx.parent.parent
            if not (source / "model.mil").exists() and (source / "bundle/model.mil").exists():
                source = source / "bundle"
            if digest(hwx) != receipt["hwx_sha256"] or any(r["hwx_sha256"] != receipt["hwx_sha256"] for r in records):
                raise ValueError("HWX checksum mismatch")
            if digest(source / "model.mil") != receipt["mil_sha256"]:
                raise ValueError("MIL checksum mismatch")
            for name, expected in receipt["source_weight_blobs"].items():
                if digest(source / name) != expected:
                    raise ValueError("MIL weight checksum mismatch: " + name)
            first = records[0]
            if any(r["inputs"] != first["inputs"] or r["input_port_names"] != first["input_port_names"] for r in records):
                raise ValueError("inconsistent shared input metadata")
            inputs = {name:tuple(shape) for name, shape in zip(first["input_port_names"], first["inputs"]) if name}
            outputs = {r["output_port_name"]:tuple(r["output"]) for r in records}
            program = E5RT.compile(source / "model.mil", cache_dir=output / f"cache-{group_index}",
                                   inputs=inputs, outputs=outputs, device_mask=4)
            fixtures = {}
            for record in records:
                for fixture in record["fixtures"]:
                    fixtures.setdefault(fixture["path"], []).append((record, fixture))
            for fixture_index, (fixture_name, checks) in enumerate(fixtures.items()):
                fixture = kit / fixture_name
                if any(digest(fixture) != f["sha256"] for _, f in checks):
                    raise ValueError("fixture checksum mismatch")
                with np.load(fixture) as data:
                    arrays = {key:data[key].copy() for key in data.files}
                for i, (name, shape) in enumerate(zip(first["input_port_names"], first["inputs"])):
                    if name:
                        array = arrays[f"input{i:02d}"].reshape(shape)
                        if array.dtype != np.float16 or not np.isfinite(array).all():
                            raise ValueError("inputs must be finite FP16")
                        program.set_input(name, array)
                started = time.perf_counter()
                program.execute()
                elapsed = time.perf_counter() - started
                report["dispatches"] += 1
                actual_outputs = {name:program.read_output(name) for name in outputs}
                captured = {}
                for record, _ in checks:
                    actual = actual_outputs[record["output_port_name"]]
                    expected = arrays[record["output_key"]].reshape(actual.shape).astype(np.float32)
                    observed = actual.astype(np.float32)
                    finite = bool(np.isfinite(observed).all() and np.isfinite(expected).all())
                    relative = float(np.linalg.norm(observed - expected) / max(np.linalg.norm(expected), 1e-40)) if finite else None
                    close = bool(np.allclose(observed, expected, rtol=manifest["gate"]["allclose_rtol"],
                                            atol=manifest["gate"]["allclose_atol"])) if finite else False
                    passed = finite and relative < manifest["gate"]["relative_l2"] and close
                    comparison = dict(name=record["name"], fixture=fixture_name, finite=finite,
                                      relative_l2=relative, allclose=close, pass_gate=passed, execute_seconds=elapsed,
                                      bitwise_equal=actual.tobytes() == arrays[record["output_key"]].reshape(actual.shape).tobytes()
                                      if arrays[record["output_key"]].dtype == np.float16 else None)
                    if "hf_output" in arrays:
                        hf = arrays["hf_output"].astype(np.float64).ravel()
                        flat = observed.astype(np.float64).ravel()
                        cosine = float(flat @ hf / max(np.linalg.norm(flat) * np.linalg.norm(hf), 1e-40)) if finite and np.isfinite(hf).all() else None
                        comparison.update(hf_cosine=cosine, passes_encoder_cosine=cosine is not None and cosine > .999)
                        comparison["pass_gate"] = passed and comparison["passes_encoder_cosine"]
                        captured["hf_output"] = arrays["hf_output"]
                    report["comparisons"].append(comparison)
                    print(json.dumps(comparison), flush=True)
                    captured[record["output_key"]] = actual
                    captured["reference_" + record["output_key"]] = expected
                captured.update({key:value for key, value in arrays.items() if key.startswith("input")})
                path = output / f"program-{group_index}-fixture-{fixture_index}.npz"
                np.savez_compressed(path, **captured)
                row.setdefault("captured_fixtures", []).append(dict(path=path.name, sha256=digest(path),
                                                                    source=fixture_name, reference_kind="macOS ANE FP16 outputs"))
            row["status"] = "executed"
        except Exception as error:
            row.update(status="failed", error=str(error))
        finally:
            if program:
                program.release()
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "pass" if len(report["comparisons"]) == expected_comparisons and all(r["status"] == "executed" for r in report["programs"]) and all(c["pass_gate"] for c in report["comparisons"]) else "failed"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = verify(a.kit.resolve(), a.output.resolve())
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
