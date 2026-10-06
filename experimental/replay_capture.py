"""Prepare and verify captured H13G fixtures with the existing guarded Asahi kernel loader."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import plistlib
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GATE = dict(relative_l2=.005, allclose_rtol=.01, allclose_atol=.03)


def validate_manifest(manifest, kit):
    """Require real, complete comparisons and intact payloads before device access."""
    records = manifest.get("records")
    if manifest.get("unsupported") or not isinstance(records, list) or not records:
        raise ValueError("capture does not satisfy replay guards")
    if manifest.get("gate") != GATE:
        raise ValueError("capture must use the unchanged replay gates")
    names, outputs, groups, checked = set(), set(), {}, set()
    comparisons = 0
    for record in records:
        if record["name"] in names:
            raise ValueError("duplicate capture record")
        names.add(record["name"])
        identity = (record["hwx"], record["output_port_name"])
        if identity in outputs:
            raise ValueError("duplicate output port")
        outputs.add(identity)
        if len(record["inputs"]) != len(record["input_port_names"]):
            raise ValueError("input port count mismatch")
        for shape in [*record["inputs"], record["output"]]:
            if len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError("invalid physical port shape")
        fixtures = record.get("fixtures")
        if not isinstance(fixtures, list) or not fixtures:
            raise ValueError("every output requires at least one fixture")
        paths = [fixture["path"] for fixture in fixtures]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate output fixture")
        group = (record["inputs"], record["input_port_names"], set(paths))
        if record["hwx"] in groups and group != groups[record["hwx"]]:
            raise ValueError("inconsistent shared program fixtures or inputs")
        groups[record["hwx"]] = group
        for name, expected in [(record["hwx"], record["hwx_sha256"]),
                               *[(f["path"], f["sha256"]) for f in fixtures]]:
            if (name, expected) not in checked:
                if hashlib.sha256((kit / name).read_bytes()).hexdigest() != expected:
                    raise ValueError("capture checksum mismatch: " + name)
                checked.add((name, expected))
        for fixture in fixtures:
            with np.load(kit / fixture["path"]) as data:
                for i, shape in enumerate(record["inputs"]):
                    array = data[f"input{i:02d}"]
                    if array.dtype != np.float16 or array.size != math.prod(shape) or not np.isfinite(array).all():
                        raise ValueError("fixture inputs must match finite FP16 ports")
                array = data[record["output_key"]]
                if array.dtype.kind != "f" or array.size != math.prod(record["output"]) or not np.isfinite(array).all():
                    raise ValueError("fixture reference must match finite output ports")
        comparisons += len(fixtures)
    return comparisons


def prepare(kit, kind):
    records, unavailable = [], []
    report_path = kit / ("capture-report.json" if kind == "training" else "report.json")
    source_report = json.loads(report_path.read_text()) if report_path.exists() else {}
    reference_kind = source_report.get("reference_kind", "macOS ANE FP16 outputs")
    if kind == "training":
        index = json.loads((kit / "kernel-index.json").read_text())
        candidates = [(r["name"], kit / Path(r["mil"]).parent / "hwx", r["input_port_names"],
                       [(r["output_port_name"], "output")], [kit / Path(r["mil"]).parent / "fixture.npz"]) for r in index]
    elif kind == "qwen":
        index = json.loads((kit / "report.json").read_text())["records"]
        candidates = [(f"heads-{r['head_start']}", kit / f"heads-{r['head_start']}-{r['head_start'] + 3}" / "hwx",
                       [p["name"] for p in r.get("input_ports", [])],
                       [(p["name"], f"output{i:02d}") for i, p in enumerate(r.get("output_ports", []))],
                       [kit / f"heads-{r['head_start']}-{r['head_start'] + 3}" / c["fixture"] for c in r["cases"]]) for r in index if r.get("export")]
        unavailable.extend(dict(name=f"heads-{r['head_start']}", reason=r.get("error", r["status"])) for r in index if not r.get("export"))
    else:
        index = json.loads((kit / "report.json").read_text())
        candidates = [("whisper-encoder", kit / "hwx", [p["name"] for p in index["ports"]["inputs"]],
                       [(p["name"], "output") for p in index["ports"]["outputs"]],
                       [kit / c["fixture"] for c in index["fixtures"]])]
    for name, directory, input_names, output_names, fixtures in candidates:
        try:
            from gpt2.hwx import parse_container, parse_tasks, relocate
            receipt = json.loads((directory / "receipt.json").read_text())
            if receipt["status"] != "exported": raise ValueError(receipt["status"])
            data = (directory / "model.hwx").read_bytes()
            if hashlib.sha256(data).hexdigest() != receipt["hwx_sha256"]: raise ValueError("export checksum mismatch")
            container = parse_container(data)
            bars, thread = container["thread"]["bars"], container["thread"]
            text = next(s for s in container["segments"] if s["name"] == "__TEXT")
            const = next(s for s in container["sections"] if s["segment"] == "__TEXT" and s["name"] == "__const")
            if not (bars[0] == thread["entry"] == text["vmaddr"] and bars[1] == const["addr"] and bars[2] == 0):
                raise ValueError("existing loader rejects command/constant BAR layout")
            if text["filesize"] % 16: raise ValueError("unaligned command boundary")
            coefficients = [s for s in container["segments"] if s["name"].startswith("__KERN")]
            if len(coefficients) > 1: raise ValueError("existing replay loader supports one coefficient bank")
            weight_bank = bars.index(coefficients[0]["vmaddr"]) if coefficients else None
            if weight_bank is not None and weight_bank < 4: raise ValueError("unsupported coefficient BAR")
            banks = {i:i for i,addr in enumerate(bars) if addr}; banks[1] = 2
            if weight_bank is not None: banks[weight_bank] = 1
            raw = data[text["fileoff"]:text["fileoff"] + text["filesize"]]
            tasks = parse_tasks(raw, thread["td_size"], thread["td_count"])
            if len(tasks) != receipt["task_count"]: raise ValueError("export task count mismatch")
            relocate(raw, tasks, banks)
            for bank,addr in enumerate(bars):
                if addr and bank not in (0, 1, weight_bank):
                    segment = next(s for s in container["segments"] if s["vmaddr"] == addr)
                    if segment["filesize"]: raise ValueError("initialized non-coefficient buffer")
            status = plistlib.loads((directory / "model.hwx.status.plist").read_bytes())
            if status["ErrorList"]: raise ValueError("compiler errors")
            network, = status["NetworkStatusList"]
            ports, roles = {}, {}
            for role, key in (("input", "LiveInputList"), ("output", "LiveOutputList")):
                for port in network[key]:
                    symbol = port["Symbol"].removesuffix("@output")
                    shape = tuple(port[k] for k in ("Batches", "Channels", "Height", "Width"))
                    if not (port["Type"] == "Float16" and port["Depth"] == 1 and port["Interleave"] == 1
                            and port["RowStride"] == shape[-1] * 2 and port["PlaneStride"] == shape[-2] * port["RowStride"]
                            and port["BatchStride"] == shape[1] * port["PlaneStride"]):
                        raise ValueError(f"existing replay loader rejects padded/non-FP16 I/O: {symbol}")
                    ports[symbol] = shape
                    roles[symbol] = role
                    bank = bars.index(container["symbols"][symbol]["addr"])
                    if bank < 3 or bank == weight_bank: raise ValueError("unsupported I/O BAR")
                    segment = next(s for s in container["segments"] if s["vmaddr"] == bars[bank])
                    if int(np.prod(shape)) * 2 > segment["vmsize"]: raise ValueError("I/O exceeds declared buffer")
            for output_name, output_key in output_names:
                if roles.get(output_name) != "output" or any(name is not None and roles.get(name) != "input" for name in input_names):
                    raise ValueError("fixture port role mismatch")
                with np.load(fixtures[0]) as fixture:
                    shapes = [ports[n] if n else tuple(fixture[f"input{i:02d}"].shape) for i, n in enumerate(input_names)]
                row = dict(name=name + "/" + output_name, hwx=str((directory / "model.hwx").relative_to(kit)),
                           reference_kind=reference_kind,
                           hwx_sha256=receipt["hwx_sha256"], task_count=receipt["task_count"],
                           input_port_names=input_names, inputs=shapes, output_port_name=output_name,
                           output=ports[output_name], output_key=output_key, fixtures=[])
                for path in fixtures:
                    with np.load(path) as fixture:
                        if any(fixture[f"input{i:02d}"].size != int(np.prod(shape)) for i, shape in enumerate(shapes)):
                            raise ValueError("logical fixture input count differs from physical metadata")
                        if fixture[output_key].size != int(np.prod(row["output"])): raise ValueError("fixture output size differs")
                        if any(fixture[f"input{i:02d}"].dtype != np.dtype("float16") or not np.isfinite(fixture[f"input{i:02d}"]).all() for i in range(len(shapes))):
                            raise ValueError("fixture inputs must be finite FP16")
                        if fixture[output_key].dtype.kind != "f" or not np.isfinite(fixture[output_key]).all():
                            raise ValueError("fixture reference must be finite floating-point values")
                    row["fixtures"].append(dict(path=str(path.relative_to(kit)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                records.append(row)
        except Exception as error:
            unavailable.append(dict(name=name, reason=str(error)))
    report = dict(kind=kind, records=records, unsupported=unavailable, hardware_execution="pending native Asahi",
                  gate=GATE.copy(),
                  reference_kind=reference_kind,
                  scope="Replay against the explicitly identified captured reference; model-level Linux validation remains separate.")
    (kit / "asahi-fixtures.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(kind=kind, replay_outputs=len(records), fixtures=sum(len(r["fixtures"]) for r in records), unsupported=unavailable)), flush=True)


def verify(kit, device, output):
    if platform.system() != "Linux": raise SystemExit("requires native Asahi Linux; no hardware submission attempted")
    manifest = json.loads((kit / "asahi-fixtures.json").read_text())
    expected_comparisons = validate_manifest(manifest, kit)
    sys.path.insert(0, str(ROOT / "gpt2/training"))
    import asahi
    from replay import device_path
    asahi.ROOT = kit
    path = device_path(device)
    backend = SimpleNamespace(fd=os.open(path, os.O_RDWR | os.O_CLOEXEC), dispatch_seconds=0., dispatches=0)
    report = dict(device=str(path), kernel=platform.release(), records=[], status="running")
    try:
        for record in manifest["records"]:
            kernel = asahi.Kernel(backend, record)
            try:
                for fixture in record["fixtures"]:
                    source = kit / fixture["path"]
                    if hashlib.sha256(source.read_bytes()).hexdigest() != fixture["sha256"]: raise ValueError("fixture checksum mismatch")
                    with np.load(source) as data:
                        arrays = [data[f"input{i:02d}"].reshape(shape) for i, shape in enumerate(record["inputs"])]
                        expected = data[record["output_key"]].reshape(record["output"]).astype(np.float32)
                        actual = kernel(*arrays)
                    relative = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-40))
                    passed = relative < .005 and bool(np.allclose(actual, expected, rtol=.01, atol=.03))
                    row = dict(name=record["name"], fixture=fixture["path"], relative_l2=relative, pass_gate=passed)
                    report["records"].append(row); print(json.dumps(row), flush=True)
                    if not passed: raise ValueError("captured-output gate failed")
            finally:
                kernel.close()
        if len(report["records"]) != expected_comparisons:
            raise ValueError("incomplete fixture execution")
        report.update(status="pass", dispatches=backend.dispatches, dispatch_seconds=backend.dispatch_seconds)
    except Exception as error:
        report.update(status="failed", error=str(error)); raise
    finally:
        os.close(backend.fd)
        output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "verify"])
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--kind", choices=["training", "qwen", "whisper"])
    p.add_argument("--device")
    p.add_argument("--output", type=Path, default=Path("asahi-fixture-result.json"))
    a = p.parse_args()
    if a.command == "prepare":
        if not a.kind: p.error("prepare requires --kind")
        prepare(a.kit.resolve(), a.kind)
    else: verify(a.kit.resolve(), a.device, a.output)


if __name__ == "__main__":
    main()
