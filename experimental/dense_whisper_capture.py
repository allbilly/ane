"""Wrap captured Whisper inputs in dense ports using shape-only MIL reshapes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

from capture_macos_program import export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((a.capture / "report.json").read_text())
    source = a.capture / "bundle"
    bundle = a.output / "bundle"
    bundle.mkdir()
    mil = (source / "model.mil").read_text()
    prefix = []
    ports = []
    for port in report["ports"]["inputs"]:
        name, shape = port["name"], tuple(port["shape"])
        count = int(np.prod(shape))
        if count % 128: raise ValueError("dense reshape requires a width-128 factor")
        dense = (1, 1, count // 128, 128)
        symbol = "capture_dense_" + name
        original_type = "tensor<fp16,[" + ",".join(map(str, shape)) + "]>"
        dense_type = "tensor<fp16,[" + ",".join(map(str, dense)) + "]>"
        dimensions = r"\s*,\s*".join(map(str, shape))
        signature = r"tensor<fp16,\s*\[" + dimensions + r"\]>\s+" + re.escape(name) + r"(?=\s*[,\)])"
        if len(re.findall(signature, mil)) != 1: raise ValueError("unexpected original input signature: " + name)
        mil = re.sub(signature, dense_type + " " + symbol, mil, count=1)
        prefix.append(f'        tensor<int32,[4]> capture_shape_{name} = const()[name=string("capture_shape_{name}"), val=tensor<int32,[4]>([{",".join(map(str, shape))}])];')
        prefix.append(f'        {original_type} {name} = reshape(shape=capture_shape_{name}, x={symbol})[name=string("{name}")];')
        ports.append(dict(name=symbol, shape=dense, original_name=name, original_shape=shape))
    # Only the function signature and two explicit reshape nodes change.
    match = re.search(r"func main<[^>]+>\([^\n]*\) \{\n", mil)
    if not match: raise ValueError("unexpected MIL main function header")
    mil = mil[:match.end()] + "\n".join(prefix) + "\n" + mil[match.end():]
    (bundle / "model.mil").write_text(mil)
    shutil.copy2(source / "weights.bin", bundle / "weights.bin")
    shutil.copy2(source / "pos.f16", bundle / "pos.f16")
    port_lines = [f"{port['name']} {int(np.prod(port['shape']))}" for port in ports]
    port_lines.extend(f"{port['name']} {int(np.prod(port['shape']))}" for port in report["ports"]["outputs"])
    (bundle / "ports.txt").write_text("\n".join(port_lines) + "\n")
    for row in report["fixtures"]:
        with np.load(a.capture / row["fixture"]) as f:
            arrays = {name:f[name].copy() for name in f.files}
        for i, port in enumerate(ports):
            before = arrays[f"input{i:02d}"]
            after = before.reshape(port["shape"])
            if before.tobytes() != after.tobytes(): raise RuntimeError("reshape changed input bytes")
            arrays[f"input{i:02d}"] = after
        np.savez_compressed(a.output / row["fixture"], **arrays)
        shutil.copy2(a.capture / (row["name"] + ".wav"), a.output / (row["name"] + ".wav"))
    report.update(ports=dict(inputs=ports, outputs=report["ports"]["outputs"]), transcriptions=[],
                  source_capture=str(a.capture), source_mil_sha256=hashlib.sha256((source / "model.mil").read_bytes()).hexdigest(),
                  layout_note="Two width-128 contiguous input ports are reshaped to the original tensors; all logical input bytes and trained operations remain unchanged.",
                  hardware_execution="not attempted; new port wrapper requires runtime validation", status="offline_export_pending")
    report["export"] = export(bundle, a.output / "hwx", Path(__file__).resolve().parents[1] / "gpt2/training/build/dump_hwx")
    report["status"] = "offline_exported" if report["export"]["status"] == "exported" else "export_failed"
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "offline_exported": raise SystemExit(1)


if __name__ == "__main__": main()
