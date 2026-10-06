"""Export an exact MIL program and retain its H13G structure for Linux replay."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import plistlib
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_spec = importlib.util.spec_from_file_location("capture_hwx_reader", Path(__file__).resolve().parents[1] / "gpt2/hwx.py")
_reader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reader)
parse_container, parse_tasks = _reader.parse_container, _reader.parse_tasks


def export(source, output, compiler):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [str(Path(compiler).resolve()), str(source), str(output)]
    with (output / "compiler.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    receipt = dict(command=command, returncode=result.returncode,
                   mil_sha256=hashlib.sha256((source / "model.mil").read_bytes()).hexdigest(),
                   source_weight_blobs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.bin")},
                   export_method="ANECCompile h13g offline export; runtime fixtures are captured separately",
                   linux_hardware_replay="pending; macOS export is not Linux execution")
    path = output / "model.hwx"
    if result.returncode or not path.exists():
        receipt["status"] = "compile_failed"
    else:
        data = path.read_bytes()
        receipt.update(hwx_sha256=hashlib.sha256(data).hexdigest(), hwx_bytes=len(data))
        try:
            container = parse_container(data)
            segment = next(s for s in container["segments"] if s["name"] == "__TEXT")
            text = data[segment["fileoff"]:segment["fileoff"] + segment["filesize"]]
            thread = container["thread"]
            tasks = parse_tasks(text, thread["td_size"], thread["td_count"])
            for task in tasks:
                task["registers"] = {f"0x{k:05x}": v for k, v in task["registers"].items()}
            (output / "container.json").write_text(json.dumps(container, indent=2) + "\n")
            (output / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
            receipt.update(task_count=len(tasks), td_size=thread["td_size"],
                           text_sha256=hashlib.sha256(text).hexdigest(),
                           coefficient_segments=[dict(s, sha256=hashlib.sha256(data[s["fileoff"]:s["fileoff"] + s["filesize"]]).hexdigest())
                                                 for s in container["segments"] if s["name"].startswith("__KERN")])
            status = path.with_name("model.hwx.status.plist")
            if status.exists():
                decoded = plistlib.loads(status.read_bytes())
                (output / "compiler-status.json").write_text(json.dumps(decoded, indent=2, default=str) + "\n")
                receipt["compiler_errors"] = decoded.get("ErrorList", [])
            receipt["status"] = "exported" if not receipt.get("compiler_errors") else "compiler_errors"
        except Exception as error:
            receipt.update(status="exported_parse_incomplete", parse_error=str(error))
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compiler", type=Path, default=Path(__file__).resolve().parents[1] / "gpt2/training/build/dump_hwx")
    a = p.parse_args()
    receipt = export(a.source, a.output, a.compiler)
    if receipt["status"] != "exported": raise SystemExit(1)


if __name__ == "__main__":
    main()
