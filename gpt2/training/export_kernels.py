"""Export exact training MIL to H13G HWX and decode task/register packets."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from hwx import parse_container, parse_tasks


def main():
  manifest = []
  for backend in ("aneforge", "orion"):
    directory = ROOT / backend
    for record in json.loads((directory / "kernel-index.json").read_text()):
      kernel = directory / record["mil"]
      out = kernel.parent / "hwx"
      subprocess.run([str(ROOT / "build/dump_hwx"), str(kernel.parent), str(out)], check=True)
      data = (out / "model.hwx").read_bytes()
      container = parse_container(data)
      segment = next(s for s in container["segments"] if s["name"] == "__TEXT")
      text = data[segment["fileoff"]:segment["fileoff"] + segment["filesize"]]
      thread = container["thread"]
      tasks = parse_tasks(text, thread["td_size"], thread["td_count"])
      for task in tasks: task["registers"] = {f"0x{k:05x}": v for k, v in task["registers"].items()}
      (out / "container.json").write_text(json.dumps(container, indent=2) + "\n")
      (out / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
      (out / "task-descriptors.bin").write_bytes(text)
      manifest.append({"backend": backend, **record, "hwx": str((out / "model.hwx").relative_to(ROOT)),
                       "hwx_sha256": hashlib.sha256(data).hexdigest(), "hwx_bytes": len(data),
                       "task_count": len(tasks), "task_descriptors_sha256": hashlib.sha256(text).hexdigest(),
                       "export_method": "ANECCompile h13g offline export of exact emitted MIL; not a readback of the signed runtime binary"})
      print(f"{backend}/{record['name']}: {len(tasks)} tasks, {len(data)} bytes", flush=True)
  (ROOT / "kernel-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
