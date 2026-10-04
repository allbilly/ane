"""Build an external ctypes adapter from unmodified Orion runtime/compiler sources."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
ORION = Path.home() / "Desktop/Orion"
C_SOURCES = ["graph", "builder", "topo", "validate", "pass_dce", "pass_identity",
             "pass_cast", "pass_conv_bias", "pass_sram", "pass_uniform_outputs", "pass_ane_validate", "pipeline"]
SOURCES = [ORION / "compiler" / (name + ".c") for name in C_SOURCES]
SOURCES += [ORION / "compiler/codegen.m", ORION / "core/ane_runtime.m",
            ORION / "core/iosurface_tensor.m", ROOT / "orion_shim.m"]


def main():
  build = ROOT / "build"
  build.mkdir(exist_ok=True)
  sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
  flags = ["-O2", "-fPIC", "-Wall", "-Wextra", "-isysroot", sdk, "-I", str(ORION), "-I", str(ORION / "compiler")]
  objects, commands = [], []
  for i, source in enumerate(SOURCES):
    obj = build / f"orion_{i}.o"
    cmd = ["xcrun", "clang", *flags, *(["-fobjc-arc"] if source.suffix == ".m" else []),
           "-c", str(source), "-o", str(obj)]
    subprocess.run(cmd, check=True)
    commands.append(cmd)
    objects.append(str(obj))
  cmd = ["xcrun", "clang", "-dynamiclib", *objects, "-framework", "Foundation", "-framework", "IOSurface",
         "-o", str(build / "liborion_training.dylib")]
  subprocess.run(cmd, check=True)
  commands.append(cmd)
  receipt = {"commands": commands, "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
             "orion_commit": subprocess.check_output(["git", "-C", str(ORION), "rev-parse", "HEAD"], text=True).strip()}
  (build / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__": main()
