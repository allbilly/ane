#!/usr/bin/env python3
"""Freshly build and benchmark external Orion on macOS; never copy weights here."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = """
core/ane_runtime.m core/ane_io.m core/ane_program_cache.m core/mil_builder.m
core/iosurface_tensor.m core/bucket.m core/kernel.m
kernels/inference/prefill_ane.m kernels/inference/decode_ane.m
kernels/inference/decode_cpu.m kernels/inference/kv_cache.m model/weight_loader.m
compiler/graph.c compiler/builder.c compiler/topo.c compiler/patterns.c
compiler/validate.c compiler/pass_dce.c compiler/pass_identity.c
compiler/pass_conv_bias.c compiler/pass_cast.c compiler/pass_sram.c
compiler/pass_uniform_outputs.c compiler/pass_ane_validate.c compiler/pipeline.c
compiler/frontends/gpt2_prefill.c compiler/frontends/gpt2_decode.c
compiler/frontends/gpt2_final.c compiler/frontends/classifier_softmax.c
compiler/frontends/stories_train.c compiler/frontends/lora.c
compiler/codegen.m compiler/kernel_adapter.m compiler/mil_diff.m
""".split()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(*command):
    return subprocess.check_output(command, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orion", type=Path, default=Path.home() / "Desktop/Orion")
    parser.add_argument("--weights", type=Path, help="External Orion BLOBFILE directory")
    parser.add_argument("--output", type=Path, default=Path("orion-prewarmed.json"))
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("This is a macOS reference benchmark; use the Linux runtime on Asahi")
    orion = args.orion.expanduser().resolve()
    weights = (args.weights or orion / "model/blobs/gpt2_124m").expanduser().resolve()
    result = args.output.expanduser().resolve()
    log_path = result.with_suffix(".log")
    if result.exists() or log_path.exists():
        parser.error("Use a new --output path to preserve previous measurements")
    expected = json.loads((ROOT / "model-checksums.json").read_text())
    for name, checksum in expected.items():
        if digest(weights / name) != checksum:
            raise ValueError(f"Weight hash mismatch: {name}")
    print(f"Verified {len(expected)} external weight tensors", flush=True)
    compiler = output("xcrun", "--find", "clang")
    sdk = output("xcrun", "--show-sdk-path")
    linker = output("xcrun", "--find", "ld")
    flags = ["-O2", "-Wall", "-Wextra", "-DACCELERATE_NEW_LAPACK",
             "-isysroot", sdk, "-I", str(orion), "-I", str(orion / "core"),
             "-I", str(orion / "compiler")]
    harness = Path(__file__).with_suffix(".m")
    inputs = [orion / name for name in SOURCES] + [harness]
    source_hashes = {name: digest(orion / name) for name in SOURCES}
    source_hashes.update({str(p.relative_to(orion)): digest(p) for p in orion.rglob("*.h")
                          if "build" not in p.relative_to(orion).parts})
    metadata = {
        "measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": {"macos": platform.mac_ver()[0], "architecture": platform.machine(),
                 "chip": output("sysctl", "-n", "machdep.cpu.brand_string"),
                 "model": output("sysctl", "-n", "hw.model")},
        "orion_commit": output("git", "-C", str(orion), "rev-parse", "HEAD"),
        "orion_worktree_status": output("git", "-C", str(orion), "status", "--short"),
        "orion_source_sha256": source_hashes,
        "harness_sha256": digest(harness), "runner_sha256": digest(Path(__file__)),
        "verified_external_tensors": len(expected),
        "model_checksums_sha256": digest(ROOT / "model-checksums.json"),
        "build": {"compiler": compiler, "version": output(compiler, "--version"),
                  "sdk": sdk, "linker": linker, "flags": flags,
                  "objc_flags": ["-fobjc-arc"], "fresh_objects": True},
    }
    result.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orion-prewarm-") as tmp, log_path.open("w") as log:
        build = Path(tmp)
        objects = []
        for i, source in enumerate(inputs):
            obj = build / f"{i}.o"
            command = [compiler, *flags]
            if source.suffix == ".m":
                command.append("-fobjc-arc")
            subprocess.run([*command, "-c", str(source), "-o", str(obj)],
                           check=True, stdout=log, stderr=log)
            objects.append(str(obj))
        binary = build / "bench"
        subprocess.run([compiler, "-isysroot", sdk, f"--ld-path={linker}",
                        *objects, "-ldl", "-framework", "Foundation", "-framework",
                        "IOSurface", "-framework", "Accelerate", "-o", str(binary)],
                       check=True, stdout=log, stderr=log)
        native_result = build / "native.json"
        print(f"Build complete; prewarming and measuring. Log: {log_path}", flush=True)
        subprocess.run([str(binary), str(weights), str(native_result)],
                       check=True, stdout=log, stderr=log)
        report = json.loads(native_result.read_text())
    report["provenance"] = metadata
    report["log_sha256"] = digest(log_path)
    result.write_text(json.dumps(report, indent=2) + "\n")
    for backend, data in report["backends"].items():
        stats = data["decode"]
        print(f"{backend}: {stats['decode_steps_per_second']:.2f} steps/s; "
              f"p50 {stats['p50_ms']:.2f} ms; p90 {stats['p90_ms']:.2f} ms")
    print(f"Timed compiles: {report['timed_compiles']}; report: {result}")


if __name__ == "__main__":
    main()
