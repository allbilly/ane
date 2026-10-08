"""Verify native transfers with real weighted exports and a fake device; no ANE runs."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

from qwen35.weights import sha256
from whisper.encoder_kernel import pack_port, require
from whisper.pr_encoder_kernel import reconstruct
from whisper.pr_encoder_replay import native_descriptor, prepare, validate_fixtures

ROOT = Path(__file__).resolve().parents[1]


def verify(checkpoints, kernels, fixtures, manifest, compiler, output, openmp_root=None):
    require(len(checkpoints) == 3, "requires tiny/base/small checkpoints in order")
    report = dict(status="PASS_HOST_NATIVE_TRANSFER", hardware_execution="unverified", models={},
                  scope="real regenerated payloads and captured arrays, in-memory fake submission transport")
    with tempfile.TemporaryDirectory(prefix="pr3905-native-", dir=ROOT / "whisper/build") as temporary:
        directory = Path(temporary)
        binary = directory / "test-native"
        command = [str(compiler), "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror", "-I" + str(ROOT),
                   "-I" + str(ROOT / "whisper/vendor/whisper.cpp/src"),
                   str(ROOT / "experimental/test_pr_native_encoder.cpp"), "-o", str(binary)]
        if openmp_root is not None:
            require(platform.system() == "Darwin", "--openmp-root is the optional Apple Clang/libomp host test")
            command += ["-Xclang", "-fopenmp", "-I" + str(openmp_root / "include"),
                        "-L" + str(openmp_root / "lib"), "-Wl,-rpath," + str(openmp_root / "lib"), "-lomp"]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        report["compile"] = dict(command=["<temporary binary>" if item == str(binary) else item for item in command], stderr=result.stderr,
                                 compiler_version=subprocess.check_output([str(compiler), "--version"], text=True).splitlines()[0])
        report["asserted_read_workers"] = 4 if openmp_root is not None else 1
        if openmp_root is not None:
            report["openmp_runtime_sha256"] = sha256(openmp_root / "lib/libomp.dylib")
        for size, checkpoint in zip(("tiny", "base", "small"), checkpoints):
            meta, hwx, positions = reconstruct(checkpoint, kernels / size)
            plan, payloads = prepare(meta, hwx, positions)
            cases = validate_fixtures(plan, manifest, fixtures / size)
            payload_dir, fixture_dir = directory / size / "payloads", directory / size / "fixtures"
            payload_dir.mkdir(parents=True)
            fixture_dir.mkdir()
            descriptor = native_descriptor(plan, payloads)
            (payload_dir / "native-layout.txt").write_text(descriptor)
            for name, data in payloads.items():
                (payload_dir / (name + ".bin")).write_bytes(data)
            buffers = {b["bank"]:b["size"] for b in plan["buffers"]}
            for name, arrays in cases:
                prefix = fixture_dir / name
                prefix.with_suffix(".mel.f32").write_bytes(arrays["mel"].astype("<f4").tobytes())
                prefix.with_suffix(".mel.f16").write_bytes(arrays["mel"].tobytes())
                prefix.with_suffix(".out.f16").write_bytes(arrays["output"].tobytes())
                prefix.with_suffix(".out.f32").write_bytes(arrays["output"].astype("<f4").tobytes())
                prefix.with_suffix(".mel.padded.bin").write_bytes(pack_port(arrays["mel"], plan["ports"]["mel"], buffers[5]))
            (fixture_dir / "positions.padded.bin").write_bytes(pack_port(cases[0][1]["positions"], plan["ports"]["positions"], buffers[4]))
            environment = os.environ.copy()
            environment.update(OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="4")
            result = subprocess.run([str(binary), str(payload_dir), str(fixture_dir)], env=environment, check=True, text=True, capture_output=True)
            report["models"][size] = dict(status="PASS_HOST_NATIVE_TRANSFER", cases=[name for name, _ in cases],
                                         td_count=plan["td_count"], coefficient_banks=len(plan["coefficient_banks"]),
                                         descriptor=descriptor, stdout=result.stdout, stderr=result.stderr,
                                         hwx_sha256=plan["hwx_sha256"])
            print(result.stdout.strip(), flush=True)
        report["source_sha256"] = {str(p.relative_to(ROOT)):sha256(p) for p in
                                     (ROOT / "whisper/asahi_full_encoder.cpp", ROOT / "experimental/test_pr_native_encoder.cpp")}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs=3, required=True)
    parser.add_argument("--kernels", type=Path, default=ROOT / "whisper/kernels/pr3905")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, default=Path("/usr/bin/c++"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--openmp-root", type=Path, help="Optional installed libomp prefix for the Mac four-worker host test")
    args = parser.parse_args()
    verify(args.checkpoints, args.kernels, args.fixtures, json.loads(args.manifest.read_text()), args.compiler, args.output, args.openmp_root)


if __name__ == "__main__":
    main()
