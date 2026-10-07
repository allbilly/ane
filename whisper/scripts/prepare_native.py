#!/usr/bin/env python3
"""Prepare the same pinned CPU math and profiling on macOS or Asahi."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from whisper.native import REVISION, accuracy_patch, instrument_patch, dot_patch, tiled_patch, macos_adapter_patch, blas_profile_patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("asahi", "macos"),
                        default="macos" if platform.system() == "Darwin" else "asahi")
    parser.add_argument("--source", type=Path, required=True, help="Isolated whisper.cpp worktree at the pinned revision")
    parser.add_argument("--precision", choices=("fp32", "original"), default="fp32")
    args = parser.parse_args()
    if args.backend == "asahi":
        if args.precision != "fp32":
            parser.error("the Asahi reference requires --precision fp32")
        from whisper.scripts.prepare_asahi import main as prepare_asahi
        sys.argv = [sys.argv[0], "--source", str(args.source)]
        return prepare_asahi()
    source = args.source.resolve()
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        parser.error("requires isolated whisper.cpp worktree at " + REVISION)
    changes = []
    for name in ("src/whisper.cpp", "src/CMakeLists.txt", "ggml/src/ggml-cpu/simd-mappings.h",
                 "ggml/src/ggml-cpu/llamafile/sgemm.cpp", "src/aneforge/whisper-aneforge.cpp",
                 "ggml/src/ggml-blas/ggml-blas.cpp"):
        before = subprocess.check_output(["git", "-C", str(source), "show", "HEAD:" + name], text=True)
        after = before
        previous = before
        if name == "src/whisper.cpp":
            after = instrument_patch(accuracy_patch(before) if args.precision == "fp32" else before)
            previous = instrument_patch(accuracy_patch(before) if args.precision == "fp32" else before, profile_matrices=False)
        elif name == "src/aneforge/whisper-aneforge.cpp":
            after = macos_adapter_patch(before)
        elif name == "src/CMakeLists.txt":
            after += '\n# Shared precision and instrumentation from the parent ane repository.\n'
            after += 'if (NOT ANE_ROOT)\n    message(FATAL_ERROR "Set ANE_ROOT to the parent ane repository")\nendif()\n'
            after += 'target_include_directories(whisper PRIVATE "${ANE_ROOT}/whisper")\n'
            if args.precision == "fp32":
                after += 'target_compile_definitions(ggml-cpu PRIVATE WHISPER_F32_DOT=1)\n'
            previous = after
            after += 'if (TARGET ggml-blas)\n    target_include_directories(ggml-blas PRIVATE "${ANE_ROOT}/whisper")\nendif()\n'
        elif name.endswith("ggml-blas.cpp"):
            after = blas_profile_patch(before)
        elif args.precision == "fp32":
            after = dot_patch(before) if name.endswith("simd-mappings.h") else tiled_patch(before)
        path = source / name
        if path.read_text() not in (before, previous, after):
            raise ValueError("refusing to overwrite other edits in " + str(path))
        changes.append((path, after))
    for path, after in changes:
        if path.read_text() != after:
            path.write_text(after)
    print(f"Prepared {args.backend} worktree with {args.precision} CPU precision: {source}")


if __name__ == "__main__":
    main()
