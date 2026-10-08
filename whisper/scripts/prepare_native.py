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
    parser.add_argument("--encoder", choices=("default", "paired", "pr"), default="default",
                        help="Shared paired arithmetic or PR replay with FP32 CPU reference math")
    args = parser.parse_args()
    if args.encoder in ("paired", "pr") and args.precision != "fp32":
        parser.error("paired/PR validation requires FP32 host precision")
    if args.encoder == "pr" and args.backend != "asahi":
        parser.error("PR validation mode supplies the Linux replay adapter")
    if args.backend == "asahi" and args.encoder == "default":
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
                 "ggml/src/ggml-blas/ggml-blas.cpp", "ggml/src/ggml-cpu/simd-gemm.h"):
        before = subprocess.check_output(["git", "-C", str(source), "show", "HEAD:" + name], text=True)
        after = before
        previous = before
        if name == "src/whisper.cpp":
            after = instrument_patch(accuracy_patch(before) if args.precision == "fp32" else before)
            previous = instrument_patch(accuracy_patch(before) if args.precision == "fp32" else before, profile_matrices=False)
            if args.encoder == "paired":
                from whisper.scripts.prepare_macos_precision import patch_source
                previous = after
                after = patch_source(after)
            elif args.encoder == "pr":
                from whisper.scripts.prepare_pr_asahi import source_patch
                previous = after
                after = source_patch(after)
        elif name == "src/aneforge/whisper-aneforge.cpp":
            if args.backend == "macos":
                after = macos_adapter_patch(before)
        elif name == "src/CMakeLists.txt":
            if args.encoder == "pr":
                from whisper.scripts.prepare_pr_asahi import cmake_patch
                after = cmake_patch(before)
            else:
                after = before
            after += '\n# Shared precision and instrumentation from the parent ane repository.\n'
            after += 'if (NOT ANE_ROOT)\n    message(FATAL_ERROR "Set ANE_ROOT to the parent ane repository")\nendif()\n'
            after += 'target_include_directories(whisper PRIVATE "${ANE_ROOT}/whisper")\n'
            if args.precision == "fp32":
                after += 'target_compile_definitions(ggml-cpu PRIVATE WHISPER_F32_DOT=1)\n'
            previous = after
            after += 'if (TARGET ggml-blas)\n    target_include_directories(ggml-blas PRIVATE "${ANE_ROOT}/whisper")\nendif()\n'
            if args.encoder == "paired":
                previous = after
                after += 'target_sources(whisper PRIVATE "${ANE_ROOT}/whisper/macos_precision.cpp")\n'
                if args.backend == "asahi":
                    after += '''target_sources(whisper PRIVATE "${ANE_ROOT}/qwen35/ane_matmul.c")
target_include_directories(whisper PRIVATE "${ANE_ROOT}/qwen35")
if (NOT GGML_OPENMP)
    message(FATAL_ERROR "Paired Linux projections require GGML_OPENMP")
endif()
find_package(OpenMP REQUIRED COMPONENTS C CXX)
target_link_libraries(whisper PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
set_source_files_properties("${ANE_ROOT}/whisper/macos_precision.cpp" "${ANE_ROOT}/qwen35/ane_matmul.c"
                           PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16")
'''
        elif name.endswith("ggml-blas.cpp"):
            after = blas_profile_patch(before)
        elif name.endswith("simd-gemm.h"):
            if args.backend == "asahi":
                after = before.replace("defined (__ARM_NEON__)", "defined(__ARM_NEON)")
        elif args.precision == "fp32":
            after = dot_patch(before) if name.endswith("simd-mappings.h") else tiled_patch(before)
        path = source / name
        compatible = after.replace('if (NOT GGML_OPENMP)\n    message(FATAL_ERROR "Paired Linux projections require GGML_OPENMP")\nendif()\n', '')
        if path.read_text() not in (before, previous, after, compatible):
            raise ValueError("refusing to overwrite other edits in " + str(path))
        changes.append((path, after))
    for path, after in changes:
        if path.read_text() != after:
            path.write_text(after)
    print(f"Prepared {args.backend} worktree with {args.precision} CPU precision: {source}")


if __name__ == "__main__":
    main()
