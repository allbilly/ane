#!/usr/bin/env python3
"""Add explicit ANE encoder projections to an isolated pinned whisper.cpp worktree."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from whisper.native import REVISION, accuracy_patch, instrument_patch, replace_once, dot_patch, tiled_patch, blas_profile_patch


def source_patch(text, complete=True, profile_matrices=True):
    text = instrument_patch(accuracy_patch(text), profile_matrices)
    text = replace_once(text, '#include "whisper-arch.h"',
                        '#include "whisper-arch.h"\n#include "asahi_encoder.h"')
    text = replace_once(text, "struct whisper_context {\n",
                        "struct whisper_context {\n    std::unique_ptr<WhisperAsahi> asahi;\n")
    start = text.index("static struct ggml_cgraph * whisper_build_graph_encoder(")
    end = text.index("// pre-compute cross-attention memory", start)
    graph = text[start:end]
    anchor = "    const auto & model   = wctx.model;"
    graph = replace_once(graph, anchor, """    if (whisper_asahi_enabled() && !wctx.asahi) {
        if (wctx.params.use_gpu) GGML_ABORT("Asahi encoder requires use_gpu=false");
        wctx.asahi.reset(new WhisperAsahi);
    }
    auto project = [&](ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
        return wctx.asahi ? wctx.asahi->project(ctx, w, x) : ggml_mul_mat(ctx, w, x);
    };
""" + anchor)
    for matrix in ("attn_q_w", "attn_k_w", "attn_v_w", "attn_ln_1_w", "mlp_0_w", "mlp_1_w"):
        before = "ggml_mul_mat(ctx0,\n                    layer." + matrix
        graph = replace_once(graph, before, "project(ctx0,\n                    layer." + matrix)
    text = text[:start] + graph + text[end:]
    text = replace_once(text, "    // cross\n    if (!whisper_cross_external(wstate)) {", """    if (wctx.asahi) {
        const int positions = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : wctx.model.hparams.n_audio_ctx;
        wctx.asahi->finish_encoder(wctx.model.hparams.n_audio_layer, positions);
    }

    // cross
    if (!whisper_cross_external(wstate)) {""")
    return complete_patch(text) if complete else text


def complete_patch(text):
    # Reuse whisper.cpp's external-encoder graph and CPU decoder. Linux supplies
    # the same narrow C API directly; no Apple dylib or runtime compilation.
    text = replace_once(text,
        '    if (const char * aneforge_dir = getenv("ANEFORGE_ENCODER")) {',
        '''    const char * asahi_encoder_dir = getenv("WHISPER_ASAHI_ENCODER");
    if (asahi_encoder_dir && (ctx->params.use_gpu || ctx->model.hparams.n_mels != 80 ||
            ctx->model.hparams.n_audio_ctx != 1500 || ctx->model.hparams.n_audio_state != 384 ||
            ctx->model.hparams.n_audio_layer != 4 || ctx->model.hparams.n_vocab != 51864)) {
        WHISPER_LOG_ERROR("complete Asahi encoder requires tiny.en with use_gpu=false\\n");
        whisper_free_state(state);
        return nullptr;
    }
    if (const char * aneforge_dir = asahi_encoder_dir ? asahi_encoder_dir : getenv("ANEFORGE_ENCODER")) {''')
    text = replace_once(text,
        '        WHISPER_LOG_INFO("%s: compiling for the ANE (one time) ...\\n", __func__);',
        '        WHISPER_LOG_INFO("%s: loading external encoder payloads ...\\n", __func__);')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "whisper/vendor/whisper-asahi")
    args = parser.parse_args()
    source = args.source.resolve()
    # Accept only the known previous generated version when upgrading the patcher.
    legacy = {
        "src/whisper.cpp": "ffa8e1240ba9cd8d5a2795e3d4291e9436e4f7abfc6739bcdd05828052472b39",
        "src/CMakeLists.txt": "70c73f550ffd6f60b284a4728e3e821bc5204297bfb34449437fd8d359d07836",
        "ggml/src/ggml-cpu/simd-mappings.h": "f83b0de9f5dfff92092c2bab9ec718450884dde77d0f9809f03a62e053d4ff88",
        "ggml/src/ggml-cpu/llamafile/sgemm.cpp": "6d35a9d1882cd9ed0e7f31e94f9a005f50816f659568d3984cf97e0db58a7981",
    }
    import hashlib
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        parser.error("requires isolated whisper.cpp worktree at " + REVISION)
    for name in ("src/whisper.cpp", "src/CMakeLists.txt", "ggml/src/ggml-cpu/simd-mappings.h",
                 "ggml/src/ggml-cpu/llamafile/sgemm.cpp", "ggml/src/ggml-cpu/simd-gemm.h",
                 "src/aneforge/whisper-aneforge.cpp", "ggml/src/ggml-blas/ggml-blas.cpp"):
        before = subprocess.check_output(["git", "-C", str(source), "show", "HEAD:" + name], text=True)
        previous = before
        older = before
        if name == "src/whisper.cpp":
            after = source_patch(before)
            previous = source_patch(before, complete=False)
            older = source_patch(before, profile_matrices=False)
        elif name == "src/aneforge/whisper-aneforge.cpp":
            after = replace_once(before,
                "#else  // not (Apple && arm64): the ANE is unavailable, so stub the entry points.",
                "#elif defined(WHISPER_ASAHI_FULL_ENCODER)\n"
                "// Native Linux definitions are supplied by asahi_full_encoder2.cpp.\n\n"
                "#else  // not (Apple && arm64): the ANE is unavailable, so stub the entry points.")
        elif name.endswith("sgemm.cpp"):
            after = tiled_patch(before)
        elif name.endswith("simd-gemm.h"):
            # GCC defines __ARM_NEON, unlike Clang's additional alias.
            after = replace_once(before, "defined (__ARM_NEON__)", "defined(__ARM_NEON)")
        elif name.endswith("simd-mappings.h"):
            after = dot_patch(before)
        elif name.endswith("ggml-blas.cpp"):
            after = blas_profile_patch(before)
        else:
            after = before + """
# Parent ane repository supplies the Linux matrix stream; no Apple compiler.
if (NOT ANE_ROOT)
    message(FATAL_ERROR "Set ANE_ROOT to the parent ane repository")
endif()
target_sources(whisper PRIVATE "${ANE_ROOT}/whisper/asahi_encoder.cpp"
                               "${ANE_ROOT}/qwen35/ane_matmul.c")
target_include_directories(whisper PRIVATE "${ANE_ROOT}/whisper" "${ANE_ROOT}/qwen35")
find_package(OpenMP REQUIRED COMPONENTS C CXX)
target_link_libraries(whisper PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
target_compile_definitions(ggml-cpu PRIVATE WHISPER_F32_DOT=1)
set_source_files_properties("${ANE_ROOT}/qwen35/ane_matmul.c"
                            PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16")
"""
            previous = after
            after += '''
target_sources(whisper PRIVATE "${ANE_ROOT}/whisper/asahi_full_encoder2.cpp")
target_compile_definitions(whisper PRIVATE WHISPER_ASAHI_FULL_ENCODER=1)
'''
            older = after
            after += '''
if (TARGET ggml-blas)
    target_include_directories(ggml-blas PRIVATE "${ANE_ROOT}/whisper")
endif()
'''
        path = source / name
        current = path.read_text()
        # Accept the same generated patch before the local runner was renamed.
        compatible = (current.replace("asahi_full_encoder.cpp", "asahi_full_encoder2.cpp")
                      if name in ("src/CMakeLists.txt", "src/aneforge/whisper-aneforge.cpp") else current)
        if compatible not in (before, previous, older, after) and hashlib.sha256(current.encode()).hexdigest() != legacy.get(name):
            raise ValueError("refusing to overwrite other edits in " + str(path))
        if current != after:
            path.write_text(after)
    print("Prepared isolated Asahi encoder worktree:", source)


if __name__ == "__main__":
    main()
