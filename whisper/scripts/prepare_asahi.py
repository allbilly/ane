#!/usr/bin/env python3
"""Add explicit ANE encoder projections to an isolated pinned whisper.cpp worktree."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REVISION = "60c0be6ac8fa71b1a2ae2dd938a31a34a508e774"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("pinned source anchor mismatch: " + before[:80])
    return text.replace(before, after, 1)


def source_patch(text):
    text = replace_once(text, '#include "whisper-arch.h"',
                        '#include "whisper-arch.h"\n#include "asahi_encoder.h"')
    text = replace_once(text, "struct whisper_context {\n",
                        "struct whisper_context {\n    std::unique_ptr<WhisperAsahi> asahi;\n")
    text = replace_once(text, "ggml_type itype = ggml_type::GGML_TYPE_F16;", "ggml_type itype = ggml_type::GGML_TYPE_F32;")
    start = text.index("static struct ggml_cgraph * whisper_build_graph_encoder(")
    end = text.index("// pre-compute cross-attention memory", start)
    graph = text[start:end]
    graph = replace_once(graph, "ggml_gelu(ctx0, cur)", "ggml_gelu_erf(ctx0, cur)")
    # CPU flash attention supports the actual sequence length. Exposing the
    # allocation's 36 zero-filled padding slots changes softmax probabilities.
    graph = replace_once(graph, "    const int n_ctx_pad = GGML_PAD(n_ctx, 256);\n\n", "")
    if graph.count("n_state_head, n_ctx_pad, n_head") != 2:
        raise ValueError("pinned encoder K/V view anchor mismatch")
    graph = graph.replace("n_state_head, n_ctx_pad, n_head", "n_state_head, n_ctx, n_head")
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
    start = text.index("static struct ggml_cgraph * whisper_build_graph_conv(")
    end = text.index("static struct ggml_cgraph * whisper_build_graph_encoder(", start)
    graph = text[start:end]
    if graph.count("ggml_gelu(ctx0, cur)") != 2:
        raise ValueError("pinned convolution GELU anchor mismatch")
    text = text[:start] + graph.replace("ggml_gelu(ctx0, cur)", "ggml_gelu_erf(ctx0, cur)") + text[end:]
    # The remaining GELU is in the CPU decoder. Match the independent HF model.
    text = replace_once(text, "ggml_gelu(ctx0, cur)", "ggml_gelu_erf(ctx0, cur)")
    # Cross-attention has the same unmasked padding issue as the encoder.
    # Keep the padded per-layer allocation stride, but expose only real keys.
    anchor = "n_state_head, n_audio_ctx_pad, n_head"
    if text.count(anchor) != 2:
        raise ValueError("pinned decoder cross K/V view anchor mismatch")
    text = text.replace(anchor, "n_state_head, n_audio_ctx, n_head")
    text = replace_once(text,
        "            ggml_backend_tensor_set(mel, wstate.inp_mel.data(), 0, ggml_nelements(mel)*sizeof(float));",
        "            ggml_backend_tensor_set(mel, wstate.inp_mel.data(), 0, ggml_nelements(mel)*sizeof(float));\n"
        '            whisper_asahi_trace_tensor("mel.f32", mel);')
    text = replace_once(text, "    // cross\n    if (!whisper_cross_external(wstate)) {", """    whisper_asahi_trace_tensor("encoder.f32", wstate.embd_enc);
    if (wctx.asahi) {
        const int positions = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : wctx.model.hparams.n_audio_ctx;
        wctx.asahi->finish_encoder(wctx.model.hparams.n_audio_layer, positions);
    }

    // cross
    if (!whisper_cross_external(wstate)) {""")
    text = replace_once(text, "    if (batch.n_tokens > 1) {\n        //printf", """    whisper_asahi_trace_logits(logits_out.data() + (n_tokens - 1)*n_vocab,
                               n_vocab, batch.token, n_tokens);

    if (batch.n_tokens > 1) {
        //printf""")
    start = text.index("static bool whisper_encode_internal(")
    end = text.index("static struct ggml_cgraph * whisper_build_graph_decoder(", start)
    encode = text[start:end]
    encode = replace_once(encode, "    const int64_t t_start_us = ggml_time_us();", """    const int64_t t_start_us = ggml_time_us();
    auto compute_stage = [&](ggml_backend_sched_t sched, ggml_cgraph * graph, const char * name) {
        const auto start = ggml_time_us();
        const bool result = ggml_graph_compute_helper(sched, graph, n_threads);
        whisper_asahi_profile_stage(name, start);
        return result;
    };""")
    before = "ggml_graph_compute_helper(sched, gf, n_threads)"
    if encode.count(before) != 3:
        raise ValueError("pinned encoder compute stage anchor mismatch")
    for name in ("convolution", "transformer", "cross_kv"):
        encode = encode.replace(before, f'compute_stage(sched, gf, "{name}")', 1)
    text = text[:start] + encode + text[end:]
    return text


def dot_patch(text):
    # Native ggml's NEON F16 dot path accumulates in F16. Keep F16 storage
    # and use its existing widened FP32 path for the Whisper CPU reference,
    # convolution, attention and decoder in both execution modes.
    start = text.index("#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)")
    before = "#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)"
    position = text.index(before, start)
    return text[:position] + text[position:].replace(before,
        before + " && !defined(WHISPER_ASAHI_F32_DOT)", 1)


def tiled_patch(text):
    # Keep the fast blocked matrix path while widening F16 operands before
    # accumulation. The original ARM half accumulator violates our reference.
    text = replace_once(text, "    if (n < 2)\n        return false;", "    if (n < 2 && !(Atype == GGML_TYPE_F16 && Btype == GGML_TYPE_F32))\n        return false;")
    start = text.index("    case GGML_TYPE_F16: {")
    tail = text[start:]
    tail = replace_once(tail,
        "#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)",
        "#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER) && !defined(WHISPER_ASAHI_F32_DOT)")
    anchor = "#elif defined(__ARM_NEON) && !defined(_MSC_VER)\n        if (Btype == GGML_TYPE_F32) {"
    tail = replace_once(tail, anchor, """#elif defined(__ARM_NEON) && !defined(_MSC_VER)
        if (Btype == GGML_TYPE_F16) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, ggml_fp16_t, float> tb{ params,
                k, (const ggml_fp16_t *)A, lda,
                (const ggml_fp16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
        if (Btype == GGML_TYPE_F32) {""")
    return text[:start] + tail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "whisper/vendor/whisper-asahi")
    args = parser.parse_args()
    source = args.source.resolve()
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        parser.error("requires isolated whisper.cpp worktree at " + REVISION)
    for name in ("src/whisper.cpp", "src/CMakeLists.txt", "ggml/src/ggml-cpu/simd-mappings.h",
                 "ggml/src/ggml-cpu/llamafile/sgemm.cpp", "ggml/src/ggml-cpu/simd-gemm.h"):
        before = subprocess.check_output(["git", "-C", str(source), "show", "HEAD:" + name], text=True)
        if name == "src/whisper.cpp":
            after = source_patch(before)
        elif name.endswith("sgemm.cpp"):
            after = tiled_patch(before)
        elif name.endswith("simd-gemm.h"):
            # GCC defines __ARM_NEON, unlike Clang's additional alias.
            after = replace_once(before, "defined (__ARM_NEON__)", "defined(__ARM_NEON)")
        elif name.endswith("simd-mappings.h"):
            after = dot_patch(before)
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
target_compile_definitions(ggml-cpu PRIVATE WHISPER_ASAHI_F32_DOT=1)
set_source_files_properties("${ANE_ROOT}/qwen35/ane_matmul.c"
                            PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16")
"""
        path = source / name
        current = path.read_text()
        if current not in (before, after):
            raise ValueError("refusing to overwrite other edits in " + str(path))
        if current != after:
            path.write_text(after)
    print("Prepared isolated Asahi encoder worktree:", source)


if __name__ == "__main__":
    main()
