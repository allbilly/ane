"""Pinned, portable whisper.cpp precision and instrumentation patches."""
REVISION = "60c0be6ac8fa71b1a2ae2dd938a31a34a508e774"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("pinned source anchor mismatch: " + before[:80])
    return text.replace(before, after, 1)



def accuracy_patch(text):
    text = replace_once(text, "ggml_type itype = ggml_type::GGML_TYPE_F16;", "ggml_type itype = ggml_type::GGML_TYPE_F32;")
    if text.count("ggml_gelu(ctx0, cur)") != 4:
        raise ValueError("pinned encoder/decoder GELU anchor mismatch")
    text = text.replace("ggml_gelu(ctx0, cur)", "ggml_gelu_erf(ctx0, cur)")
    start = text.index("static struct ggml_cgraph * whisper_build_graph_encoder(")
    end = text.index("// pre-compute cross-attention memory", start)
    graph = text[start:end]
    graph = replace_once(graph, "    const int n_ctx_pad = GGML_PAD(n_ctx, 256);\n\n", "")
    if graph.count("n_state_head, n_ctx_pad, n_head") != 2:
        raise ValueError("pinned encoder K/V view anchor mismatch")
    graph = graph.replace("n_state_head, n_ctx_pad, n_head", "n_state_head, n_ctx, n_head")
    text = text[:start] + graph + text[end:]
    anchor = "n_state_head, n_audio_ctx_pad, n_head"
    if text.count(anchor) != 2:
        raise ValueError("pinned decoder cross K/V view anchor mismatch")
    return text.replace(anchor, "n_state_head, n_audio_ctx, n_head")


def instrument_patch(text, profile_matrices=True):
    text = replace_once(text, '#include "whisper-arch.h"',
                        '#include "whisper-arch.h"\n#include "validation.h"')
    text = replace_once(text,
        "            ggml_backend_tensor_set(mel, wstate.inp_mel.data(), 0, ggml_nelements(mel)*sizeof(float));",
        "            ggml_backend_tensor_set(mel, wstate.inp_mel.data(), 0, ggml_nelements(mel)*sizeof(float));\n"
        '            whisper_trace_tensor("mel.f32", mel);')
    text = replace_once(text, "    // cross\n    if (!whisper_cross_external(wstate)) {",
        '    whisper_trace_tensor("encoder.f32", wstate.embd_enc);\n\n'
        "    // cross\n    if (!whisper_cross_external(wstate)) {")
    text = replace_once(text, "    if (batch.n_tokens > 1) {\n        //printf",
        '    whisper_trace_logits(logits_out.data() + (n_tokens - 1)*n_vocab,\n'
        '                         n_vocab, batch.token, n_tokens);\n\n'
        "    if (batch.n_tokens > 1) {\n        //printf")
    start = text.index("static bool whisper_encode_internal(")
    end = text.index("static struct ggml_cgraph * whisper_build_graph_decoder(", start)
    encode = text[start:end]
    encode = replace_once(encode, "    const int64_t t_start_us = ggml_time_us();", """    const int64_t t_start_us = ggml_time_us();
    auto compute_stage = [&](ggml_backend_sched_t sched, ggml_cgraph * graph, const char * name) {
        const auto start = ggml_time_us();
        const bool result = ggml_graph_compute_helper(sched, graph, n_threads);
        whisper_profile_stage(name, start);
        return result;
    };""")
    before = "ggml_graph_compute_helper(sched, gf, n_threads)"
    if encode.count(before) != 3:
        raise ValueError("pinned encoder compute stage anchor mismatch")
    for name in ("convolution", "transformer", "cross_kv"):
        encode = encode.replace(before, f'compute_stage(sched, gf, "{name}")', 1)
    text = text[:start] + encode + text[end:]
    if profile_matrices:
        for name, matrix in (("K", "k"), ("V", "v")):
            anchor = f"                layer.cross_attn_{matrix}_w,\n                cur);"
            text = replace_once(text, anchor, anchor +
                f'\n        ggml_format_name({name}cross, "whisper.cross_kv.%d.{matrix}", il);')
    return text


def blas_profile_patch(text):
    """Time the actual eight BLAS products without replacing their arithmetic."""
    text = replace_once(text, '#include "ggml-blas.h"', '#include "ggml-blas.h"\n#include "validation.h"')
    text = replace_once(text, "    const enum ggml_type type = src0->type;", """    const bool profile = whisper_profile_matrix(dst);
    if (profile) whisper_profile_matrix_input(src1);
    const auto t0 = profile ? ggml_time_us() : 0;
    const enum ggml_type type = src0->type;""")
    text = replace_once(text, "    void * wdata = ctx->work_data.get();",
                        "    void * wdata = ctx->work_data.get();\n    const auto t1 = profile ? ggml_time_us() : 0;")
    text = replace_once(text, "#if defined(GGML_BLAS_USE_OPENBLAS)\n    openblas_set_num_threads",
                        "    const auto t2 = profile ? ggml_time_us() : 0;\n\n"
                        "#if defined(GGML_BLAS_USE_OPENBLAS)\n    openblas_set_num_threads")
    text = replace_once(text, "    for (int64_t i13 = 0; i13 < ne13; i13++) {",
                        "    const auto t3 = profile ? ggml_time_us() : 0;\n"
                        "    for (int64_t i13 = 0; i13 < ne13; i13++) {")
    anchor = "\n}\n\nstatic void ggml_backend_blas_out_prod("
    text = replace_once(text, anchor, '''
    if (profile) {
        const auto t4 = ggml_time_us();
        const char * backend = "other BLAS";
        int blas_threads = -1; // Apple has no per-call thread-count query here.
#if defined(GGML_BLAS_USE_OPENBLAS)
        backend = "OpenBLAS";
        blas_threads = openblas_get_num_threads();
#elif defined(GGML_BLAS_USE_ACCELERATE)
        backend = "Accelerate";
#endif
#ifdef GGML_USE_OPENMP
        const char * conversion = "OpenMP to_float rows";
#else
        const char * conversion = "std::async to_float rows";
#endif
        whisper_profile_matrix_result(dst, backend, conversion, ctx->n_threads, blas_threads,
            reinterpret_cast<void *>(cblas_sgemm), t1-t0, t2-t1, t3-t2, t4-t3, t4-t0);
    }
}

static void ggml_backend_blas_out_prod(''')
    return text


def macos_adapter_patch(text):
    """Check every native E5RT call and report its encoder host stages."""
    text = replace_once(text, '#include "whisper-aneforge.h"',
                        '#include "whisper-aneforge.h"\n#include "validation.h"')
    text = replace_once(text,
        "    ctx->set_input(ctx->prog, ctx->pos_port.c_str(), pos.data(), ctx->pos_n);",
        '    if (ctx->set_input(ctx->prog, ctx->pos_port.c_str(), pos.data(), ctx->pos_n))\n'
        '        GGML_ABORT("E5RT position feed failed");')
    text = replace_once(text, "    f32_to_f16(mel, ctx->mel16.data(), n);",
                        "    const auto t0 = ggml_time_us();\n    f32_to_f16(mel, ctx->mel16.data(), n);\n"
                        "    const auto t1 = ggml_time_us();")
    text = replace_once(text,
        "    ctx->set_input(ctx->prog, ctx->mel_port.c_str(), ctx->mel16.data(), ctx->mel_n);\n"
        "    ctx->execute(ctx->prog);\n"
        "    ctx->get_output(ctx->prog, ctx->out_port.c_str(), ctx->out16.data(), ctx->out_n);",
        '''    if (ctx->set_input(ctx->prog, ctx->mel_port.c_str(), ctx->mel16.data(), ctx->mel_n))
        GGML_ABORT("E5RT mel feed failed");
    const auto t2 = ggml_time_us();
    if (ctx->execute(ctx->prog)) GGML_ABORT("E5RT encoder execute failed");
    const auto t3 = ggml_time_us();
    if (ctx->get_output(ctx->prog, ctx->out_port.c_str(), ctx->out16.data(), ctx->out_n))
        GGML_ABORT("E5RT encoder read failed");
    const auto t4 = ggml_time_us();''')
    text = replace_once(text, "    f16_to_f32(ctx->out16.data(), out, ctx->out_n);",
        '''    f16_to_f32(ctx->out16.data(), out, ctx->out_n);
    const auto t5 = ggml_time_us();
    std::fprintf(stderr, "MACOS_ANE encoder: submissions=1\\n");
    if (std::getenv("WHISPER_PROFILE"))
        std::fprintf(stderr, "WHISPER_PROFILE encoder: input=%.3f feed=%.3f dispatch=%.3f read=%.3f convert=%.3f ms\\n",
            (t1-t0)/1000., (t2-t1)/1000., (t3-t2)/1000., (t4-t3)/1000., (t5-t4)/1000.);''')
    return text


def dot_patch(text):
    # Native ggml's NEON F16 dot path accumulates in F16. Keep F16 storage
    # and use its existing widened FP32 path for the Whisper CPU reference,
    # convolution, attention and decoder in both execution modes.
    start = text.index("#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)")
    before = "#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)"
    position = text.index(before, start)
    return text[:position] + text[position:].replace(before,
        before + " && !defined(WHISPER_F32_DOT)", 1)


def tiled_patch(text):
    # Keep the fast blocked matrix path while widening F16 operands before
    # accumulation. The original ARM half accumulator violates our reference.
    text = replace_once(text, "    if (n < 2)\n        return false;", "    if (n < 2 && !(Atype == GGML_TYPE_F16 && Btype == GGML_TYPE_F32))\n        return false;")
    start = text.index("    case GGML_TYPE_F16: {")
    tail = text[start:]
    tail = replace_once(tail,
        "#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)",
        "#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER) && !defined(WHISPER_F32_DOT)")
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
