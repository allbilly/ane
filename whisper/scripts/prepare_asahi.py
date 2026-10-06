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
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "whisper/vendor/whisper-asahi")
    args = parser.parse_args()
    source = args.source.resolve()
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        parser.error("requires isolated whisper.cpp worktree at " + REVISION)
    for name in ("src/whisper.cpp", "src/CMakeLists.txt"):
        before = subprocess.check_output(["git", "-C", str(source), "show", "HEAD:" + name], text=True)
        if name.endswith(".cpp"):
            after = source_patch(before)
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
set_source_files_properties("${ANE_ROOT}/qwen35/ane_matmul.c"
                            PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16")
"""
        path = source / name
        current = path.read_text()
        if current not in (before, after):
            raise ValueError("refusing to overwrite other edits in " + str(path))
        path.write_text(after)
    print("Prepared isolated Asahi encoder worktree:", source)


if __name__ == "__main__":
    main()
