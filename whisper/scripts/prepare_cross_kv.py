"""Fuse the cross-K/V graph in an already prepared, isolated tiny.en worktree."""
import argparse
from pathlib import Path
import subprocess

from whisper.native import REVISION, accuracy_patch, instrument_patch, replace_once


def patch(text, original):
    start = "static struct ggml_cgraph * whisper_build_graph_cross("
    end = "// evaluate the encoder with the given state"
    expected = original[original.index(start):original.index(end)]
    # Some existing Asahi preparations have not yet named individual products.
    unnamed = expected
    for kind in ("k", "v"):
        unnamed = unnamed.replace(f'\n        ggml_format_name({kind.upper()}cross, "whisper.cross_kv.%d.{kind}", il);', "")
    begin, finish = text.index(start), text.index(end)
    graph = text[begin:finish]
    after = replace_once(expected, "    const float  Kscale = pow(float(n_state_head), -0.25);", """    struct ggml_tensor * fused = nullptr;
    if (whisper_fused_cross_kv_enabled()) {
        if (wctx.params.use_gpu || !wctx.params.flash_attn || n_state != 384 || n_ctx != 1500 ||
            model.hparams.n_text_layer != 4) GGML_ABORT("fused cross-K/V requires tiny.en, CPU and flash attention");
        if (!wstate.fused_cross_kv) {
            std::vector<ggml_tensor *> weights;
            for (const auto & layer : model.layers_decoder) {
                weights.push_back(layer.cross_attn_k_w);
                weights.push_back(layer.cross_attn_v_w);
            }
            wstate.fused_cross_kv.reset(new WhisperCrossKV(weights));
        }
        fused = wstate.fused_cross_kv->project(ctx0, cur);
    }

    const float  Kscale = pow(float(n_state_head), -0.25);""")
    for kind in ("k", "v"):
        letter = kind.upper()
        before = f"""        struct ggml_tensor * {letter}cross = ggml_mul_mat(ctx0,
                layer.cross_attn_{kind}_w,
                cur);
        ggml_format_name({letter}cross, "whisper.cross_kv.%d.{kind}", il);"""
        after = replace_once(after, before, f"""        struct ggml_tensor * {letter}cross = fused
            ? WhisperCrossKV::view(ctx0, fused, 2*il + {int(kind == 'v')})
            : ggml_mul_mat(ctx0, layer.cross_attn_{kind}_w, cur);
        if (!fused) ggml_format_name({letter}cross, "whisper.cross_kv.%d.{kind}", il);""")
    if graph == after:
        if '#include "cross_kv.h"' not in text or 'std::unique_ptr<WhisperCrossKV> fused_cross_kv;' not in text:
            raise ValueError("incomplete existing fusion patch")
        return text
    if graph not in (expected, unnamed):
        raise ValueError("refusing to overwrite other cross-K/V graph edits")
    if '#include "cross_kv.h"' in text or 'std::unique_ptr<WhisperCrossKV>' in text:
        raise ValueError("incomplete existing fusion patch")
    text = text[:begin] + after + text[finish:]
    text = replace_once(text, '#include "ggml-backend.h"', '#include "ggml-backend.h"\n#include "cross_kv.h"')
    return replace_once(text, "struct whisper_state {\n", "struct whisper_state {\n    std::unique_ptr<WhisperCrossKV> fused_cross_kv;\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        parser.error("requires pinned whisper.cpp revision " + REVISION)
    original = subprocess.check_output(["git", "-C", str(source), "show", "HEAD:src/whisper.cpp"], text=True)
    path = source / "src/whisper.cpp"
    before = path.read_text()
    if 'ggml_type itype = ggml_type::GGML_TYPE_F32;' not in before:
        parser.error("prepare the shared FP32 CPU worktree first")
    after = patch(before, instrument_patch(accuracy_patch(original)))
    if after != before:
        path.write_text(after)
    print("Prepared opt-in fused cross-K/V:", source)


if __name__ == "__main__":
    main()
