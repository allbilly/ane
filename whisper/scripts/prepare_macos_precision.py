"""Stage verified paired projection programs and patch an isolated Mac worktree."""
import argparse
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np

from whisper.native import REVISION, accuracy_patch, instrument_patch, replace_once
from whisper.precision_encoder import CHECKPOINT_SHA256
from whisper.validation import digest

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


def checkpoint_weights(checkpoint):
    from safetensors.numpy import load_file
    if digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("requires the pinned tiny.en checkpoint")
    tensors = load_file(checkpoint)
    result = {}
    for layer in range(4):
        for name in PROJECTIONS:
            key = f"model.encoder.layers.{layer}." + ("self_attn." if name.endswith("proj") else "") + name + ".weight"
            weights = tensors[key]
            if not np.array_equal(weights, weights.astype("<f2").astype(np.float32)):
                raise ValueError("checkpoint projection requires weight residual planes")
            result[f"layer{layer}-{name}"] = weights
    return result


def validate_programs(directory, checkpoint):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["checkpoint_sha256"] != CHECKPOINT_SHA256 or manifest["device_mask"] != 4 or
            manifest["gains"] != [1., 1.375] or manifest["contraction_partitions"] != 2 or
            manifest["temporal_planes"] != 4 or manifest["positions"] != 1500):
        raise ValueError("precision program contract changed")
    expected = checkpoint_weights(checkpoint)
    if len(manifest["programs"]) != 24 or {p["name"] for p in manifest["programs"]} != set(expected):
        raise ValueError("precision encoder requires all 24 projections")
    for record in manifest["programs"]:
        path = directory / record["name"]
        weights = expected[record["name"]]
        n, k = weights.shape
        if (record["input_features"], record["output_features"]) != (k, n):
            raise ValueError("precision projection dimensions changed")
        for name in ("model.mil", "weights.bin"):
            if digest(path / name) != record["file_sha256"][name]:
                raise ValueError("precision program file changed: " + str(path / name))
        verify_blob(path / "weights.bin", weights)
    return manifest


def verify_blob(path, weights):
    n, k = weights.shape
    grouped = np.concatenate((weights[:, :k//2], weights[:, k//2:]), axis=0).astype("<f2").tobytes()
    blob = path.read_bytes()
    if (len(blob) != 128+len(grouped) or struct.unpack_from("<II", blob) != (1, 2) or
            struct.unpack_from("<IIQQ", blob, 64) != (0xdeadbeef, 1, len(grouped), 128) or
            blob[128:] != grouped):
        raise ValueError("paired projection weights differ from checkpoint: " + str(path))


def patch_source(text):
    text = replace_once(text, '#include "validation.h"', '#include "validation.h"\n#include "macos_precision.h"')
    text = replace_once(text, "struct whisper_state {\n", "struct whisper_state {\n    std::unique_ptr<WhisperMacPrecision> precision;\n")
    start = text.index("static struct ggml_cgraph * whisper_build_graph_encoder(")
    end = text.index("// pre-compute cross-attention memory", start)
    graph = text[start:end]
    anchor = "    const auto & model   = wctx.model;"
    graph = replace_once(graph, anchor, """    if (whisper_macos_precision_enabled() && !wstate.precision) {
        if (wctx.params.use_gpu || wstate.ctx_aneforge || wctx.model.hparams.n_audio_layer != 4 ||
            wctx.model.hparams.n_audio_state != 384) GGML_ABORT("precision encoder requires tiny.en and CPU host");
        wstate.precision.reset(new WhisperMacPrecision);
    }
    auto project = [&](ggml_tensor * weight, ggml_tensor * input, int layer, const char * name) {
        return wstate.precision ? wstate.precision->project(ctx0, weight, input, layer, name)
                                : ggml_mul_mat(ctx0, weight, input);
    };
""" + anchor)
    # ctx0 exists only after graph metadata allocation; place the lambda there.
    begin = graph.index("    auto project =")
    finish = graph.index("    const auto & model", begin)
    project = graph[begin:finish]
    graph = graph[:begin]+graph[finish:]
    graph = replace_once(graph, "    struct ggml_context * ctx0 = ggml_init(params);\n", "    struct ggml_context * ctx0 = ggml_init(params);\n" + project)
    for matrix, name in zip(("attn_q_w", "attn_k_w", "attn_v_w", "attn_ln_1_w", "mlp_0_w", "mlp_1_w"), PROJECTIONS):
        before = "ggml_mul_mat(ctx0,\n                    layer." + matrix + ",\n                    cur)"
        graph = replace_once(graph, before, f'project(layer.{matrix}, cur, il, "{name}")')
    text = text[:start]+graph+text[end:]
    return replace_once(text, '    whisper_trace_tensor("encoder.f32", wstate.embd_enc);',
        '    if (wstate.precision) wstate.precision->finish_encoder();\n'
        '    whisper_trace_tensor("encoder.f32", wstate.embd_enc);')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--programs", type=Path, required=True, help="Existing paired-host-fp32/projections directory")
    parser.add_argument("--receipt", type=Path, required=True, help="Passing paired Python precision report")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        parser.error("requires pinned whisper.cpp revision " + REVISION)
    receipt = json.loads(args.receipt.read_text())
    variant, = [v for v in receipt["variants"] if v["name"] == "paired-host-fp32"]
    counts = {"jfk":25, "jfk-first-5s":8, "jfk-repeat":47}
    if (receipt["checkpoint_sha256"] != CHECKPOINT_SHA256 or receipt["full_logit_gate_nrmse"] != .005 or
            {c["audio"]:c["vectors"] for c in variant["clips"]} != counts):
        parser.error("requires the complete pinned 80-vector paired accuracy receipt")
    for clip in variant["clips"]:
        if len(clip["logit_checks"]) != counts[clip["audio"]] or any(
                not math.isfinite(c["nrmse"]) or c["nrmse"] >= .005 or not c["argmax_match"]
                for c in clip["logit_checks"]):
            parser.error("paired Python program receipt fails the unchanged full-logit gate")
    records = {p["name"]:p for p in variant["projections"]}
    weights = checkpoint_weights(args.checkpoint)
    if len(records) != 24 or set(records) != set(weights):
        parser.error("requires all 24 paired projection receipts")
    manifest = dict(checkpoint_sha256=CHECKPOINT_SHA256, device_mask=4, gains=[1.,1.375],
                    positions=1500, contraction_partitions=2, temporal_planes=4, programs=[])
    for name, weight in weights.items():
        path = args.programs / name
        if digest(path / "model.mil") != records[name]["mil_sha256"]:
            raise ValueError("paired MIL differs from accuracy receipt: " + name)
        verify_blob(path / "weights.bin", weight)
        n, k = weight.shape
        manifest["programs"].append(dict(name=name, input_features=k, output_features=n,
            file_sha256={filename:digest(path/filename) for filename in ("model.mil", "weights.bin")}))
    original = subprocess.check_output(["git", "-C", str(source), "show", "HEAD:src/whisper.cpp"], text=True)
    before = instrument_patch(accuracy_patch(original))
    after = patch_source(before)
    cpp = source / "src/whisper.cpp"
    if cpp.read_text() not in (before, after):
        parser.error("requires an unmodified shared Mac FP32 preparation")
    cmake = source / "src/CMakeLists.txt"
    cmake_before = cmake.read_text()
    line = '\ntarget_sources(whisper PRIVATE "${ANE_ROOT}/whisper/macos_precision.cpp")\n'
    if 'macos_precision.cpp' in cmake_before and not cmake_before.endswith(line):
        parser.error("unknown native precision build edits")
    args.output.mkdir(parents=True, exist_ok=False)
    for record in manifest["programs"]:
        dst = args.output / record["name"]
        dst.mkdir()
        (dst / "native-cache").mkdir()
        for filename in ("model.mil", "weights.bin"):
            shutil.copyfile(args.programs / record["name"] / filename, dst / filename)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    validate_programs(args.output, args.checkpoint)
    if cpp.read_text() != after:
        cpp.write_text(after)
    if not cmake_before.endswith(line):
        cmake.write_text(cmake_before+line)
    print("Staged 24 checkpoint-verified paired projection programs:", args.output)


if __name__ == "__main__":
    main()
