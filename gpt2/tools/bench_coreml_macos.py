#!/usr/bin/env python3
"""Benchmark external more-ane-transformers GPT-2 on macOS; keep weights external.

Requires coremltools 9, torch, transformers, numpy, and safetensors. The native
Swift runner retains Core ML KV output arrays directly, avoiding Python cache
round-trips. Results are native prediction throughput, not application speed.
"""
import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
WEIGHT_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def output(*cmd):
    return subprocess.check_output(cmd, text=True).strip()


def references(weights, steps, trials, warmups, warmup_steps, directory):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(weights), local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(
        str(weights), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32, attn_implementation="eager").eval()
    cases, offset = [], 0
    raw = directory / "reference.f32"
    with raw.open("wb") as stream, torch.inference_mode():
        for prompt, count in [("Hello world", steps), ("The capital of France is", 16),
                              ("Before boarding your rocket to Mars, remember to pack these items:", 16)]:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            result = model(torch.tensor([ids]), use_cache=True)
            next_tokens = []
            for i in range(count + 1):
                logits = result.logits[0, -1].float().numpy()
                if not np.isfinite(logits).all():
                    raise ValueError("Nonfinite HF reference logits")
                logits.astype("<f4").tofile(stream)
                token = int(logits.argmax()); next_tokens.append(token)
                if i < count:
                    result = model(torch.tensor([[token]]), past_key_values=result.past_key_values,
                                   use_cache=True)
            cases.append(dict(prompt=prompt, prompt_ids=ids, next_tokens=next_tokens,
                              reference_offset=offset, steps=count))
            offset += (count + 1) * model.config.vocab_size
    trace = dict(cases=cases, vocabulary=model.config.vocab_size, trials=trials,
                 warmups=warmups, warmup_steps=warmup_steps)
    path = directory / "trace.json"
    path.write_text(json.dumps(trace) + "\n")
    del result, model
    gc.collect()
    return path, raw, trace, tokenizer


def convert(repo, weights, destination):
    """Run the upstream model and precision policy with current coremltools.

    Only the checkpoint location is overridden; upstream model sources stay intact.
    A fresh artifact is useful when the old release or custom proxy is unavailable.
    """
    import coremltools as ct
    import numpy as np
    import torch
    from transformers import GPT2LMHeadModel
    from unittest.mock import patch
    sys.path.insert(0, str(repo))
    from models.gpt2 import GPT
    original = GPT2LMHeadModel.from_pretrained
    with patch.object(GPT2LMHeadModel, "from_pretrained", side_effect=lambda *a, **kw:
                      original(str(weights), local_files_only=True, use_safetensors=True)):
        model = GPT.from_pretrained("gpt2").eval()
    torch.manual_seed(0)
    samples = model.sample_inputs()
    samples["full_sequence_length"] = torch.tensor([64], dtype=torch.int32)
    with torch.inference_mode():
        traced = torch.jit.trace(model, list(samples.values()))
    arrays = {k: v.numpy() for k, v in samples.items()}
    # Upstream convert.py keeps all layer_norm ops in fp32 for GPT-2 124M.
    converted = ct.convert(traced, inputs=[ct.TensorType(
        name=k, shape=v.shape, dtype=v.dtype,
        default_value=np.zeros(v.shape, v.dtype) if k == "kv_cache" else None)
        for k, v in arrays.items()], outputs=[ct.TensorType(name=k, dtype=np.float16)
        for k in model.output_types()],
        compute_precision=ct.transform.FP16ComputePrecision(lambda op: op.op_type != "layer_norm"),
        minimum_deployment_target=ct.target.iOS16, convert_to="mlprogram", skip_model_load=True)
    converted.user_defined_metadata["benchmark_hf_revision"] = REVISION
    converted.user_defined_metadata["benchmark_upstream_commit"] = output("git", "-C", str(repo), "rev-parse", "HEAD")
    converted.save(str(destination))
    del converted, traced, model, samples
    gc.collect()


def placement(compiled):
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.compute_device import (
        MLCPUComputeDevice, MLGPUComputeDevice, MLNeuralEngineComputeDevice)
    plan = MLComputePlan.load_from_path(str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = dict(ane=0, cpu=0, gpu=0, unknown=0)
    ops = []

    def visit(block):
        for op in block.operations:
            if op.operator_name == "const" or op.operator_name.startswith("constexpr_"):
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            device = getattr(usage, "preferred_compute_device", None)
            key = "ane" if isinstance(device, MLNeuralEngineComputeDevice) else (
                "cpu" if isinstance(device, MLCPUComputeDevice) else (
                    "gpu" if isinstance(device, MLGPUComputeDevice) else "unknown"))
            counts[key] += 1
            ops.append(dict(operator=op.operator_name, preferred_device=key))
            for child in getattr(op, "blocks", []):
                visit(child)
    for function in plan.model_structure.program.functions.values():
        visit(function.block)
    if counts["gpu"] or not counts["ane"]:
        raise ValueError(f"CPU_AND_NE plan did not establish ANE placement: {counts}")
    return dict(compute_units="CPU_AND_NE", nonconstant_operation_counts=counts,
                interpretation="Static preferred-device counts, not percentage of runtime or FLOPs",
                operations=ops)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.home() / "more-ane-transformers")
    parser.add_argument("--weights", type=Path, default=Path.home() / ".cache/huggingface/hub/models--openai-community--gpt2/snapshots" / REVISION)
    parser.add_argument("--model", type=Path, required=True, help="External .mlpackage path")
    parser.add_argument("--convert", action="store_true", help="Create a fresh external model from the unchanged upstream model code")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=16)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Requires Apple Silicon macOS")
    if min(args.steps, args.trials, args.warmups, args.warmup_steps) < 1 or args.warmup_steps > args.steps:
        parser.error("Positive counts required; warmup steps must not exceed measured steps")
    repo, weights, model, result = [p.expanduser().resolve() for p in
                                   (args.repo, args.weights, args.model, args.output)]
    log_path = result.with_suffix(".log")
    if result.exists() or log_path.exists():
        parser.error("Use new output and log paths to preserve previous measurements")
    if args.convert and model.exists():
        parser.error("--convert requires a new model path")
    if digest(weights / "model.safetensors") != WEIGHT_SHA:
        raise ValueError("Checkpoint differs from the portable GPT-2 checkpoint")
    result.parent.mkdir(parents=True, exist_ok=True)
    print("Verified cached checkpoint; creating HF float32 reference trace", flush=True)
    with tempfile.TemporaryDirectory(prefix="coreml-gpt2-") as temporary, log_path.open("w") as log:
        directory = Path(temporary)
        trace_file, raw_file, trace, tokenizer = references(
            weights, args.steps, args.trials, args.warmups, args.warmup_steps, directory)
        if args.convert:
            print("Converting upstream GPT-2 with coremltools 9; weights stay external", flush=True)
            convert(repo, weights, model)
        import coremltools as ct
        compiled = Path(str(model).removesuffix(".mlpackage") + ".mlmodelc")
        started = time.perf_counter()
        if not compiled.exists():
            ct.models.utils.compile_model(str(model), destination_path=str(compiled))
        compile_ms = (time.perf_counter() - started) * 1000
        print("Profiling the CPU+ANE compute plan", flush=True)
        plan = placement(compiled)
        swift = ROOT / "tools/bench_coreml_macos.swift"
        binary = directory / "bench"
        command = ["xcrun", "swiftc", "-O", str(swift), "-o", str(binary), "-framework", "CoreML"]
        subprocess.run(command, check=True, stdout=log, stderr=log)
        native = directory / "native.json"
        print("Running native CPU and CPU+ANE trials; warmup excluded", flush=True)
        subprocess.run([str(binary), str(compiled), str(trace_file), str(raw_file), str(native)],
                       check=True, stdout=log, stderr=log)
        report = json.loads(native.read_text())
        for backend in report["backends"].values():
            for smoke in backend["free_greedy_smoke"]:
                smoke["continuation"] = tokenizer.decode(smoke["generated_token_ids"])
        spec = ct.models.MLModel(str(model), skip_model_load=True)
        artifact_kind = ("Fresh coremltools-9 conversion from pinned HF weights"
                         if spec.user_defined_metadata.get("benchmark_hf_revision") == REVISION
                         else "External preconverted release artifact")
        report["reference"] = dict(checkpoint="openai-community/gpt2", revision=REVISION,
            weight_sha256=WEIGHT_SHA, precision="HF PyTorch float32 eager attention, KV cache",
            cases=[dict(**item, greedy_continuation=tokenizer.decode(item["next_tokens"])) for item in trace["cases"]])
        report["compute_plan"] = plan
        report["configuration"] = dict(steps=args.steps, trials_per_backend=args.trials,
            warmup_generations=args.warmups, warmup_steps=args.warmup_steps,
            cached_or_new_compile_ms_excluded=compile_ms,
            timing="Synchronous MLModel.prediction(from:) only; embeddings, attention, FFN, head, runtime dispatch included; input construction, argmax, logit checks, printing, model loading, compilation and warmup excluded",
            kv_handoff="Retain generation_kv_cache MLMultiArray from prior prediction, no Python/numpy round-trip",
            scope="Native macOS CoreML reference; not Python CLI, Orion port, or Asahi",
            gpu_allowed=False, ane_only_claim=False)
        report["provenance"] = dict(
            measured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            host=dict(chip=output("sysctl", "-n", "machdep.cpu.brand_string"),
                      model=output("sysctl", "-n", "hw.model"), memory_bytes=int(output("sysctl", "-n", "hw.memsize")),
                      macos=platform.mac_ver()[0], macos_build=output("sw_vers", "-buildVersion"), architecture=platform.machine()),
            repo_url="https://github.com/smpanaro/more-ane-transformers",
            repo_commit=output("git", "-C", str(repo), "rev-parse", "HEAD"),
            repo_status=output("git", "-C", str(repo), "status", "--short"),
            artifact_kind=artifact_kind,
            external_model=str(model), external_compiled_model=str(compiled),
            model_files_sha256={str(p.relative_to(model)): digest(p) for p in sorted(model.rglob("*")) if p.is_file()},
            upstream_model_source_sha256=digest(repo / "models/gpt2.py"),
            upstream_converter_sha256=digest(repo / "convert.py"),
            harness_sha256=digest(swift), runner_sha256=digest(Path(__file__)),
            swift_version=output("xcrun", "swiftc", "--version"), build_command=command,
            packages={name: importlib.metadata.version(name) for name in ("coremltools", "torch", "transformers", "numpy", "safetensors")},
            command_line=sys.argv)
        report["provenance"]["reference_logits_sha256"] = digest(raw_file)
        report["provenance"]["reference_trace_sha256"] = digest(trace_file)
    report["log_sha256"] = digest(log_path)
    result.write_text(json.dumps(report, indent=2) + "\n")
    for backend, data in report["backends"].items():
        stats = data["decode"]
        mismatch = sum(t["diagnostics"]["top1_mismatches"] for t in data["trials"])
        rmse = max(t["diagnostics"]["max_normalized_logit_rmse"] for t in data["trials"])
        print(f"{backend}: {stats['steps_per_second']:.2f} steps/s; p50 {stats['p50_ms']:.2f} ms; "
              f"p90 {stats['p90_ms']:.2f} ms; top1 mismatches {mismatch}; max NRMSE {rmse:.5f}")
    print(f"Compute plan: {plan['nonconstant_operation_counts']}; report: {result}")


if __name__ == "__main__":
    main()
