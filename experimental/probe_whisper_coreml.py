"""Convert the cached trained Whisper encoder and test Core ML routes on real audio."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import wave


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import coremltools as ct
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    torch.set_num_threads(4)
    hf = WhisperForConditionalGeneration.from_pretrained(a.model, attn_implementation="eager").eval()
    processor = WhisperProcessor.from_pretrained(a.model)
    with wave.open(str(a.audio)) as audio:
        signal = np.frombuffer(audio.readframes(audio.getnframes()), "<i2").astype(np.float32) / 32768
    mel = processor(signal, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad(): reference = hf.model.encoder(mel).last_hidden_state.numpy()
    np.savez_compressed(a.output / "jfk-encoder.npz", logmel_data=mel.numpy(), hf_output=reference)
    report = dict(model=str(a.model), routes=[], status="running",
                  versions=dict(coremltools=ct.__version__, torch=torch.__version__), temporary_directory=os.environ.get("TMPDIR"),
                  checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.model.iterdir() if p.suffix in (".bin", ".safetensors")})
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    class Encoder(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        def forward(self, features):
            return self.encoder(features, return_dict=False)[0]
    traced = torch.jit.trace(Encoder(hf.model.encoder).eval(), mel, check_trace=False)
    package = a.output / "encoder.mlpackage"
    converted = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13,
                           inputs=[ct.TensorType(name="logmel_data", shape=(1, 80, 3000), dtype=np.float32)],
                           outputs=[ct.TensorType(name="output", dtype=np.float32)],
                           compute_precision=ct.precision.FLOAT16, skip_model_load=True)
    converted.save(package)
    report["package"] = str(package)
    try:
        compiled = Path(ct.models.utils.compile_model(str(package)))
    except Exception as error:
        report.update(status="compile_failed", error=str(error), hardware_execution="not reached")
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    # Place the compiled directory beside a link to the exact GGML checkpoint,
    # following whisper.cpp's encoder filename convention.
    import shutil
    compiled_target = a.output / "ggml-tiny.en-encoder.mlmodelc"
    shutil.copytree(compiled, compiled_target)
    reports = []
    saved_outputs = {}
    for label, units in [("cpu", ct.ComputeUnit.CPU_ONLY), ("cpu_gpu", ct.ComputeUnit.CPU_AND_GPU),
                         ("cpu_ane", ct.ComputeUnit.CPU_AND_NE), ("all", ct.ComputeUnit.ALL)]:
        try:
            runtime = ct.models.CompiledMLModel(str(compiled_target), compute_units=units)
            for _ in range(2): runtime.predict({"logmel_data": mel.numpy()})
        except Exception as error:
            reports.append(dict(route=label, status="runtime_failed", error=str(error), passes_encoder_cosine=False))
            continue
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            output = runtime.predict({"logmel_data": mel.numpy()})["output"]
            samples.append((time.perf_counter() - start) * 1000)
        actual = np.asarray(output, np.float64).ravel()
        saved_outputs["output_" + label] = np.asarray(output).copy()
        expected = reference.astype(np.float64).ravel()
        cosine = float(actual @ expected / (np.linalg.norm(actual) * np.linalg.norm(expected)))
        row = dict(route=label, prediction_ms=samples, median_ms=float(np.median(samples)), cosine=cosine,
                   normalized_rmse=float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
                   passes_encoder_cosine=cosine > .999)
        try:
            plan = ct.models.compute_plan.MLComputePlan.load_from_path(str(compiled_target), compute_units=units)
            counts = {}
            def visit(block):
                for op in block.operations:
                    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    if usage:
                        name = type(usage.preferred_compute_device).__name__
                        counts[name] = counts.get(name, 0) + 1
                    for child in op.blocks: visit(child)
            for function in plan.model_structure.program.functions.values(): visit(function.block)
            row["preferred_device_counts"] = counts
        except Exception as error:
            row["compute_plan_error"] = str(error)
        reports.append(row)
        print(json.dumps(row), flush=True)
        del runtime
    np.savez_compressed(a.output / "jfk-encoder.npz", logmel_data=mel.numpy(), hf_output=reference, **saved_outputs)
    report.update(model=str(a.model), routes=reports, package=str(package), compiled=str(compiled_target),
                  checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.model.iterdir() if p.suffix in (".bin", ".safetensors")},
                  timing_boundary="Python synchronous encoder prediction including API I/O; excludes compilation/load/warmup",
                  placement_note="Preferred-device counts are static placement, not engine utilization",
                  status="pass" if all(row["passes_encoder_cosine"] for row in reports) else "failed")
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "pass": raise SystemExit(1)


if __name__ == "__main__":
    main()
