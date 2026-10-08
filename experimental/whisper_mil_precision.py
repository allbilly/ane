"""Measure mean/GELU corrections to the original MIL with the shared HF decoder."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

import numpy as np

from whisper.encoder_kernel import reconstruct, write_bundle
from whisper.validation import compare, digest, hardware_locks, logits_records

ROOT = Path(__file__).resolve().parents[1]


def rewrite_mil(source, mean_matmul=False, exact_gelu=False):
    """Preserve original operations except the explicitly selected corrections."""
    counts = dict(mean_matmul=0, exact_gelu=0)
    if mean_matmul:
        constants = '''
        tensor<int32,[4]> precision_perm = const()[name=string("precision_perm"), val=tensor<int32,[4]>([0,2,3,1])];
        tensor<int32,[4]> precision_inverse = const()[name=string("precision_inverse"), val=tensor<int32,[4]>([0,3,1,2])];
        bool precision_true = const()[name=string("precision_true"), val=bool(true)];
        bool precision_false = const()[name=string("precision_false"), val=bool(false)];
        tensor<fp16,[1,384]> precision_mean = const()[name=string("precision_mean"), val=tensor<fp16,[1,384]>(BLOBFILE(path=string("@model_path/reduction-weights.bin"), offset=uint64(64)))];
'''
        header = re.search(r"func main<[^>]+>\([^\n]+\) \{\n", source)
        if header is None:
            raise ValueError("unexpected original MIL function signature")
        source = source[:header.end()] + constants + source[header.end():]
        pattern = r'        tensor<fp16,\s*\[1,1,1,1500\]> (\w+) = reduce_mean\(axes=(\w+), keep_dims=(\w+), x=(\w+)\)\[name=string\("\1"\)\];'

        def mean(match):
            name, axis, _, operand = match.groups()
            if not re.search(rf'{axis} = const\(\).*?val=tensor<int32,\[1\]>\(\[1\]\)', source):
                raise ValueError("mean rewrite only supports the original 384-channel reductions")
            return f'''        tensor<fp16,[1,1,1500,384]> {name}_tr = transpose(perm=precision_perm, x={operand})[name=string("{name}_tr")];
        tensor<fp16,[1,1,1500,1]> {name}_mm = matmul(transpose_x=precision_false, transpose_y=precision_true, x={name}_tr, y=precision_mean)[name=string("{name}_mm")];
        tensor<fp16,[1,1,1,1500]> {name} = transpose(perm=precision_inverse, x={name}_mm)[name=string("{name}")];'''

        source, counts["mean_matmul"] = re.subn(pattern, mean, source)
        if counts["mean_matmul"] != 18 or "= reduce_mean(" in source:
            raise ValueError("requires all 18 original mean/variance reductions")
    if exact_gelu:
        pattern = r'        (tensor<fp16, \[[\d, ]+\]>) (\w+) = gelu\(x = (\w+), mode = (\w+)\)\[name = string\("\2"\)\];'

        def gelu(match):
            tensor, name, operand, _ = match.groups()
            reciprocal = float(np.float16(2 ** -.5)).hex()
            return f'''        fp16 {name}_scale = const()[name=string("{name}_scale"), val=fp16({reciprocal})];
        fp16 {name}_one = const()[name=string("{name}_one"), val=fp16(1.0)];
        fp16 {name}_half = const()[name=string("{name}_half"), val=fp16(0.5)];
        {tensor} {name}_z = mul(x={operand}, y={name}_scale)[name=string("{name}_z")];
        {tensor} {name}_erf = erf(x={name}_z)[name=string("{name}_erf")];
        {tensor} {name}_shift = add(x={name}_erf, y={name}_one)[name=string("{name}_shift")];
        {tensor} {name}_factor = mul(x={name}_shift, y={name}_half)[name=string("{name}_factor")];
        {tensor} {name} = mul(x={operand}, y={name}_factor)[name=string("{name}")];'''

        source, counts["exact_gelu"] = re.subn(pattern, gelu, source)
        if counts["exact_gelu"] != 6 or "= gelu(" in source:
            raise ValueError("requires the six original encoder GELU operations")
    return source, counts


def run(args):
    import torch
    from transformers import WhisperForConditionalGeneration
    from aneforge._runtime import E5RT
    from aneforge._blob import BlobWriter, FP16

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    meta, payloads = reconstruct(args.hf_model / "model.safetensors")
    hf = WhisperForConditionalGeneration.from_pretrained(args.hf_model,
        local_files_only=True, attn_implementation="eager").eval()
    prepared = []
    for label, count in (("jfk", 25), ("jfk-first-5s", 8), ("jfk-repeat", 47)):
        trace = args.traces / (label + "-cpu")
        mel = np.fromfile(trace / "mel.f32", "<f4").reshape(1, 80, 3000)
        records = logits_records(trace / "logits.bin")
        if len(records) != count:
            raise ValueError("requires all 80 full native decoder histories")
        history, expected = [], []
        with torch.inference_mode():
            reference = hf.model.encoder(torch.from_numpy(mel)).last_hidden_state
            for tokens, _ in records:
                history.extend(tokens.tolist())
                hidden = hf.model.decoder(input_ids=torch.tensor([history]),
                    encoder_hidden_states=reference, use_cache=False).last_hidden_state
                expected.append(hf.proj_out(hidden[:, -1, :]).numpy().ravel())
        prepared.append((label, mel.astype("<f2").reshape(1, 80, 1, 3000), reference.numpy(), records, expected))
    report = dict(status="running", checkpoint_sha256=digest(args.hf_model / "model.safetensors"),
        full_logit_gate_nrmse=.005, variants=[],
        scope="Numerical experiments on existing native mel/history traces, with the same HF FP32 decoder. Original compact kernels remain unchanged. Native transcription must be rebenchmarked for any accepted correction.",
        timing_scope=("Entire Python encoder: CPU convolution/attention/normalization/nonlinearities and 24 paired ANE projections, including host packing and output combination."
            if args.paired else "Blocking E5RT encoder execution plus output read/conversion; input feed excluded."))
    variants = ([("paired-host-fp32", False, False)] if args.paired else
        [("original", False, False), ("mean-matmul", True, False),
         ("erf-gelu", False, True), ("mean-matmul-erf-gelu", True, True)])
    try:
        with hardware_locks():
            for name, means, gelu in variants:
                directory = args.output / name
                write_bundle(directory, meta, payloads)
                source, counts = rewrite_mil(payloads["source-mil"].decode(), means, gelu)
                (directory / "model.mil").write_text(source)
                if means:
                    blob = BlobWriter()
                    blob.add(np.full((1, 384), 1/384, "<f2").tobytes(), FP16)
                    (directory / "reduction-weights.bin").write_bytes(blob.build())
                if args.paired:
                    from whisper.precision_encoder import PairedMacEncoder
                    program = PairedMacEncoder(hf.model.encoder, args.hf_model / "model.safetensors", directory / "projections")
                else:
                    program = E5RT.compile(directory / "model.mil", cache_dir=directory / "compiled",
                        inputs={"t1":(1,80,1,3000), "t0":(1,384,1,1500)}, outputs={"t1383":(1500,384)}, device_mask=4)
                item = dict(name=name, rewrites=counts, mil_sha256=digest(directory / "model.mil"), clips=[])
                try:
                    if args.paired:
                        item["projections"] = program.receipts
                    else:
                        program.set_input("t0", np.frombuffer(payloads["positions"], "<f2").reshape(1,384,1,1500))
                    for label, mel, reference, records, expected in prepared:
                        # Paired arithmetic receives the original FP32 native mel;
                        # the original full graph retains its FP16 input boundary.
                        paired_mel = np.fromfile(args.traces / (label + "-cpu/mel.f32"), "<f4").reshape(1,80,3000)
                        if not args.paired:
                            program.set_input("t1", mel)
                        for _ in range(2):
                            program(paired_mel) if args.paired else program.execute()
                        times, hashes = [], []
                        for _ in range(args.runs):
                            start = time.perf_counter()
                            if args.paired:
                                actual = program(paired_mel)
                            else:
                                program.execute()
                                actual = program.read_output("t1383").astype(np.float32).reshape(1,1500,384)
                            times.append((time.perf_counter()-start)*1000)
                            hashes.append(hashlib.sha256(actual.tobytes()).hexdigest())
                        np.save(directory / (label + "-encoder.npy"), actual)
                        checks, history = [], []
                        with torch.inference_mode():
                            for (tokens, _), target in zip(records, expected):
                                history.extend(tokens.tolist())
                                hidden = hf.model.decoder(input_ids=torch.tensor([history]),
                                    encoder_hidden_states=torch.from_numpy(actual), use_cache=False).last_hidden_state
                                logits = hf.proj_out(hidden[:, -1, :]).numpy().ravel()
                                checks.append(dict(**compare(target, logits),
                                    argmax_match=int(np.argmax(target)) == int(np.argmax(logits))))
                        clip = dict(audio=label, vectors=len(checks), logit_checks=checks,
                            maximum_logit_nrmse=max(c["nrmse"] for c in checks),
                            all_argmaxes_match=all(c["argmax_match"] for c in checks),
                            repeat_bitwise=len(set(hashes)) == 1, encoder_vs_hf=compare(reference, actual),
                            warm_encoder_ms=times, median_encoder_ms=statistics.median(times))
                        clip["passes_logit_gate"] = all(c["nrmse"] < .005 and c["argmax_match"] for c in checks)
                        item["clips"].append(clip)
                        print(name, label, clip["maximum_logit_nrmse"], clip["passes_logit_gate"], flush=True)
                finally:
                    program.close() if args.paired else program.release()
                item["passes_all_80_vectors"] = all(c["passes_logit_gate"] for c in item["clips"])
                report["variants"].append(item)
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "measured"
    except Exception as error:
        report.update(status="error", error=str(error))
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--paired", action="store_true", help="Test all 24 paired ANE projections with FP32 HF host encoder math")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    run(args)


if __name__ == "__main__":
    main()
