"""Compare encoder attention algorithms after both pass full native validation."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

import numpy as np

from whisper.attention import add_attention_profiles
from whisper.scripts.benchmark_cross_kv import compact_receipt
from whisper.scripts.benchmark_native import parse_audio_runs, write_log
from whisper.scripts.prepare_macos_precision import validate_programs
from whisper.scripts.summarize_cpu_ablations import medians
from whisper.scripts.summarize_native_precision import EXPECTED, add_profiles, collect
from whisper.validation import compare, digest, hardware_locks, logits_records

ROOT = Path(__file__).resolve().parents[1]


def profile_medians(rows):
    result = medians(rows)
    if "precision_api_ms" in rows[0]:
        result["precision_api_ms"] = {key:statistics.median(r["precision_api_ms"][key] for r in rows)
                                      for key in ("pack","dispatch","combine")}
        result["transformer_remaining_ms"] = statistics.median(r["transformer_remaining_ms"] for r in rows)
    if "attention_matrices" in rows[0]:
        result["attention_matrix_ms"] = {key:statistics.median(
            sum(m[key+"_us"] for m in r["attention_matrices"])/1000 for r in rows)
            for key in ("allocate","convert","thread_setup","gemm","total")}
    return result


def compare_algorithms(reference, validation):
    checks = []
    for audio,count in EXPECTED.items():
        for backend in ("cpu","ane"):
            left,right = (p/f"{audio}-{backend}" for p in (reference,validation))
            if digest(left/"mel.f32") != digest(right/"mel.f32"):
                raise ValueError("attention algorithm changed the mel input")
            encoder = compare(np.fromfile(left/"encoder.f32","<f4"),np.fromfile(right/"encoder.f32","<f4"))
            a,b = logits_records(left/"logits.bin"),logits_records(right/"logits.bin")
            if len(a) != count or len(b) != count:
                raise ValueError("cross-algorithm comparison lost decoder vectors")
            vectors = []
            for (tokens,logits),(actual_tokens,actual) in zip(a,b):
                np.testing.assert_array_equal(tokens,actual_tokens)
                error = compare(logits,actual)
                error["argmax_match"] = int(logits.argmax()) == int(actual.argmax())
                if error["nrmse"] >= .005 or not error["argmax_match"]:
                    raise ValueError("attention algorithm changed native logits beyond the gate")
                vectors.append(error)
            checks.append(dict(audio=audio,backend=backend,mel_sha256=digest(left/"mel.f32"),
                               encoder_error=encoder,histories_match=True,logit_checks=vectors))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True,help="Completed flash-attention native paired validation")
    parser.add_argument("--validation",type=Path,required=True,help="Completed batched-attention native paired validation")
    parser.add_argument("--model",type=Path,default=ROOT/"models/ggml-tiny.en.bin")
    parser.add_argument("--precision-programs",type=Path,required=True)
    parser.add_argument("--dylib",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--compact-receipt",type=Path,required=True)
    args = parser.parse_args()
    validated = {algorithm:collect(path) for algorithm,path in (("flash",args.reference),("blas",args.validation))}
    for algorithm,report in validated.items():
        if report["encoder_attention"] != algorithm:
            parser.error("native validation used a different encoder attention algorithm")
    flash,blas = validated.values()
    for key in ("model_sha256","hf_checkpoint_sha256","precision_manifest_sha256","runtime_sha256",
                "binary_sha256","driver_sha256","library_sha256"):
        if flash[key] != blas[key]:
            parser.error("validation identity changed: "+key)
    build = args.build.resolve()
    for filename,checksum in blas["library_sha256"].items():
        if digest(build/"bin"/filename) != checksum:
            parser.error("native library changed since validation: "+filename)
    driver = build/"bin/benchmark-whisper"
    if (digest(driver) != blas["driver_sha256"] or digest(args.model) != blas["model_sha256"] or
            digest(args.dylib) != blas["runtime_sha256"] or
            digest(args.precision_programs/"manifest.json") != blas["precision_manifest_sha256"]):
        parser.error("native driver/model/runtime/programs changed since validation")
    validate_programs(args.precision_programs,ROOT/"models/hf-tiny.en/model.safetensors")
    checks = compare_algorithms(args.reference,args.validation)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    report = dict(status="RUNNING",timings_accepted=False,host_backend="macos",full_logit_gate_nrmse=.005,
        encoder_cpu_gate_nrmse=.005,hf_encoder_gate_cosine=.999,
        correctness={algorithm:data["correctness"] for algorithm,data in validated.items()},
        cross_algorithm_checks=checks,
        validation_summary_sha256={algorithm:data["validation_summary_sha256"] for algorithm,data in validated.items()},
        method="Same native binary/libraries; two persistent contexts per algorithm/backend, two excluded warmups and ten measures per clip per round. Algorithm and CPU/ANE order reversed in round two; four workers.",
        limitations="Active M1 desktop and unfixed clocks; Apple BLAS. Asahi execution/performance and wider audio accuracy remain unverified. Decoder flash attention is unchanged.",
        configurations={algorithm:{kind:{} for kind in ("cpu_cpu","ane_precision_cpu")} for algorithm in validated})
    for key in ("model_sha256","hf_checkpoint_sha256","precision_manifest_sha256","runtime_sha256",
                "binary_sha256","driver_sha256","library_sha256","precision_programs"):
        report[key] = blas[key]
    env = os.environ.copy()
    for key in ("ANEFORGE_ENCODER","ANEFORGE_DYLIB","WHISPER_MACOS_PRECISION","WHISPER_ASAHI_ANE",
                "WHISPER_ASAHI_ENCODER","WHISPER_TRACE","WHISPER_ASAHI_TRACE","WHISPER_ASAHI_PROFILE",
                "WHISPER_PROFILE_INPUT","WHISPER_FUSED_CROSS_KV","WHISPER_ENCODER_BLAS_ATTENTION",
                "WHISPER_PROFILE_ENCODER_ATTENTION"):
        env.pop(key,None)
    env.update(WHISPER_PROFILE="1",WHISPER_PROFILE_MATMUL="1",OPENBLAS_NUM_THREADS="1",OMP_WAIT_POLICY="PASSIVE")
    pcm = [str((args.validation/(name+".f32")).resolve()) for name in EXPECTED]
    try:
        with hardware_locks():
            for round_index in range(2):
                algorithms = ("flash","blas") if round_index == 0 else ("blas","flash")
                modes = (0,3) if round_index == 0 else (3,0)
                for algorithm in algorithms:
                    for mode in modes:
                        kind = "ane_precision_cpu" if mode else "cpu_cpu"
                        run_env = dict(env)
                        if algorithm == "blas":
                            run_env.update(WHISPER_ENCODER_BLAS_ATTENTION="1",WHISPER_PROFILE_ENCODER_ATTENTION="1")
                        if mode:
                            run_env.update(WHISPER_MACOS_PRECISION=str(args.precision_programs.resolve()),
                                           ANEFORGE_DYLIB=str(args.dylib.resolve()))
                        result = subprocess.run([str(driver),str(args.model.resolve()),pcm[0],"0","2","10",*pcm[1:]],
                            env=run_env,capture_output=True,text=True,timeout=180)
                        write_log(output/f"{algorithm}-{kind}-{round_index+1}.log",result)
                        result.check_returncode()
                        clips = parse_audio_runs(result,mode,blas["correctness"],1779,"macos")
                        add_profiles(clips,result.stderr,bool(mode))
                        add_attention_profiles(clips,result.stderr,algorithm == "blas")
                        for audio,rows in clips.items():
                            if len(rows) != 12 or sum(r["phase"] == "measure" for r in rows) != 10:
                                raise ValueError("paired warm run count changed")
                            for row in rows:
                                row["round"] = round_index+1
                                if len(row.get("cross_kv_matrices",[])) != 8:
                                    raise ValueError("cross-K/V execution evidence missing")
                            report["configurations"][algorithm][kind].setdefault(audio,[]).extend(rows)
                        print(f"round {round_index+1}: {algorithm} {kind} captured",flush=True)
        for kinds in report["configurations"].values():
            for clips in kinds.values():
                for audio,rows in list(clips.items()):
                    measured = [r for r in rows if r["phase"] == "measure"]
                    clips[audio] = dict(runs=measured,warmups=[r for r in rows if r["phase"] == "warmup"],
                        median=profile_medians(measured),round_medians={str(i):profile_medians([r for r in measured if r["round"] == i]) for i in (1,2)})
        report["source_sha256"] = {str(p):digest(p) for p in (Path(__file__),ROOT/"attention.py",
            ROOT/"validation.h",ROOT/"scripts/prepare_encoder_attention.py",ROOT/"scripts/summarize_native_precision.py")}
        report["artifacts"] = {p.name:digest(p) for p in output.iterdir() if p.is_file()}
        report.update(status="PASS",timings_accepted=True)
        with args.compact_receipt.open("x") as stream:
            stream.write(json.dumps(compact_receipt(report),indent=2)+"\n")
    except Exception as error:
        report.update(status="ERROR",error=str(error))
        raise
    finally:
        (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print("Evidence:",output/"summary.json",flush=True)


if __name__ == "__main__":
    main()
