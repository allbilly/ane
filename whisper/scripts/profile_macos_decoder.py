"""Verify diagnostic labels preserve all native arrays, then sample three clips."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess

from whisper.scripts.benchmark_macos import parse_runs
from whisper.scripts.profile_macos_cpu import run as sample_cpu
from whisper.scripts.summarize_cpu_ablations import completed_validation
from whisper.validation import digest, hardware_locks, logits_records, words

ROOT = Path(__file__).resolve().parents[1]


def vocabulary_profiles(log, positions):
    rows = [json.loads(line.split("\t",1)[1]) for line in log.splitlines() if line.startswith("VOCABULARY_PROFILE\t")]
    if len(rows) != len(positions) or [r["m"] for r in rows] != positions:
        raise ValueError("missing vocabulary prompt/token execution evidence")
    for row in rows:
        if ((row["n"],row["k"],row["threads"]) != (51864,384,4) or
                (row["weight_type"],row["input_type"],row["output_type"]) != ("f16","f32","f32") or
                row["phase"] != ("prompt" if row["m"] > 1 else "token")):
            raise ValueError("unexpected vocabulary projection geometry or arithmetic")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build","source","reference","model","payloads","dylib","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--runs",type=int,default=200)
    parser.add_argument("--seconds",type=int,default=10)
    args = parser.parse_args()
    raw = completed_validation(args.reference)
    if (raw["encoder_backend"] != "complete" or digest(args.model) != raw["model_sha256"] or
            digest(args.dylib) != raw["runtime_sha256"] or min(args.runs,args.seconds) < 1 or args.seconds > 15):
        parser.error("requires the complete matched Mac reference and existing model/runtime")
    for clip in raw["correctness"]:
        if any(v["cpu"]["nrmse"] >= .005 or not v["cpu_argmax_match"] for v in clip["hf_logit_checks"]):
            parser.error("CPU reference did not pass every independent HF logit vector")
    for name,checksum in raw["payload_sha256"].items():
        filename = "pos.f16" if name == "positions" else name+".bin"
        if digest(args.payloads/filename) != checksum:
            parser.error("complete encoder payload changed: "+name)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    report = dict(status="RUNNING",utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reference_summary_sha256=digest(args.reference/"summary.json"),full_logit_gate_nrmse=.005,
        numerical_failures=raw["numerical_failures"],correctness=raw["correctness"],
        scope="Diagnostic stack attribution; original ANE numerical failures retained. Native trace files must remain byte identical to the shared reference.",
        boundaries=[],profiles={})
    env = os.environ.copy()
    for key in ("ANEFORGE_ENCODER","ANEFORGE_DYLIB","WHISPER_MACOS_PRECISION","WHISPER_FUSED_CROSS_KV",
                "WHISPER_ENCODER_BLAS_ATTENTION","WHISPER_PROFILE_ENCODER_ATTENTION","WHISPER_PROFILE_INPUT"):
        env.pop(key,None)
    env.update(WHISPER_PROFILE_VOCABULARY="1",OPENBLAS_NUM_THREADS="1",OMP_WAIT_POLICY="PASSIVE")
    try:
        with hardware_locks():
            for clip in raw["correctness"]:
                positions = [len(tokens) for tokens,_ in logits_records(args.reference/(clip["audio"]+"-cpu")/"logits.bin")]
                for backend in ("cpu","ane"):
                    path = output/(clip["audio"]+"-"+backend)
                    path.mkdir()
                    run_env = dict(env,WHISPER_TRACE=str(path))
                    if backend == "ane":
                        run_env.update(ANEFORGE_ENCODER=str(args.payloads.resolve()),ANEFORGE_DYLIB=str(args.dylib.resolve()))
                    command = [str((args.build/"bin/whisper-cli").resolve()),"-m",str(args.model.resolve()),
                        "-f",str((args.reference/(clip["audio"]+".wav")).resolve()),"-l","en","-t","4",
                        "-bs","1","-bo","1","-tp","0","-nf","-nt","-ng"]
                    process = subprocess.run(command,env=run_env,capture_output=True,text=True,timeout=120)
                    (path/"run.log").write_text(process.stdout+"\n"+process.stderr)
                    process.check_returncode()
                    metadata = vocabulary_profiles(process.stderr,positions)
                    hashes = {}
                    for name in ("mel.f32","encoder.f32","logits.bin"):
                        hashes[name] = digest(path/name)
                        if hashes[name] != digest(args.reference/path.name/name):
                            raise ValueError("diagnostic labels changed native bytes: "+str(path/name))
                    if len(logits_records(path/"logits.bin")) != clip["decoder_calls"]:
                        raise ValueError("diagnostic verification lost full vectors")
                    report["boundaries"].append(dict(audio=clip["audio"],backend=backend,sha256=hashes,
                        decoder_calls=clip["decoder_calls"],vocabulary=metadata))
                print(clip["audio"]+": all CPU/ANE native arrays byte identical",flush=True)
        libdir = (args.build/"bin").resolve()
        driver = libdir/"benchmark-whisper"
        command = ["/usr/bin/c++","-std=c++17","-O2",str(ROOT/"scripts/benchmark_whisper.cpp"),
            "-I"+str(args.source/"include"),"-I"+str(args.source/"ggml/include"),"-L"+str(libdir),
            "-Wl,-rpath,"+str(libdir),"-lwhisper","-lggml","-lggml-cpu","-lggml-base","-o",str(driver)]
        subprocess.run(command,check=True,capture_output=True,text=True)
        for clip in raw["correctness"]:
            positions = [len(tokens) for tokens,_ in logits_records(args.reference/(clip["audio"]+"-cpu")/"logits.bin")]
            directory = output/(clip["audio"]+"-sample")
            sample_cpu(argparse.Namespace(build=args.build,model=args.model,
                pcm=args.reference/(clip["audio"]+".f32"),payloads=args.payloads,dylib=args.dylib,
                output=directory,runs=args.runs,seconds=args.seconds,vocabulary=True))
            log = (directory/"driver.log").read_text()
            records = parse_runs(subprocess.CompletedProcess([],0,log,log),"ane_cpu",clip["audio_seconds"],
                words(clip["cpu_transcript"]),require_dispatch=True)
            if len(records) != args.runs+2:
                raise ValueError("sampling workload lost transcriptions")
            blocks = re.findall(r"BENCH_BEGIN\t(\w+)\t(\d+)\n(.*?)BENCH_END\t\1\t\2",log,re.S)
            for _,_,block in blocks:
                vocabulary_profiles(block,positions)
            report["profiles"][clip["audio"]] = dict(transcriptions=len(records),
                sample_sha256=digest(directory/"sample.txt"),environment=json.loads((directory/"environment.json").read_text()))
            print(clip["audio"]+": warmed sample and vocabulary geometry verified",flush=True)
        report["hashes"] = dict(model=digest(args.model),runtime=digest(args.dylib),driver=digest(driver),
            libraries={p.name:digest(p) for p in libdir.glob("*.dylib") if not p.is_symlink()})
        report["source_sha256"] = {str(p):digest(p) for p in (Path(__file__),ROOT/"scripts/prepare_decoder_profile.py",
            ROOT/"scripts/profile_macos_cpu.py",args.source/"src/whisper.cpp",args.source/"ggml/src/ggml-cpu/ggml-cpu.c")}
        report["artifacts"] = {str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file()}
        report["status"] = "PASS_DIAGNOSTIC"
    finally:
        (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print("Evidence:",output/"summary.json",flush=True)


if __name__ == "__main__":
    main()
