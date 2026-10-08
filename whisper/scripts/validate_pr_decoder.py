"""Check multilingual PR replay and native decoding on frozen local CPU histories.

This does not substitute local mel for the missing exact Mac speech fixtures.
The independent HF implementation uses every stored GGML weight in FP32.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import wave

import numpy as np

from whisper.ggml_encoder_weights import reference_model
from whisper.pr_encoder_kernel import reconstruct
from whisper.pr_encoder_replay import prepare, native_descriptor
from whisper.validation import compare, digest, hardware_locks, logits_records
from whisper.scripts.benchmark_native import write_log

ROOT = Path(__file__).resolve().parents[2]
GATE = .005


def runtime(log, dimensions, tasks, ane):
    if not re.search(r"use gpu\s*=\s*0", log):
        raise ValueError("CPU host configuration evidence missing")
    ready = f"ASAHI_PR_ANE ready: state={dimensions['state']} layers={dimensions['layers']} tasks={tasks}"
    stages = re.findall(r"ASAHI_PR_PROFILE encoder: .* submissions=1 read_workers=(\d+)", log)
    if ane:
        if ready not in log or stages != ["4"]:
            raise ValueError("missing complete native PR task/readback execution")
    elif stages or "ASAHI_PR_ANE ready:" in log:
        raise ValueError("CPU reference unexpectedly executed ANE")


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model=args.model, full_logit_gate_nrmse=GATE, records=[], timings_accepted=False,
        scope="Independent HF FP32 arithmetic loaded from the matching stored GGML weights; native CPU and exported ANE encoder/CPU decoder on identical local mel and frozen CPU token histories.",
        cross_host_scope="Exact Mac speech inputs/output arrays and E5RT executable identity are absent; this local gate does not qualify cross-host replay.")
    try:
        checkpoint = ROOT/f"whisper/models/ggml-{args.model}.bin"
        kernels = ROOT/f"whisper/kernels/pr3905/{args.model}"
        meta, hwx, positions = reconstruct(checkpoint, kernels)
        plan, payloads = prepare(meta, hwx, positions)
        if (args.payloads/"native-layout.txt").read_text() != native_descriptor(plan, payloads):
            raise ValueError("native PR layout differs from captured checkpoint/template")
        for name, data in payloads.items():
            if (args.payloads/(name+".bin")).read_bytes() != data:
                raise ValueError("native PR payload changed: " + name)
        report.update(checkpoint_sha256=digest(checkpoint), hwx_sha256=meta["hwx_sha256"],
                      task_count=plan["td_count"], payload_sha256=plan["payloads"],
                      position_sha256=meta["position_sha256"])
        del hwx, positions, payloads
        cache = args.build/"CMakeCache.txt"
        source = Path(re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cache.read_text(), re.M)[1])
        text = (source/"src/whisper.cpp").read_text()
        if ('ggml_type itype = ggml_type::GGML_TYPE_F32;' not in text or
            'ggml_gelu(ctx0, cur)' in text or 'n_state_head, n_audio_ctx_pad, n_head' in text):
            raise ValueError("requires shared --encoder pr --precision fp32 preparation")
        paths = [Path(__file__), ROOT/"whisper/ggml_encoder_weights.py", ROOT/"whisper/validation.py",
                 ROOT/"whisper/asahi_full_encoder.cpp", ROOT/"whisper/scripts/probe_encoder_logits.cpp",
                 ROOT/"whisper/scripts/prepare_native.py", ROOT/"whisper/scripts/prepare_pr_asahi.py",
                 ROOT/"whisper/native.py", kernels/"meta.json", source/"src/whisper.cpp",
                 source/"src/CMakeLists.txt", source/"ggml/src/ggml-cpu/simd-mappings.h",
                 source/"ggml/src/ggml-cpu/llamafile/sgemm.cpp", source/"ggml/src/ggml-cpu/simd-gemm.h"]
        report["source_sha256"] = {str(p):digest(p) for p in paths}
        report["build_cache_sha256"] = digest(cache)
        libdir = args.build.resolve()/"bin"
        probe = libdir/"probe-encoder-logits"
        compiled = subprocess.run(["c++", "-std=c++17", "-O3", str(ROOT/"whisper/scripts/probe_encoder_logits.cpp"),
            "-I"+str(source/"include"), "-I"+str(source/"ggml/include"), "-L"+str(libdir),
            "-Wl,-rpath,"+str(libdir), "-lwhisper", "-lggml", "-lggml-cpu", "-lggml-base",
            "-o", str(probe)], capture_output=True, text=True)
        write_log(output/"probe-build.log", compiled)
        compiled.check_returncode()
        report["probe_sha256"] = digest(probe)
        report["library_sha256"] = {p.name:digest(p) for p in libdir.glob("*.so")}
        env = os.environ.copy()
        for key in ("WHISPER_ASAHI_ANE", "WHISPER_ASAHI_ENCODER", "WHISPER_ASAHI_PRECISION", "ANEFORGE_ENCODER",
                    "ANEFORGE_DYLIB", "WHISPER_MACOS_PRECISION", "WHISPER_TRACE", "WHISPER_ASAHI_TRACE"):
            env.pop(key, None)
        env.update(WHISPER_PROFILE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4",
                   OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE")
        report["thread_environment"] = {k:env[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_WAIT_POLICY")}
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        with wave.open(str(ROOT/"whisper/vendor/whisper.cpp/samples/jfk.wav")) as wav:
            if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getnframes()) != (1,16000,2,176000):
                raise ValueError("requires pinned 11-second JFK PCM16 audio")
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").copy()
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        model, dimensions = reference_model(checkpoint)
        report["reference_dimensions"] = dimensions
        for name, samples in (("jfk",pcm), ("jfk-first-5s",pcm[:80000]),
                              ("jfk-repeat",np.concatenate((pcm,np.zeros(16000,"<i2"),pcm)))):
            audio = output/(name+".wav")
            with wave.open(str(audio),"wb") as wav:
                wav.setparams((1,2,16000,len(samples),"NONE","not compressed"))
                wav.writeframes(samples.tobytes())
            cpu = output/(name+"-cpu"); cpu.mkdir()
            result = subprocess.run([str(libdir/"whisper-cli"), "-m", str(checkpoint), "-f", str(audio),
                "-l", "en", "-t", "4", "-bs", "1", "-bo", "1", "-tp", "0", "-nf", "-nt", "-ng",
                "-otxt", "-of", str(cpu/"transcript")],
                env=dict(env, WHISPER_TRACE=str(cpu)), capture_output=True, text=True, timeout=300)
            write_log(cpu/"run.log",result); result.check_returncode()
            runtime(result.stderr, meta["dimensions"], plan["td_count"], False)
            records = logits_records(cpu/"logits.bin", vocabulary=51865)
            if records[0][0].size != 4 or any(tokens.size != 1 for tokens,_ in records[1:]):
                raise ValueError("expected one complete English prompt followed by single-token histories")
            tokens = np.concatenate([tokens for tokens,_ in records]).astype("<i4")
            token_path = output/(name+"-tokens.i32"); tokens.tofile(token_path)
            native_features = []
            for suffix in ("ane", "ane-repeat"):
                folder = output/(name+"-"+suffix); folder.mkdir()
                result = subprocess.run([str(probe), str(checkpoint), str(cpu/"mel.f32"), str(token_path),
                                         str(folder/"probe-logits.bin")],
                    env=dict(env, WHISPER_TRACE=str(folder), WHISPER_ASAHI_ENCODER=str(args.payloads.resolve())),
                    capture_output=True, text=True, timeout=300)
                write_log(folder/"run.log",result); result.check_returncode()
                runtime(result.stderr, meta["dimensions"], plan["td_count"], True)
                if (folder/"probe-logits.bin").read_bytes() != (folder/"logits.bin").read_bytes():
                    raise ValueError("probe and shared native logit trace disagree")
                (folder/"probe-logits.bin").unlink()
                native_features.append(folder)
            ane, repeated = native_features
            for filename in ("mel.f32", "encoder.f32", "logits.bin"):
                if (ane/filename).read_bytes() != (repeated/filename).read_bytes():
                    raise ValueError("native PR replay is not bitwise repeatable: " + name+"/"+filename)
                # Retain both capture paths without storing identical arrays twice.
                (repeated/filename).unlink()
                os.link(ane/filename, repeated/filename)
            if (cpu/"mel.f32").read_bytes() != (ane/"mel.f32").read_bytes():
                raise ValueError("CPU and ANE mel inputs differ")
            mel = np.fromfile(cpu/"mel.f32", "<f4").reshape(1,80,3000)
            width = dimensions["audio_state"]
            features = {"cpu":torch.from_numpy(np.fromfile(cpu/"encoder.f32", "<f4").reshape(1,1500,width)),
                        "ane":torch.from_numpy(np.fromfile(ane/"encoder.f32", "<f4").reshape(1,1500,width))}
            with torch.no_grad():
                features["hf"] = model.model.encoder(torch.from_numpy(mel)).last_hidden_state
            native = {"cpu":records,"ane":logits_records(ane/"logits.bin", vocabulary=51865)}
            if len(native["ane"]) != len(records):
                raise ValueError("frozen native decoder vector count differs")
            checks, history = [], []
            for index,(batch,_) in enumerate(records):
                history.extend(batch.tolist())
                np.testing.assert_array_equal(batch, native["ane"][index][0])
                values = {kind:native[kind][index][1] for kind in native}
                with torch.no_grad():
                    for kind, feature in features.items():
                        hidden = model.model.decoder(input_ids=torch.tensor([history]),
                            encoder_hidden_states=feature, use_cache=False).last_hidden_state
                        values["hf_"+kind] = model.proj_out(hidden[:,-1,:]).numpy().ravel()
                comparisons = {}
                for kind in ("cpu", "ane", "hf_cpu", "hf_ane"):
                    error = compare(values["hf_hf"], values[kind])
                    error["argmax_match"] = int(values["hf_hf"].argmax()) == int(values[kind].argmax())
                    comparisons[kind] = error
                checks.append(dict(index=index, history_length=len(history), comparisons=comparisons))
            passed = all(v["nrmse"] < GATE and v["argmax_match"] for c in checks for v in c["comparisons"].values())
            row = dict(audio=name, status="PASS_LOCAL_GATE" if passed else "FAIL_LOCAL_GATE", vectors=len(records),
                checkpoint_sha256=report["checkpoint_sha256"], audio_sha256=digest(audio),
                mel_sha256=digest(cpu/"mel.f32"), frozen_tokens_sha256=digest(token_path),
                frozen_cpu_logit_trace_sha256=digest(cpu/"logits.bin"), repeat_bitwise_equal=True,
                all_fixed_histories_match=True, logit_checks=checks,
                maximum_logit_nrmse={k:max(c["comparisons"][k]["nrmse"] for c in checks) for k in checks[0]["comparisons"]},
                raw_argmax_matches={k:sum(c["comparisons"][k]["argmax_match"] for c in checks) for k in checks[0]["comparisons"]},
                encoder_errors={k:compare(features["hf"].numpy(), features[k].numpy()) for k in ("cpu","ane")})
            report["records"].append(row)
            print(name, row["status"], "vectors", row["vectors"], "maximum NRMSE", row["maximum_logit_nrmse"], flush=True)
        report["artifacts"] = {str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file()}
        if any(digest(Path(p)) != h for p,h in report["source_sha256"].items()):
            raise ValueError("validation sources changed during the run")
        report["status"] = "PASS_LOCAL_GATE" if all(r["status"] == "PASS_LOCAL_GATE" for r in report["records"]) else "FAIL_LOCAL_GATE"
    except Exception as error:
        report.update(status="ERROR", error=str(error))
        raise
    finally:
        (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print("Evidence:", output/"summary.json",flush=True)
    return 0 if report["status"] == "PASS_LOCAL_GATE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tiny","base","small"), required=True)
    parser.add_argument("--build", type=Path, default=ROOT/"whisper/build/pr3905-asahi-accuracy")
    parser.add_argument("--payloads",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    with hardware_locks():
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
