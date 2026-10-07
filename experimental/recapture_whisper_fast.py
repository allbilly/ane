"""Recapture the unchanged fast MIL, its native ports, and independent logit gates."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whisper/scripts"))

from experimental.capture_macos_program import export
from experimental.replay_capture import GATE
from experimental.verify_macos_capture import verify
from whisper.scripts.benchmark_asahi import compare, logits_records
from whisper.scripts.benchmark_macos import parse_runs
from whisper.encoder_kernel import require


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def checked(command, output, env=None, timeout=180):
    try:
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        def string(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        output.write_text(string(error.stdout) + "\n" + string(error.stderr) + "\nTIMEOUT\n")
        raise
    output.write_text(result.stdout + "\n" + result.stderr)
    result.check_returncode()
    return result


def capture(a):
    from aneforge._runtime import E5RT, _find_dylib
    historical = json.loads((ROOT / "whisper/results/benchmark-macos/summary.json").read_text())
    require(digest(a.source / "model.mil") == historical["mil_sha256"], "source differs from fast benchmark MIL")
    require(digest(a.hf_model / "model.safetensors") == "db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad", "wrong checkpoint")
    require(digest(ROOT / "whisper/models/ggml-tiny.en.bin") == historical["model_sha256"], "wrong native checkpoint")
    if a.output.exists():
        require(a.resume, "output exists; use --resume for an intact partial capture")
        for name in ("model.mil", "weights.bin", "pos.f16", "ports.txt"):
            require(digest(a.output / "bundle" / name) == digest(a.source / name), "partial capture source changed: " + name)
        previous = json.loads((a.output / "report.json").read_text())
    else:
        a.output.mkdir(parents=True)
        previous = {}
    bundle = a.output / "bundle"
    if not bundle.exists():
        bundle.mkdir()
        for name in ("model.mil", "weights.bin", "pos.f16", "ports.txt"):
            shutil.copy2(a.source / name, bundle / name)
    source_report = json.loads((a.fixtures / "report.json").read_text())
    rows = source_report["fixtures"]
    require(len(rows) == 3, "expected all three audio fixtures")
    for row in rows:
        for name in (row["fixture"], row["name"] + ".wav"):
            shutil.copy2(a.fixtures / name, a.output / name)
    report = dict(status="running", utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  host=platform.platform(), chip=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                  macos=subprocess.check_output(["sw_vers"], text=True).strip(),
                  hf_model=str(a.hf_model), checkpoint_sha256=digest(a.hf_model / "model.safetensors"),
                  ggml_model_sha256=digest(ROOT / "whisper/models/ggml-tiny.en.bin"),
                  whisper_library_sha256=digest(ROOT / "whisper/build/metal/bin/libwhisper.dylib"),
                  fixtures=rows, ports=source_report["ports"], source_capture=str(a.fixtures),
                  original_fast_mil_sha256=historical["mil_sha256"], runtime_sha256=digest(_find_dylib()),
                  reference_kind="macOS ANE FP16 outputs", linux_hardware_replay="pending",
                  executable_identity_limit="ANECCompile exports and E5RT execution compile the same exact MIL separately; E5RT hardware command identity is not established.",
                  compiler_binary_sha256=digest(ROOT / "gpt2/training/build/dump_hwx"))
    if (a.output / "hwx/receipt.json").exists():
        report["export"] = json.loads((a.output / "hwx/receipt.json").read_text())
        require(digest(a.output / "hwx/model.hwx") == report["export"]["hwx_sha256"], "partial HWX changed")
        require(report["export"]["mil_sha256"] == digest(bundle / "model.mil"), "partial HWX source changed")
    else:
        report["export"] = export(bundle, a.output / "hwx", ROOT / "gpt2/training/build/dump_hwx")
    require(report["export"]["status"] == "exported", "offline export failed")
    old = json.loads((a.fixtures / "hwx/receipt.json").read_text())
    report["recapture_matches_original_hwx"] = report["export"]["hwx_sha256"] == old["hwx_sha256"]
    report["recapture_matches_original_payloads"] = (
        report["export"]["text_sha256"] == old["text_sha256"] and
        [s["sha256"] for s in report["export"]["coefficient_segments"]] == [s["sha256"] for s in old["coefficient_segments"]])
    require(report["recapture_matches_original_payloads"], "recaptured command/coefficient payloads changed")
    save(a.output / "report.json", report)
    fixtures = [dict(path=r["fixture"], sha256=digest(a.output / r["fixture"])) for r in rows]
    manifest = dict(kind="whisper", gate=GATE, unsupported=[], reference_kind=report["reference_kind"],
        native_strided_ports=True, replay_entrypoint="whisper.replay_encoder",
        records=[dict(name="whisper-encoder", hwx="hwx/model.hwx", hwx_sha256=report["export"]["hwx_sha256"],
            input_port_names=[p["name"] for p in report["ports"]["inputs"]],
            inputs=[p["shape"] for p in report["ports"]["inputs"]], output_port_name="t1383",
            output=[1, 1, 1500, 384], output_key="output", fixtures=fixtures)])
    save(a.output / "asahi-fixtures.json", manifest)
    runtime_path = a.output / "runtime-validation/report.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else verify(a.output, runtime_path.parent)
    require(runtime["status"] == "pass", "private-runtime encoder validation failed")
    report["runtime_validation"] = dict(report="runtime-validation/report.json",
                                        sha256=digest(a.output / "runtime-validation/report.json"))
    save(a.output / "report.json", report)
    inputs = {p["name"]:tuple(p["shape"]) for p in report["ports"]["inputs"]}
    # Reuse the just-validated compilation; a new cache would compile again.
    program = E5RT.compile(bundle / "model.mil", cache_dir=a.output / "runtime-validation/cache-0",
                           inputs=inputs, outputs={"t1383":(1500, 384)}, device_mask=4)
    report["warm_encoder"] = []
    try:
        program.set_input("t0", np.fromfile(bundle / "pos.f16", "<f2").reshape(inputs["t0"]))
        for row in rows:
            with np.load(a.output / row["fixture"], allow_pickle=False) as data:
                mel, expected = data["input00"], data["output"].reshape(1500, 384)
            program.set_input("t1", mel)
            for _ in range(2):
                program.execute()
            times = []
            for _ in range(a.runs):
                start = time.perf_counter()
                program.execute()
                times.append((time.perf_counter() - start) * 1000)
                require(np.array_equal(program.read_output("t1383"), expected), "warm output changed")
            report["warm_encoder"].append(dict(audio=row["name"], execute_ms=times, median_ms=statistics.median(times)))
    finally:
        program.release()
    if not (bundle / "cache").exists():
        shutil.copytree(a.output / "runtime-validation/cache-0", bundle / "cache")
    save(a.output / "report.json", report)
    env = dict(os.environ, HF_HUB_OFFLINE="1", OPENBLAS_NUM_THREADS="1", OMP_WAIT_POLICY="PASSIVE")
    env.pop("ANEFORGE_ENCODER", None)
    env.pop("ANEFORGE_DYLIB", None)
    libdir = ROOT / "whisper/build/metal/bin"
    probe = a.output / "probe-encoder-logits"
    checked(["/usr/bin/c++", "-std=c++17", "-O3", ROOT / "whisper/scripts/probe_encoder_logits.cpp",
        "-I" + str(ROOT / "whisper/vendor/whisper.cpp/include"),
        "-I" + str(ROOT / "whisper/vendor/whisper.cpp/ggml/include"),
        "-L" + str(libdir), "-Wl,-rpath," + str(libdir), "-lwhisper", "-lggml", "-lggml-base", "-o", probe], a.output / "probe-build.log")
    report["decoder_validation"] = previous.get("decoder_validation", [])
    import torch
    from transformers import WhisperForConditionalGeneration
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    import transformers
    report["versions"] = dict(numpy=np.__version__, torch=torch.__version__, transformers=transformers.__version__)
    hf = WhisperForConditionalGeneration.from_pretrained(a.hf_model, local_files_only=True, attn_implementation="eager").eval()
    with torch.inference_mode():
        for row in rows:
            label = row["name"]
            if any(r["audio"] == label for r in report["decoder_validation"]):
                continue
            with np.load(a.output / row["fixture"], allow_pickle=False) as data:
                mel = data["input00"].reshape(1, 80, 3000).astype(np.float32)
                ane = data["output"].reshape(1, 1500, 384).astype(np.float32)
            reference = hf.model.encoder(torch.from_numpy(mel)).last_hidden_state
            feature_error = compare(reference.numpy(), ane)
            # Generate one HF history, then evaluate both native backends and
            # both HF encoder variants on exactly those prefixes. No sampling
            # filters are applied to the captured 51,864 raw logits.
            generated = hf.generate(torch.from_numpy(mel), do_sample=False, max_new_tokens=100,
                                    return_dict_in_generate=True)
            tokens = generated.sequences[0].tolist()
            require(tokens[:2] == [hf.config.decoder_start_token_id, hf.generation_config.no_timestamps_token_id], "unexpected English prompt")
            require(tokens[-1] == hf.config.eos_token_id, "generation did not finish")
            tokens = tokens[:-1]
            mel.tofile(a.output / (label + "-mel.f32"))
            np.asarray(tokens, "<i4").tofile(a.output / (label + "-tokens.i32"))
            expected = hf.proj_out(hf.model.decoder(input_ids=torch.tensor([tokens]),
                encoder_hidden_states=reference, use_cache=False).last_hidden_state)[0, 1:].numpy()
            from_ane = hf.proj_out(hf.model.decoder(input_ids=torch.tensor([tokens]),
                encoder_hidden_states=torch.from_numpy(ane), use_cache=False).last_hidden_state)[0, 1:].numpy()
            np.savez_compressed(a.output / (label + "-hf-logits.npz"), tokens=tokens, reference=expected, from_ane=from_ane)
            native = {}
            for mode in ("cpu", "ane"):
                run_env = dict(env)
                if mode == "ane":
                    run_env.update(ANEFORGE_ENCODER=str(bundle), ANEFORGE_DYLIB=str(_find_dylib()))
                path = a.output / (label + "-" + mode + "-logits.bin")
                result = checked([probe, ROOT / "whisper/models/ggml-tiny.en.bin", a.output / (label + "-mel.f32"),
                    a.output / (label + "-tokens.i32"), path], a.output / (label + "-" + mode + "-probe.log"), run_env)
                if mode == "ane":
                    require("aneforge: encoder ready" in result.stderr and "aneforge: compile failed" not in result.stderr,
                            "native probe did not initialize the ANE encoder")
                captured = logits_records(path)
                require([int(t) for ts, _ in captured for t in ts] == tokens, "native histories changed")
                native[mode] = np.stack([value for _, value in captured])
            checks = []
            for index, logits in enumerate(expected):
                values = {"hf_decoder_ane_encoder":from_ane[index], "native_cpu":native["cpu"][index], "native_ane":native["ane"][index]}
                item = dict(prefix_length=index + 2)
                for key, actual in values.items():
                    item[key] = dict(**compare(logits, actual), argmax_match=int(np.argmax(logits)) == int(np.argmax(actual)))
                item["native_ane_vs_cpu"] = dict(**compare(native["cpu"][index], native["ane"][index]),
                    argmax_match=int(np.argmax(native["cpu"][index])) == int(np.argmax(native["ane"][index])))
                checks.append(item)
            keys = ("hf_decoder_ane_encoder", "native_cpu", "native_ane", "native_ane_vs_cpu")
            result = dict(audio=label, encoder_vs_hf=feature_error, token_history=tokens, logit_checks=checks,
                summaries={key:dict(max_nrmse=max(c[key]["nrmse"] for c in checks), argmax_matches=sum(c[key]["argmax_match"] for c in checks),
                    vectors=len(checks), pass_gate=all(c[key]["nrmse"] < .005 and c[key]["argmax_match"] for c in checks)) for key in keys})
            report["decoder_validation"].append(result)
            save(a.output / "report.json", report)
            print(json.dumps(dict(audio=label, encoder_cosine=feature_error["cosine"], decoder=result["summaries"])), flush=True)
    del hf
    report["accuracy_status"] = "pass" if all(r["encoder_vs_hf"]["cosine"] >= .999 and
        all(s["pass_gate"] for s in r["summaries"].values()) for r in report["decoder_validation"]) else "failed_logit_gate"
    save(a.output / "report.json", report)
    benchmark = a.output / "benchmark-whisper"
    checked(["/usr/bin/c++", "-std=c++17", "-O3", ROOT / "whisper/scripts/benchmark_whisper.cpp",
        "-I" + str(ROOT / "whisper/vendor/whisper.cpp/include"),
        "-I" + str(ROOT / "whisper/vendor/whisper.cpp/ggml/include"),
        "-L" + str(libdir), "-Wl,-rpath," + str(libdir), "-lwhisper", "-lggml", "-lggml-base", "-o", benchmark], a.output / "benchmark-build.log")
    report["warm_transcriptions"] = []
    pcm_paths = []
    for row in rows:
        with wave.open(str(a.output / (row["name"] + ".wav"))) as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype("<f4") / 32768
        pcm_path = a.output / (row["name"] + "-audio.f32")
        pcm.tofile(pcm_path)
        pcm_paths.append(pcm_path)
    for mode in ("cpu_cpu", "ane_cpu"):
        run_env = dict(env)
        if mode == "ane_cpu":
            run_env.update(ANEFORGE_ENCODER=str(bundle), ANEFORGE_DYLIB=str(_find_dylib()))
        print("Warm transcription: initializing one persistent " + mode + " context for all three clips", flush=True)
        result = checked([benchmark, ROOT / "whisper/models/ggml-tiny.en.bin",
            pcm_paths[0], "0", "2", str(a.runs), *pcm_paths[1:]], a.output / (mode + "-warm.log"), run_env, timeout=600)
        initialization = result.stderr.split("BENCH_AUDIO\t", 1)[0]
        outputs = result.stdout.split("BENCH_AUDIO\t")[1:]
        errors = result.stderr.split("BENCH_AUDIO\t")[1:]
        require(len(outputs) == len(errors) == len(rows), "missing warm audio blocks")
        for index, row in enumerate(rows):
            stdout_index, stdout = outputs[index].split("\n", 1)
            stderr_index, stderr = errors[index].split("\n", 1)
            require(int(stdout_index) == int(stderr_index) == index, "warm audio order changed")
            clip_result = subprocess.CompletedProcess(result.args, result.returncode, stdout, initialization + stderr)
            from whisper.scripts import benchmark_macos
            texts = [line.split("\t", 4)[4] for line in stdout.splitlines() if line.startswith("BENCH_RESULT\t")]
            require(bool(texts) and len(set(texts)) == 1, "warm transcription changed")
            expected_text = next(r["transcript"] for r in source_report["transcriptions"]
                                 if r["audio"] == row["name"] and r["route"] == "cpu")
            runs = parse_runs(clip_result, mode, row["audio_seconds"], benchmark_macos.words(expected_text))
            measured = [r for r in runs if r["phase"] == "measure"]
            report["warm_transcriptions"].append(dict(audio=row["name"], backend=mode, runs=runs,
                median={k:statistics.median(r[k] for r in measured) for k in ("encode_ms", "prompt_ms", "batchd_ms", "decode_ms", "decode_ms_per_token", "decoder_ms", "wall_ms")}, transcript=texts[0].strip()))
            save(a.output / "report.json", report)
    report.update(status="captured", accuracy_status="pass" if all(r["encoder_vs_hf"]["cosine"] >= .999 and
        all(s["pass_gate"] for s in r["summaries"].values()) for r in report["decoder_validation"]) else "failed_logit_gate",
        timing_scope="Warm encoder execute excludes feeds, reads and cross-K/V; native encode includes ANE/CPU encoder and CPU cross-K/V; whole transcription excludes setup. Four workers; active desktop.",
        executable_identity_limit="ANECCompile exports and E5RT execution compile the same exact MIL separately; E5RT hardware command identity is not established.")
    save(a.output / "report.json", report)
    print(json.dumps(dict(output=str(a.output), status=report["status"], accuracy_status=report["accuracy_status"],
        task_count=report["export"]["task_count"], recapture_matches_original_hwx=report["recapture_matches_original_hwx"])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "whisper/models/whisper-tiny.en-ane")
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--hf-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--resume", action="store_true", help="Reuse an intact capture and validated private-runtime cache")
    a = p.parse_args()
    if platform.system() != "Darwin" or a.runs < 1:
        p.error("requires macOS and positive runs")
    a.source, a.fixtures, a.hf_model, a.output = (path.resolve() for path in (a.source, a.fixtures, a.hf_model, a.output))
    sys.path.insert(0, str(Path.home() / "Desktop/ANEForge"))
    os.environ.setdefault("ANEFORGE_NO_AUTOBUILD", "1")
    from contextlib import ExitStack
    with ExitStack() as stack:
        for name in ("ane.lock", "gpu.lock"):
            lock = stack.enter_context((Path.home() / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        capture(a)


if __name__ == "__main__":
    main()
