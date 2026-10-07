"""Profile original/wrapped Asahi encoder replay using checkpoint and audio only."""
import argparse
from contextlib import ExitStack
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import statistics
import wave

import numpy as np

from whisper.encoder_kernel import ROOT as KERNELS, require
from whisper.replay_encoder import Encoder

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "whisper/results/fast-recapture-20261007/encoder-reference.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(model, audio, reference):
    """Recreate the exact Mac inputs; report drift before hardware allocation."""
    from transformers import WhisperFeatureExtractor
    require(digest((model / "model.safetensors").read_bytes()) == reference["checkpoint_sha256"], "wrong HF checkpoint")
    with wave.open(str(audio)) as wav:
        require((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (16000, 1, 2, 176000),
                "expected 11-second 16 kHz mono PCM16 JFK")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").copy()
    require(digest(pcm.tobytes()) == reference["pcm_sha256"], "requires the original JFK PCM samples")
    frontend = WhisperFeatureExtractor.from_pretrained(model, local_files_only=True)
    clips = (("jfk", pcm), ("jfk-first-5s", pcm[:80000]),
             ("jfk-repeat", np.concatenate((pcm, np.zeros(16000, "<i2"), pcm))))
    prepared = []
    for label, samples in clips:
        mel = frontend(samples.astype(np.float32) / 32768, sampling_rate=16000, return_tensors="np").input_features
        mel = mel.reshape(80, 3000).astype("<f2")
        expected = next(r for r in reference["records"] if r["audio"] == label)
        actual = digest(mel.tobytes())
        prepared.append((mel, dict(audio=label, audio_seconds=len(samples) / 16000,
            mel_sha256=actual, matches_mac_input=actual == expected["mel_sha256"],
            expected_mac_output_sha256=expected["output_sha256"])))
    return prepared


def run(a):
    reference = json.loads(REFERENCE.read_text())
    report = dict(status="running", host=platform.platform(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint_sha256=reference["checkpoint_sha256"], warmups=a.warmups, runs=a.runs,
        scope="Complete encoder only. No CPU cross-K/V, decoder or whole-transcription timing. Python readback is a reference, not the optimized native four-worker implementation.",
        metric_notes=dict(prepare_ms="FP16 conversion, input packing/upload, scratch zeroing and output sentinel",
            dispatch_ms="Synchronous ioctl: driver work, scheduling/wait and ANE execution",
            readback_ms="Logical FP16 read, copy and finite-output check", total_ms="Sum of the three replay stages"),
        decoder_gate="not measured; original fast Mac graph fails the 0.005 full-logit gate", records=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        prepared = prepare(a.hf_model, a.audio, reference)
        report["input_checks"] = [r for _, r in prepared]
        require(all(r["matches_mac_input"] for _, r in prepared), "frontend differs from Mac inputs; hardware comparison would be invalid")
        if a.prepare_only:
            report.update(status="prepared", hardware_execution="not attempted")
            return report
        import torch
        from transformers import WhisperForConditionalGeneration
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        hf = WhisperForConditionalGeneration.from_pretrained(a.hf_model, local_files_only=True, attn_implementation="eager").eval()
        references = []
        with torch.inference_mode():
            for mel, _ in prepared:
                features = hf.model.encoder(torch.from_numpy(mel.astype(np.float32)).reshape(1, 80, 3000)).last_hidden_state.numpy()[0]
                references.append(features)
        del hf
        packages = [a.kernels]
        if a.compare_baseline:
            packages.append(ROOT / "whisper/kernels/tiny-en-encoder")
        with ExitStack() as stack:
            for path in (Path.home() / "ane.lock", Path.home() / "gpu.lock", Path("/tmp/m1-gpu.lock")):
                lock = stack.enter_context(path.open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for package in packages:
                encoder = Encoder(a.hf_model / "model.safetensors", a.device, package)
                try:
                    require(encoder.meta["position_sha256"] == reference["position_sha256"], "position input differs from Mac")
                    for (mel, info), expected in zip(prepared, references):
                        for _ in range(a.warmups):
                            encoder(mel)
                        times = []
                        hashes = []
                        for _ in range(a.runs):
                            actual = encoder(mel)
                            times.append(dict(encoder.last_timing_ms))
                            hashes.append(digest(actual.astype("<f2").tobytes()))
                        av, ev = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
                        cosine = float(av @ ev / (np.linalg.norm(av) * np.linalg.norm(ev)))
                        bitwise = all(h == info["expected_mac_output_sha256"] for h in hashes)
                        item = dict(**info, kernels=package.name, task_count=encoder.meta["td_count"],
                            timings=times, median_ms={k:statistics.median(r[k] for r in times) for k in times[0]},
                            output_sha256=hashes[-1], all_match_mac_output=bitwise,
                            hf_encoder_cosine=cosine, pass_encoder_gate=bitwise and cosine >= .999)
                        report["records"].append(item)
                        a.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps(item), flush=True)
                finally:
                    encoder.close()
        report["status"] = "pass_encoder_replay" if all(r["pass_encoder_gate"] for r in report["records"]) else "failed_encoder_gate"
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        a.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-model", type=Path, required=True)
    p.add_argument("--audio", type=Path, default=ROOT / "whisper/vendor/whisper.cpp/samples/jfk.wav")
    p.add_argument("--kernels", type=Path, default=KERNELS)
    p.add_argument("--device")
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--compare-baseline", action="store_true")
    p.add_argument("--prepare-only", action="store_true", help="Check portable inputs without allocating/submitting hardware")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(a.runs, a.warmups) < 1:
        p.error("positive runs and warmups required")
    if not a.prepare_only and platform.system() != "Linux":
        p.error("hardware profiling requires native M1 Asahi; use --prepare-only on this host")
    if a.output.exists():
        p.error("output already exists")
    report = run(a)
    print(json.dumps(dict(status=report["status"], report=str(a.output))))
    if report["status"] not in ("prepared", "pass_encoder_replay"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
