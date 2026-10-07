"""Benchmark the same complete encoder, inputs and gates on macOS and Asahi."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import statistics
import wave

import numpy as np

from whisper.encoder_kernel import ROOT as KERNELS, require
from whisper.encoder_runtime import open_encoder
from whisper.validation import hardware_locks

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "whisper/results/fast-recapture-20261007/encoder-reference.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(model, audio, reference, fixtures=None):
    """Recreate the exact Mac inputs; report drift before hardware allocation."""
    from transformers import WhisperFeatureExtractor
    require(digest((model / "model.safetensors").read_bytes()) == reference["checkpoint_sha256"], "wrong HF checkpoint")
    with wave.open(str(audio)) as wav:
        require((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (16000, 1, 2, 176000),
                "expected 11-second 16 kHz mono PCM16 JFK")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").copy()
    require(digest(pcm.tobytes()) == reference["pcm_sha256"], "requires the original JFK PCM samples")
    frontend = WhisperFeatureExtractor.from_pretrained(model, local_files_only=True) if fixtures is None else None
    clips = (("jfk", pcm), ("jfk-first-5s", pcm[:80000]),
             ("jfk-repeat", np.concatenate((pcm, np.zeros(16000, "<i2"), pcm))))
    prepared = []
    for label, samples in clips:
        expected = next(r for r in reference["records"] if r["audio"] == label)
        if fixtures is None:
            mel = frontend(samples.astype(np.float32) / 32768, sampling_rate=16000, return_tensors="np").input_features
            mel = mel.reshape(80, 3000).astype("<f2")
        else:
            with np.load(fixtures / (label + ".npz"), allow_pickle=False) as data:
                mel = data["input00"].reshape(80, 3000).astype("<f2")
                require(np.isfinite(mel).all() and np.isfinite(data["output"]).all(), "nonfinite captured fixture")
                require(data["output"].size == 1500*384 and
                        digest(data["output"].astype("<f2").tobytes()) == expected["output_sha256"],
                        "captured encoder output checksum mismatch")
                require(digest(data["input01"].astype("<f2").tobytes()) == reference["position_sha256"],
                        "captured position input checksum mismatch")
        actual = digest(mel.tobytes())
        if fixtures is not None:
            require(actual == expected["mel_sha256"], "captured mel checksum mismatch")
        prepared.append((mel, dict(audio=label, audio_seconds=len(samples) / 16000,
            mel_sha256=actual, matches_mac_input=actual == expected["mel_sha256"],
            input_source="captured fixture" if fixtures else "regenerated frontend",
            expected_mac_output_sha256=expected["output_sha256"])))
    return prepared


def run(a):
    reference = json.loads(REFERENCE.read_text())
    report = dict(status="running", host=platform.platform(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint_sha256=reference["checkpoint_sha256"], backend=a.backend, warmups=a.warmups, runs=a.runs,
        scope="Complete encoder only. No CPU cross-K/V, decoder or whole-transcription timing. Python readback is a reference, not the optimized native four-worker implementation.",
        metric_notes=dict(prepare_ms="FP16 conversion, backend input packing/upload, Asahi scratch reset and output sentinel",
            dispatch_ms="Blocking backend execute only: DRM ioctl on Asahi, E5RT execute on macOS; includes runtime/driver work, scheduling/wait and ANE execution",
            readback_ms="Logical FP16 read, copy and finite-output check", total_ms="Sum of the three replay stages"),
        decoder_gate="not measured; original fast Mac graph fails the 0.005 full-logit gate", records=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        prepared = prepare(a.hf_model, a.audio, reference, a.fixtures)
        report["input_checks"] = [r for _, r in prepared]
        inputs_match = all(r["matches_mac_input"] for _, r in prepared)
        require(inputs_match or a.diagnostic_inputs, "frontend differs from Mac inputs; hardware comparison would be invalid")
        report["mac_comparison_valid"] = inputs_match
        if a.prepare_only:
            report.update(status="prepared" if inputs_match else "diagnostic_input_drift", hardware_execution="not attempted")
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
        with hardware_locks():
            original_hashes = {}
            for package in packages:
                encoder = open_encoder(a.hf_model / "model.safetensors", package, a.backend, a.device,
                    a.work_dir / package.name if a.work_dir else None)
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
                        original_hash = original_hashes.setdefault(info["audio"], hashes[-1])
                        item = dict(**info, kernels=package.name, task_count=encoder.meta["td_count"],
                            timings=times, median_ms={k:statistics.median(r[k] for r in times) for k in times[0]},
                            output_sha256=hashes[-1], all_match_mac_output=bitwise,
                            all_match_first_package=all(h == original_hash for h in hashes),
                            hf_encoder_cosine=cosine, pass_encoder_gate=bitwise and cosine >= .999)
                        item["backend"] = encoder.backend
                        report["records"].append(item)
                        a.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps(item), flush=True)
                finally:
                    encoder.close()
        report["status"] = ("diagnostic_input_drift" if not inputs_match else
            "pass_encoder_replay" if all(r["pass_encoder_gate"] for r in report["records"]) else "failed_encoder_gate")
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
    p.add_argument("--fixtures", type=Path, help="Existing jfk*.npz captures; use exact mel/position/output hashes on either host")
    p.add_argument("--backend", choices=("auto", "asahi", "macos"), default="auto")
    p.add_argument("--work-dir", type=Path, help="Generated macOS MIL/weights/compiler outputs; default whisper/build/encoder-runtime")
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--compare-baseline", action="store_true")
    p.add_argument("--prepare-only", action="store_true", help="Check portable inputs without allocating/submitting hardware")
    p.add_argument("--diagnostic-inputs", action="store_true", help="Profile locally generated inputs despite hash drift; retains failed Mac comparison and exits nonzero")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(a.runs, a.warmups) < 1:
        p.error("positive runs and warmups required")
    if not a.prepare_only and platform.system() not in ("Linux", "Darwin"):
        p.error("hardware profiling requires native base-M1 Asahi/macOS; use --prepare-only on this host")
    if a.output.exists():
        p.error("output already exists")
    report = run(a)
    print(json.dumps(dict(status=report["status"], report=str(a.output))))
    if report["status"] not in ("prepared", "pass_encoder_replay"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
