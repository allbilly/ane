#!/usr/bin/env python3
"""Measure ANEForge's unmodified Python Whisper transcribe path after warmup."""
import argparse
import datetime
import json
from pathlib import Path
import platform
import statistics
import time
import wave
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from whisper.validation import digest, words

ROOT = Path(__file__).resolve().parents[1]


class TimedProgram:
    """Time decoder feeds + execution + logit reads, preserving the original calls."""
    def __init__(self, program, token_port, last_logit_port):
        self.program = program
        self.token_port = token_port
        self.last_logit_port = last_logit_port
        self.reset()

    def reset(self):
        self.steps = []
        self.started = None
        self.first_step_start = None

    def set_input(self, name, array):
        if name == self.token_port:
            self.started = time.perf_counter()
            if self.first_step_start is None:
                self.first_step_start = self.started
        return self.program.set_input(name, array)

    def execute(self):
        return self.program.execute()

    def read_output(self, name):
        value = self.program.read_output(name)
        if name == self.last_logit_port:
            self.steps.append((time.perf_counter() - self.started) * 1000)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, default=ROOT / "vendor/whisper.cpp/samples/jfk.wav")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("requires a physical Apple Silicon Mac")
    if min(args.warmups, args.runs) < 1:
        parser.error("warmups and runs must be positive")
    if args.output.exists():
        parser.error("output already exists")
    import aneforge as af
    import numpy as np
    import torch

    torch.set_num_threads(4)
    with wave.open(str(args.audio)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            parser.error("requires 16 kHz mono PCM16 WAV")
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if len(audio) > 30 * 16000:
        parser.error("covers one clip of at most 30 seconds")
    expected = words("And so my fellow Americans ask not what your country can do for you ask what you can do for your country")
    report = {
        "status": "RUNNING", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": str(args.model.resolve()), "audio_seconds": len(audio) / 16000,
        "audio_sha256": digest(args.audio), "checkpoint_sha256": digest(args.model / "model.safetensors"),
        "aneforge_source": af.__file__, "warmups": [], "runs": [],
        "method": "One persistent model, two or more full warmups excluded, unchanged greedy transcribe path with timing wrappers, four Torch host threads, fixed 30-second encoder context.",
        "metric_notes": {
            "encode_ms": "encode call minus feature extraction; includes encoder input/output copies but excludes cross-K/V preparation",
            "decoder_ms": "sum of decoder step feeds, execute and logit reads, including prompt steps; excludes host cross-K/V projection, embedding/mask construction and sampling",
            "decoder_phase_ms": "wall time after encode returns, including host cross-K/V projection, cache reset, all decoder steps and tokenization",
            "decode_ms_per_token": "mean step evaluation after all start-prompt tokens have been consumed",
            "wall_ms": "whole warm transcribe call, including host work; excludes loading/compilation/file I/O",
            "comparison": "Different runtime and stage boundaries from whisper.cpp; report separately, no direct hardware ranking from this row",
        },
    }
    model = None
    try:
        started = time.perf_counter()
        model = af.load_whisper(str(args.model.resolve()))
        report["load_ms"] = (time.perf_counter() - started) * 1000
        # Run once to compile the lazy decoder, then wrap the actual program.
        started = time.perf_counter()
        cold = model.transcribe(audio)
        report["first_transcription_ms"] = (time.perf_counter() - started) * 1000
        if words(cold) != expected:
            raise RuntimeError("first transcript did not match JFK")
        decoder = model._decoder
        if model._encoder._prog._device_mask != 4 or decoder["net"].prog._device_mask != 4:
            raise RuntimeError("encoder/decoder is not configured for ANE dispatch")
        timed = TimedProgram(decoder["net"].prog, decoder["x"], decoder["logits"][-1])
        decoder["net"].prog = timed
        original_features, original_encode = model._features, model.encode
        stage = {}

        def features(value):
            started = time.perf_counter()
            result = original_features(value)
            stage["mel_ms"] = (time.perf_counter() - started) * 1000
            return result

        def encode(value):
            started = time.perf_counter()
            result = original_encode(value)
            stage["encode_with_mel_ms"] = (time.perf_counter() - started) * 1000
            return result

        model._features, model.encode = features, encode
        for index in range(args.warmups + args.runs):
            timed.reset()
            stage.clear()
            started = time.perf_counter()
            transcript = model.transcribe(audio)
            wall_ms = (time.perf_counter() - started) * 1000
            if words(transcript) != expected:
                raise RuntimeError(f"transcript mismatch on run {index + 1}: {transcript!r}")
            prompt_steps = len(model.sot)
            generation_steps = timed.steps[prompt_steps:]
            if not generation_steps:
                raise RuntimeError("no autoregressive token timing evidence")
            run = {
                "index": index + 1, "transcript": transcript, "wall_ms": wall_ms,
                "mel_ms": stage["mel_ms"],
                "encode_ms": stage["encode_with_mel_ms"] - stage["mel_ms"],
                "decoder_ms": sum(timed.steps),
                "decoder_phase_ms": wall_ms - stage["encode_with_mel_ms"],
                "prefill_ms": sum(timed.steps[:prompt_steps]), "prompt_tokens": prompt_steps,
                "decode_ms": sum(generation_steps), "decode_units": len(generation_steps),
                "decode_ms_per_token": statistics.mean(generation_steps),
                "decoder_step_count": len(timed.steps), "rtf": wall_ms / (len(audio) / 16),
            }
            phase = "warmup" if index < args.warmups else "measure"
            report["warmups" if phase == "warmup" else "runs"].append(run)
            print(f"{phase} {index + 1}: PASS encode={run['encode_ms']:.2f} ms, decoder={run['decoder_ms']:.2f} ms, whole={wall_ms:.2f} ms", flush=True)
        keys = ("encode_ms", "decoder_ms", "decoder_phase_ms", "prefill_ms", "decode_ms",
                "decode_ms_per_token", "mel_ms", "wall_ms", "rtf")
        report["median"] = {key: statistics.median(run[key] for run in report["runs"]) for key in keys}
        report["wall_range_ms"] = [min(run["wall_ms"] for run in report["runs"]), max(run["wall_ms"] for run in report["runs"])]
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        raise
    finally:
        if model is not None:
            # Restore the original program so release follows the original runtime path.
            if model._decoder is not None and isinstance(model._decoder["net"].prog, TimedProgram):
                model._decoder["net"].prog = model._decoder["net"].prog.program
            model.release()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Evidence: {args.output}", flush=True)


if __name__ == "__main__":
    main()
