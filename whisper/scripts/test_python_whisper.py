#!/usr/bin/env python3
"""Check ANEForge's Python Whisper encoder and decoder against Hugging Face on JFK."""
import argparse
import datetime
import json
from pathlib import Path
import platform
import re
import time
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument("--audio", type=Path, default=Path(__file__).resolve().parents[1] / "vendor/whisper.cpp/samples/jfk.wav")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("this test requires a physical Apple Silicon Mac")
    import aneforge as af
    import numpy as np
    import torch
    import transformers
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(4)
    with wave.open(str(args.audio)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            parser.error("audio must be 16 kHz mono PCM16 WAV")
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if len(audio) > 30 * 16000:
        parser.error("this test covers one clip of at most 30 seconds")
    print(f"ANEForge source: {af.__file__}", flush=True)
    x = af.input((1, 4, 1, 4))
    net = af.compile(x * 2 + 1)
    sample = np.arange(16, dtype=np.float16).reshape(1, 4, 1, 4)
    np.testing.assert_array_equal(np.asarray(net(sample)), sample * 2 + 1)
    net.release()
    print("ANE hardware dispatch: PASS", flush=True)
    started = time.perf_counter()
    whisper = af.load_whisper(args.model)
    load_ms = (time.perf_counter() - started) * 1000
    try:
        started = time.perf_counter()
        cold = whisper.transcribe(audio)
        first_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        text = whisper.transcribe(audio)
        warm_ms = (time.perf_counter() - started) * 1000
        features = whisper.encode(audio)
        proc = WhisperProcessor.from_pretrained(args.model)
        hf = WhisperForConditionalGeneration.from_pretrained(args.model).eval()
        mel = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            reference = hf.model.encoder(mel).last_hidden_state[0].numpy()
            reference_text = proc.batch_decode(hf.generate(mel), skip_special_tokens=True)[0]
        cosine = float(features.ravel() @ reference.ravel() / (np.linalg.norm(features) * np.linalg.norm(reference)))
        normalize = lambda value: re.findall(r"[a-z0-9']+", value.lower())
        matched = normalize(text) == normalize(reference_text)
        passed = cosine > 0.999 and matched and cold == text
        report = {
            "status": "PASS" if passed else "FAIL", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": args.model, "audio": str(args.audio.resolve()), "audio_seconds": len(audio) / 16000,
            "aneforge_source": af.__file__, "numpy": np.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "transcript": text, "reference_transcript": reference_text,
            "transcript_words_match": matched, "repeat_transcript_identical": cold == text,
            "encoder_cosine": cosine, "load_ms": load_ms, "first_transcription_ms": first_ms,
            "warm_transcription_ms": warm_ms,
            "scope": "ANE encoder and decoder graphs; feature extraction, embeddings, cross-K/V preparation and sampling on host",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not passed:
            raise RuntimeError("Whisper parity failed; see JSON report")
    finally:
        whisper.release()


if __name__ == "__main__":
    main()
