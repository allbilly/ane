#!/usr/bin/env python3
"""Compatibility entry point for the shared macOS/Asahi native benchmark."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from whisper.scripts.benchmark_native import (
    main, check_runtime, parse_warm_runs, parse_audio_runs, write_log,
)
from whisper.validation import compare, digest, logits_records, words

if __name__ == "__main__":
    # Historical callers hold flock around this compatibility entry point.
    main(default_backend="asahi", default_encoder="projections", acquire_locks=False)
