"""Shared numerical gates and timing records for macOS and Asahi benchmarks."""
import hashlib
from contextlib import contextmanager, ExitStack
import fcntl
import math
import json
from pathlib import Path
import re
import struct

import numpy as np


@contextmanager
def hardware_locks():
    with ExitStack() as stack:
        for path in (Path.home()/"ane.lock", Path.home()/"gpu.lock", Path("/tmp/m1-gpu.lock")):
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def compare(reference, actual):
    if reference.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("invalid numerical comparison arrays")
    a, b = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    norm, actual_norm = np.linalg.norm(a), np.linalg.norm(b)
    if norm == 0 or actual_norm == 0:
        raise ValueError("zero numerical comparison norm")
    return dict(nrmse=float(np.linalg.norm(a-b)/norm),
                cosine=float(np.dot(a, b)/(norm*actual_norm)),
                max_abs=float(np.max(np.abs(a-b))))


def logits_records(path):
    """Read every full tiny.en vocabulary vector, rejecting truncated captures."""
    data, offset, records = Path(path).read_bytes(), 0, []
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError("incomplete logit header")
        count, vocabulary = struct.unpack_from("<2i", data, offset)
        offset += 8
        if not 1 <= count <= 448 or vocabulary != 51864:
            raise ValueError("unexpected tiny.en logit record")
        size = (count + vocabulary)*4
        if len(data) - offset < size:
            raise ValueError("incomplete logit capture")
        tokens = np.frombuffer(data, "<i4", count, offset).copy()
        values = np.frombuffer(data, "<f4", vocabulary, offset + count*4).copy()
        if (tokens < 0).any() or (tokens >= vocabulary).any() or not np.isfinite(values).all():
            raise ValueError("invalid token history or nonfinite logits")
        offset += size
        records.append((tokens, values))
    if not records:
        raise ValueError("empty logit capture")
    return records


def matrix_profiles(log, required=False, layout="separate"):
    if layout not in ("separate", "fused"):
        raise ValueError("unknown cross-K/V matrix layout")
    records = [json.loads(line.split("\t", 1)[1]) for line in log.splitlines()
               if line.startswith("MATRIX_PROFILE\t")]
    if not records and not required:
        return []
    expected = ({"whisper.cross_kv.fused"} if layout == "fused" else
                {f"whisper.cross_kv.{i}.{kind}" for i in range(4) for kind in ("k", "v")})
    if len(records) != len(expected) or {r["name"] for r in records} != expected:
        raise ValueError("cross-K/V profile does not match the requested " + layout + " layout")
    for row in records:
        if (row["m"], row["n"], row["k"]) != (1500, 3072 if layout == "fused" else 384, 384):
            raise ValueError("unexpected tiny.en cross-K/V matrix dimensions")
        if layout == "fused" and (row["weight_type"] != "f32" or row["weight"] != "whisper.cross_kv.cached_weights"):
            raise ValueError("fused cross-K/V is not using its cached FP32 weights")
        if row["order"] != "row_major" or row["transpose_a"] or not row["transpose_b"]:
            raise ValueError("unexpected cross-K/V operand packing")
        if any(row[k] != "f32" for k in ("input_type", "output_type", "gemm_type")):
            raise ValueError("cross-K/V is not using FP32 GEMM")
        stages = ("allocate_us", "convert_us", "thread_setup_us", "gemm_us")
        if any(type(row[k]) is not int or row[k] < 0 for k in (*stages, "total_us")):
            raise ValueError("invalid matrix stage timing")
        if sum(row[k] for k in stages) != row["total_us"]:
            raise ValueError("matrix stage timing boundaries do not add up")
    return records


def parse_runs(result, audio_seconds, expected_words, check_runtime=None, encoder_marker=None,
               matrix_layout="separate"):
    """Parse the common benchmark_whisper.cpp protocol; adapters verify dispatch."""
    if not math.isfinite(audio_seconds) or audio_seconds <= 0:
        raise ValueError("invalid benchmark audio duration")
    records = {}
    for line in result.stdout.splitlines():
        if not line.startswith("BENCH_RESULT\t"):
            continue
        _, phase, index, wall_ms, text = line.split("\t", 4)
        key = (phase, int(index))
        elapsed = float(wall_ms)
        if phase not in ("warmup", "measure") or key[1] < 1 or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("invalid benchmark record")
        if key in records:
            raise ValueError("duplicate benchmark record")
        if words(text) != expected_words:
            raise ValueError("warm transcript mismatch: " + repr(text))
        records[key] = dict(phase=phase, index=key[1], wall_ms=elapsed, transcript=text.strip())
    if check_runtime:
        check_runtime(result.stderr, len(records))
    seen = set()
    for block in re.finditer(r"BENCH_BEGIN\t(\w+)\t(\d+)\n(.*?)BENCH_END\t\1\t\2", result.stderr, re.S):
        key = (block[1], int(block[2]))
        if key not in records or key in seen:
            raise ValueError("missing record or duplicate timing block")
        seen.add(key)
        record = records[key]
        for name in ("mel", "sample", "encode", "decode", "batchd", "prompt"):
            match = re.search(rf"\b{name} time\s*=\s*([\d.]+) ms(?: /\s*(\d+) runs)?", block[3])
            if not match:
                raise ValueError("missing stage timing: " + name)
            elapsed = float(match[1])
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid stage timing: " + name)
            record[name + "_ms"] = elapsed
            if match[2]:
                record[name + "_units"] = int(match[2])
        fallback = re.findall(r"fallbacks\s*=\s*(\d+) p /\s*(\d+) h", block[3])
        if fallback != [("0", "0")]:
            raise ValueError("missing fallback evidence or decoding fallback detected")
        if record.get("decode_units", 0) <= 0:
            raise ValueError("missing positive decode call count")
        if encoder_marker and block[3].count(encoder_marker) != 1:
            raise ValueError("timed transcription did not execute the ANE encoder exactly once")
        matrices = matrix_profiles(block[3], layout=matrix_layout)
        if matrices:
            record["cross_kv_matrices"] = matrices
        record["decoder_ms"] = record["decode_ms"] + record["batchd_ms"] + record["prompt_ms"]
        record["decode_ms_per_token"] = record["decode_ms"]/record["decode_units"]
        record["rtf"] = record["wall_ms"]/(audio_seconds*1000)
    if not records or seen != records.keys():
        raise ValueError("incomplete benchmark timings")
    return list(records.values())
