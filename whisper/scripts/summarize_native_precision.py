"""Verify and compact a completed native paired-encoder accuracy/performance run."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

from whisper.scripts.benchmark_cross_kv import compact_receipt
from whisper.scripts.benchmark_native import parse_audio_runs
from whisper.scripts.summarize_cpu_ablations import medians
from whisper.validation import digest
from whisper.attention import add_attention_profiles

EXPECTED = {"jfk":25, "jfk-first-5s":8, "jfk-repeat":47}


def add_profiles(clips, stderr, paired):
    sections = stderr.split("BENCH_AUDIO\t")[1:]
    if len(sections) != len(clips):
        raise ValueError("missing native audio profile boundaries")
    for index, (audio, records) in enumerate(clips.items()):
        number, log = sections[index].split("\n", 1)
        if number != str(index):
            raise ValueError("native profile audio order changed")
        blocks = {(phase,int(number)):text for phase,number,text in re.findall(
            r"BENCH_BEGIN\t(\w+)\t(\d+)\n(.*?)BENCH_END\t\1\t\2", log, re.S)}
        if len(blocks) != len(records):
            raise ValueError("native profile repetition count changed")
        for row in records:
            block = blocks[row["phase"],row["index"]]
            stages = re.findall(r"WHISPER_PROFILE stage: (\w+)=([\d.]+) ms",block)
            if len(stages) != 3 or {key for key,_ in stages} != {"convolution","transformer","cross_kv"}:
                raise ValueError("native precision requires all three encoder stage profiles")
            row["stage_ms"] = {key:float(value) for key,value in stages}
            projections = re.findall(r"WHISPER_PROFILE precision: pack=([\d.]+) dispatch=([\d.]+) combine=([\d.]+) ms",block)
            if len(projections) != int(paired):
                raise ValueError("missing or unexpected paired projection timings")
            if paired:
                row["precision_api_ms"] = dict(zip(("pack","dispatch","combine"),map(float,projections[0])))
                remainder = row["stage_ms"]["transformer"]-sum(row["precision_api_ms"].values())
                if remainder < -.01:
                    raise ValueError("projection timing exceeds its enclosing transformer stage")
                row["transformer_remaining_ms"] = remainder


def collect(validation):
    raw = json.loads((validation/"summary.json").read_text())
    if (raw["status"] != "PASS" or not raw["timings_accepted"] or raw["numerical_failures"] or
            raw["host_backend"] != "macos" or raw["encoder_backend"] != "paired" or
            raw["encoder_cpu_gate_nrmse"] != .005 or raw["full_logit_gate_nrmse"] != .005 or
            raw["hf_encoder_gate_cosine"] != .999 or
            {c["audio"]:c["decoder_calls"] for c in raw["correctness"]} != EXPECTED or
            (raw["warmups_per_context"],raw["runs_per_context"],raw["rounds"]) != (2,10,2)):
        raise ValueError("requires a completed native paired run with unchanged gates and warmup policy")
    for clip in raw["correctness"]:
        count = EXPECTED[clip["audio"]]
        if len(clip["hf_logit_checks"]) != count or len(clip["logit_checks"]) != count:
            raise ValueError("native accuracy receipt lost full vectors")
        for vector in clip["hf_logit_checks"]:
            for backend in ("cpu","ane"):
                error = vector[backend]["nrmse"]
                if not math.isfinite(error) or error >= .005 or not vector[backend+"_argmax_match"]:
                    raise ValueError("independent HF full-logit vector failed")
        for vector in clip["logit_checks"]:
            if not math.isfinite(vector["nrmse"]) or vector["nrmse"] >= .005 or not vector["argmax_match"]:
                raise ValueError("native CPU/ANE full-logit vector failed")
        if not clip["all_histories_match"] or clip["encoder_vs_cpu"]["nrmse"] >= .005:
            raise ValueError("encoder boundary or native token history gate failed")
        if any(clip[key]["cosine"] < .999 for key in ("encoder_vs_hf","cpu_encoder_vs_hf")):
            raise ValueError("HF encoder cosine gate failed")
    for path, checksum in raw["artifacts"].items():
        if digest(validation/path) != checksum:
            raise ValueError("native validation artifact changed: " + path)
    result = dict(status="PASS",timings_accepted=True,numerical_failures=[],
        validation_summary_sha256=digest(validation/"summary.json"),
        utc=raw["utc"], host_backend=raw["host_backend"], scope=raw["scope"],
        full_logit_gate_nrmse=.005,encoder_cpu_gate_nrmse=.005,hf_encoder_gate_cosine=.999,
        method=raw["method"],limitations=raw["limitations"],correctness=raw["correctness"],
        model_sha256=raw["model_sha256"],hf_checkpoint_sha256=raw["hf_checkpoint_sha256"],
        precision_programs=raw["precision_programs"],precision_manifest_sha256=raw["precision_manifest_sha256"],
        library_sha256=raw["library_sha256"],runtime_sha256=raw["runtime_sha256"],
        binary_sha256=raw["binary_sha256"],driver_sha256=raw["driver_sha256"],
        source_sha256=raw["source_sha256"],artifacts=raw["artifacts"],
        encoder_attention=raw.get("encoder_attention", "flash"),
        configurations={"native_paired":{kind:{} for kind in raw["backends"]}})
    for kind, original in raw["backends"].items():
        paired = kind == "ane_precision_cpu"
        clips = {audio:dict(runs=[],warmups=[]) for audio in EXPECTED}
        for round_index in (1,2):
            log = (validation/f"{kind}-{round_index}.log").read_text()
            stdout,stderr = log.split("\n\n",1)
            process = subprocess.CompletedProcess([],0,stdout,stderr)
            parsed = parse_audio_runs(process,3 if paired else 0,raw["correctness"],1779,"macos")
            add_profiles(parsed,stderr,paired)
            add_attention_profiles(parsed,stderr,raw.get("encoder_attention") == "blas")
            for audio, rows in parsed.items():
                if len(rows) != 12:
                    raise ValueError("native warm count changed")
                for row in rows:
                    row["round"] = round_index
                    phase = "runs" if row["phase"] == "measure" else "warmups"
                    expected, = [r for r in original["clips"][audio][phase] if r["round"] == round_index and r["index"] == row["index"]]
                    if any(row[key] != value for key,value in expected.items()):
                        raise ValueError("reparsed native measurement differs from the validation receipt")
                    clips[audio][phase].append(row)
        for clip in clips.values():
            if len(clip["runs"]) != 20 or len(clip["warmups"]) != 4:
                raise ValueError("native round measurements incomplete")
            clip["median"] = medians(clip["runs"])
            if raw.get("encoder_attention") == "blas":
                clip["median"]["attention_matrix_ms"] = {key:statistics.median(
                    sum(m[key+"_us"] for m in row["attention_matrices"])/1000 for row in clip["runs"])
                    for key in ("allocate","convert","thread_setup","gemm","total")}
            if paired:
                clip["median"]["precision_api_ms"] = {key:statistics.median(row["precision_api_ms"][key] for row in clip["runs"])
                                                       for key in ("pack","dispatch","combine")}
                clip["median"]["transformer_remaining_ms"] = statistics.median(row["transformer_remaining_ms"] for row in clip["runs"])
            clip["round_medians"] = {str(i):medians([r for r in clip["runs"] if r["round"] == i]) for i in (1,2)}
        result["configurations"]["native_paired"][kind] = clips
    result["collector_source_sha256"] = digest(Path(__file__))
    return compact_receipt(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    report = collect(args.validation)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report,indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
