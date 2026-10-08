"""Consolidate completed native CPU ablations without running more experiments."""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess

from whisper.scripts.benchmark_native import parse_audio_runs
from whisper.validation import digest


def add_stages(clips, stderr, ane):
    sections = stderr.split("BENCH_AUDIO\t")[1:]
    if len(sections) != len(clips):
        raise ValueError("missing stage audio boundaries")
    for index, (_, records) in enumerate(clips.items()):
        clip_index, log = sections[index].split("\n", 1)
        if clip_index != str(index):
            raise ValueError("stage audio order changed")
        blocks = {(phase, int(number)): text for phase, number, text in re.findall(
            r"BENCH_BEGIN\t(\w+)\t(\d+)\n(.*?)BENCH_END\t\1\t\2", log, re.S)}
        if len(blocks) != len(records):
            raise ValueError("stage repetition count changed")
        for row in records:
            block = blocks[row["phase"], row["index"]]
            stages = re.findall(r"WHISPER_PROFILE stage: (\w+)=([\d.]+) ms", block)
            if len(stages) != len({key for key, _ in stages}) or "cross_kv" not in dict(stages):
                raise ValueError("missing or duplicate host stage")
            row["stage_ms"] = {key: float(value) for key, value in stages}
            encoder = re.findall(
                r"WHISPER_PROFILE encoder: input=([\d.]+) feed=([\d.]+) "
                r"dispatch=([\d.]+) read=([\d.]+) convert=([\d.]+) ms", block)
            if len(encoder) != int(ane):
                raise ValueError("missing or unexpected ANE API stage profile")
            if ane:
                row["ane_api_ms"] = dict(zip(
                    ("input", "feed", "dispatch", "read", "convert"), map(float, encoder[0])))


def medians(rows):
    keys = ("encode_ms", "decoder_ms", "decode_ms", "batchd_ms", "prompt_ms", "wall_ms")
    result = {key: statistics.median(row[key] for row in rows) for key in keys}
    result["stage_ms"] = {key: statistics.median(row["stage_ms"][key] for row in rows)
        for key in rows[0]["stage_ms"]}
    if "ane_api_ms" in rows[0]:
        result["ane_api_ms"] = {key: statistics.median(row["ane_api_ms"][key] for row in rows)
            for key in rows[0]["ane_api_ms"]}
    if "cross_kv_matrices" in rows[0]:
        result["cross_kv_matrix_ms"] = {key: statistics.median(
            sum(matrix[key + "_us"] for matrix in row["cross_kv_matrices"]) / 1000 for row in rows)
            for key in ("allocate", "convert", "thread_setup", "gemm", "total")}
    elif any("cross_kv_matrices" in row for row in rows):
        raise ValueError("inconsistent cross-K/V profiling across measurements")
    return result


def completed_validation(path):
    report = json.loads((path / "summary.json").read_text())
    expected = {"jfk": 25, "jfk-first-5s": 8, "jfk-repeat": 47}
    if {row["audio"]: row["decoder_calls"] for row in report["correctness"]} != expected:
        raise ValueError("configuration has not checked all 80 decoder vectors")
    if report["host_backend"] != "macos" or report["full_logit_gate_nrmse"] != .005:
        raise ValueError("requires the shared macOS full-logit gate")
    if (report["warmups_per_context"], report["runs_per_context"], report["rounds"]) != (2, 10, 2):
        raise ValueError("requires two warmups and ten measures per clip per round")
    if "binary_sha256" not in report or report["status"] not in ("PASS", "FAIL"):
        raise ValueError("incomplete validation report")
    for backend in report["backends"].values():
        for audio in expected:
            records = backend["clips"][audio]
            if len(records["runs"]) != 20 or len(records["warmups"]) != 4:
                raise ValueError("configuration did not finish its native benchmark")
    return report


def recorded_asahi():
    source = Path(__file__).resolve().parents[2] / "todo.md"
    text = source.read_text()
    block = text.split("- [ ] Asahi: replay", 1)[1].split("- [ ] Qwen:", 1)[0]
    def number(pattern):
        found = re.search(pattern, block)
        if found is None:
            raise ValueError("missing recorded Asahi timing: " + pattern)
        return float(found[1])
    return dict(source="todo.md, Asahi section", source_sha256=digest(source),
        cross_kv_ms=number(r"reduces CPU cross-K/V to ([\d.]+) ms"),
        encode_ms=number(r"Latest 11-second encode is ([\d.]+) ms"),
        wall_ms=number(r"whole transcription ([\d.]+) ms"),
        dispatch_ms=number(r"dispatch ([\d.]+) ms"),
        readback_ms=number(r"readback ([\d.]+) ms"), precision_gate="FAIL")


def collect(configurations):
    output = dict(status="measured_diagnostic", timings_accepted=False,
        workers=4, rounds=2, warmups_per_clip_per_round=2, measures_per_clip_per_round=10,
        method="Existing benchmark_native results only; CPU/ANE order reversed in round two for each configuration. No further hardware experiments performed by this summarizer.",
        limitations="Configurations ran sequentially, not interleaved; active desktop and unfixed clocks limit causal size estimates. The provider comparison also changes algorithms and threading. Asahi numbers are prior recorded measurements, not a fresh Linux run.",
        scope="Original 1,779-task FP16 encoder, shared FP32 CPU precision patches. Numerical failures retained. No private AMX-disable flag assumed. LLAMAFILE=OFF changes ggml's matrix algorithm; it does not disable NEON.",
        recorded_asahi=recorded_asahi(),
        configurations={})
    first, first_path = None, None
    for name, build, path in configurations:
        build, path = Path(build).resolve(), Path(path).resolve()
        checked = completed_validation(path)
        if name in output["configurations"]:
            raise ValueError("duplicate configuration name")
        if first is None:
            first, first_path = checked, path
        for key in ("model_sha256", "hf_checkpoint_sha256", "payload_sha256", "runtime_sha256"):
            if checked[key] != first[key]:
                raise ValueError("configuration inputs/runtime differ: " + key)
        for clip in checked["correctness"]:
            for filename in ("mel.f32", "encoder.f32"):
                relative = clip["audio"] + "-ane/" + filename
                if digest(path / relative) != digest(first_path / relative):
                    raise ValueError("ANE boundary arrays changed: " + relative)
        for library, sha in checked["library_sha256"].items():
            if digest(build / "bin" / library) != sha:
                raise ValueError("library changed since validation: " + library)
        cache = (build / "CMakeCache.txt").read_text()
        source = Path(re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cache, re.M)[1])
        config = dict(validation_report=str(path / "summary.json"),
            validation_report_sha256=digest(path / "summary.json"),
            accuracy_status=checked["status"], timings_accepted=checked["timings_accepted"],
            numerical_failures=checked["numerical_failures"],
            accuracy=[{key: row[key] for key in ("audio", "decoder_calls", "all_histories_match",
                "raw_argmax_matches", "maximum_cpu_logit_nrmse_vs_hf",
                "maximum_ane_logit_nrmse_vs_hf")} for row in checked["correctness"]],
            cache_flags=dict(re.findall(r"^(GGML_\w+|CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER):\w+=(.*)$", cache, re.M)),
            library_sha256=checked["library_sha256"], source_sha256=checked["source_sha256"],
            cpu_build_flags=(build / "ggml/src/CMakeFiles/ggml-cpu.dir/flags.make").read_text(),
            additional_source_sha256={str(source / "ggml/src/ggml-cpu/vec.cpp"): digest(source / "ggml/src/ggml-cpu/vec.cpp")},
            artifacts={name: checked[name] for name in ("model_sha256", "hf_checkpoint_sha256", "payload_sha256", "runtime_sha256")},
            backends={})
        for backend in ("cpu_cpu", "ane_complete_cpu"):
            clips = {row["audio"]: dict(runs=[], warmups=[]) for row in checked["correctness"]}
            for round_index in (1, 2):
                # write_log saves stdout, a blank line, then stderr. BENCH_AUDIO
                # appears in both streams; split at the stream separator.
                log_path = path / f"{backend}-{round_index}.log"
                stdout, stderr = log_path.read_text().split("\n\n", 1)
                result = subprocess.CompletedProcess([], 0, stdout, stderr)
                rows = parse_audio_runs(result, 2 if backend.startswith("ane") else 0,
                    checked["correctness"], 1779, "macos")
                add_stages(rows, stderr, backend.startswith("ane"))
                for audio, records in rows.items():
                    for row in records:
                        row["round"] = round_index
                        phase = "warmups" if row["phase"] == "warmup" else "runs"
                        clips[audio][phase].append(row)
            for audio, clip in clips.items():
                if len(clip["runs"]) != 20 or len(clip["warmups"]) != 4:
                    raise ValueError("incomplete repetition data")
                for phase in ("runs", "warmups"):
                    for original, parsed in zip(checked["backends"][backend]["clips"][audio][phase], clip[phase]):
                        for key in ("phase", "index", "round", "wall_ms", "encode_ms", "decoder_ms", "cross_kv_matrices"):
                            if original[key] != parsed[key]:
                                raise ValueError("reparsed timing disagrees with validation: " + key)
                clip["median"] = medians(clip["runs"])
                clip["round_medians"] = [medians([row for row in clip["runs"] if row["round"] == n])
                    for n in (1, 2)]
            config["backends"][backend] = clips
        output["configurations"][name] = config
    return output


def compact(report):
    """Keep every timing repetition, sharing unchanged matrix metadata once."""
    columns = [key + "_us" for key in ("allocate", "convert", "thread_setup", "gemm", "total")]
    report["matrix_timing_columns"] = columns
    for config in report["configurations"].values():
        for clips in config["backends"].values():
            for clip in clips.values():
                profiles = clip["runs"][0]["cross_kv_matrices"]
                clip["matrix_metadata"] = {row["name"]: {
                    key: value for key, value in row.items()
                    if key not in columns and key not in ("address", "name")}
                    for row in profiles}
                for row in clip["runs"] + clip["warmups"]:
                    matrices = row.pop("cross_kv_matrices")
                    row["cross_kv_matrix_timings_us"] = {
                        matrix["name"]: [matrix[key] for key in columns] for matrix in matrices}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", nargs=3, action="append", required=True,
        metavar=("NAME", "BUILD", "VALIDATION_DIRECTORY"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact", action="store_true", help="Share matrix metadata; retain every measured and warmup timing")
    args = parser.parse_args()
    report = collect(args.configuration)
    if args.compact:
        report = compact(report)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
