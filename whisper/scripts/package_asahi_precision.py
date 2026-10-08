"""Derive weight-free H13G templates from the validated paired Mac programs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

import numpy as np

from experimental.capture_macos_program import parse_container, parse_tasks, _reader
from whisper.encoder_kernel import port_view, require
from whisper.paired_kernel import FORMAT, ROOT, pack_coefficients, reconstruct, validate_layout
from whisper.precision_encoder import CHECKPOINT_SHA256
from whisper.scripts.prepare_macos_precision import checkpoint_weights, validate_programs
from whisper.validation import digest


def accuracy_receipt(path, programs):
    receipt = json.loads(path.read_text())
    require(receipt["status"] == "PASS" and receipt["host_backend"] == "macos"
            and receipt["full_logit_gate_nrmse"] == .005
            and receipt["hf_checkpoint_sha256"] == CHECKPOINT_SHA256
            and receipt["precision_programs"] == programs,
            "requires the passing native receipt for these exact paired programs")
    counts = {"jfk": 25, "jfk-first-5s": 8, "jfk-repeat": 47}
    clips = receipt["correctness"]
    require(len(clips) == 3 and {c["audio"] for c in clips} == set(counts),
            "requires all three native accuracy cases")
    for clip in clips:
        require(clip["all_histories_match"] and len(clip["hf_logit_checks"]) == counts[clip["audio"]],
                "requires every native prefix and matching token histories")
        for vector in clip["hf_logit_checks"]:
            for backend in ("cpu", "ane"):
                error = vector[backend]["nrmse"]
                require(math.isfinite(error) and error < .005 and vector[backend + "_argmax_match"],
                        "native paired program receipt fails the unchanged full-vector gate")
    return receipt


def coefficient_tiles(coefficients, weights):
    """Locate complete transposed tiles and prove all learned bytes are covered."""
    n, k = weights.shape
    grouped = np.concatenate((weights[:, :k//2], weights[:, k//2:]), axis=0).astype("<f2")
    first, operations = 0, []
    while first < 2*n:
        found = []
        for count in (16, 8):
            if first + count > 2*n:
                continue
            value = grouped[first:first+count].T.copy().tobytes()
            offset = coefficients.find(value)
            if offset >= 0 and coefficients.find(value, offset+1) < 0:
                found.append((count, offset))
        require(bool(found), "unrecognized or ambiguous paired coefficient tile")
        count, offset = found[0]
        operations.append(dict(first=first, count=count, offset=offset))
        first += count
    require(pack_coefficients(weights, operations, len(coefficients)) == coefficients,
            "paired coefficient reconstruction is not byte exact")
    return operations


def recover(path, record, weight):
    receipt = json.loads((path / "receipt.json").read_text())
    require(receipt["status"] == "exported" and receipt["returncode"] == 0
            and receipt.get("compiler_errors") == []
            and receipt["mil_sha256"] == record["file_sha256"]["model.mil"]
            and receipt["source_weight_blobs"] == {"weights.bin": record["file_sha256"]["weights.bin"]},
            "export differs from the validated paired MIL/checkpoint")
    data = (path / "model.hwx").read_bytes()
    require(hashlib.sha256(data).hexdigest() == receipt["hwx_sha256"], "paired HWX checksum mismatch")
    container = parse_container(data)
    thread = container["thread"]
    text, = [s for s in container["segments"] if s["name"] == "__TEXT"]
    const, = [s for s in container["sections"] if s["segment"] == "__TEXT" and s["name"] == "__const"]
    kern, = [s for s in container["segments"] if s["name"].startswith("__KERN")]
    commands = data[text["fileoff"]:text["fileoff"]+text["filesize"]]
    tasks = parse_tasks(commands, thread["td_size"], thread["td_count"])
    require(receipt["task_count"] == len(tasks) and receipt["td_size"] == thread["td_size"],
            "paired exported task geometry changed")
    require(thread["bars"][0] == text["vmaddr"] and thread["bars"][1] == const["addr"]
            and thread["bars"][6] == kern["vmaddr"]
            and {i for i, value in enumerate(thread["bars"]) if value} == {0, 1, 4, 5, 6},
            "unsupported paired source BAR layout")
    bank_map = {0: 0, 1: 2, 4: 4, 5: 5, 6: 1}
    relocated = _reader.relocate(commands, tasks, bank_map)
    updated_tasks = parse_tasks(relocated, thread["td_size"], thread["td_count"])
    allowed = {task["offset"] + delta + byte for task in tasks for delta in (32, 36) for byte in range(4)}
    require(all(a == b or index in allowed for index, (a, b) in enumerate(zip(commands, relocated))),
            "paired relocation changed register packets or task dependencies")
    relocations = []
    for old, new in zip(tasks, updated_tasks):
        require(old["registers"] == new["registers"] and old["header"][:8] == new["header"][:8],
                "paired relocation changed task arithmetic")
        for delta in (32, 36):
            offset = old["offset"] + delta
            before, = struct.unpack_from("<I", commands, offset)
            after, = struct.unpack_from("<I", relocated, offset)
            if before != after:
                relocations.append(dict(offset=offset, before=before, after=after))
    constants = data[const["offset"]:const["offset"]+const["size"]]
    coefficients = data[kern["fileoff"]:kern["fileoff"]+kern["filesize"]]
    status_path = path / "compiler-status.json"
    status = json.loads(status_path.read_text())
    require(status["ErrorList"] == [], "paired compiler reported errors")
    network, = status["NetworkStatusList"]
    ports, buffers = [], []
    for role, key, bank in (("input", "LiveInputList", 4), ("output", "LiveOutputList", 5)):
        port, = network[key]
        name = port["Symbol"].removesuffix("@output")
        address = container["symbols"][name]["addr"]
        require(address == thread["bars"][bank], "paired port is not at the BAR base")
        segment, = [s for s in container["segments"] if s["vmaddr"] == address]
        require(segment["filesize"] == 0, "paired activation port unexpectedly contains static bytes")
        ports.append(dict(name=name, role=role, original_bank=bank, replay_bank=bank,
                          byte_offset=0, compiler_layout=port))
        buffers.append(dict(replay_bank=bank, size=segment["vmsize"], role=role))
        port_view(bytearray(segment["vmsize"]), ports[-1])
    n, k = weight.shape
    meta = dict(format=FORMAT, target="apple,t8103", input_features=k, output_features=n,
                positions=1500, temporal_planes=4, contraction_partitions=2, gains=[1., 1.375],
                td_count=thread["td_count"], td_size=thread["td_size"],
                coefficient_bytes=len(coefficients), coefficient_tiles=coefficient_tiles(coefficients, weight),
                layout=dict(ports=ports, buffers=buffers), bank_map=bank_map, relocations=relocations)
    validate_layout(meta)
    return meta, dict(commands=relocated, constants=constants, coefficients=coefficients), dict(
        hwx_sha256=receipt["hwx_sha256"], export_receipt_sha256=digest(path/"receipt.json"),
        compiler_status_sha256=digest(status_path), original_commands_sha256=hashlib.sha256(commands).hexdigest(),
        mil_sha256=receipt["mil_sha256"], source_weight_sha256=receipt["source_weight_blobs"]["weights.bin"])


def package(programs, exports, checkpoint, native_receipt, output):
    manifest = validate_programs(programs, checkpoint)
    accuracy_receipt(native_receipt, manifest)
    weights = checkpoint_weights(checkpoint)
    shapes, records, evidence = {}, [], []
    for record in manifest["programs"]:
        name = record["name"]
        meta, payloads, provenance = recover(exports/name/"hwx", record, weights[name])
        shape = f"{meta['input_features']}-{meta['output_features']}"
        if shape in shapes:
            previous_meta, previous_payloads = shapes[shape]
            require(meta == previous_meta and payloads["commands"] == previous_payloads["commands"]
                    and payloads["constants"] == previous_payloads["constants"],
                    "paired kernel layout/instructions depend on learned values")
        else:
            shapes[shape] = (meta, payloads)
        records.append(dict(name=name, kernel=shape, coefficients_sha256=hashlib.sha256(payloads["coefficients"]).hexdigest()))
        evidence.append(dict(name=name, kernel=shape, coefficient_bytes=len(payloads["coefficients"]),
                             tiles=len(meta["coefficient_tiles"]), td_count=meta["td_count"], **provenance))
    output.mkdir(parents=True, exist_ok=False)
    result = dict(format=FORMAT, checkpoint_sha256=CHECKPOINT_SHA256, programs=records, kernel_meta_sha256={})
    for shape, (meta, payloads) in shapes.items():
        directory = output/shape
        directory.mkdir()
        for name in ("commands", "constants"):
            filename = name + ".zlib"
            data = payloads[name]
            (directory/filename).write_bytes(zlib.compress(data, 9))
            meta[name] = dict(file=filename, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        (directory/"meta.json").write_text(json.dumps(meta, separators=(",", ":"))+"\n")
        result["kernel_meta_sha256"][shape] = digest(directory/"meta.json")
    (output/"manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    rebuilt_manifest, rebuilt = reconstruct(checkpoint, output)
    require(rebuilt_manifest == result and len(rebuilt) == 24, "paired reconstruction incomplete")
    # Compare again to independently parsed original captures, including every byte.
    for record in manifest["programs"]:
        name = record["name"]
        _, original, _ = recover(exports/name/"hwx", record, weights[name])
        require(all(rebuilt[name][key] == value for key, value in original.items()),
                "paired replay payload differs from capture: "+name)
    files = {str(path.relative_to(output)): dict(bytes=path.stat().st_size, sha256=digest(path))
             for path in sorted(output.rglob("*")) if path.is_file()}
    proof = dict(status="PASS_RECONSTRUCTION", checkpoint_sha256=CHECKPOINT_SHA256,
                 source_program_manifest_sha256=digest(programs/"manifest.json"),
                 native_accuracy_receipt_sha256=digest(native_receipt), programs=evidence, files=files,
                 learned_coefficient_bytes_reconstructed=sum(p["coefficient_bytes"] for p in evidence),
                 hardware_tasks_per_encode=sum(p["td_count"] for p in evidence), submissions_per_encode=24,
                 compiler_binary_sha256=digest(Path(__file__).resolve().parents[2]/"gpt2/training/build/dump_hwx"),
                 executable_identity_limit="Offline ANECCompile and E5RT compile the same validated MIL separately; the exported HWX has not been executed on either host.",
                 native_linux_integration="pending", linux_hardware_accuracy_and_performance="pending",
                 scope="All replay command, constant and learned coefficient bytes reconstruct exactly. Templates contain no learned coefficients; checkpoint stays external.")
    (output/"proof.json").write_text(json.dumps(proof, indent=2)+"\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--programs", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    proof = package(args.programs, args.exports, args.checkpoint, args.native_receipt, args.output)
    print(json.dumps({key: proof[key] for key in ("status", "hardware_tasks_per_encode",
        "submissions_per_encode", "learned_coefficient_bytes_reconstructed", "linux_hardware_accuracy_and_performance")}))


if __name__ == "__main__":
    main()
