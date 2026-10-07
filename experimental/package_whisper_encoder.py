"""Recover the validated complete Whisper encoder as a portable Asahi replay kit."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import plistlib
import shutil
import struct
import subprocess
import tarfile

from experimental.capture_macos_program import parse_container, parse_tasks, _reader
from experimental.replay_capture import validate_manifest


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def recover(capture, output):
    capture, output = Path(capture).resolve(), Path(output).resolve()
    manifest = json.loads((capture / "asahi-fixtures.json").read_text())
    checks = validate_manifest(manifest, capture)
    report = json.loads((capture / "report.json").read_text())
    receipt = json.loads((capture / "hwx/receipt.json").read_text())
    count = receipt["task_count"]
    if checks != 3 or len(manifest["records"]) != 1 or count not in (1779, 1783):
        raise ValueError("expected a complete original/wrapped encoder and three fixtures")
    runtime_path = (capture / report["runtime_validation"]["report"]).resolve()
    runtime = json.loads(runtime_path.read_text())
    if (digest(runtime_path) != report["runtime_validation"]["sha256"]
            or runtime["status"] != "pass" or runtime["device_mask"] != 4
            or len(runtime["comparisons"]) != 3
            or not all(c["pass_gate"] and c["bitwise_equal"] and c["passes_encoder_cosine"]
                       for c in runtime["comparisons"])):
        raise ValueError("missing validated ANE/HF fixture evidence")
    if digest(capture / "bundle/model.mil") != receipt["mil_sha256"]:
        raise ValueError("MIL checksum mismatch")
    for name, expected in receipt["source_weight_blobs"].items():
        if digest(capture / "bundle" / name) != expected:
            raise ValueError("source weight checksum mismatch: " + name)
    data = (capture / "hwx/model.hwx").read_bytes()
    if hashlib.sha256(data).hexdigest() != receipt["hwx_sha256"]:
        raise ValueError("HWX checksum mismatch")
    container = parse_container(data)
    text = next(s for s in container["segments"] if s["name"] == "__TEXT")
    const = next(s for s in container["sections"] if s["segment"] == "__TEXT" and s["name"] == "__const")
    thread = container["thread"]
    commands = data[text["fileoff"]:text["fileoff"] + text["filesize"]]
    tasks = parse_tasks(commands, thread["td_size"], thread["td_count"])
    if len(tasks) != count:
        raise ValueError("incomplete encoder task chain")
    coefficients, = [s for s in container["segments"] if s["name"].startswith("__KERN")]
    bars = thread["bars"]
    weight_bank = bars.index(coefficients["vmaddr"])
    bank_map = {i:i for i, addr in enumerate(bars) if addr}
    bank_map[1], bank_map[weight_bank] = 2, 1
    relocated = _reader.relocate(commands, tasks, bank_map)
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("bundle", "hwx"):
        shutil.copytree(capture / directory, output / directory)
    for name in ("report.json", "asahi-fixtures.json"):
        shutil.copy2(capture / name, output / name)
    for row in report["fixtures"]:
        shutil.copy2(capture / row["fixture"], output / row["fixture"])
        audio = capture / (row["name"] + ".wav")
        if digest(audio) != row["audio_sha256"]:
            raise ValueError("audio checksum mismatch")
        shutil.copy2(audio, output / audio.name)
    shutil.copy2(runtime_path, output / "macos-runtime-validation.json")
    replay = output / "replay"
    replay.mkdir()
    (replay / "commands-original.bin").write_bytes(commands)
    (replay / "commands-asahi.bin").write_bytes(relocated)
    (replay / "task-descriptors.bin").write_bytes(b"".join(commands[t["offset"]:t["offset"] + t["size"]] for t in tasks))
    (replay / "constants.bin").write_bytes(data[const["offset"]:const["offset"] + const["size"]])
    (replay / "coefficients.bin").write_bytes(data[coefficients["fileoff"]:coefficients["fileoff"] + coefficients["filesize"]])
    relocations = []
    task_index = []
    coefficient_registers = []
    for index, task in enumerate(tasks):
        task_index.append(dict(index=index, offset=task["offset"], size=task["size"],
                               next_offset=task["header"][7], header=task["header"],
                               descriptor_sha256=hashlib.sha256(commands[task["offset"]:task["offset"] + task["size"]]).hexdigest()))
        for delta in (32, 36):
            offset = task["offset"] + delta
            before, = struct.unpack_from("<I", commands, offset)
            after, = struct.unpack_from("<I", relocated, offset)
            if before != after:
                relocations.append(dict(task=index, byte_offset=offset, original=before, relocated=after))
        coefficient_registers.append(dict(task=index, registers={f"0x{address:05x}":value
                                            for address, value in task["registers"].items()
                                            if 0x5500 <= address < 0x5600}))
    write_json(replay / "task-index.json", task_index)
    write_json(replay / "relocations.json", dict(bank_map=bank_map, words=relocations,
                                                scope="Only active BAR selectors in task header words 8/9 change; packets, dependencies and NextPtr are preserved."))
    symbols = sorted((dict(name=name, offset=symbol["addr"] - coefficients["vmaddr"])
                      for name, symbol in container["symbols"].items()
                      if coefficients["vmaddr"] <= symbol["addr"] < coefficients["vmaddr"] + coefficients["filesize"]),
                     key=lambda s:s["offset"])
    for i, symbol in enumerate(symbols):
        symbol["bytes_until_next_symbol"] = (symbols[i + 1]["offset"] if i + 1 < len(symbols) else coefficients["filesize"]) - symbol["offset"]
    write_json(replay / "coefficient-layout.json", dict(segment=coefficients, symbols=symbols,
                per_task_kernel_dma_registers=coefficient_registers,
                layout="Compiler-packed coefficient bytes; exact symbol offsets and consuming register streams are retained. Do not reinterpret as a flat logical matrix."))
    status = plistlib.loads((capture / "hwx/model.hwx.status.plist").read_bytes())
    network, = status["NetworkStatusList"]
    ports = []
    for role, key in (("input", "LiveInputList"), ("output", "LiveOutputList")):
        for p in network[key]:
            name = p["Symbol"].removesuffix("@output")
            bank = bars.index(container["symbols"][name]["addr"])
            ports.append(dict(name=name, role=role, original_bank=bank, replay_bank=bank_map[bank],
                              byte_offset=0, compiler_layout=p))
    buffers = []
    for bank, address in enumerate(bars):
        if address and bank not in (0, 1, weight_bank):
            segment = next(s for s in container["segments"] if s["vmaddr"] == address)
            role = next((p["role"] for p in ports if p["original_bank"] == bank), "scratch/intermediate")
            buffers.append(dict(original_bank=bank, replay_bank=bank_map[bank], size=segment["vmsize"],
                                role=role, segment=segment, sections=[s for s in container["sections"]
                                if address <= s["addr"] < address + segment["vmsize"]]))
    write_json(replay / "buffers.json", dict(ports=ports, buffers=buffers,
                commands_and_weights=dict(bank=0, command_bytes=len(commands), weight_offset=len(commands),
                                          coefficient_bytes=coefficients["filesize"]),
                constants=dict(bank=2, bytes=const["size"]),
                bootstrap=dict(bytes=thread["td_size"], source="First relocated descriptor; replace bits 16..23 of its first word with 0x40.")))
    compiler_strings = [line for line in subprocess.check_output(["strings", "-a", str(capture / "hwx/model.hwx")], text=True).splitlines()
                        if "zin_ane_compiler" in line or "ANEC v" in line]
    recovery = dict(status="recovered_and_verified", captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    source_capture=str(capture), task_count=len(tasks), task_descriptor_bytes=sum(t["size"] for t in tasks),
                    hwx_sha256=receipt["hwx_sha256"], target=dict(soc="apple,t8103", family="H13G", cpu_subtype=4),
                    compiler_strings=compiler_strings,
                    runtime=dict(path=runtime["runtime"], sha256=runtime["runtime_sha256"], device_mask=4),
                    checkpoint=dict(revision=Path(report["hf_model"]).name, sha256=report["checkpoint_sha256"]),
                    fixture_comparisons=checks, macos_ane_bitwise_equal=True, hf_cosine_gate=.999,
                    linux_hardware_replay="pending native Asahi",
                    replay_command="python -m whisper.replay_encoder --checkpoint <pinned-checkpoint> --fixtures <extracted-kit> --output <new-result.json>",
                    scope="Complete encoder command chain, not the single-convolution old HWX or projection-only path. Historical capture paths in copied evidence identify provenance.")
    write_json(output / "recovery.json", recovery)
    files = {str(p.relative_to(output)):dict(bytes=p.stat().st_size, sha256=digest(p))
             for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "artifact-sha256.json", files)
    archive = output.with_suffix(".tar.gz")
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    with tarfile.open(archive, "r:gz") as tar:
        for name, item in files.items():
            with tar.extractfile(output.name + "/" + name) as f:
                if hashlib.sha256(f.read()).hexdigest() != item["sha256"]:
                    raise ValueError("archive payload checksum mismatch: " + name)
    result = dict(recovery=recovery, archive=dict(path=str(archive), bytes=archive.stat().st_size,
                                                 sha256=digest(archive), verified_payloads=len(files)))
    write_json(output.with_name(output.name + "-receipt.json"), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(recover(a.capture, a.output)), flush=True)


if __name__ == "__main__":
    main()
