"""Prepare or replay PR 3905 encoder packets with the guarded base-M1 DRM ABI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import struct
import sys
import time
import zlib

import numpy as np

from gpt2.hwx import parse_container, parse_tasks, relocate
from qwen35.weights import sha256
from whisper.encoder_kernel import require, pack_port, port_view, port_shape
from whisper.pr_encoder_kernel import reconstruct

GATE = dict(relative_l2=.005, allclose_rtol=.01, allclose_atol=.03)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def native_descriptor(plan, payloads):
    """Validated text layout for the dependency-free native full-encoder runner."""
    dims = plan["dimensions"]
    buffers = {item["bank"]:item for item in plan["buffers"]}
    require(set(buffers) == ({0, 2, 3, 4, 5, 6, 8} if dims["state"] == 768 else {0, 2, 3, 4, 5, 6}),
            "unsupported native PR buffer banks")
    require(buffers[3]["role"] == "scratch", "unsupported native PR scratch bank")
    lines = ["ANE_WHISPER_PR_V1", " ".join(map(str, (dims["state"], dims["layers"], plan["td_count"],
              plan["td_size"], plan["command_bytes"], len(buffers), len(payloads))))]
    for bank, item in sorted(buffers.items()):
        lines.append(f"{bank} {item['size']}")
    for name, bank, shape in (("mel", 5, (1, 80, 1, 3000)),
                              ("positions", 4, (1, dims["state"], 1, 1500)),
                              ("output", 6, (1, 1, 1500, dims["state"]))):
        port = plan["ports"][name]
        require(port["replay_bank"] == bank and port["byte_offset"] == 0 and port_shape(port) == shape,
                "unsupported native PR port mapping")
        layout = port["compiler_layout"]
        require(name != "output" or layout["RowStride"] == dims["state"] * 2,
                "native PR output must have tight rows")
        lines.append(" ".join(map(str, (bank, port["byte_offset"], *shape[1:], layout["PlaneStride"], layout["RowStride"]))))
    destinations = dict(commands=(0, 0), constants=(2, 0), positions=(-1, 0), bootstrap=(-2, 0),
                        **{"coefficients-1":(0, plan["command_bytes"]), "coefficients-8":(8, 0)})
    for name, data in sorted(payloads.items()):
        require(name in destinations and len(data) == plan["payloads"][name]["bytes"]
                and digest(data) == plan["payloads"][name]["sha256"], "native PR payload identity mismatch")
        bank, offset = destinations[name]
        lines.append(f"{name}.bin {len(data)} {zlib.crc32(data)} {bank} {offset}")
    return "\n".join(lines) + "\n"


def prepare(meta, hwx, positions):
    """Preserve a second coefficient BAR; only the first uses synthesized BAR 1."""
    require(meta["target"] == "apple,t8103" and digest(hwx) == meta["hwx_sha256"], "encoder HWX target/checksum mismatch")
    require(digest(positions) == meta["position_sha256"], "encoder positions checksum mismatch")
    container = parse_container(hwx)
    thread, bars = container["thread"], container["thread"]["bars"]
    require(thread == meta["layout"]["thread"] and container["segments"] == meta["layout"]["segments"]
            and container["sections"] == meta["layout"]["sections"], "encoder container metadata mismatch")
    text = next(s for s in container["segments"] if s["name"] == "__TEXT")
    const = next(s for s in container["sections"] if s["segment"] == "__TEXT" and s["name"] == "__const")
    require(bars[0] == thread["entry"] == text["vmaddr"] and bars[1] == const["addr"] and bars[2] == 0,
            "unsupported command/constant BAR layout")
    require(text["filesize"] > 0 and text["filesize"] % 16 == 0
            and text["fileoff"] <= const["offset"] < const["offset"] + const["size"] <= text["fileoff"] + text["filesize"],
            "invalid command boundary/constants")
    coefficients = [s for s in container["segments"] if s["name"].startswith("__KERN")]
    require(1 <= len(coefficients) <= 2 and [s["name"] for s in coefficients] == [f"__KERN_{i}" for i in range(len(coefficients))],
            "unsupported coefficient segments")
    weight_banks = [bars.index(s["vmaddr"]) for s in coefficients]
    require(len(set(weight_banks)) == len(weight_banks) and all(bank >= 3 for bank in weight_banks), "invalid coefficient BARs")
    bank_map = {i:i for i, address in enumerate(bars) if address}
    bank_map[1], bank_map[weight_banks[0]] = 2, 1
    require(len(set(bank_map.values())) == len(bank_map), "overlapping replay BARs")
    original = hwx[text["fileoff"]:text["fileoff"] + text["filesize"]]
    tasks = parse_tasks(original, thread["td_size"], thread["td_count"])
    commands = relocate(original, tasks, bank_map)
    relocated_tasks = parse_tasks(commands, thread["td_size"], thread["td_count"])
    require(all(a["offset"] == b["offset"] and a["size"] == b["size"] and a["header"][:8] == b["header"][:8]
                and a["registers"] == b["registers"] for a, b in zip(tasks, relocated_tasks)), "task packets/dependencies changed")
    require(relocate(commands, relocated_tasks, {v:k for k, v in bank_map.items()}) == original, "non-BAR command bytes changed")
    payloads = dict(commands=commands, constants=bytes(hwx[const["offset"]:const["offset"] + const["size"]]), positions=positions)
    coefficient_layout = []
    for index, segment in enumerate(coefficients):
        bank = bank_map[weight_banks[index]]
        name = f"coefficients-{bank}"
        payloads[name] = bytes(hwx[segment["fileoff"]:segment["fileoff"] + segment["filesize"]])
        require(segment["filesize"] > 0 and segment["filesize"] == segment["vmsize"], "unsupported coefficient padding")
        coefficient_layout.append(dict(original_bank=weight_banks[index], replay_bank=bank, payload=name,
                                       bytes=segment["filesize"], sha256=digest(payloads[name])))
    buffers = [dict(bank=0, size=len(commands) + len(payloads["coefficients-1"]) + 1, role="commands+coefficients"),
               dict(bank=2, size=len(payloads["constants"]), role="constants")]
    ports = []
    for port in meta["layout"]["ports"]:
        bank = port["original_bank"]
        symbol = container["symbols"][port["name"]]["addr"]
        require(port["role"] in ("input", "output") and bank >= 3 and bank not in weight_banks
                and symbol == bars[bank] + port["byte_offset"], "invalid encoder port BAR/offset")
        ports.append(dict(port, replay_bank=bank_map[bank]))
    for bank, address in enumerate(bars):
        if not address or bank in (0, 1, weight_banks[0]):
            continue
        segment, = [s for s in container["segments"] if s["vmaddr"] == address]
        if bank in weight_banks:
            role = "coefficients"
        else:
            require(segment["filesize"] == 0, "initialized workspace is unsupported")
            roles = [p["role"] for p in ports if p["original_bank"] == bank]
            require(len(roles) <= 1, "aliased encoder ports")
            role = roles[0] if roles else "scratch"
        require(segment["vmsize"] > 0, "empty replay buffer")
        buffers.append(dict(bank=bank_map[bank], size=segment["vmsize"], role=role))
    require(1 not in {b["bank"] for b in buffers} and len({b["bank"] for b in buffers}) == len(buffers), "invalid reserved/duplicate replay BAR")
    dims = meta["dimensions"]
    require(dims["mels"] == 80 and dims["context"] == 1500 and dims["frames"] == 3000
            and dims["state"] in (384, 512, 768), "unsupported PR encoder dimensions")
    require(len(positions) == dims["state"] * dims["context"] * 2, "invalid position payload size")
    selected = {}
    for name, role, shape in (("mel", "input", (1, 80, 1, 3000)),
                              ("positions", "input", (1, dims["state"], 1, 1500)),
                              ("output", "output", (1, 1, 1500, dims["state"]))):
        matches = [p for p in ports if p["role"] == role and port_shape(p) == shape]
        require(len(matches) == 1, "missing/ambiguous encoder port: " + name)
        selected[name] = matches[0]
        size, = [b["size"] for b in buffers if b["bank"] == matches[0]["replay_bank"]]
        # Validate bounds and padded strides using the same view as live staging.
        port_view(bytearray(size), matches[0])
    require(len(ports) == 3, "unexpected extra encoder ports")
    bootstrap = bytearray(commands[:thread["td_size"]])
    header, = struct.unpack_from("<I", bootstrap)
    struct.pack_into("<I", bootstrap, 0, (header & ~(0xFF << 16)) | (0x40 << 16))
    payloads["bootstrap"] = bytes(bootstrap)
    plan = dict(format="whisper-pr3905-drm-replay/v1", model=meta["checkpoint"]["model"], target=meta["target"],
                dimensions=dims, hwx_sha256=meta["hwx_sha256"], source_mil_sha256=meta["source_mil_sha256"],
                td_count=thread["td_count"], td_size=thread["td_size"], command_bytes=len(commands),
                bank_map=bank_map, buffers=buffers, ports=selected, coefficient_banks=coefficient_layout,
                payloads={n:dict(bytes=len(raw), sha256=digest(raw)) for n, raw in payloads.items()},
                hardware_validation="pending native Asahi; plan validation is not execution")
    return plan, payloads


class Encoder:
    """One ioctl submits the full encoder; input packing/readback remain separate."""
    def __init__(self, checkpoint, kernels, device=None):
        require(platform.system() == "Linux", "encoder replay requires native M1 Asahi Linux")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpt2"))
        from replay import Buffer, Submit, SUBMIT, ioctl, device_path
        path = device_path(device)
        meta, hwx, positions = reconstruct(checkpoint, kernels)
        self.plan, payloads = prepare(meta, hwx, positions)
        self.fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        self.buffers, self.bootstrap = {}, None
        self.ioctl, self.submit_opcode = ioctl, SUBMIT
        self.submissions, self.last_timing_ms = 0, None
        try:
            self.allocate(payloads, Buffer, Submit)
        except BaseException:
            self.close()
            raise

    def allocate(self, payloads, Buffer, Submit):
        for item in self.plan["buffers"]:
            self.buffers[item["bank"]] = Buffer(self.fd, item["size"])
        self.buffers[0].write(payloads["commands"])
        self.buffers[0].write(payloads["coefficients-1"], self.plan["command_bytes"])
        self.buffers[2].write(payloads["constants"])
        for coefficient in self.plan["coefficient_banks"][1:]:
            self.buffers[coefficient["replay_bank"]].write(payloads[coefficient["payload"]])
        position_port = self.plan["ports"]["positions"]
        buffer = self.buffers[position_port["replay_bank"]]
        buffer.write(pack_port(np.frombuffer(payloads["positions"], "<f2"), position_port, buffer.size))
        self.bootstrap = Buffer(self.fd, self.plan["td_size"])
        self.bootstrap.write(payloads["bootstrap"])
        self.request = Submit(tsk_size=self.plan["command_bytes"], td_count=self.plan["td_count"],
                              td_size=self.plan["td_size"], btsp_handle=self.bootstrap.handle)
        for bank, buffer in self.buffers.items():
            self.request.handles[bank] = buffer.handle
        require(self.request.handles[1] == 0, "driver coefficient BAR must be synthesized")

    def __call__(self, mel):
        began = time.perf_counter()
        dims = self.plan["dimensions"]
        mel = np.asarray(mel, dtype="<f2")
        require(mel.shape == (dims["mels"], dims["frames"]) and bool(np.isfinite(mel).all()), "invalid encoder mel input")
        port = self.plan["ports"]["mel"]
        buffer = self.buffers[port["replay_bank"]]
        buffer.write(pack_port(mel, port, buffer.size))
        for item in self.plan["buffers"]:
            if item["role"] == "scratch":
                self.buffers[item["bank"]].write(bytes(self.buffers[item["bank"]].size))
        port = self.plan["ports"]["output"]
        buffer = self.buffers[port["replay_bank"]]
        port_view(buffer.map, port)[...] = np.nan
        start = time.perf_counter()
        self.ioctl(self.fd, self.submit_opcode, self.request)
        completed = time.perf_counter()
        self.submissions += 1
        output = port_view(buffer.map, port).reshape(dims["context"], dims["state"]).copy()
        require(bool(np.isfinite(output).all()), "encoder produced nonfinite/unwritten output")
        finished = time.perf_counter()
        self.last_timing_ms = dict(prepare_ms=(start - began) * 1000, dispatch_ms=(completed - start) * 1000,
                                  readback_ms=(finished - completed) * 1000, total_ms=(finished - began) * 1000)
        return output

    def close(self):
        buffers = list(self.buffers.values()) + ([self.bootstrap] if self.bootstrap else [])
        self.buffers, self.bootstrap = {}, None
        try:
            for buffer in reversed(buffers):
                try:
                    buffer.close()
                except OSError:
                    pass  # Closing the device FD releases remaining GEM objects.
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


def validate_fixtures(plan, manifest, root):
    require(manifest["format"] == "whisper-pr3905-replay-fixtures/v1" and manifest["gate"] == GATE, "fixture format/gate mismatch")
    record = manifest["models"][plan["model"]]
    require(record["hwx_sha256"] == plan["hwx_sha256"] and record["mil_sha256"] == plan["source_mil_sha256"], "fixture model mismatch")
    dims = plan["dimensions"]
    cases = []
    names = set()
    require(bool(record["cases"]), "no encoder fixtures")
    for case in record["cases"]:
        require(case["name"] not in names, "duplicate encoder fixture")
        names.add(case["name"])
        path = (root / case["file"]).resolve()
        require(path.is_relative_to(root.resolve()) and sha256(path) == case["sha256"], "fixture path/checksum mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            values = {}
            for key, shape in (("mel", (dims["mels"], dims["frames"])), ("positions", (dims["state"], dims["context"])),
                               ("output", (dims["context"], dims["state"]))):
                value = arrays[key]
                require(value.shape == shape and value.dtype == np.dtype("<f2") and bool(np.isfinite(value).all())
                        and digest(value.tobytes()) == case["arrays"][key]["sha256"], "invalid fixture array: " + key)
                values[key] = value.copy()
            require(digest(values["positions"].tobytes()) == plan["payloads"]["positions"]["sha256"], "fixture positions mismatch")
        cases.append((case["name"], values))
    return cases


def benchmark(encoder, cases, warmups, runs):
    require(type(warmups) is int and warmups >= 0 and type(runs) is int and runs > 0, "invalid replay repetitions")
    records = []
    for name, arrays in cases:
        previous = None
        samples = []
        for index in range(warmups + runs):
            actual = encoder(arrays["mel"])
            reference = arrays["output"].astype(np.float32)
            current = actual.astype(np.float32)
            relative = float(np.linalg.norm(current - reference) / max(float(np.linalg.norm(reference)), 1e-40))
            require(relative < GATE["relative_l2"] and bool(np.allclose(current, reference, rtol=GATE["allclose_rtol"], atol=GATE["allclose_atol"])),
                    "Mac captured encoder output mismatch: " + name)
            bits = actual.tobytes()
            require(previous is None or previous == bits, "nonrepeatable encoder output: " + name)
            previous = bits
            samples.append(dict(phase="warmup" if index < warmups else "measured", index=index,
                                relative_l2=relative, output_sha256=digest(bits), **encoder.last_timing_ms))
        measured = [s for s in samples if s["phase"] == "measured"]
        records.append(dict(name=name, samples=samples, median_ms={k:statistics.median(s[k] for s in measured)
                       for k in ("prepare_ms", "dispatch_ms", "readback_ms", "total_ms")}))
    return dict(status="PASS_CAPTURED_ENCODER_REPLAY", records=records, submissions=encoder.submissions,
                warmups=warmups, measured_runs=runs, gate=GATE,
                timing_scope="dispatch is blocking Linux ioctl wall time; total includes input packing, scratch clear and readback; CPU cross-K/V/decoder excluded",
                strict_decoder_accuracy="unverified; captured encoder agreement is a separate gate")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "verify"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device")
    p.add_argument("--fixtures", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--runs", type=int, default=20)
    a = p.parse_args()
    if a.command == "prepare":
        meta, hwx, positions = reconstruct(a.checkpoint, a.kernels)
        plan, payloads = prepare(meta, hwx, positions)
        descriptor = native_descriptor(plan, payloads)
        a.output.mkdir(parents=True, exist_ok=False)
        for name, raw in payloads.items():
            (a.output / (name + ".bin")).write_bytes(raw)
        (a.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        (a.output / "native-layout.txt").write_text(descriptor)
        print(json.dumps(dict(status="PASS_REPLAY_PLAN_PREPARATION", model=plan["model"], tasks=plan["td_count"],
                             coefficient_banks=len(plan["coefficient_banks"]), hardware_validation="pending")))
        return
    require(platform.system() == "Linux", "verification requires native M1 Asahi Linux; no device access attempted")
    require(a.fixtures is not None and a.manifest is not None, "verify requires --fixtures and --manifest")
    # Validate all payloads and fixtures before opening the device or allocating BOs.
    meta, hwx, positions = reconstruct(a.checkpoint, a.kernels)
    plan, payloads = prepare(meta, hwx, positions)
    cases = validate_fixtures(plan, json.loads(a.manifest.read_text()), a.fixtures)
    del meta, hwx, positions, payloads
    encoder = None
    report = dict(status="FAILED_REPLAY", model=plan["model"], kernel=platform.release())
    try:
        encoder = Encoder(a.checkpoint, a.kernels, a.device)
        report.update(benchmark(encoder, cases, a.warmups, a.runs), model=plan["model"], kernel=platform.release(),
                      hwx_sha256=plan["hwx_sha256"], plan_payloads=plan["payloads"])
    except BaseException as error:
        report.update(status="FAILED_REPLAY", error=str(error))
        raise
    finally:
        if encoder is not None:
            encoder.close()
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
