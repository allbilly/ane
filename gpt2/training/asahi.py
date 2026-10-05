#!/usr/bin/env python3
"""Replay the captured GPT-2 training kernels through the Asahi DRM driver."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import resource
import struct
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from external_weights import find_weights, load_weights as external_load_weights, verify_weights
from model import LAYER_SHAPES
from hwx import parse_container, parse_tasks, relocate, require
from replay import Buffer, Submit, SUBMIT, device_path, ioctl
from backends import Primitives
from train import Adam, GPT2, TEXT, Tokenizer, load_weights, metrics


class Kernel:
    def __init__(self, backend, record):
        self.backend, self.record = backend, record
        self.buffers, self.bootstrap = {}, None
        path = ROOT / record["hwx"]
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == record["hwx_sha256"],
                f"training HWX checksum mismatch: {path}")
        container = parse_container(data)
        thread, segments = container["thread"], container["segments"]
        bars = thread["bars"]
        text = next(s for s in segments if s["name"] == "__TEXT")
        const = next(s for s in container["sections"]
                     if s["segment"] == "__TEXT" and s["name"] == "__const")
        require(bars[0] == thread["entry"] == text["vmaddr"] and bars[1] == const["addr"]
                and bars[2] == 0, "unexpected training command/constant BARs")
        raw = data[text["fileoff"]:text["fileoff"] + text["filesize"]]
        require(len(raw) % 16 == 0, "unaligned training command boundary")
        tasks = parse_tasks(raw, thread["td_size"], thread["td_count"])
        require(len(tasks) == record["task_count"], "training task count mismatch")
        banks = {i: i for i, addr in enumerate(bars) if addr}
        banks[1] = 2  # Linux synthesizes BAR 1 after the command buffer.
        coefficients = [s for s in segments if s["name"].startswith("__KERN")]
        require(len(coefficients) <= 1, "multiple training coefficient banks")
        weights, weight_bank = b"", None
        if coefficients:
            segment, = coefficients
            weight_bank = bars.index(segment["vmaddr"])
            require(weight_bank >= 4, "unexpected coefficient bank")
            banks[weight_bank] = 1
            weights = data[segment["fileoff"]:segment["fileoff"] + segment["filesize"]]
        program = relocate(raw, tasks, banks)
        status = plistlib.loads(path.with_name("model.hwx.status.plist").read_bytes())
        require(not status["ErrorList"], "compiler recorded training errors")
        network, = status["NetworkStatusList"]
        ports = {}
        for role, key in (("input", "LiveInputList"), ("output", "LiveOutputList")):
            for item in network[key]:
                name = item["Symbol"].removesuffix("@output")
                shape = tuple(item[k] for k in ("Batches", "Channels", "Height", "Width"))
                require(item["Type"] == "Float16" and item["Depth"] == 1
                        and item["Interleave"] == 1 and item["RowStride"] == shape[-1] * 2
                        and item["PlaneStride"] == shape[-2] * item["RowStride"]
                        and item["BatchStride"] == shape[1] * item["PlaneStride"],
                        f"unsupported training I/O strides: {name}")
                bank = bars.index(container["symbols"][name]["addr"])
                require(bank >= 3 and bank != weight_bank, "invalid training I/O bank")
                ports[name] = dict(role=role, shape=shape, bank=bank)
        self.inputs = [ports[name] if name is not None else None
                       for name in record["input_port_names"]]
        self.output = ports[record["output_port_name"]]
        require(self.output["role"] == "output"
                and self.output["shape"] == tuple(record["output"]), "output shape mismatch")
        for port, shape in zip(self.inputs, record["inputs"]):
            require(port is None or (port["role"] == "input" and port["shape"] == tuple(shape)),
                    "input shape mismatch")
        io_banks = {p["bank"] for p in ports.values()}
        self.scratch = []
        self.td_size, self.td_count, self.tsk_size = thread["td_size"], len(tasks), len(program)
        try:
            self.buffers[0] = Buffer(backend.fd, len(program) + len(weights) + 1)
            self.buffers[0].write(program + weights)
            self.buffers[2] = Buffer(backend.fd, const["size"])
            self.buffers[2].write(data[const["offset"]:const["offset"] + const["size"]])
            for bank, addr in enumerate(bars):
                if not addr or bank in (0, 1, weight_bank):
                    continue
                segment = next(s for s in segments if s["vmaddr"] == addr)
                require(segment["filesize"] == 0, "unexpected initialized training buffer")
                self.buffers[bank] = Buffer(backend.fd, segment["vmsize"])
                if bank not in io_banks:
                    self.scratch.append(bank)
            for port in ports.values():
                require(np.prod(port["shape"]) * 2 <= self.buffers[port["bank"]].size,
                        "training I/O exceeds buffer")
            self.bootstrap = Buffer(backend.fd, self.td_size)
            bootstrap = bytearray(program[:self.td_size])
            header, = struct.unpack_from("<I", bootstrap)
            struct.pack_into("<I", bootstrap, 0, (header & ~(0xFF << 16)) | (0x40 << 16))
            self.bootstrap.write(bootstrap)
            self.request = Submit(tsk_size=self.tsk_size, td_count=self.td_count,
                                  td_size=self.td_size, btsp_handle=self.bootstrap.handle)
            for bank, buffer in self.buffers.items():
                self.request.handles[bank] = buffer.handle
        except BaseException:
            self.close()
            raise

    def __call__(self, *arrays):
        require(len(arrays) == len(self.inputs), "training input count mismatch")
        for port, shape, array in zip(self.inputs, self.record["inputs"], arrays):
            array = np.asarray(array, dtype="<f2")
            require(array.shape == tuple(shape) and bool(np.isfinite(array).all()),
                    f"invalid input: {self.record['name']}")
            if port is not None:
                self.buffers[port["bank"]].write(array.tobytes(order="C"))
        for bank in self.scratch:
            buffer = self.buffers[bank]
            buffer.write(bytes(buffer.size))
        output = self.buffers[self.output["bank"]]
        count = int(np.prod(self.output["shape"]))
        output.write(np.full(count, np.nan, dtype="<f2").tobytes())
        start = time.perf_counter()
        ioctl(self.backend.fd, SUBMIT, self.request)
        self.backend.dispatch_seconds += time.perf_counter() - start
        self.backend.dispatches += 1
        result = np.frombuffer(output.map, dtype="<f2", count=count).reshape(self.output["shape"]).astype(np.float32)
        require(bool(np.isfinite(result).all()), f"nonfinite/unwritten output: {self.record['name']}")
        return result

    def close(self):
        buffers = list(self.buffers.values()) + ([self.bootstrap] if self.bootstrap else [])
        self.buffers, self.bootstrap = {}, None
        for buffer in reversed(buffers):
            buffer.close()


class Backend:
    def __init__(self, source="aneforge", device=None):
        self.path = device_path(device)
        records = json.loads((ROOT / "kernel-manifest.json").read_text())
        self.records = {r["name"]: r for r in records if r["backend"] == source}
        require(len(self.records) == 27, "expected 27 captured training templates")
        # Port identities were added to the capture index after HWX export.
        for record in json.loads((ROOT / source / "kernel-index.json").read_text()):
            self.records[record["name"]].update(record)
        self.source, self.cache = source, {}
        self.dispatches, self.dispatch_seconds = 0, 0.0
        self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)

    def program(self, name):
        require(name in self.records, f"no captured kernel for {name}; only sequence length 32 is supported")
        if name not in self.cache:
            self.cache[name] = Kernel(self, self.records[name])
        return self.cache[name]

    def run(self, name, arrays, builder):
        return self.program(name)(*arrays)

    def verify(self):
        results = {}
        for name, record in self.records.items():
            fixture = ROOT / self.source / Path(record["mil"]).parent / "fixture.npz"
            with np.load(fixture) as reference:
                actual = self.program(name)(*[reference[f"input{i:02d}"] for i in range(len(record["inputs"]))])
                expected = reference["output"].astype(np.float32)
            error = metrics(actual, expected)
            require(np.allclose(actual, expected, rtol=0.01, atol=0.03)
                    and error["relative_l2"] < 0.005, f"training fixture mismatch: {name}: {error}")
            results[name] = error
            print(f"PASS {self.source}/{name}: max_abs={error['max_abs']:.6g}", flush=True)
        return results

    def close(self):
        try:
            for kernel in self.cache.values():
                kernel.close()
            self.cache.clear()
        finally:
            os.close(self.fd)


def training_weights(source, prepared=None):
    if source.is_dir():
        return load_weights(source)
    prepared = prepared or external_load_weights(source)
    require(np.array_equal(prepared.get("wte", (50257, 768)), prepared.get("lm_head", (50257, 768))),
            "training requires tied token embeddings and output weights")
    weights = {}
    shapes = {"wte": (50257, 768), "wpe": (1024, 768), "ln_f_g": (768,), "ln_f_b": (768,)}
    shapes.update({f"layer{i}/{name}": shape for i in range(12) for name, shape in LAYER_SHAPES.items()})
    for name, shape in shapes.items():
        array = prepared.get(name, shape)
        if name in ("wte", "wpe"):
            pass
        elif name.rsplit("/", 1)[-1].startswith("w"):
            array = array.T.copy().reshape(1, 1, array.shape[1], array.shape[0])
        else:
            array = array.reshape(1, 1, 1, -1)
        weights[name] = array
    return weights


def train(backend, args, report):
    source = find_weights(args.weights)
    require(source is not None, "cached GPT-2 weights missing; run gpt2/gpt2.py setup")
    prepared = external_load_weights(source)
    report["verified_weight_files"] = verify_weights(source, ROOT.parent, weights=prepared)
    reference = getattr(prepared, "reference", True)
    report["reference_checkpoint"] = reference
    weights = training_weights(source, prepared)
    tokens = np.array(Tokenizer(ROOT.parent / "tokenizer").encode(TEXT)[:33], np.int64)
    protocol = json.loads((ROOT / backend.source / "results.json").read_text())
    require(tokens.tolist() == protocol["tokens"], "training batch differs from captured reference")
    model = GPT2(weights, Primitives(backend), tokens)
    optimizer = Adam(args.lr)
    report.update(model="GPT-2 124M", parameter_count=sum(a.size for a in weights.values()),
                  sequence_length=32, batch_size=1, steps=args.steps, lr=args.lr,
                  tokens=tokens.tolist(), initial_weights=str(source),
                  scope="Transformer forward/backward and parameter gradients on ANE; embeddings, vocabulary head/loss and fp32 Adam on CPU.",
                  phase_times=[])
    loss, cache = model.forward()
    report["initial_loss"] = loss
    report["macos_reference_loss"] = protocol["initial_loss"]
    if reference:
        require(abs(loss - protocol["initial_loss"]) < 0.01, "initial loss differs from macOS reference")
    del cache
    print(f"Initial loss: {loss:.8f}", flush=True)
    with (args.output / "loss.csv").open("w") as stream:
        fields = ["step", "loss", "forward_ms", "backward_ms", "optimizer_ms", "total_ms",
                  "ane_execute_ms", "ane_dispatches", "gradient_norm"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for step in range(1, args.steps + 1):
            before_count, before_time = backend.dispatches, backend.dispatch_seconds
            start = time.perf_counter()
            loss, cache = model.forward()
            forward = time.perf_counter()
            grads = model.backward(cache)
            backward = time.perf_counter()
            norm = optimizer.update(weights, grads)
            finish = time.perf_counter()
            if step == 1 and reference:
                reference_norm = protocol["phase_times"][0]["gradient_norm"]
                report["initial_gradient_norm"] = norm
                report["macos_reference_gradient_norm"] = reference_norm
                require(abs(norm / reference_norm - 1.0) < 0.01,
                        "initial full-model gradient norm differs from macOS reference")
            row = dict(step=step, loss=loss, forward_ms=(forward - start) * 1000,
                       backward_ms=(backward - forward) * 1000, optimizer_ms=(finish - backward) * 1000,
                       total_ms=(finish - start) * 1000, ane_execute_ms=(backend.dispatch_seconds - before_time) * 1000,
                       ane_dispatches=backend.dispatches - before_count, gradient_norm=norm)
            require(row["ane_dispatches"] == 580 and np.isfinite(norm), "incomplete training dispatch/gradient")
            writer.writerow(row)
            stream.flush()
            report["phase_times"].append(row)
            print(f"step {step}: loss={loss:.6f}; {row['total_ms']:.1f} ms; {row['ane_dispatches']} ANE dispatches", flush=True)
            del grads, cache
    final_loss, cache = model.forward()
    del cache
    report["final_loss_after_updates"] = final_loss
    report["loss_decreased"] = final_loss < report["initial_loss"]
    require(report["loss_decreased"], "training loss did not decrease")
    if args.checkpoint:
        name = f"checkpoint-step{args.steps}.npz"
        np.savez(args.output / name, **weights)
        report["checkpoint"] = name
    print(f"PASS: loss {report['initial_loss']:.8f} -> {final_loss:.8f} after {args.steps} updates", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "train"])
    parser.add_argument("--kernel-source", choices=["aneforge", "orion"], default="aneforge")
    parser.add_argument("--device")
    parser.add_argument("--weights")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "asahi-output")
    args = parser.parse_args()
    require(args.steps > 0 and args.lr > 0, "steps and learning rate must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(backend="asahi", kernel_source=args.kernel_source, kernel=platform.release(),
                  command=args.command, status="running")
    backend = Backend(args.kernel_source, args.device)
    report["device"] = str(backend.path)
    try:
        report["fixtures"] = backend.verify()
        if args.command == "train":
            train(backend, args, report)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["total_dispatches"] = backend.dispatches
        report["ane_execute_seconds"] = backend.dispatch_seconds
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        backend.close()


if __name__ == "__main__":
    main()
