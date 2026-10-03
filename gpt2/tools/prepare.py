#!/usr/bin/env python3
"""Build the portable package from an Orion checkout and its verified dump."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import plistlib
import re
import shutil
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwx import parse_container, parse_tasks, relocate, require


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def object_file(root, data):
    name = f"objects/{sha(data)}.bin"
    target = root / name
    if not target.exists():
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
    return name


def verify_model_binding(bundle, mil, model, kernel):
    """Check CPU parameters are byte-identical to all packed MIL tensors."""
    names = {"ln1_g": "ln1_g", "ln1_beta": "ln1_b", "ln2_g": "ln2_g", "ln2_beta": "ln2_b",
             "q_W": "wq", "k_W": "wk", "v_W": "wv", "q_b": "bq", "k_b": "bk", "v_b": "bv",
             "proj_W": "wo", "proj_b": "bo", "ffn_fc_W": "wfc", "ffn_fc_b": "bfc",
             "ffn_proj_W": "wproj", "ffn_proj_b": "bproj", "lnf_g": "ln_f_g", "lnf_beta": "ln_f_b"}
    packed = (bundle / "weights/packed.bin").read_bytes()
    layer = None if kernel.endswith("L-1") else int(kernel.rsplit("_L", 1)[1])
    count = 0
    for line in mil.decode().splitlines():
        if "BLOBFILE" not in line:
            continue
        field = re.search(r"> (\w+) = const", line).group(1)
        if field == "attn_mask":
            continue  # fixed causal mask, not a CPU model parameter
        require(field in names, f"unknown model parameter {field}")
        offset = int(re.search(r"offset=uint64\((\d+)\)", line).group(1))
        require(offset + 64 <= len(packed), "truncated BLOB metadata")
        magic, dtype, size, payload = struct.unpack_from("<IIQQ", packed, offset)
        require((magic, dtype) == (0xDEADBEEF, 1) and payload + size <= len(packed), "invalid fp16 BLOB")
        path = model / ((f"layer{layer}/" if layer is not None else "") + names[field] + ".bin")
        raw = path.read_bytes()
        require(raw[128:] == packed[payload:payload + size], f"CPU/ANE model weight mismatch: {kernel}/{field}")
        count += 1
    return count


def export_kernel(root, dump, record, model):
    name = record["kernel"]
    source = dump / "hwx" / name / "model.hwx"
    data = source.read_bytes()
    require(sha(data) == record["hwx_sha256"], f"HWX checksum mismatch: {name}")
    bundle = dump / "bundles" / (name + "_loaded")
    mil = (bundle / "model.mil").read_bytes()
    require(sha(mil) == record["mil_sha256"], f"MIL checksum mismatch: {name}")
    bound_tensors = verify_model_binding(bundle, mil, model, name)
    require(sha((bundle / "weights/packed.bin").read_bytes()) == record["weights_sha256"], f"weight checksum mismatch: {name}")
    container = parse_container(data)
    segments, sections, thread = container["segments"], container["sections"], container["thread"]
    text = next(s for s in segments if s["name"] == "__TEXT")
    kern = next(s for s in segments if s["name"] == "__KERN_0")
    require(len([s for s in segments if s["name"].startswith("__KERN")]) == 1, "multiple weight banks unsupported")
    const = next(s for s in sections if s["segment"] == "__TEXT" and s["name"] == "__const")
    executable = next(s for s in sections if s["segment"] == "__TEXT" and s["name"] == "__text")
    bars = thread["bars"]
    require(bars[0] == text["vmaddr"] == thread["entry"] == executable["addr"], "unexpected entry address")
    require(bars[1] == const["addr"] and bars[2] == 0, "BAR 2 must be available for constants")
    kernel_bank = bars.index(kern["vmaddr"])
    require(kernel_bank >= 4 and text["filesize"] % 16 == 0, "unexpected kernel layout")
    raw_text = data[text["fileoff"]:text["fileoff"] + text["filesize"]]
    tasks = parse_tasks(raw_text, thread["td_size"], thread["td_count"])
    require(len(tasks) == record["tasks"], "source manifest task count mismatch")
    bank_map = {i: i for i, addr in enumerate(bars) if addr}
    bank_map[1], bank_map[kernel_bank] = 2, 1
    program = relocate(raw_text, tasks, bank_map)
    attrs = plistlib.loads((bundle / "attributes.plist").read_bytes())
    network, = attrs["NetworkStatusList"]
    io = []
    buffers = []
    for old, addr in enumerate(bars):
        if not addr or old in (0, 1, kernel_bank):
            continue
        segment = next(s for s in segments if s["vmaddr"] == addr)
        require(segment["filesize"] == 0 and segment["vmsize"] % 0x4000 == 0, "unexpected I/O segment")
        buffers.append(dict(bank=old, size=segment["vmsize"], role="scratch" if old == 3 else "unknown"))
    for role, key in (("input", "LiveInputList"), ("output", "LiveOutputList")):
        for item in network[key]:
            symbol = item["Symbol"].removesuffix("@output")
            addr = container["symbols"][symbol]["addr"]
            bank = bars.index(addr)
            buffer = next(b for b in buffers if b["bank"] == bank)
            require(buffer["role"] == "unknown" and item["Type"] == "Float16", "unexpected/duplicate I/O")
            shape = [item[k] for k in ("Batches", "Channels", "Depth", "Height", "Width")]
            require(shape == [1, 768, 1, 1, 32], "expected 768 channels with 32 fp16 positions")
            require(item["PlaneStride"] == 64 and item["RowStride"] == 64 and item["BatchStride"] == 49152, "unexpected I/O strides")
            require(buffer["size"] >= 49152, "I/O buffer too small")
            buffer.update(role=role, name=symbol)
            io.append(dict(bank=bank, role=role, name=symbol, shape=[1, 768, 1, 32], bytes=49152))
    require(all(b["role"] != "unknown" for b in buffers), "unresolved buffer")
    constants = data[const["offset"]:const["offset"] + const["size"]]
    # BAR 1 is synthesized by the Linux driver at align16(tsk_size). The
    # entire __KERN_0 bank, including padding, follows the original __TEXT.
    kernel = data[kern["fileoff"]:kern["fileoff"] + kern["filesize"]]
    meta = dict(kernel=name, td_count=len(tasks), td_size=thread["td_size"], tsk_size=len(program),
                program=object_file(root, program), weights=object_file(root, kernel),
                constants=object_file(root, constants), bank_map=bank_map, buffers=buffers, io=io,
                source_hwx_sha256=record["hwx_sha256"], source_weights_sha256=record["weights_sha256"],
                source_bars=bars, verified_model_tensors=bound_tensors,
                extended_headers=sum(bool(t["header"][6] & (1 << 24)) for t in tasks))
    destination = root / "kernels" / name
    write_json(destination / "meta.json", meta)
    (destination / "model.mil").write_bytes(mil)
    # Corrected reference: do not let the old parser consume alignment padding
    # or treat the extended-header word as a register packet.
    ref = dict(format="strict H13G tasks; original unrelocated headers/registers", tasks=tasks)
    (destination / "registers.json.gz").write_bytes(gzip.compress(json.dumps(ref).encode(), mtime=0))
    return meta


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dump", type=Path, required=True)
    cli.add_argument("--orion", type=Path, required=True)
    cli.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    args = cli.parse_args()
    root, dump, orion = args.output.resolve(), args.dump.resolve(), args.orion.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dump / "manifest.json").read_text())
    expected = {f"{kind}_L{i}" for kind in ("decode_proj", "decode_ffn", "prefill_attn", "prefill_ffn") for i in range(12)} | {"prefill_final_ln_L-1"}
    require({r["kernel"] for r in manifest["kernels"]} == expected and len(manifest["kernels"]) == 49, "expected all 49 kernels")
    metas = [export_kernel(root, dump, r, orion / "model/blobs/gpt2_124m") for r in manifest["kernels"]]
    model_records = {}
    for source in (orion / "model/blobs/gpt2_124m").rglob("*.bin"):
        rel = source.relative_to(orion / "model/blobs/gpt2_124m")
        model_records[str(rel)] = sha(source.read_bytes())
    write_json(root / "model-checksums.json", model_records)
    tokenizer = root / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    for name in ("vocab.json", "merges.txt"):
        shutil.copyfile(orion / "tokenizer/data" / name, tokenizer / name)
    shutil.copyfile(orion / "LICENSE", root / "LICENSE-Orion")
    license_source = Path(__file__).resolve().parents[1] / "LICENSE-GPT2"
    if license_source != root / "LICENSE-GPT2":
        shutil.copyfile(license_source, root / "LICENSE-GPT2")
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=True)
    shutil.copyfile(dump / "manifest.json", provenance / "original-manifest.json")
    shutil.copyfile(dump / "inference.log", provenance / "original-inference.log")
    revision = subprocess.check_output(["git", "-C", str(orion), "rev-parse", "HEAD"], text=True).strip()
    write_json(root / "package.json", dict(format_version=1, architecture="H13G/M1", context=1024,
               decode_positions=32, kernels=[m["kernel"] for m in metas],
               source_orion_commit=revision, source_orion_dirty=bool(subprocess.check_output(["git", "-C", str(orion), "status", "--porcelain"])),
               source_dump_manifest_sha256=sha((dump / "manifest.json").read_bytes()),
               verified_model_tensors=sum(m["verified_model_tensors"] for m in metas),
               linux_hardware_verified=False, runtime="ANE decode_proj/FFN; CPU attention/embeddings/logits; sequential prompt ingest",
               weight_storage="external HF checkpoint plus generated H13G cache",
               packing="reference-verified fp16 matrix/bias tiles and folded LN affine",
               learned_weights_packaged=False))
    # Strip learned parameters only after every reference payload has been
    # reconstructed exactly. This prevents rebuilding a weight-bundled port.
    subprocess.run([sys.executable, str(Path(__file__).with_name("derive_packing.py")),
                    "--dump", str(dump), "--root", str(root),
                    "--weights", str(orion / "model/blobs/gpt2_124m"), "--prune"], check=True)
    print(f"Prepared {len(metas)} kernels, {sum(m['td_count'] for m in metas)} tasks; original dump untouched.")


if __name__ == "__main__":
    main()
