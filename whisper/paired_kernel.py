"""Checkpoint-only reconstruction of the 24 paired H13G projection programs."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np

from gpt2.hwx import parse_tasks
from whisper.encoder_kernel import port_shape, port_view, require, unpack
from whisper.precision_encoder import CHECKPOINT_SHA256
from whisper.scripts.prepare_macos_precision import checkpoint_weights

ROOT = Path(__file__).resolve().parent / "kernels/tiny-en-paired"
FORMAT = "whisper-tiny-en-paired-h13g/v1"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pack_coefficients(weights, operations, size):
    """Fill every coefficient byte from exact grouped checkpoint FP16 values."""
    weights = np.asarray(weights)
    require(weights.ndim == 2 and bool(np.isfinite(weights).all()), "invalid paired projection weights")
    n, k = weights.shape
    require(k % 2 == 0 and np.array_equal(weights, weights.astype("<f2").astype(np.float32)),
            "paired checkpoint weights require residual planes")
    grouped = np.concatenate((weights[:, :k//2], weights[:, k//2:]), axis=0).astype("<f2")
    require(size == weights.size*2, "paired coefficient size differs from checkpoint")
    result = bytearray(size)
    occupied = np.zeros(size, bool)
    covered_rows = np.zeros(2*n, bool)
    for row in operations:
        first, count, offset = (row[key] for key in ("first", "count", "offset"))
        require(all(type(v) is int for v in (first,count,offset)) and count in (8,16)
                and 0 <= first <= 2*n-count, "invalid paired coefficient tile")
        value = grouped[first:first+count].T.copy().tobytes()
        require(0 <= offset <= size-len(value) and not occupied[offset:offset+len(value)].any()
                and not covered_rows[first:first+count].any(), "overlapping paired coefficient tiles")
        result[offset:offset+len(value)] = value
        occupied[offset:offset+len(value)] = True
        covered_rows[first:first+count] = True
    require(occupied.all() and covered_rows.all(), "incomplete paired coefficient coverage")
    return bytes(result)


def validate_layout(meta):
    require(meta["target"] == "apple,t8103" and meta["format"] == FORMAT,
            "paired kernels require base-M1 H13G")
    require((meta["positions"],meta["temporal_planes"],meta["contraction_partitions"],meta["gains"])
            == (1500,4,2,[1.,1.375]), "paired arithmetic contract changed")
    k,n = meta["input_features"],meta["output_features"]
    require((k,n) in ((384,384),(384,1536),(1536,384)), "unsupported paired projection dimensions")
    expected_tasks = 1 if (k,n)==(384,384) else 2
    require(meta["td_count"] == expected_tasks and meta["td_size"] == 628,
            "paired task geometry changed")
    buffers = {b["replay_bank"]:b["size"] for b in meta["layout"]["buffers"]}
    require(len(meta["layout"]["buffers"]) == 2 and set(buffers)=={4,5}, "paired buffer banks changed")
    ports = meta["layout"]["ports"]
    require(len(ports)==2, "paired projection requires one input and one output")
    for role,bank,channels,name in (("input",4,k,"t0"),("output",5,2*n,"t1")):
        matches = [p for p in ports if p["role"]==role]
        require(len(matches)==1, "ambiguous paired port")
        port = matches[0]
        require(port["name"]==name and port["replay_bank"]==bank and port["byte_offset"]==0
                and port_shape(port)==(1,channels,1,6000), "paired port geometry changed")
        view = port_view(bytearray(buffers[bank]),port)
        require(view.strides==(channels*12032,12032,12032,2)
                and buffers[bank]==channels*12032, "paired padded port strides changed")


def reconstruct(checkpoint, root=ROOT):
    """Verify pinned external weights and reconstruct all 24 replay payloads."""
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    require(manifest["format"]==FORMAT and manifest["checkpoint_sha256"]==CHECKPOINT_SHA256,
            "paired manifest identity changed")
    weights = checkpoint_weights(Path(checkpoint))
    require(len(manifest["programs"])==24 and {p["name"] for p in manifest["programs"]}==set(weights),
            "paired encoder requires all 24 projections")
    kernels = {}
    for name in sorted({p["kernel"] for p in manifest["programs"]}):
        require(name in ("384-384","384-1536","1536-384"), "unknown paired kernel")
        path = root/name
        meta = json.loads((path/"meta.json").read_text())
        require(sha256((path/"meta.json").read_bytes())==manifest["kernel_meta_sha256"][name],
                "paired kernel metadata changed")
        validate_layout(meta)
        commands = unpack(path,meta["commands"]["file"],meta["commands"])
        constants = unpack(path,meta["constants"]["file"],meta["constants"])
        parse_tasks(commands,meta["td_size"],meta["td_count"])
        require(len(commands)%16==0 and len(constants)==16384, "paired command/constant boundaries changed")
        # Driver BAR1 points immediately after commands in BAR0's allocation.
        bootstrap = bytearray(commands[:meta["td_size"]])
        header, = struct.unpack_from("<I",bootstrap)
        struct.pack_into("<I",bootstrap,0,(header & ~(0xFF<<16)) | (0x40<<16))
        kernels[name] = (meta,commands,constants,bytes(bootstrap))
    results = {}
    for record in manifest["programs"]:
        name = record["name"]
        weight = weights[name]
        n,k = weight.shape
        require(record["kernel"]==f"{k}-{n}", "paired weight/kernel dimensions differ")
        meta,commands,constants,bootstrap = kernels[record["kernel"]]
        coefficients = pack_coefficients(weight,meta["coefficient_tiles"],meta["coefficient_bytes"])
        require(sha256(coefficients)==record["coefficients_sha256"],
                "paired coefficients differ from captured checkpoint: "+name)
        results[name] = dict(meta=meta,commands=commands,constants=constants,
                            coefficients=coefficients,bootstrap=bootstrap)
        # Same grouped BLOBFILE that the shared native Mac/Linux adapter checks
        # against whisper.cpp's F16 weights before accepting a projection.
        grouped = np.concatenate((weight[:, :k//2], weight[:, k//2:]), axis=0).astype("<f2").tobytes()
        header = bytearray(128)
        struct.pack_into("<II", header, 0, 1, 2)
        struct.pack_into("<IIQQ", header, 64, 0xdeadbeef, 1, len(grouped), 128)
        results[name]["weights"] = bytes(header) + grouped
    return manifest,results


def native_descriptor(program):
    meta = program["meta"]
    layout = [meta["td_count"], meta["td_size"], len(program["commands"]),
              len(program["coefficients"]), len(program["constants"]), meta["input_features"],
              meta["output_features"], 12032, meta["input_features"]*12032, 2*meta["output_features"]*12032]
    checksums = [zlib.crc32(program[payload]) for payload in
                 ("commands", "constants", "coefficients", "bootstrap", "weights")]
    return "ANE_WHISPER_PAIRED_V1 " + " ".join(map(str, layout + checksums)) + "\n"


def validate_payloads(output, checkpoint, root=ROOT):
    """Bind native payload files to the pinned weights and captured templates."""
    output = Path(output)
    manifest, programs = reconstruct(checkpoint, root)
    require(json.loads((output/"manifest.json").read_text()) == manifest, "paired payload manifest changed")
    for name, program in programs.items():
        path = output/name
        for payload in ("commands", "constants", "coefficients", "bootstrap", "weights"):
            require((path/(payload+".bin")).read_bytes() == program[payload],
                    "paired native payload differs from checkpoint/template: " + name + "/" + payload)
        require((path/"native-layout.txt").read_text() == native_descriptor(program),
                "paired native descriptor changed: " + name)
    return manifest


def write_payloads(output, manifest, programs):
    """Write native-runner payloads into a new directory; no compiler required."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    for name,program in programs.items():
        path=output/name
        path.mkdir()
        for payload in ("commands","constants","coefficients","bootstrap","weights"):
            (path/(payload+".bin")).write_bytes(program[payload])
        meta=program["meta"]
        layout=[meta["td_count"],meta["td_size"],len(program["commands"]),
                len(program["coefficients"]),len(program["constants"]),meta["input_features"],
                meta["output_features"],12032,meta["input_features"]*12032,2*meta["output_features"]*12032]
        (path/"layout.txt").write_text(" ".join(map(str,layout))+"\n")
        (path/"native-layout.txt").write_text(native_descriptor(program))
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--kernels",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    manifest,programs=reconstruct(args.checkpoint,args.kernels)
    write_payloads(args.output,manifest,programs)
    print(json.dumps(dict(status="PASS_RECONSTRUCTION",programs=len(programs),
        tasks_per_encode=sum(p["meta"]["td_count"] for p in programs.values()),
        submissions_per_encode=24,linux_hardware_replay="pending",output=str(args.output))))


if __name__=="__main__":
    main()
