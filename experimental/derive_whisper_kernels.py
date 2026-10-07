"""Strip learned values from the validated encoder; derive exact pinned recipes."""
import argparse
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

from experimental.capture_macos_program import parse_container
from qwen35.weights import SafeTensors, sha256
from whisper.encoder_kernel import reconstruct, require


def convolution_scatter(coefficients, weights):
    """Locate compiler packet values without storing any learned coefficient bits."""
    vectors = weights.reshape(96, 4, 384, 3).transpose(0, 2, 3, 1).copy().reshape(-1, 4)
    start, end = 187392, 1228800
    sub = np.frombuffer(coefficients[start:end], np.uint8)
    index = np.lib.stride_tricks.as_strided(sub, shape=(len(sub) - 3, 4), strides=(1, 1)).copy().view("<u4").ravel()
    order = index.argsort()
    values = index[order]
    keys = vectors.view("<u4").reshape(-1, 2)
    positions = np.full(keys.shape, -1, dtype="<i4")
    pending = []
    for i in range(len(keys)):
        for j in range(2):
            lo, hi = np.searchsorted(values, keys[i, j], side="left"), np.searchsorted(values, keys[i, j], side="right")
            candidates = order[lo:hi] + start
            if len(candidates) == 1:
                positions[i, j] = candidates[0]
            else:
                pending.append((i, j, candidates))
    occupied = np.zeros(len(coefficients), bool)
    for offset in positions[positions >= 0]:
        require(not occupied[offset:offset + 4].any(), "ambiguous unique convolution value")
        occupied[offset:offset + 4] = True
    for i, j, candidates in pending:
        candidates = np.array([p for p in candidates if not occupied[p:p + 4].any()])
        if not len(candidates):
            continue
        neighbours = [positions[k, j] for k in range(max(i // 1152 * 1152, i - 4), min((i // 1152 + 1) * 1152, i + 5))
                      if k != i and positions[k, j] >= 0]
        require(bool(neighbours), "convolution value has no packet neighbour")
        distances = np.min(np.abs(candidates[:, None] - np.array(neighbours)), axis=1)
        require(distances.min() <= 150 and np.sum(distances == distances.min()) == 1, "ambiguous convolution packet")
        positions[i, j] = candidates[distances.argmin()]
        occupied[positions[i, j]:positions[i, j] + 4] = True
    extra = []
    for i, j in np.argwhere(positions < 0):
        require((vectors[i, j * 2:j * 2 + 2] == 0).sum() == 1, "unexpected sparse convolution pair")
        near = min(positions[k, j] for k in range(i - 3, i + 4) if positions[k, j] >= 0)
        for k in range(2):
            scalar = vectors[i, j * 2 + k]
            if scalar == 0:
                continue
            locations = [p for p in range(near - 40, near + 150)
                         if coefficients[p:p + 2] == scalar.tobytes() and not occupied[p:p + 2].any()]
            require(len(locations) == 1, "ambiguous sparse convolution scalar")
            extra.append([int(i), int(j * 2 + k), int(locations[0])])
            occupied[locations[0]:locations[0] + 2] = True
    require(int(occupied.sum()) == weights.nbytes - 2, "incomplete convolution packing coverage")
    return positions, extra


def derive(capture, checkpoint, scatter_path, output):
    capture, output = Path(capture), Path(output)
    hwx = (capture / "hwx/model.hwx").read_bytes()
    container = parse_container(hwx)
    receipt = json.loads((capture / "hwx/receipt.json").read_text())
    task_count = container["thread"]["td_count"]
    require(task_count in (1779, 1783) and hashlib.sha256(hwx).hexdigest() == receipt["hwx_sha256"],
            "expected the intact complete encoder capture")
    recovery = json.loads((capture / "recovery.json").read_text())
    require(recovery["status"] == "recovered_and_verified" and recovery["macos_ane_bitwise_equal"], "unvalidated encoder capture")
    text = next(s for s in container["segments"] if s["name"] == "__TEXT")
    const = next(s for s in container["sections"] if s["segment"] == "__TEXT" and s["name"] == "__const")
    kern = next(s for s in container["segments"] if s["name"] == "__KERN_0")
    tensors = SafeTensors(checkpoint)
    replay = capture / "replay"
    original = {name:(replay / filename).read_bytes() for name, filename in
                (("commands", "commands-asahi.bin"), ("constants", "constants.bin"), ("coefficients", "coefficients.bin"))}
    original["source-weights"] = (capture / "bundle/weights.bin").read_bytes()
    templates = {k:bytearray(v) for k, v in original.items()}
    used = {k:np.zeros(len(v), bool) for k, v in original.items()}
    recipes = {k:[] for k in original}
    def add(name, operation, value):
        offset = operation["offset"]
        require(original[name][offset:offset + len(value)] == value and not used[name][offset:offset + len(value)].any(),
                "missing/overlapping source tensor: " + str(operation))
        operation["bytes"] = len(value)
        recipes[name].append(operation)
        used[name][offset:offset + len(value)] = True
        templates[name][offset:offset + len(value)] = bytes(len(value))
    coeff = original["coefficients"]
    names = sorted(n for n in tensors.header if n.startswith("model.encoder.") and n.endswith(".weight")
                   and len(tensors.header[n]["shape"]) >= 2 and "embed_positions" not in n)
    for name in names:
        if name.endswith("conv2.weight"):
            continue
        w = tensors.tensor(name).astype("<f2")
        bias_name = name[:-6] + "bias"
        b = tensors.tensor(bias_name).astype("<f2") if bias_name in tensors.header else None
        first = 0
        while first < len(w):
            found = []
            for count in (16, 8):
                if first + count > len(w):
                    continue
                value = w[first:first + count].reshape(count, -1).T.tobytes()
                offset = coeff.find(value)
                if offset >= 0 and coeff.find(value, offset + 1) < 0:
                    found.append((count, offset, value))
            require(bool(found), "unrecognized matrix tile: " + name + ":" + str(first))
            count, offset, value = found[0]
            add("coefficients", dict(kind="tile", tensor=name, first=first, count=count, offset=offset), value)
            if b is not None:
                add("coefficients", dict(kind="tile", tensor=bias_name, first=first, count=count, offset=offset - count * 2),
                    b[first:first + count].tobytes())
            first += count
    b = tensors.tensor("model.encoder.conv2.bias").astype("<f2")
    for first, count in [(i, 10) for i in range(0, 320, 10)] + [(i, 4) for i in range(320, 384, 4)]:
        value = b[first:first + count].tobytes()
        offset = coeff.find(value)
        require(offset >= 0 and coeff.find(value, offset + 1) < 0, "ambiguous conv2 bias")
        add("coefficients", dict(kind="tile", tensor="model.encoder.conv2.bias", first=first, count=count, offset=offset), value)
    conv2 = tensors.tensor("model.encoder.conv2.weight").astype("<f2")
    if scatter_path:
        with np.load(scatter_path, allow_pickle=False) as scatter:
            offsets, extra = scatter["positions"].astype("<i4"), scatter["extra"].tolist()
    else:
        offsets, extra = convolution_scatter(coeff, conv2)
    source = conv2.reshape(96, 4, 384, 3).transpose(0, 2, 3, 1).copy().reshape(-1, 4)
    require(offsets.shape == (len(source), 2), "bad convolution map")
    target = np.frombuffer(templates["coefficients"], np.uint8)
    original_view = np.frombuffer(coeff, np.uint8)
    for pair in range(2):
        valid = offsets[:, pair] >= 0
        addresses = offsets[valid, pair, None].astype(np.int64) + np.arange(4)
        value = source[:, pair * 2:pair * 2 + 2].copy().view(np.uint8).reshape(-1, 4)[valid]
        require(addresses.min() >= 0 and addresses.max() < len(coeff) and len(np.unique(addresses)) == addresses.size
                and not used["coefficients"][addresses].any() and np.array_equal(original_view[addresses], value),
                "invalid learned convolution map")
        target[addresses] = 0
        used["coefficients"][addresses] = True
    for index, channel, offset in extra:
        value = source[index, channel].tobytes()
        require(coeff[offset:offset + 2] == value and not used["coefficients"][offset:offset + 2].any(), "bad sparse convolution scalar")
        templates["coefficients"][offset:offset + 2] = b"\0\0"
        used["coefficients"][offset:offset + 2] = True
    coefficient_bytes = sum(tensors.tensor(n).size * 2 for n in tensors.header
                            if n.startswith("model.encoder.") and "embed_positions" not in n and "layer_norm" not in n)
    require(int(used["coefficients"].sum()) == coefficient_bytes - 2, "incomplete encoder coefficient coverage")
    for name in sorted(n for n in tensors.header if n.startswith("model.encoder.") and "layer_norm" in n and n.endswith(".weight")):
        gamma = tensors.tensor(name).astype("<f2")
        beta_name = name[:-6] + "bias"
        ratio = (tensors.tensor(beta_name).astype("<f2").astype(np.float32) / gamma.astype(np.float32)).astype("<f2")
        for key, value, kind in ((name, gamma, "tensor"), (beta_name, ratio, "ratio")):
            raw = value.tobytes()
            offset = original["constants"].find(raw)
            require(offset >= 0 and original["constants"].find(raw, offset + 1) < 0, "ambiguous layer norm")
            operation = dict(kind=kind, tensor=key, offset=offset)
            if kind == "ratio":
                operation["gamma"] = name
            add("constants", operation.copy(), raw)
            operation["offset"] += const["offset"] - text["fileoff"]
            add("commands", operation, raw)
    source_bytes = 0
    for name in sorted(n for n in tensors.header if n.startswith("model.encoder.") and "embed_positions" not in n):
        value = tensors.tensor(name).astype("<f2").tobytes()
        offset = original["source-weights"].find(value)
        require(offset >= 0 and original["source-weights"].find(value, offset + 1) < 0, "ambiguous MIL source tensor")
        add("source-weights", dict(kind="tensor", tensor=name, offset=offset), value)
        source_bytes += len(value)
    output.mkdir(parents=True, exist_ok=False)
    def asset(name, raw):
        (output / name).write_bytes(zlib.compress(raw, 9))
        return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    payloads = {}
    for name, raw in original.items():
        filename = name + ".template.zlib"
        payloads[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), template=filename,
                              template_metadata=asset(filename, templates[name]), operations=recipes[name])
    filename = "conv2-offsets.i32.zlib"
    deltas = np.diff(offsets.ravel().astype(np.int64), prepend=0).astype("<i4")
    payloads["coefficients"]["scatter"] = dict(file=filename, tensor="model.encoder.conv2.weight", extra=extra,
                                                encoding="delta-i32", **asset(filename, deltas.tobytes()))
    packing = {name:dict(operations=recipe.pop("operations")) for name, recipe in payloads.items()}
    packing["coefficients"]["scatter"] = payloads["coefficients"].pop("scatter")
    packing_file = "packing.json.zlib"
    packing_metadata = asset(packing_file, json.dumps(packing, separators=(",", ":")).encode())
    mil = (capture / "bundle/model.mil").read_bytes()
    layout = json.loads((replay / "buffers.json").read_text())
    source_report = json.loads((capture / "report.json").read_text())
    meta = dict(format="whisper-tiny-en-h13g/v1", target="apple,t8103", generation="H13G", td_count=task_count,
                td_size=container["thread"]["td_size"], checkpoint=dict(model="openai/whisper-tiny.en",
                 revision="87c7102498dcde7456f24cfd30239ca606ed9063", sha256=sha256(checkpoint)),
                capture_hwx_sha256=hashlib.sha256(hwx).hexdigest(), payloads=payloads, layout=layout,
                packing=dict(file=packing_file, **packing_metadata),
                compiler_strings=recovery["compiler_strings"], runtime=recovery["runtime"],
                mil=dict(file="model.mil.zlib", **asset("model.mil.zlib", mil)),
                position_sha256=sha256(capture / "bundle/pos.f16"),
                scope=f"All {task_count} tasks. Templates retain instructions, compiler tables, sparse masks and padding; checkpoint tensors are external.",
                linux_hardware_validation="pending", original_thread=container["thread"], coefficient_segment=kern)
    if "original_fast_mil_sha256" in source_report:
        meta.update(original_fast_mil_sha256=source_report["original_fast_mil_sha256"],
                    compiler_binary_sha256=source_report["compiler_binary_sha256"],
                    recapture_matches_original_hwx=source_report["recapture_matches_original_hwx"],
                    recapture_matches_original_payloads=source_report["recapture_matches_original_payloads"],
                    executable_identity_limit=source_report["executable_identity_limit"])
    (output / "meta.json").write_text(json.dumps(meta, separators=(",", ":")) + "\n")
    _, rebuilt = reconstruct(checkpoint, output)
    require(all(rebuilt[k] == v for k, v in original.items()), "byte-exact reconstruction failed")
    proof = dict(status="pass", checkpoint_sha256=sha256(checkpoint), task_count=task_count,
                 source_capture=recovery["source_capture"], recovery_report_sha256=sha256(capture / "recovery.json"),
                 byte_exact_payloads={k:payloads[k]["sha256"] for k in original},
                 stripped_coefficient_bytes=int(used["coefficients"].sum()),
                 stripped_source_weight_bytes=source_bytes,
                 package_bytes_excluding_proof=sum(p.stat().st_size for p in output.iterdir()),
                 runtime_validation=json.loads((capture / "macos-runtime-validation.json").read_text()),
                 fixtures=json.loads((capture / "asahi-fixtures.json").read_text()))
    if "accuracy_status" in source_report:
        proof.update(accuracy_status=source_report["accuracy_status"],
                     validation_report_sha256=sha256(capture / "report.json"),
                     decoder_gate_summaries=[dict(audio=r["audio"], summaries=r["summaries"])
                                            for r in source_report["decoder_validation"]])
    (output / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return dict(status="pass", package_bytes=sum(p.stat().st_size for p in output.iterdir()), task_count=task_count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--conv2-scatter", type=Path, help="Optional previously derived offset map; every mapped byte is verified")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(derive(a.capture, a.checkpoint, a.conv2_scatter, a.output)))


if __name__ == "__main__":
    main()
