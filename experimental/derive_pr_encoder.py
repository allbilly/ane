"""Learn PR 3905 packet locations; save instructions and zeroed weight templates."""
import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import zlib

import numpy as np

from gpt2.hwx import parse_container, parse_tasks
from qwen35.weights import sha256
from whisper.encoder_kernel import require
from whisper.encoder_checkpoints import tensor_manifest
from whisper.ggml_encoder_weights import read_encoder
from whisper.pr_encoder_kernel import reconstruct, scatter_vectors, goc_affine


def convolution_scatter(coeff, weight, start, end, occupied):
    """Locate four-byte pairs using packet neighbours to resolve repeated values."""
    vectors = scatter_vectors(weight)
    sub = np.frombuffer(coeff[start:end], np.uint8)
    index = np.lib.stride_tricks.sliding_window_view(sub, 4).copy().view("<u4").ravel()
    order = index.argsort()
    values = index[order]
    keys = vectors.view("<u4").reshape(-1, 2)
    lo = np.searchsorted(values, keys, side="left")
    hi = np.searchsorted(values, keys, side="right")
    positions = np.full(keys.shape, -1, dtype="<i4")
    sparse = (vectors.reshape(-1, 2, 2) == 0).any(axis=2)
    # A zero may be omitted. A coincidental pair elsewhere is not its location.
    unique = (hi - lo == 1) & ~sparse
    positions[unique] = order[lo[unique]] + start
    for offset in positions[positions >= 0]:
        require(not occupied[offset:offset + 4].any(), "overlapping unique convolution value")
        occupied[offset:offset + 4] = True
    group = weight.shape[1] * 3
    pending = list(map(tuple, np.argwhere(~unique & ~sparse)))
    # Repeated values may need a neighbour resolved in the preceding pass.
    while pending:
        unresolved = []
        for i, j in pending:
            candidates = order[lo[i, j]:hi[i, j]] + start
            candidates = np.array([p for p in candidates if not occupied[p:p + 4].any()])
            if not len(candidates):
                continue
            neighbours = [positions[k, j] for k in range(max(i // group * group, i - 4), min((i // group + 1) * group, i + 5))
                          if k != i and positions[k, j] >= 0]
            if not neighbours:
                unresolved.append((i, j))
                continue
            distances = np.min(np.abs(candidates[:, None] - np.array(neighbours)), axis=1)
            require(distances.min() <= 150 and np.sum(distances == distances.min()) == 1, "ambiguous convolution packet")
            offset = candidates[distances.argmin()]
            positions[i, j] = offset
            occupied[offset:offset + 4] = True
        require(not unresolved or len(unresolved) < len(pending), "convolution values lack packet neighbours")
        pending = unresolved
    extra = []
    for i, j in np.argwhere(positions < 0):
        pair = vectors[i, j * 2:j * 2 + 2]
        require(bool((pair == 0).any()), "unmapped nonzero convolution pair")
        near = [positions[k, j] for k in range(max(0, i - 3), min(len(vectors), i + 4)) if positions[k, j] >= 0]
        require(bool(near), "sparse scalar lacks packet neighbour")
        for k, scalar in enumerate(pair):
            if scalar == 0:
                continue
            locations = [p for p in range(max(start, min(near) - 40), min(end - 1, min(near) + 150))
                         if coeff[p:p + 2] == scalar.tobytes() and not occupied[p:p + 2].any()]
            require(len(locations) == 1, "ambiguous sparse convolution scalar")
            extra.append([int(i), int(j * 2 + k), int(locations[0])])
            occupied[locations[0]:locations[0] + 2] = True
    return positions, extra


def derive(checkpoint, capture, bundle, model, output):
    dims, tensors = read_encoder(checkpoint)
    raw = (capture / "model.hwx").read_bytes()
    container = parse_container(raw)
    thread = container["thread"]
    text = next(s for s in container["segments"] if s["name"] == "__TEXT")
    const = next(s for s in container["sections"] if s["segment"] == "__TEXT" and s["name"] == "__const")
    tasks = parse_tasks(raw[text["fileoff"]:text["fileoff"] + text["filesize"]], thread["td_size"], thread["td_count"])
    segments = [s for s in container["segments"] if s["name"].startswith("__KERN")]
    coeff = b"".join(raw[s["fileoff"]:s["fileoff"] + s["filesize"]] for s in segments)
    starts = np.cumsum([0] + [s["filesize"] for s in segments])
    def address(offset, length):
        index = int(np.searchsorted(starts, offset, side="right") - 1)
        require(0 <= index < len(segments) and offset + length <= starts[index + 1], "packing crosses coefficient bank")
        return int(segments[index]["fileoff"] + offset - starts[index])
    symbols = []
    for index, segment in enumerate(segments):
        for name, value in container["symbols"].items():
            if segment["vmaddr"] <= value["addr"] < segment["vmaddr"] + segment["filesize"]:
                symbols.append((int(starts[index] + value["addr"] - segment["vmaddr"]), name.rsplit("_ne_", 1)[0]))
    symbols.sort()
    groups = {}
    for index, (offset, name) in enumerate(symbols):
        end = symbols[index + 1][0] if index + 1 < len(symbols) else len(coeff)
        groups.setdefault(name, [offset, end])[1] = end
    groups = sorted(groups.values())
    template = bytearray(raw)
    used = np.zeros(len(raw), bool)
    coefficient_used = np.zeros(len(coeff), bool)
    operations = []
    def add(op, value):
        offset = op["offset"]
        require(raw[offset:offset + len(value)] == value and not used[offset:offset + len(value)].any(),
                "missing/overlapping learned tensor: " + str(op))
        op["bytes"] = len(value)
        operations.append(op)
        used[offset:offset + len(value)] = True
        template[offset:offset + len(value)] = bytes(len(value))
    matrix_regions = []
    names = sorted(n for n, value in tensors.items() if n.endswith(".weight") and value.ndim >= 2
                   and "embed_positions" not in n and not n.endswith("conv2.weight"))
    for name in names:
        weight = tensors[name]
        bias_name = name[:-6] + "bias"
        first = 0
        region = None
        preferred_count = 16
        while first < len(weight):
            found = []
            counts = list(dict.fromkeys([preferred_count, *range(16, 1, -2)]))
            for start, end in ([region, (0, len(coeff))] if region else [(0, len(coeff))]):
                for count in counts:
                    if first + count > len(weight):
                        continue
                    value = weight[first:first + count].reshape(count, -1).T.tobytes()
                    offset = coeff.find(value, start, end)
                    if offset >= 0 and coeff.find(value, offset + 1, end) < 0:
                        found.append((count, offset, value))
                        break
                if found:
                    break
            require(bool(found), "unrecognized matrix tile: " + name + ":" + str(first))
            count, offset, value = found[0]
            preferred_count = count
            if region is None or not region[0] <= offset < region[1]:
                region = next((a, b) for a, b in groups if a <= offset < b)
                matrix_regions.append((name, region))
            add(dict(kind="tile", tensor=name, first=first, count=count, offset=address(offset, len(value))), value)
            coefficient_used[offset:offset + len(value)] = True
            if bias_name in tensors:
                bias = tensors[bias_name][first:first + count].tobytes()
                p = offset - len(bias)
                add(dict(kind="tile", tensor=bias_name, first=first, count=count, offset=address(p, len(bias))), bias)
                coefficient_used[p:p + len(bias)] = True
            first += count
        print(json.dumps(dict(model=model, tensor=name, status="mapped")), flush=True)
    conv_start = next(b for n, (a, b) in matrix_regions if n.endswith("conv1.weight"))
    conv_end = min(a for n, (a, b) in matrix_regions if not n.endswith("conv1.weight"))
    require(conv_end <= segments[0]["filesize"], "convolution spans coefficient banks")
    bias_name = "model.encoder.conv2.bias"
    bias = tensors[bias_name]
    first = 0
    while first < len(bias):
        found = None
        for count in range(min(32, len(bias) - first), 1, -1):
            value = bias[first:first + count].tobytes()
            p = coeff.find(value, conv_start, conv_end)
            if p >= 0 and coeff.find(value, p + 1, conv_end) < 0:
                found = count, p, value
                break
        require(found is not None, "unrecognized conv2 bias: " + str(first))
        count, offset, value = found
        add(dict(kind="tile", tensor=bias_name, first=first, count=count, offset=address(offset, len(value))), value)
        coefficient_used[offset:offset + len(value)] = True
        first += count
    conv_name = "model.encoder.conv2.weight"
    before = int(coefficient_used.sum())
    offsets, extra = convolution_scatter(coeff, tensors[conv_name], conv_start, conv_end, coefficient_used)
    source = scatter_vectors(tensors[conv_name])
    view = np.frombuffer(template, np.uint8)
    original = np.frombuffer(raw, np.uint8)
    for pair in range(2):
        valid = offsets[:, pair] >= 0
        offsets[valid, pair] += segments[0]["fileoff"]
        addresses = offsets[valid, pair, None].astype(np.int64) + np.arange(4)
        value = source[:, pair * 2:pair * 2 + 2].copy().view(np.uint8).reshape(-1, 4)[valid]
        require(np.unique(addresses).size == addresses.size and not used[addresses].any()
                and np.array_equal(original[addresses], value), "invalid convolution map")
        view[addresses] = 0
        used[addresses] = True
    for row in extra:
        i, channel, offset = row
        row[2] = address(offset, 2)
        add(dict(kind="tensor-scalar", tensor=conv_name, offset=row[2]), source[i, channel].tobytes())
        operations.pop()  # Sparse scalars live in the scatter recipe instead.
    expected = sum(v.nbytes for n, v in tensors.items() if "layer_norm" not in n and "embed_positions" not in n)
    conv_mapped = int(coefficient_used.sum()) - before
    omitted = tensors[conv_name].nbytes - conv_mapped
    require(int(coefficient_used.sum()) == expected - omitted, "incomplete coefficient coverage")
    for name in sorted(n for n in tensors if "layer_norm" in n and n.endswith(".weight")):
        gamma = tensors[name]
        beta_name = name[:-6] + "bias"
        if raw.find(gamma.tobytes(), const["offset"], const["offset"] + const["size"]) < 0:
            value = goc_affine(gamma, tensors[beta_name], 16).tobytes()
            offset = coeff.find(value)
            require(offset >= 0 and coeff.find(value, offset + 1) < 0, "unrecognized GOC affine: " + name)
            add(dict(kind="goc-affine", tensor=name, beta=beta_name, engines=16,
                     offset=address(offset, len(value))), value)
            continue
        ratio = (tensors[beta_name].astype(np.float32) / gamma.astype(np.float32)).astype("<f2")
        for key, value, kind in ((name, gamma, "tensor"), (beta_name, ratio, "ratio")):
            block = value.tobytes()
            p = raw.find(block, const["offset"], const["offset"] + const["size"])
            require(p >= 0 and raw.find(block, p + 1, const["offset"] + const["size"]) < 0, "ambiguous layer norm")
            op = dict(kind=kind, tensor=key, offset=p)
            if kind == "ratio":
                op["gamma"] = name
            add(op, block)
    learned_bytes = sum(v.nbytes for n, v in tensors.items() if "embed_positions" not in n)
    require(int(used.sum()) == learned_bytes - omitted, "incomplete learned tensor coverage")
    output.mkdir(parents=True, exist_ok=False)
    def asset(name, value):
        (output / name).write_bytes(zlib.compress(value, 9))
        return dict(file=name, bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
    packing = dict(operations=operations, stripped_bytes=int(used.sum()),
                   scatter=dict(tensor=conv_name, extra=extra, encoding="delta-i32"))
    status = plistlib.loads((capture / "model.hwx.status.plist").read_bytes())
    network, = status["NetworkStatusList"]
    ports = []
    for role, key in (("input", "LiveInputList"), ("output", "LiveOutputList")):
        for port in network[key]:
            symbol = port["Symbol"].removesuffix("@output")
            ports.append(dict(role=role, name=symbol, compiler_layout=port,
                              original_bank=thread["bars"].index(container["symbols"][symbol]["addr"]), byte_offset=0))
    meta = dict(format="whisper-pr3905-h13g/v1", target="apple,t8103", generation="H13G", dimensions=dims,
                checkpoint=dict(model=model, format="whisper.cpp GGML F16", sha256=sha256(checkpoint)),
                encoder_tensors=tensor_manifest(tensors),
                hwx_sha256=hashlib.sha256(raw).hexdigest(), position_sha256=sha256(bundle / "pos.f16"),
                source_mil_sha256=sha256(bundle / "model.mil"), source_weights_sha256=sha256(bundle / "weights.bin"),
                template=asset("model.hwx.template.zlib", template),
                packing=asset("packing.json.zlib", json.dumps(packing, separators=(",", ":")).encode()),
                scatter=asset("conv2-offsets.i32.zlib", np.diff(offsets.ravel().astype(np.int64), prepend=0).astype("<i4").tobytes()),
                layout=dict(thread=thread, segments=container["segments"], sections=container["sections"], ports=ports,
                            coefficient_banks=[dict(segment=s, bank=thread["bars"].index(s["vmaddr"])) for s in segments]),
                linux_hardware_validation="pending", accuracy_validation="not performed in this speed-only benchmark",
                executable_identity_limit="E5RT and offline HWX compile the same MIL/weights separately; offline HWX has not executed.")
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    _, rebuilt, positions = reconstruct(checkpoint, output)
    require(rebuilt == raw and positions == (bundle / "pos.f16").read_bytes(), "byte-exact reconstruction failed")
    proof = dict(status="PASS_BYTE_EXACT_REPACK", model=model, task_count=len(tasks), coefficient_banks=len(segments),
                 hwx_bytes=len(raw), hwx_sha256=meta["hwx_sha256"], stripped_bytes=int(used.sum()),
                 omitted_zero_conv2_bytes=omitted, package_bytes_excluding_proof=sum(p.stat().st_size for p in output.iterdir()),
                 all_matrix_bias_and_layer_norm_values_external=True, linux_hardware_validation="pending")
    (output / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(derive(a.checkpoint, a.capture, a.bundle, a.model, a.output)), flush=True)


if __name__ == "__main__":
    main()
