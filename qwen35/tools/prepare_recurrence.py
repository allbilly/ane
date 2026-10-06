"""Prepare real four-head recurrence inputs and CPU references without macOS."""
import argparse
import json
from pathlib import Path

import numpy as np

from qwen35.model import Model
from qwen35.weights import DEFAULT_MODEL, MODEL_ID, MODEL_SHA256, REVISION, sha256

ROOT = Path(__file__).resolve().parents[1]
POSITIONS = (0, 1, 7, 22)
INPUT_NAMES = ("state", "q", "k", "v", "z", "beta", "g")
GAIN = 128.
GATE = .005


def fp16(value):
    with np.errstate(over="ignore"):
        result = np.ascontiguousarray(value, dtype="<f2")
    if not np.isfinite(result).all():
        raise ValueError("recurrence input or constant exceeds finite FP16 range")
    return result


def pack_case(capture, layer, start):
    """Dense physical I/O; native state is [head,value,key], MIL is [head,key,value]."""
    projected = capture["projected"]
    q, k, v = projected[:6144].reshape(3, 16, 128)[:, start:start + 4]
    arrays = (capture["state"][start:start + 4].transpose(0, 2, 1),
              q[:, None, :], k[:, None, :], v[:, None, :],
              projected[6144:8192].reshape(16, 128)[start:start + 4, None, :],
              np.repeat(projected[8192:8208][start:start + 4, None, None], 128, axis=-1),
              np.repeat(projected[8208:8224][start:start + 4, None, None], 128, axis=-1))
    inputs = {name: fp16(array[None]) for name, array in zip(INPUT_NAMES, arrays)}
    constants = dict(dt=fp16(layer["dt"][start:start + 4].reshape(4, 1, 1)),
                     decay_scale=fp16(-np.exp(layer["a_log"][start:start + 4]).reshape(4, 1, 1)),
                     norm=fp16(layer["norm"].reshape(1, 1, 128)))
    return inputs, constants


def rounded_reference(inputs, constants):
    """FP32 evaluation of the selected exp-SiLU MIL on FP16 I/O and constants.

    Matches capture_qwen_recurrence.py's offline expression. Intermediate ANE
    rounding is deliberately absent: actual device outputs must be tested.
    """
    state, q, k, v, z, beta, g = [inputs[name][0].astype(np.float32) for name in INPUT_NAMES]
    dt, decay_scale, norm = [constants[name].astype(np.float32) for name in ("dt", "decay_scale", "norm")]
    floor = np.float32(np.float16(1e-6 ** .5))
    q = q / np.maximum(np.sqrt((q * q).sum(-1, keepdims=True)), floor)
    q *= np.float32(np.float16(128 ** -.5 * GAIN))
    k = k / np.maximum(np.sqrt((k * k).sum(-1, keepdims=True)), floor)
    with np.errstate(over="ignore"):
        decayed = state * GAIN * np.exp(np.logaddexp(0, g + dt) * decay_scale)
        change = (v * GAIN - k @ decayed) / (1 + np.exp(-beta))
        scaled_state = decayed + k.transpose(0, 2, 1) @ change
        value = (q @ scaled_state) * (128. / (GAIN * GAIN))
        value /= np.sqrt((value * value).mean(-1, keepdims=True) + np.float32(np.float16(128 ** 2 * 1e-6)))
        value *= norm * z / (1 + np.exp(-z))
    return dict(state=(scaled_state / GAIN)[None], output=value[None])


def error(actual, expected):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid recurrent output or reference")
    return dict(normalized_rmse=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-40)),
                max_abs=float(np.abs(a - b).max()))


def capture_native(model):
    """Use unmodified packed floating projections and capture layer zero only."""
    original = model.native.gdn
    captures = []

    def capture(state, projected, a_log, dt, norm):
        wanted = norm is model.layers[0]["norm"] and model.position in POSITIONS
        before = state.copy() if wanted else None
        result = original(state, projected, a_log, dt, norm)
        if wanted:
            captures.append(dict(position=model.position, state=before, projected=projected.copy(),
                                 output=result.copy(), next_state=state.copy()))
        return result

    manifest = json.loads((ROOT / "provenance/vendor-validation.json").read_text())
    tokens = next(item["tokens"] for item in manifest["checks"] if item["label"] == "arithmetic")
    model.native.gdn = capture
    try:
        for token in tokens:
            model.step(token, logits=False)
    finally:
        model.native.gdn = original
    if [c["position"] for c in captures] != list(POSITIONS):
        raise RuntimeError("missing native prefix captures")
    return tokens, captures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.threads < 1:
        p.error("threads must be positive")
    checkpoint_hash = sha256(a.model / "model.safetensors")
    if checkpoint_hash != MODEL_SHA256:
        raise ValueError("use the pinned Mirai Qwen3.5-0.8B-M checkpoint")
    a.output.mkdir(parents=True, exist_ok=False)
    model = Model(a.model, kernels="native", threads=a.threads, context=128)
    tokens, captures = capture_native(model)
    layer = model.layers[0]
    native_path = a.output / "native-inputs.npz"
    np.savez_compressed(native_path, tokens=tokens, positions=POSITIONS,
                        **{name: np.array([c[name] for c in captures])
                           for name in ("state", "next_state", "projected", "output")},
                        **{name: layer[name] for name in ("a_log", "dt", "norm")})
    records = []
    for start in (0, 4, 8, 12):
        directory = a.output / f"heads-{start}-{start + 3}"
        directory.mkdir()
        for capture in captures:
            inputs, constants = pack_case(capture, layer, start)
            reference = rounded_reference(inputs, constants)
            # Gate the rounded boundary result as well, not only FP32 expressions.
            rounded = {name: fp16(value) for name, value in reference.items()}
            expected = dict(state=capture["next_state"][start:start + 4].transpose(0, 2, 1)[None],
                            output=capture["output"].reshape(16, 128)[start:start + 4, None, :][None])
            checks = {name: error(rounded[name], expected[name]) for name in expected}
            path = directory / f"fixture-{capture['position']}.npz"
            np.savez_compressed(path, **{f"input_{n}": v for n, v in inputs.items()},
                                **{f"rounded_{n}": v for n, v in rounded.items()},
                                **{f"cpu_mil_{n}": v for n, v in reference.items()},
                                **{f"native_{n}": v for n, v in expected.items()})
            records.append(dict(head_start=start, position=capture["position"],
                                fixture=str(path.relative_to(a.output)), sha256=sha256(path),
                                checks=checks, status="pass" if all(c["normalized_rmse"] <= GATE for c in checks.values()) else "fail"))
        path = directory / "constants.npz"
        np.savez_compressed(path, **constants)
        for record in records[-len(captures):]:
            record.update(constants=str(path.relative_to(a.output)), constants_sha256=sha256(path))
    report = dict(model_id=MODEL_ID, model_revision=REVISION, model_sha256=checkpoint_hash,
                  native_inputs_sha256=sha256(native_path), native_kernels="floating packed W4",
                  scope="Layer-0 four-head updates at real arithmetic positions 0,1,7,22; 16 cases, 32 component checks",
                  input_shapes={n: list(v.shape) for n, v in inputs.items()}, input_dtype="little-endian FP16",
                  reference_kind="Independent native FP32 DeltaNet; CPU FP32 MIL expression on exact FP16 inputs/constants",
                  compensated_gains=dict(query=GAIN, state=GAIN), silu_expression="exp",
                  full_component_nrmse_limit=GATE, records=records,
                  hardware_execution="not attempted; CPU preparation only",
                  compiled_port_mapping="Semantic names; map through extracted MIL/status port names before dispatch",
                  coefficient_layout="FP16 semantic constants; compiled coefficient-bank packing requires its template layout",
                  status="pass" if all(r["status"] == "pass" for r in records) else "fail")
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], cases=len(records), model_sha256=checkpoint_hash,
                         max_state_nrmse=max(r["checks"]["state"]["normalized_rmse"] for r in records),
                         max_output_nrmse=max(r["checks"]["output"]["normalized_rmse"] for r in records))), flush=True)
    if report["status"] != "pass":
        raise RuntimeError("rounded recurrence boundary gate failed; captures retained")


if __name__ == "__main__":
    main()
