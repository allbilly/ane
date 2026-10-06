"""Probe fused four-head DeltaNet updates using real Mirai decoder inputs."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "Desktop/ANEForge"))
from qwen35.model import Model
from qwen35.weights import REVISION, sha256
from capture_macos_program import export


def error(actual, expected):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if a.shape != b.shape: raise RuntimeError("recurrent output shape mismatch")
    if not np.isfinite(a).all() or not np.isfinite(b).all(): return dict(finite=False, normalized_rmse=None, max_abs=None)
    return dict(finite=True, normalized_rmse=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-40)),
                max_abs=float(np.abs(a - b).max()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--offline", action="store_true", help="offline export with FP32 CPU reference fixtures")
    p.add_argument("--inputs", type=Path, help="reuse a previously captured native-inputs.npz")
    p.add_argument("--dense-scalars", action=argparse.BooleanOptionalAction, default=True, help="repeat each head's beta/g over width 128 to avoid padded scalar I/O")
    p.add_argument("--query-gain", type=float, default=128., help="power-of-two gain before the output matmul, compensated afterwards")
    p.add_argument("--state-gain", type=float, default=128., help="power-of-two gain inside the state update, compensated at its output")
    p.add_argument("--silu-mode", choices=["native", "exp", "tanh"], default="exp", help="equivalent SiLU expressions for hardware precision diagnosis")
    a = p.parse_args()
    for gain in (a.query_gain, a.state_gain):
        if not np.isfinite(gain) or not 1 <= gain <= 32768 or not np.log2(gain).is_integer():
            p.error("gains must be finite FP16 powers of two from 1 to 32768")
    checkpoint_sha256 = sha256(a.model / "model.safetensors")
    if a.inputs:
        source_report = json.loads(a.inputs.with_name("report.json").read_text())
        if source_report.get("model_sha256") != checkpoint_sha256:
            raise RuntimeError("reused native inputs belong to a different checkpoint")
        if "native_inputs_sha256" in source_report and sha256(a.inputs) != source_report["native_inputs_sha256"]:
            raise RuntimeError("reused native input checksum mismatch")
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ["ANEFORGE_CACHE_DIR"] = str(a.output / "cache")
    os.environ.setdefault("ANEFORGE_NO_AUTOBUILD", "1")
    model = None if a.inputs else Model(a.model, kernels="native", threads=4, context=128)
    original = model.native.gdn if model else None
    captures = []
    def capture(state, projected, a_log, dt, norm):
        wanted = norm is model.layers[0]["norm"] and model.position in (0, 1, 7, 22)
        before = state.copy() if wanted else None
        result = original(state, projected, a_log, dt, norm)
        if wanted: captures.append(dict(position=model.position, state=before,
                                        projected=projected.copy(), output=result.copy(), next_state=state.copy()))
        return result
    manifest = json.loads((ROOT / "qwen35/provenance/vendor-validation.json").read_text())
    tokens = next(item["tokens"] for item in manifest["checks"] if item["label"] == "arithmetic")
    if a.inputs:
        import shutil
        with np.load(a.inputs) as saved:
            if saved["tokens"].tolist() != tokens: raise RuntimeError("captured prompt token mismatch")
            layer = {name:saved[name].copy() for name in ("a_log", "dt", "norm")}
            for i, position in enumerate(saved["positions"]):
                captures.append(dict(position=int(position), **{name:saved[name][i].copy() for name in ("state", "projected", "output", "next_state")}))
        shutil.copy2(a.inputs, a.output / "native-inputs.npz")
    else:
        model.native.gdn = capture
        for token in tokens: model.step(token, logits=False)
        layer = model.layers[0]
    if [r["position"] for r in captures] != [0, 1, 7, 22]:
        raise RuntimeError("expected distinct real prefix captures at positions 0, 1, 7 and 22")
    for name, shape in (("a_log", (16,)), ("dt", (16,)), ("norm", (128,))):
        if layer[name].shape != shape or not np.isfinite(layer[name]).all():
            raise RuntimeError("invalid captured layer constant: " + name)
    for capture in captures:
        for name, shape in (("state", (16, 128, 128)), ("next_state", (16, 128, 128)),
                            ("projected", (8224,)), ("output", (2048,))):
            if capture[name].shape != shape or not np.isfinite(capture[name]).all():
                raise RuntimeError("invalid native capture array: " + name)
    if not a.inputs: np.savez_compressed(a.output / "native-inputs.npz", tokens=tokens,
                        positions=[r["position"] for r in captures],
                        state=[r["state"] for r in captures], next_state=[r["next_state"] for r in captures],
                        projected=[r["projected"] for r in captures], output=[r["output"] for r in captures],
                        a_log=layer["a_log"], dt=layer["dt"], norm=layer["norm"])
    import aneforge as af
    from aneforge._compile import compile_multi
    records = []
    for start in (0, 4, 8, 12):
        directory = a.output / f"heads-{start}-{start + 3}"
        S = af.input((4, 128, 128))
        q, k, v, z = [af.input((4, 1, 128)) for _ in range(4)]
        b, g = [af.input((4, 1, 128 if a.dense_scalars else 1)) for _ in range(2)]
        qn = q.l2_norm(-1, 1e-6) * (128 ** -.5 * a.query_gain)
        kn = k.l2_norm(-1, 1e-6)
        dt = layer["dt"][start:start + 4].reshape(4, 1, 1)
        decay_scale = (-np.exp(layer["a_log"][start:start + 4])).reshape(4, 1, 1)
        decay = ((g + dt).softplus() * decay_scale).exp()
        decayed = (S * a.state_gain) * decay
        change = ((v * a.state_gain) - kn @ decayed) * b.sigmoid()
        scaled_updated = decayed + kn.transpose([0, 2, 1]) @ change
        updated = scaled_updated / a.state_gain
        # Keep epsilon as an addition. Scale by 128 so its FP16 constant is
        # normal rather than a coarsely rounded subnormal (1e-6).
        scaled_value = (qn @ scaled_updated) * (128. / (a.query_gain * a.state_gain))
        value = scaled_value * ((scaled_value * scaled_value).mean((-1,)) + (128 ** 2 * 1e-6)).rsqrt()
        if a.silu_mode == "native":
            gate = z.silu()
        elif a.silu_mode == "exp":
            gate = z / ((z * -1.).exp() + 1.)
        else:
            gate = z * ((z * .5).tanh() * .5 + .5)
        value = value * layer["norm"].reshape(1, 1, 128) * gate
        net = None
        record = dict(head_start=start, heads=4, cases=[])
        try:
            if a.offline:
                from offline_backend import emit_multi
                live = emit_multi([updated, value], directory)
                input_ports = [(t, t._name) for t in live]
                output_ports = [(t, t._name) for t in (updated, value)]
            else:
                net = compile_multi([updated, value], build_dir=directory)
                if net.prog._device_mask != 4: raise RuntimeError("ANE-only device mask required")
                input_ports, output_ports = net.input_ports, net.output_ports
            for fixture in captures:
                projected = fixture["projected"]
                vectors = projected[:6144].reshape(3, 16, 128)[:, start:start + 4]
                arrays = [fixture["state"][start:start + 4].transpose(0, 2, 1).copy(),
                          *[vector[:, None, :] for vector in vectors],
                          projected[6144:8192].reshape(16, 128)[start:start + 4, None, :],
                          projected[8192:8208][start:start + 4, None, None],
                          projected[8208:8224][start:start + 4, None, None]]
                if a.dense_scalars:
                    arrays[-2:] = [np.repeat(array, 128, axis=-1) for array in arrays[-2:]]
                by_tensor = {id(tensor): array for tensor, array in zip((S, q, k, v, z, b, g), arrays)}
                if a.offline:
                    sr, qr, kr, vr, zr, br, gr = [np.asarray(array, np.float16).astype(np.float32) for array in arrays]
                    # Follow emitted MIL's safe-divide floor and rounded constants.
                    floor = np.float32(np.float16(1e-6 ** .5))
                    qr = qr / np.maximum(np.sqrt((qr * qr).sum(-1, keepdims=True)), floor) * np.float32(np.float16(128 ** -.5 * a.query_gain))
                    kr = kr / np.maximum(np.sqrt((kr * kr).sum(-1, keepdims=True)), floor)
                    dt_rounded = dt.astype(np.float16).astype(np.float32)
                    scale_rounded = decay_scale.astype(np.float16).astype(np.float32)
                    dec = (sr * a.state_gain) * np.exp(np.logaddexp(0, gr + dt_rounded) * scale_rounded)
                    change_ref = (vr * a.state_gain - kr @ dec) / (1 + np.exp(-br))
                    scaled_state_ref = dec + kr.transpose(0, 2, 1) @ change_ref
                    state_ref = scaled_state_ref / a.state_gain
                    value_ref = (qr @ scaled_state_ref) * (128. / (a.query_gain * a.state_gain))
                    value_ref = value_ref / np.sqrt((value_ref * value_ref).mean(-1, keepdims=True) + np.float32(np.float16(128 ** 2 * 1e-6)))
                    value_ref *= layer["norm"].astype(np.float16).astype(np.float32)
                    value_ref *= zr * (.5 + .5 * np.tanh(zr * .5)) if a.silu_mode == "tanh" else zr / (1 + np.exp(-zr))
                    results = [state_ref, value_ref]
                else:
                    for (tensor, name), array in zip(input_ports, [by_tensor[id(t)] for t, _ in input_ports]):
                        net.prog.set_input(name, np.asarray(array, np.float16))
                    net.prog.execute()
                    results = [net.prog.read_output(name).copy() for tensor, name in output_ports]
                arrays = [by_tensor[id(tensor)] for tensor, name in input_ports]
                by_output = {id(tensor): result for (tensor, name), result in zip(output_ports, results)}
                expected_state = fixture["next_state"][start:start + 4].transpose(0, 2, 1)
                expected_output = fixture["output"].reshape(16, 128)[start:start + 4, None, :]
                name = f"fixture-{fixture['position']}.npz"
                np.savez_compressed(directory / name,
                                    **{f"input{i:02d}": np.asarray(array, np.float16) for i, array in enumerate(arrays)},
                                    **{f"output{i:02d}": result for i, result in enumerate(results)},
                                    fp32_state=expected_state, fp32_output=expected_output)
                checks = [error(by_output[id(updated)], expected_state), error(by_output[id(value)], expected_output)]
                record["cases"].append(dict(position=fixture["position"], fixture=name,
                                            state=checks[0], output=checks[1],
                                            passes_full_component_gate=all(check["finite"] and check["normalized_rmse"] <= .005 for check in checks)))
            record["input_ports"] = [dict(name=name, shape=tensor.shape) for tensor, name in input_ports]
            record["output_ports"] = [dict(name=name, shape=tensor.shape) for tensor, name in output_ports]
            record["export"] = export(directory, directory / "hwx", ROOT / "gpt2/training/build/dump_hwx")
            record["status"] = ("cpu_reference_and_export_pass" if a.offline else "pass") if all(case["passes_full_component_gate"] for case in record["cases"]) and record["export"]["status"] == "exported" else "numerical_or_export_failure"
        except Exception as failure:
            record.update(status="failed", error=str(failure))
        finally:
            if net is not None: net.release()
        records.append(record)
        print(json.dumps(record), flush=True)
    report = dict(scope="Layer-0 recurrent update only; real inputs at four arithmetic prefix positions; four programs cover 16 heads.",
                  model_revision=REVISION, model_sha256=checkpoint_sha256,
                  native_inputs_sha256=sha256(a.output / "native-inputs.npz"),
                  full_component_nrmse_limit=.005, records=records,
                  reference_kind="CPU FP32 on exact rounded MIL inputs/constants" if a.offline else "macOS ANE FP16 outputs",
                  normalization="RMS epsilon addition with power-of-two scaling by 128; generic RMS floor helper is unsuitable for these small recurrent values.",
                  compensated_gains=dict(query=a.query_gain, state=a.state_gain),
                  silu_expression=a.silu_mode,
                  scalar_input_layout="Per-head beta/g repeated across width 128" if a.dense_scalars else "Width-one beta/g inputs; compiler may pad their rows",
                  hardware_execution="not attempted (--offline)" if a.offline else "ANE-only E5RT",
                  linux_replay="pending; this does not enable a fused Linux decoder",
                  status=("cpu_reference_and_export_pass" if a.offline else "pass") if all(record["status"] in ("pass", "cpu_reference_and_export_pass") for record in records) else "failed")
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] == "failed": raise SystemExit(1)


if __name__ == "__main__":
    main()
