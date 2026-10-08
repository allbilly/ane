"""Capture fast-graph Mac outputs for later Asahi replay; no timing ablations."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from qwen35.weights import sha256
from whisper.encoder_kernel import require, port_shape
from whisper.pr_encoder_replay import GATE


def capture(bundles, kernels, reference, handoff, fixtures, aneforge, output):
    require(platform.system() == "Darwin" and platform.machine() == "arm64", "capture requires the Apple Silicon Mac")
    sys.path.insert(0, str(aneforge))
    from aneforge._runtime import E5RT
    direct = json.loads(reference.read_text())
    original = json.loads(handoff.read_text())
    mels = {"zero-mel":np.zeros((80, 3000), "<f2")}
    sources = {}
    for case in original["whisper"]["fixtures"]:
        path = fixtures / case["file"]
        require(sha256(path) == case["sha256"], "existing Mac fixture checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            mel = data["input00"].reshape(80, 3000).copy()
        require(mel.dtype == np.dtype("<f2") and bool(np.isfinite(mel).all())
                and hashlib.sha256(mel.tobytes()).hexdigest() == case["arrays"]["input00"]["sha256"], "existing Mac mel changed")
        mels[case["audio"]] = mel
        sources[case["audio"]] = dict(file=case["file"], file_sha256=case["sha256"],
                                     mel_sha256=case["arrays"]["input00"]["sha256"])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(format="whisper-pr3905-replay-fixtures/v1", gate=GATE.copy(), models={},
                  source_handoff_sha256=sha256(handoff), source_direct_dispatch_sha256=sha256(reference),
                  runtime_sha256=sha256(aneforge / "aneforge/_lib/libane_e5rt_dispatch.dylib"),
                  scope="Mac E5RT outputs of exact timed PR fast graphs; encoder-only replay gate, not strict decoder accuracy",
                  hardware_replay="pending Asahi; offline HWX and E5RT compile the same source separately")
    for size in ("tiny", "base", "small"):
        bundle = bundles / size
        meta = json.loads((kernels / size / "meta.json").read_text())
        require(sha256(bundle / "model.mil") == meta["source_mil_sha256"] == direct["models"][size]["mil_sha256"]
                and sha256(bundle / "weights.bin") == meta["source_weights_sha256"]
                and sha256(bundle / "pos.f16") == meta["position_sha256"], "timed bundle identity mismatch")
        dims = meta["dimensions"]
        ports = meta["layout"]["ports"]
        inputs = {p["name"]:port_shape(p) for p in ports if p["role"] == "input"}
        outputs = {p["name"]:port_shape(p) for p in ports if p["role"] == "output"}
        mel_port, = [n for n, shape in inputs.items() if shape == (1, 80, 1, 3000)]
        pos_port, = [n for n, shape in inputs.items() if shape == (1, dims["state"], 1, 1500)]
        out_port, = outputs
        positions = np.fromfile(bundle / "pos.f16", "<f2").reshape(dims["state"], 1500)
        directory = output / size
        directory.mkdir()
        cases = []
        with E5RT.compile(bundle / "model.mil", cache_dir=bundle / "cache", inputs=inputs, outputs=outputs, device_mask=4) as program:
            for name, mel in mels.items():
                feed = {mel_port:mel.reshape(inputs[mel_port]), pos_port:positions.reshape(inputs[pos_port])}
                actual = program.eval(feed)[out_port].reshape(1500, dims["state"]).copy()
                require(actual.dtype == np.dtype("<f2") and bool(np.isfinite(actual).all()), "nonfinite Mac output")
                if name == "zero-mel":
                    require(hashlib.sha256(actual.tobytes()).hexdigest() == direct["models"][size]["output_fp16_sha256"],
                            "zero-mel output differs from the timed direct-dispatch capture")
                    again = program.eval(feed)[out_port].reshape(actual.shape).copy()
                    require(actual.tobytes() == again.tobytes(), "zero-mel output is not repeatable")
                path = directory / (name + ".npz")
                np.savez_compressed(path, mel=mel, positions=positions, output=actual)
                arrays = {key:dict(shape=list(value.shape), strides=list(value.strides), dtype=value.dtype.str,
                            sha256=hashlib.sha256(value.tobytes()).hexdigest())
                          for key, value in (("mel", mel), ("positions", positions), ("output", actual))}
                cases.append(dict(name=name, file=path.name, bytes=path.stat().st_size, sha256=sha256(path), arrays=arrays,
                                  mel_source=sources.get(name, "synthetic zeros, same as stock whisper-bench")))
                print(json.dumps(dict(model=size, case=name, status="PASS_CAPTURED_MAC_REFERENCE", output_sha256=arrays["output"]["sha256"])), flush=True)
        report["models"][size] = dict(hwx_sha256=meta["hwx_sha256"], mil_sha256=meta["source_mil_sha256"],
                                     weights_sha256=meta["source_weights_sha256"], cases=cases,
                                     zero_mel_matches_timed_output=True, zero_mel_repeat_bitwise_equal=True)
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundles", type=Path, required=True)
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--handoff", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--aneforge", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    capture(a.bundles, a.kernels, a.reference, a.handoff, a.fixtures, a.aneforge, a.output)


if __name__ == "__main__":
    main()
