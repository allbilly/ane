"""Replay independent vendor fixtures and compare optimized FP32 execution."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .model import Model


def verify(directory):
    root = Path(__file__).parent / "provenance"
    manifest = json.loads((root / "vendor-validation.json").read_text())
    model = Model(directory, precision="bf16", kernels="native", context=128)
    vendor = []
    for item in manifest["checks"]:
        fixture = root / item["fixture"]
        if hashlib.sha256(fixture.read_bytes()).hexdigest() != item["fixture_sha256"]:
            raise ValueError("corrupt independent reference fixture")
        model.reset()
        for token in item["tokens"][:-1]:
            model.step(token, logits=False)
        trace = []
        actual = model.step(item["tokens"][-1], trace=trace)
        with np.load(fixture) as data:
            if not np.array_equal(actual, data["logits"]):
                raise RuntimeError(f"vendor logits differ: {item['label']}")
            if "layers" in data and not np.array_equal(np.array(trace), data["layers"]):
                raise RuntimeError(f"vendor layer outputs differ: {item['label']}")
        vendor.append(dict(label=item["label"], logits_bitwise_equal=True,
                           layers_bitwise_equal=item.get("layers_bitwise_equal")))
    del model
    floating = Model(directory, kernels="native", context=128)
    integer = Model(directory, kernels="dot", context=128)
    errors, matches = [], []
    token = 248045
    for _ in range(16):
        expected, actual = floating.step(token), integer.step(token)
        errors.append(float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)))
        matches.append(int(actual.argmax()) == int(expected.argmax()))
        token = int(expected.argmax())
    if max(errors) >= .005 or not all(matches):
        raise RuntimeError("integer dot numerical gate failed")
    return dict(vendor=vendor, dot_steps=len(errors), dot_argmax_matches=sum(matches),
                dot_max_normalized_rmse=max(errors), status="pass")
