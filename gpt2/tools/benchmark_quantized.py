#!/usr/bin/env python3
"""Compare Q4/Q8 on identical teacherforced traces, with rotated fresh processes."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def child(args):
    sys.path.insert(0, str(ROOT))
    from checks import integrity
    from cpu_matvec import CPUMatvec, performance_cpus
    from external_weights import load_weights, verify_weights
    from model import ANEKernels, CPUKernels, GPT2
    from packing import PackedAssets
    from replay import Device
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(performance_cpus()))
    integrity(ROOT)
    weights = load_weights(args.weights)
    verify_weights(args.weights, ROOT, weights=weights)
    matvec = CPUMatvec(weights, args.cpu_kernels)
    if args.cpu_kernels == 'numpy' and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {performance_cpus()[0]})
    device = Device(ROOT) if args.backend == 'ane' else None
    if device:
        device.assets = PackedAssets(ROOT, weights)
        for name in json.loads((ROOT / 'package.json').read_text())['kernels']:
            if name.startswith('decode_'):
                device.kernel(name)
    model = GPT2(weights, ANEKernels(device) if device else CPUKernels(weights, matvec), matvec)
    case = json.loads((ROOT / 'provenance/orion-performance/m1-contexts.json').read_text())['reference']['cases'][args.context_case]
    trials = []
    try:
        for _ in range(args.trials):
            model.reset()
            for token in case['prompt_ids'] + case['next_tokens'][:32]:
                model.step(token)
            model.reset()
            for token in case['prompt_ids']:
                model.step(token)
            samples = []
            for token in case['next_tokens'][:64]:
                start = time.perf_counter_ns()
                model.step(token)
                samples.append((time.perf_counter_ns() - start) / 1e6)
            trials.append(dict(step_ms=samples, steps_per_second=64000 / sum(samples)))
        # Vocabulary timing uses the actual same activation vector for both
        # formats, independent of model-specific hidden states.
        import numpy as np
        x = np.random.default_rng(98234).normal(size=768).astype(np.float32)
        samples = []
        for i in range(110):
            start = time.perf_counter_ns()
            matvec('lm_head', (50257, 768), x)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if i >= 10:
                samples.append(elapsed)
        actual = matvec('lm_head', (50257, 768), x)
        expected = weights.get('lm_head', (50257, 768)) @ x
        difference = actual.astype(np.float64) - expected
        error = dict(max_abs=float(np.max(np.abs(difference))),
                     normalized_rmse=float(np.linalg.norm(difference) / np.linalg.norm(expected)),
                     argmax_equal=int(actual.argmax()) == int(expected.argmax()))
        return dict(trials=trials, vocabulary_ms=samples, cpu_kernels=matvec.description,
                    cpu_threads=matvec.threads, prefix_tokens=len(case['prompt_ids']), vocabulary_error=error)
    finally:
        if device:
            device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q4', type=Path)
    parser.add_argument('--q8', type=Path)
    parser.add_argument('--safetensors', type=Path, help='optional reference checkpoint for the same decoder comparison')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--backend', choices=('ane', 'cpu'), default='ane')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--context-case', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--include-numpy', action='store_true')
    parser.add_argument('--include-exact', action='store_true')
    parser.add_argument('--weights', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--cpu-kernels', choices=('native', 'exact', 'numpy'), default='native', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.weights:
        print(json.dumps(child(args)))
        return
    if not (args.q4 and args.q8 and args.output) or args.rounds < 1 or args.trials < 1:
        parser.error('--q4, --q8, --output and positive rounds/trials are required')
    formats = [('Q4_0', args.q4), ('Q8_0', args.q8)]
    if args.safetensors:
        formats.append(('safetensors', args.safetensors))
    modes = ['native'] + (['exact'] if args.include_exact else []) + (['numpy'] if args.include_numpy else [])
    cases = [(kind, mode, path) for mode in modes for kind, path in formats]
    report = dict(backend=args.backend, openblas_threads=1, rounds=args.rounds,
                  conditions='Warm coefficient and OS file caches; fastest available CPU cluster; active desktop, no resource isolation.',
                  timing='model.step on identical saved 64-token traces after 32 warmup steps; includes CPU work and ANE transfers/submission; excludes checkpoint loading, checks and prompt ingestion.',
                  cases={kind + '/' + mode: dict(weights=str(path.resolve()), runs=[]) for kind, mode, path in cases})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for round_id in range(args.rounds):
        offset = round_id % len(cases)
        for kind, mode, path in cases[offset:] + cases[:offset]:
            command = [sys.executable, str(Path(__file__)), '--weights', str(path), '--cpu-kernels', mode,
                       '--backend', args.backend, '--trials', str(args.trials), '--context-case', str(args.context_case)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
            run = json.loads(result.stdout)
            run['round'] = round_id + 1
            name = kind + '/' + mode
            report['cases'][name]['runs'].append(run)
            print(f"round {round_id + 1}: {name}: {statistics.mean(t['steps_per_second'] for t in run['trials']):.2f} steps/s", flush=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
    for name, case in report['cases'].items():
        rates = [trial['steps_per_second'] for run in case['runs'] for trial in run['trials']]
        heads = [statistics.median(run['vocabulary_ms']) for run in case['runs']]
        case['summary'] = dict(median_steps_per_second=statistics.median(rates),
                               min_steps_per_second=min(rates), max_steps_per_second=max(rates),
                               median_vocabulary_ms=statistics.median(heads))
        print(name, json.dumps(case['summary']), flush=True)
    q4, q8 = (report['cases'][kind + '/native']['summary'] for kind in ('Q4_0', 'Q8_0'))
    report['q4_native_speedup_over_q8'] = q4['median_steps_per_second'] / q8['median_steps_per_second']
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
