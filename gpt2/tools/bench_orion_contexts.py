#!/usr/bin/env python3
"""Measure native Orion CPU/ANE with the same 2/32/64-token HF traces as CoreML/MLX."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from bench_orion_macos import SOURCES, digest, output
from bench_coreml_macos import REVISION, WEIGHT_SHA

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--orion', type=Path, default=Path.home()/'Desktop/Orion')
    p.add_argument('--weights', type=Path, help='External Orion BLOBFILE directory')
    p.add_argument('--hf-weights', type=Path, default=Path.home()/'.cache/huggingface/hub/models--openai-community--gpt2/snapshots'/REVISION)
    p.add_argument('--reference-python', type=Path, default=Path.home()/'more-ane-transformers/.venv/bin/python')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        p.error('Requires Apple Silicon macOS')
    repo = args.orion.expanduser().resolve()
    weights = (args.weights or repo/'model/blobs/gpt2_124m').expanduser().resolve()
    hf = args.hf_weights.expanduser().resolve()
    result = args.output.expanduser().resolve(); log_path = result.with_suffix('.log')
    if result.exists() or log_path.exists(): p.error('Use new output and log paths')
    lock = (Path.home()/'gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = json.loads((ROOT/'model-checksums.json').read_text())
    for name, checksum in expected.items():
        if digest(weights/name) != checksum: raise ValueError('Wrong external weights: '+name)
    if digest(hf/'model.safetensors') != WEIGHT_SHA: raise ValueError('Wrong HF checkpoint')
    print('Verified external GPT-2 weights; 2/32/64 prompt tokens; no downloads', flush=True)
    compiler = output('xcrun', '--find', 'clang'); sdk = output('xcrun', '--show-sdk-path')
    linker = output('xcrun', '--find', 'ld')
    flags = ['-O2', '-Wall', '-Wextra', '-DACCELERATE_NEW_LAPACK', '-isysroot', sdk,
             '-I', str(repo), '-I', str(repo/'core'), '-I', str(repo/'compiler')]
    harness = Path(__file__).with_suffix('.m')
    helper = ROOT/'tools/bench_coreml_fair.py'
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source_hashes = {name: digest(repo/name) for name in SOURCES}
    source_hashes.update({str(f.relative_to(repo)): digest(f) for f in repo.rglob('*.h')
                         if 'build' not in f.relative_to(repo).parts})
    result.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='orion-contexts-') as tmp, log_path.open('w') as log:
        build = Path(tmp)
        subprocess.run([str(args.reference_python.expanduser()), str(helper), '--reference-only',
                        str(build), '--weights', str(hf), '--trials', '4'],
                       check=True, stdout=log, stderr=log)
        trace = json.loads((build/'trace.json').read_text())
        saved = json.loads((ROOT/'provenance/coreml-performance/m1-fair.json').read_text())
        if trace['cases'] != saved['reference']['cases']:
            raise ValueError('Reference differs from saved CoreML/MLX prompt traces')
        objects = []
        print('Building fresh Orion objects', flush=True)
        for i, source in enumerate([repo/name for name in SOURCES]+[harness]):
            obj = build/f'{i}.o'; command = [compiler, *flags]
            if source.suffix == '.m': command.append('-fobjc-arc')
            subprocess.run([*command, '-c', str(source), '-o', str(obj)],
                           check=True, stdout=log, stderr=log)
            objects.append(str(obj))
        binary = build/'bench'; native = build/'native.json'
        subprocess.run([compiler, '-isysroot', sdk, f'--ld-path={linker}', *objects,
                        '-ldl', '-framework', 'Foundation', '-framework', 'IOSurface',
                        '-framework', 'Accelerate', '-o', str(binary)],
                       check=True, stdout=log, stderr=log)
        background_start = output('ps', '-axo', 'pid,ppid,%cpu,comm')
        memory_before = dict(swap=output('sysctl', 'vm.swapusage'), counters=output('vm_stat'))
        print('Running alternating trials; two fresh warmups per trial; separate HF diagnostics', flush=True)
        subprocess.run([str(binary), str(weights), str(build/'trace.json'),
                        str(build/'reference.f32'), str(native)], check=True, stdout=log, stderr=log)
        report = json.loads(native.read_text())
        report['configuration'] = dict(batch_size=1, prompt_lengths=[2, 32, 64], decode_steps=64,
            trials_per_backend_per_prompt=4, warmups_per_trial=2, warmup_steps=16, context_capacity=1024,
            order='CPU/ANE order alternates per trial; four trials, two first positions per backend',
            prefix_cache='disabled; fresh KV allocation per request',
            cache_residency='One prompt bucket at a time; ANE program cache cleared between prompt cases',
            precision='Orion FP16 stored weights and ANE operations; CPU arithmetic float32',
            engine_timer='Synchronous native decode call with cache-length guard; embeddings, full vocabulary logits, CPU attention, ANE transfers/dispatch included; input trace lookup, argmax, prefill, allocation, printing and numerical checks excluded',
            diagnostics='Separate 65-prediction HF replay and 16-token free greedy run per backend/prompt; no logit checks between timed steps',
            quality_gate='Finite logits, zero HF top1 mismatches, max KL <= 0.01 nats; raw logit NRMSE <= 0.005 reported separately; not general accuracy',
            comparison_scope='Implementation references on one Mac; cross-runtime sessions and graphs differ')
        report['reference'] = dict(checkpoint_revision=REVISION, checkpoint_sha256=WEIGHT_SHA,
                                   cases=trace['cases'])
        report['provenance'] = dict(started_at_utc=started,
            finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            host=dict(chip=output('sysctl', '-n', 'machdep.cpu.brand_string'), model=output('sysctl', '-n', 'hw.model'),
                      macos=platform.mac_ver()[0], macos_build=output('sw_vers', '-buildVersion'),
                      memory_bytes=int(output('sysctl', '-n', 'hw.memsize'))),
            orion_commit=output('git', '-C', str(repo), 'rev-parse', 'HEAD'),
            orion_worktree_status=output('git', '-C', str(repo), 'status', '--short'),
            orion_source_sha256=source_hashes, harness_sha256=digest(harness),
            runner_sha256=digest(Path(__file__)), build_helper_sha256=digest(ROOT/'tools/bench_orion_macos.py'),
            reference_runner_sha256=digest(helper), reference_trace_sha256=digest(build/'trace.json'),
            reference_logits_sha256=digest(build/'reference.f32'), verified_external_tensors=len(expected),
            model_checksums_sha256=digest(ROOT/'model-checksums.json'),
            build=dict(compiler=compiler, version=output(compiler, '--version'), sdk=sdk, linker=linker,
                       flags=flags, objc_flags=['-fobjc-arc'], fresh_objects=True),
            background_processes_start=background_start,
            background_processes_end=output('ps', '-axo', 'pid,ppid,%cpu,comm'),
            memory_before=memory_before,
            memory_after=dict(swap=output('sysctl', 'vm.swapusage'), counters=output('vm_stat')),
            command_line=sys.argv, gpu_lock=str(Path.home()/'gpu.lock'))
    report['log_sha256'] = digest(log_path)
    result.write_text(json.dumps(report, indent=2)+'\n')
    for case in report['cases']:
        for backend, data in case['backends'].items():
            print(case['prompt_tokens'], backend, round(data['decode']['steps_per_second'], 2),
                  'steps/s; HF mismatches', data['top1_mismatches'], 'KL', data['max_kl_hf_to_orion_nats'])
    print('Saved', result)


if __name__ == '__main__': main()
