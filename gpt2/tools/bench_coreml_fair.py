#!/usr/bin/env python3
"""Matched-artifact GPT-2 CoreML CPU/GPU/ANE benchmark; weights stay external."""
import argparse
import datetime
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from bench_coreml_macos import REVISION, WEIGHT_SHA, digest, output

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('cpu_only', 'cpu_and_gpu', 'cpu_and_ne', 'all')


def references(weights, directory, trials):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(weights), local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(str(weights), local_files_only=True,
        use_safetensors=True, torch_dtype=torch.float32, attn_implementation='eager').eval()
    corpus = ('The scientist carefully measured the experiment, recorded the observations, '
              'and compared the results with the predictions. ') * 20
    corpus_ids = tokenizer.encode(corpus, add_special_tokens=False)
    cases, offset = [], 0
    raw = directory / 'reference.f32'
    with raw.open('wb') as stream, torch.inference_mode():
        for length in (2, 32, 64):
            ids = tokenizer.encode('Hello world', add_special_tokens=False) if length == 2 else corpus_ids[:length]
            assert len(ids) == length
            result = model(torch.tensor([ids]), use_cache=True)
            next_tokens = []
            for step in range(65):
                logits = result.logits[0, -1].float().numpy()
                if not np.isfinite(logits).all():
                    raise ValueError('Nonfinite HF reference')
                logits.astype('<f4').tofile(stream)
                token = int(logits.argmax()); next_tokens.append(token)
                if step < 64:
                    result = model(torch.tensor([[token]]), past_key_values=result.past_key_values, use_cache=True)
            cases.append(dict(prompt=tokenizer.decode(ids), prompt_ids=ids, next_tokens=next_tokens,
                reference_offset=offset, steps=64, hf_continuation=tokenizer.decode(next_tokens)))
            offset += 65 * model.config.vocab_size
    trace = dict(cases=cases, vocabulary=50257, trials=trials, warmups=2, warmup_steps=16)
    trace_path = directory / 'trace.json'; trace_path.write_text(json.dumps(trace)+'\n')
    del result, model
    gc.collect()
    return trace_path, raw, trace, tokenizer


def placement(compiled, units):
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.compute_device import MLCPUComputeDevice, MLGPUComputeDevice, MLNeuralEngineComputeDevice
    plan = MLComputePlan.load_from_path(str(compiled), compute_units=units)
    counts = dict(ane=0, cpu=0, gpu=0, unknown=0)
    def visit(block):
        for op in block.operations:
            if op.operator_name == 'const' or op.operator_name.startswith('constexpr_'): continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            device = getattr(usage, 'preferred_compute_device', None)
            key = 'ane' if isinstance(device, MLNeuralEngineComputeDevice) else (
                'gpu' if isinstance(device, MLGPUComputeDevice) else (
                    'cpu' if isinstance(device, MLCPUComputeDevice) else 'unknown'))
            counts[key] += 1
            for child in getattr(op, 'blocks', []): visit(child)
    for function in plan.model_structure.program.functions.values(): visit(function.block)
    return dict(compute_units=units.name, nonconstant_operation_counts=counts,
                interpretation='Static preferred-device counts; not runtime utilization or FLOP shares')


def paired_ratio(baseline, candidate):
    """Resample whole paired trial blocks, never correlated individual tokens."""
    import numpy as np
    a = np.array([t['decode']['total_ms'] for t in baseline['trials']])
    b = np.array([t['decode']['total_ms'] for t in candidate['trials']])
    rng = np.random.default_rng(20261003)
    indices = rng.integers(0, len(a), size=(10000, len(a)))
    ratios = a[indices].sum(axis=1) / b[indices].sum(axis=1)
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    return dict(throughput_ratio=float(a.sum()/b.sum()),
        paired_trial_bootstrap_95_interval=[float(lo), float(hi)],
        paired_trials=len(a), bootstrap_resamples=10000, seed=20261003,
        interpretation='Exploratory trial-block interval; one process/session, not independent machine sessions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Existing pinned more-ane-transformers .mlpackage')
    parser.add_argument('--weights', type=Path, default=Path.home()/'.cache/huggingface/hub/models--openai-community--gpt2/snapshots'/REVISION)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference-only', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--plan-report', type=Path, help='Reuse plans from a prior report only if every model artifact hash matches')
    parser.add_argument('--trials', type=int, default=4, help='Positive multiple of four for balanced trial order')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64': parser.error('Requires Apple Silicon macOS')
    if args.trials < 4 or args.trials % 4: parser.error('--trials must be a positive multiple of four')
    if args.reference_only:
        references(args.weights.expanduser().resolve(), args.reference_only, args.trials)
        return
    if args.model is None or args.output is None: parser.error('--model and --output are required')
    model, weights, result = [p.expanduser().resolve() for p in (args.model,args.weights,args.output)]
    log_path = result.with_suffix('.log')
    if result.exists() or log_path.exists(): parser.error('Use new output and log paths')
    if digest(weights/'model.safetensors') != WEIGHT_SHA: raise ValueError('Wrong checkpoint')
    import coremltools as ct
    spec = ct.models.MLModel(str(model), skip_model_load=True)
    if spec.user_defined_metadata.get('benchmark_hf_revision') != REVISION:
        raise ValueError('Use the previously verified pinned-checkpoint model artifact')
    compiled = Path(str(model).removesuffix('.mlpackage')+'.mlmodelc')
    if not compiled.is_dir(): raise ValueError('Compile the verified model before running this comparison')
    result.parent.mkdir(parents=True, exist_ok=True)
    swift = Path(__file__).with_suffix('.swift')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print('Same GPT-2 artifact; four compute configurations; 2/32/64 prompt tokens; no downloads', flush=True)
    with tempfile.TemporaryDirectory(prefix='gpt2-fair-') as tmp, log_path.open('w') as log:
        directory=Path(tmp)
        print('Creating exact-length HF traces', flush=True)
        subprocess.run([sys.executable,str(Path(__file__)), '--reference-only',str(directory),
            '--weights',str(weights),'--trials',str(args.trials)],check=True,stdout=log,stderr=log)
        trace_path=directory/'trace.json'; raw=directory/'reference.f32'; trace=json.loads(trace_path.read_text())
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(weights),local_files_only=True)
        units=[ct.ComputeUnit.CPU_ONLY,ct.ComputeUnit.CPU_AND_GPU,ct.ComputeUnit.CPU_AND_NE,ct.ComputeUnit.ALL]
        print('Recording compute plans', flush=True)
        plan_source=None
        if args.plan_report:
            plan_source=args.plan_report.expanduser().resolve()
            prior=json.loads(plan_source.read_text())
            current_hashes={str(p.relative_to(model)):digest(p) for p in sorted(model.rglob('*')) if p.is_file()}
            if current_hashes != prior['provenance']['model_files_sha256']: raise ValueError('Plan report model artifact differs')
            plans=prior['compute_plans']
            if set(plans)!=set(BACKENDS): raise ValueError('Plan report lacks required configurations')
        else:
            plans={name:placement(compiled,unit) for name,unit in zip(BACKENDS,units)}
        if plans['cpu_and_gpu']['nonconstant_operation_counts']['ane'] or plans['cpu_and_ne']['nonconstant_operation_counts']['gpu']:
            raise ValueError('Unexpected device placement')
        binary=directory/'bench'; native=directory/'native.json'
        command=['xcrun','swiftc','-O',str(swift),'-o',str(binary),'-framework','CoreML']
        subprocess.run(command,check=True,stdout=log,stderr=log)
        activity_before='\n'.join(line for line in output('ps','-axo','pid,pcpu,etime,comm').splitlines() if 'omlx' in line.lower())
        swap_before=output('sysctl','vm.swapusage'); vm_before=output('vm_stat')
        print('Running balanced native trials; detailed progress goes to '+str(log_path),flush=True)
        subprocess.run([str(binary),str(compiled),str(trace_path),str(raw),str(native)],check=True,stdout=log,stderr=log)
        report=json.loads(native.read_text())
        for case in report['cases']:
            for name,backend in case['backends'].items():
                smoke=backend['free_greedy_smoke'];smoke['continuation']=tokenizer.decode(smoke['generated_token_ids'])
                diag=backend['diagnostics']
                backend['strict_trace_choice_and_kl_gate_passed']=(diag['all_logits_finite'] and diag['top1_mismatches']==0 and diag['max_kl_hf_to_coreml_nats']<=0.01)
            case['ratios']={
                'cpu_and_ne_vs_cpu_only':paired_ratio(case['backends']['cpu_only'],case['backends']['cpu_and_ne']),
                'cpu_and_gpu_vs_cpu_only':paired_ratio(case['backends']['cpu_only'],case['backends']['cpu_and_gpu']),
                'all_vs_cpu_and_gpu':paired_ratio(case['backends']['cpu_and_gpu'],case['backends']['all'])}
        report['configuration']=dict(model='GPT-2 124M',artifact_identical=True,
            precision='Same FP16 artifact with FP32 layer_norm policy; backend arithmetic may differ',
            batch_size=1,prompt_lengths=[2,32,64],decode_steps=64,trials_per_backend_per_prompt=args.trials,
            warmup_generations_per_trial=2,warmup_steps=16,resident_models=1,prefix_cache='disabled; zero KV cache before every request',
            decoding='Shared HF float32 greedy teacher-forced trace; 64 decode inputs and 65 reference logits',
            engine_timer='Synchronous MLModel.prediction only; full-logit diagnostics excluded from every timed trial',
            request_timer='Warm in-process cache reset, input preparation, synchronous prediction, native cache handoff and full-vocabulary greedy argmax; no printing, diagnostics, token detokenization, HTTP or UI',
            ttft='Warm in-process request start through first greedy token selection; excludes model loading/compilation',
            prefill_rate='Actual prompt token count / prompt prediction duration; static input width remains 64',
            order='Four-row balanced Latin square; one backend loaded per block, each prompt trial individually prewarmed',
            quality_gate='Per-prompt HF replay: finite logits, zero top1 mismatches, max KL <= 0.01 nats; does not establish general accuracy',
            comparison_scope='CoreML backend configurations on identical graph; not isolated hardware peaks or Orion/oMLX ranking')
        report['compute_plans']=plans
        report['compute_plan_source']=(dict(report=str(plan_source),sha256=digest(plan_source),same_model_files_verified=True) if plan_source else dict(kind='Fresh MLComputePlan query'))
        report['reference']=dict(checkpoint_revision=REVISION,checkpoint_sha256=WEIGHT_SHA,cases=trace['cases'])
        report['provenance']=dict(started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            host=dict(chip=output('sysctl','-n','machdep.cpu.brand_string'),model=output('sysctl','-n','hw.model'),
                memory_bytes=int(output('sysctl','-n','hw.memsize')),macos=platform.mac_ver()[0],macos_build=output('sw_vers','-buildVersion')),
            external_model=str(model),external_compiled_model=str(compiled),
            model_files_sha256={str(p.relative_to(model)):digest(p) for p in sorted(model.rglob('*')) if p.is_file()},
            harness_sha256=digest(swift),runner_sha256=digest(Path(__file__)),helper_sha256=digest(ROOT/'tools/bench_coreml_macos.py'),
            reference_trace_sha256=digest(trace_path),reference_logits_sha256=digest(raw),build_command=command,
            packages={n:importlib.metadata.version(n) for n in ('coremltools','torch','transformers','numpy','safetensors')},
            existing_omlx_process_before=activity_before,swap_before=swap_before,swap_counters_before=vm_before,swap_counters_after=output('vm_stat'),swap_after=output('sysctl','vm.swapusage'),command_line=sys.argv)
    report['log_sha256']=digest(log_path)
    result.write_text(json.dumps(report,indent=2)+'\n')
    for case in report['cases']:
        print('Prompt tokens:',case['prompt_tokens'])
        for name,b in case['backends'].items():
            print(name,round(b['decode']['steps_per_second'],2),'engine steps/s;',round(b['request_decode']['steps_per_second'],2),'request steps/s; gate',b['strict_trace_choice_and_kl_gate_passed'])
    print('Saved',result)


if __name__=='__main__':main()
