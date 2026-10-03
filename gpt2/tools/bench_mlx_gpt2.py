#!/usr/bin/env python3
"""GPT-2 MLX-LM GPU implementation reference with synchronized timing."""
import argparse
import datetime
import ctypes
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
WEIGHT_SHA = '248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def output(*args):return subprocess.check_output(args,text=True).strip()


def stats(samples):
    ordered=sorted(samples);total=sum(samples)
    return dict(raw_ms=samples,samples=len(samples),total_ms=total,mean_ms=total/len(samples),
        p50_ms=ordered[len(samples)//2],p90_ms=ordered[len(samples)*9//10],
        steps_per_second=1000*len(samples)/total)


def thermal():
    ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
    objc=ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    objc.objc_getClass.argtypes=[ctypes.c_char_p];objc.objc_getClass.restype=ctypes.c_void_p
    objc.sel_registerName.argtypes=[ctypes.c_char_p];objc.sel_registerName.restype=ctypes.c_void_p
    address=ctypes.cast(objc.objc_msgSend,ctypes.c_void_p).value
    sendobj=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)(address)
    sendint=ctypes.CFUNCTYPE(ctypes.c_ulong,ctypes.c_void_p,ctypes.c_void_p)(address)
    info=sendobj(objc.objc_getClass(b'NSProcessInfo'),objc.sel_registerName(b'processInfo'))
    state=sendint(info,objc.sel_registerName(b'thermalState'))
    return ['nominal','fair','serious','critical'][state] if state<4 else 'unknown'


def background_snapshot():
    # Capture names and CPU activity, not potentially sensitive process arguments.
    return output('ps','-axo','pid,ppid,%cpu,comm')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.home()/'mlx-lm')
    p.add_argument('--weights',type=Path,default=Path.home()/'.cache/huggingface/hub/models--openai-community--gpt2/snapshots'/REVISION)
    p.add_argument('--reference-python',type=Path,default=Path.home()/'more-ane-transformers/.venv/bin/python')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dtype',choices=['float16','float32'],default='float16')
    args=p.parse_args();repo=args.repo.expanduser().resolve();weights=args.weights.expanduser().resolve();result=args.output.expanduser().resolve()
    if platform.system()!='Darwin' or platform.machine()!='arm64':p.error('Requires Apple Silicon macOS')
    log_path=result.with_suffix('.log')
    if result.exists() or log_path.exists():p.error('Use new output and log paths')
    if digest(weights/'model.safetensors')!=WEIGHT_SHA:raise ValueError('Wrong GPT-2 checkpoint')
    lock_path=Path.home()/'gpu.lock';gpu_lock=lock_path.open('a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import mlx.core as mx
    import mlx_lm
    import numpy as np
    from mlx_lm.models.gpt2 import Model,ModelArgs
    from mlx_lm.models.cache import make_prompt_cache
    from mlx.utils import tree_flatten
    mx.set_default_device(mx.gpu)
    dtype=getattr(mx,args.dtype)
    if not mx.metal.is_available():raise ValueError('Metal GPU unavailable')
    installed=Path(mlx_lm.__file__).resolve().parent
    source_hashes={}
    for name in ('gpt2.py','base.py','cache.py'):
        actual=installed/'models'/name;cloned=repo/'mlx_lm/models'/name
        if digest(actual)!=digest(cloned):raise ValueError(f'Installed runtime differs from cloned source: {name}')
        source_hashes[name]=digest(actual)
    print('MLX-LM GPU '+args.dtype+'; same pinned checkpoint and HF traces; no model download',flush=True)
    background_start=background_snapshot()
    started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gpt2-mlx-') as tmp,log_path.open('w') as log:
        directory=Path(tmp)
        subprocess.run([str(args.reference_python.expanduser()),str(ROOT/'tools/bench_coreml_fair.py'),
            '--reference-only',str(directory),'--weights',str(weights),'--trials','4'],
            check=True,stdout=log,stderr=log)
        trace=json.loads((directory/'trace.json').read_text())
        saved=json.loads((ROOT/'provenance/coreml-performance/m1-fair.json').read_text())
        assert all(a['prompt_ids']==b['prompt_ids'] and a['next_tokens']==b['next_tokens']
            for a,b in zip(trace['cases'],saved['reference']['cases']))
        reference=np.memmap(directory/'reference.f32',dtype='<f4',mode='r')
        begin=time.perf_counter()
        config=json.loads((weights/'config.json').read_text())
        model=Model(ModelArgs.from_dict(config))
        weight_dict=mx.load(str(weights/'model.safetensors'))
        weight_dict={k.removeprefix('transformer.'):v.astype(dtype)
            for k,v in weight_dict.items() if k!='lm_head.weight'}
        model.load_weights(list(model.sanitize(weight_dict).items()),strict=True)
        model.eval();mx.eval(model.parameters());
        assert all(v.dtype==dtype for _,v in tree_flatten(model.parameters()) if isinstance(v,mx.array))
        load_ms=(time.perf_counter()-begin)*1000
        del weight_dict

        def predict(ids,cache):
            begin=time.perf_counter()
            logits=model(ids,cache=cache)[:, -1, :]
            # Evaluate every cache buffer as well as logits; no timing of lazy enqueue only.
            buffers=[v for c in cache for _,v in tree_flatten(c.state) if isinstance(v,mx.array)]
            mx.eval(logits,*buffers)
            return logits,(time.perf_counter()-begin)*1000

        def argmax(logits):
            token=mx.argmax(logits,axis=-1);mx.eval(token)
            return int(token.item())

        def run(case,steps,check=False,greedy=False):
            initial_thermal=thermal()
            t0=time.perf_counter();cache=make_prompt_cache(model)
            logits,prefill=predict(mx.array([case['prompt_ids']],dtype=mx.int32),cache)
            chosen=argmax(logits);ttft=(time.perf_counter()-t0)*1000
            engine=[];request=[];errors=[];generated=[chosen]
            def verify(offset):
                values=np.asarray(logits,dtype=np.float64).reshape(-1)
                ref=np.asarray(reference[case['reference_offset']+offset*50257:case['reference_offset']+(offset+1)*50257],dtype=np.float64)
                if not np.isfinite(values).all():raise ValueError('Nonfinite MLX logits')
                lp=ref-ref.max();lp-=np.log(np.exp(lp).sum())
                lq=values-values.max();lq-=np.log(np.exp(lq).sum())
                errors.append(dict(top1=int(values.argmax()),reference_top1=int(ref.argmax()),
                    kl_hf_to_mlx_nats=max(0.,float((np.exp(lp)*(lp-lq)).sum())),
                    normalized_logit_rmse=float(np.linalg.norm(values-ref)/max(np.linalg.norm(ref),1e-6))))
            if check:verify(0)
            for step in range(steps):
                t0=time.perf_counter()
                token=chosen if greedy else case['next_tokens'][step]
                logits,elapsed=predict(mx.array([[token]],dtype=mx.int32),cache)
                chosen=argmax(logits)
                request.append((time.perf_counter()-t0)*1000);engine.append(elapsed);generated.append(chosen)
                if check:verify(step+1)
            return dict(prefill_prediction_ms=prefill,warm_ttft_ms=ttft,decode=stats(engine),
                request_decode=stats(request),diagnostics=errors,generated_token_ids=generated,thermal_start=initial_thermal,thermal_end=thermal())

        cases=[]
        for case in trace['cases']:
            checked=run(case,64,check=True)
            free=run(case,15,greedy=True)['generated_token_ids']
            trials=[]
            for trial in range(4):
                for _ in range(2):run(case,16)
                measured=run(case,64);measured.pop('diagnostics');measured.pop('generated_token_ids');trials.append(measured)
                print(f"prompt={len(case['prompt_ids'])} trial={trial+1} {measured['decode']['steps_per_second']:.2f} engine steps/s",flush=True)
            diag=checked['diagnostics'];mismatch=sum(v['top1']!=v['reference_top1'] for v in diag);kl=max(v['kl_hf_to_mlx_nats'] for v in diag)
            cases.append(dict(prompt_tokens=len(case['prompt_ids']),trials=trials,
                decode=stats([x for t in trials for x in t['decode']['raw_ms']]),
                request_decode=stats([x for t in trials for x in t['request_decode']['raw_ms']]),
                prefill=stats([t['prefill_prediction_ms'] for t in trials]),warm_ttft=stats([t['warm_ttft_ms'] for t in trials]),
                checked_predictions=len(diag),top1_mismatches=mismatch,max_kl_hf_to_mlx_nats=kl,
                max_normalized_logit_rmse=max(v['normalized_logit_rmse'] for v in diag),diagnostics=diag,
                strict_trace_choice_and_kl_gate_passed=(mismatch==0 and kl<=.01),
                free_greedy_token_ids=free,free_greedy_exact_hf_match=free==case['next_tokens'][:16]))
        report=dict(format_version=1,backend='MLX-LM GPU, '+args.dtype+', unquantized',cases=cases,
            configuration=dict(batch_size=1,decode_steps=64,trials_per_prompt=4,warmups_per_trial=2,warmup_steps=16,
                prompt_lengths=[2,32,64],context_capacity=1024,decode_input_width=1,prefix_cache='disabled; fresh KV cache per request',
                precision='All model parameter tensors cast from the pinned HF checkpoint to '+args.dtype+'; upstream MLX layer norm/attention kernels',
                engine_timer='Python model call/graph construction and synchronized mx.eval of last-token logits plus all KV buffers; input construction and argmax excluded',
                request_timer='Python input construction, model call, cache update, sync GPU evaluation and greedy argmax; no printing, full-logit diagnostics or detokenization',
                accuracy='Untimed 65-prediction HF replay per prompt; not universal quality validation',
                comparison_scope='Same checkpoint and decode traces; different graph/cache/arithmetic from Orion/CoreML; separate session'),
            reference=dict(checkpoint_revision=REVISION,checkpoint_sha256=WEIGHT_SHA,cases=trace['cases'],same_traces_as_coreml=True),
            provenance=dict(started_at_utc=started_at,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                host=dict(chip=output('sysctl','-n','machdep.cpu.brand_string'),model=output('sysctl','-n','hw.model'),macos=platform.mac_ver()[0],macos_build=output('sw_vers','-buildVersion')),
                source_repo='https://github.com/ml-explore/mlx-lm',repo=str(repo),repo_commit=output('git','-C',str(repo),'rev-parse','HEAD'),
                repo_status=output('git','-C',str(repo),'status','--short'),runtime_path=str(installed),runtime_mlx_lm_version=mlx_lm.__version__,
                model_attention_cache_sources_match_clone=True,source_sha256=source_hashes,
                packages={n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','numpy','transformers','safetensors')},
                runner_sha256=digest(Path(__file__)),reference_runner_sha256=digest(ROOT/'tools/bench_coreml_fair.py'),
                reference_trace_sha256=digest(directory/'trace.json'),reference_logits_sha256=digest(directory/'reference.f32'),
                background_processes_start=background_start,background_processes_end=background_snapshot(),
                model_load_and_cast_ms_excluded=load_ms,python=sys.executable,gpu_lock=str(lock_path),command_line=sys.argv))
    report['log_sha256']=digest(log_path)
    result.write_text(json.dumps(report,indent=2)+'\n')
    for c in cases:print(c['prompt_tokens'],round(c['decode']['steps_per_second'],2),'steps/s; gate',c['strict_trace_choice_and_kl_gate_passed'],'HF top1 mismatches',c['top1_mismatches'])
    print('Saved',result)


if __name__=='__main__':main()
