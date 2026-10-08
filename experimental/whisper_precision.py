"""Precision diagnostic using shared HF checks and a Linux register adapter.

Generated tensors/results are data; all executable experiment code lives here.
This partial-task adapter requires the captured dense wrapper on Asahi. The
production encoder benchmark supports both macOS and Asahi.
"""
import argparse
import json
from pathlib import Path
import struct
import time
import zlib

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration

from gpt2.hwx import parse_tasks
from whisper.replay_encoder import Encoder
from whisper.validation import compare, digest, logits_records, hardware_locks

ROOT = Path(__file__).resolve().parents[1]


def patchreg(cmd,t,address,value):
    cur=t['offset']+(44 if t['header'][6]&(1<<24) else 40)
    while cur<t['offset']+t['size']:
        packet,=struct.unpack_from('<I',cmd,cur);cur+=4
        if not packet:continue
        n=(packet>>26)+1;a=packet&0x3ffffff
        if a<=address<a+n*4:
            struct.pack_into('<I',cmd,cur+address-a,value);return
        cur+=n*4
    raise ValueError('missing register '+hex(address))

class Segments:
    def __init__(self,checkpoint,kernels,attention_mode="ane",projection_mode="ane",ffn_output_mode="ane",qkv_mode="ane"):
        self.encoder=Encoder(checkpoint,kernels=kernels)
        self.commands=self.encoder.payloads["commands"]
        self.tasks=parse_tasks(self.commands,504,1783)
        self.programs={};self.count=0;self.dispatch_ms=0;self.attention_mode=attention_mode;self.projection_mode=projection_mode;self.ffn_output_mode=ffn_output_mode;self.qkv_mode=qkv_mode
        self.commands_uploaded=False;self.command_upload_bytes=0
    def program(self,first,last):
        key=(first,last)
        if key in self.programs:return self.programs[key]
        cmd=bytearray(self.commands);base=self.tasks[first]['offset'];end=self.tasks[last]['offset']+self.tasks[last]['size']
        for i,t in enumerate(self.tasks[first:last+1]):
            h=t['header'].copy();h[0]=(h[0]&~65535)|i
            if not i:h[0]&=~(1<<24)
            if h[7]:h[7]-=base
            if first+i==last:h[0]|=1<<25;h[7]=0;h[1]&=65535
            struct.pack_into('<10I',cmd,t['offset'],*h)
            if h[6]&(1<<24):
                dep,=struct.unpack_from('<I',cmd,t['offset']+40)
                assert dep>=first
                struct.pack_into('<I',cmd,t['offset']+40,dep-first)
            if first+i in (6,11,12,13,448,449,884,885,1320,1321,1756,1757):
                patchreg(cmd,t,0xc804,t['registers'][0xc804]&~(3<<16))
                for address in range(0x1f848,0x1f888,4):
                    if t['registers'].get(0x1f808+address-0x1f848,0)&1:
                        patchreg(cmd,t,address,t['registers'][address]+128)
                for address in range(0x1f888,0x1f8c8,4):
                    if t['registers'].get(0x1f808+address-0x1f888,0)&1:
                        patchreg(cmd,t,address,t['registers'][address]-128)
        t=self.tasks[last]
        if last==16:
            patchreg(cmd,t,0x8800,0)
        cmd[:end-base]=cmd[base:end]
        size=self.tasks[first]['size'];count=last-first+1
        cmd=bytes(cmd[:end-base])
        parse_tasks(cmd,size,count)
        self.programs[key]=(cmd,size,count);return self.programs[key]
    def run(self,first,last):
        cmd,size,count=self.program(first,last);e=self.encoder
        if not self.commands_uploaded:
            # Keep absolute command-constant addresses resident. Each dispatch
            # changes only its short active prefix; coefficient BAR placement
            # still uses the full pinned command allocation in tsk_size.
            e.buffers[0].write(self.commands)
            self.commands_uploaded=True
        e.buffers[0].write(cmd)
        self.command_upload_bytes+=len(cmd)
        boot=bytearray(cmd[:size]);word,=struct.unpack_from('<I',boot)
        struct.pack_into('<I',boot,0,(word&~(255<<16))|(64<<16));e.bootstrap.write(boot)
        e.request.td_count=count;e.request.td_size=size
        print(json.dumps(dict(segment=[first,last],phase='submit')),flush=True)
        start=time.perf_counter();e.ioctl(e.fd,e.submit_opcode,e.request)
        self.dispatch_ms+=(time.perf_counter()-start)*1000;self.count+=1
    def read(self,offset,channels=384,width=1500):
        a=np.frombuffer(self.encoder.buffers[3].map[offset:offset+2*channels*width],'<f2').astype('f4')
        assert np.isfinite(a).all()
        return torch.from_numpy(a.reshape(channels//8,width,8).transpose(1,0,2).copy().reshape(width,channels))
    def write(self,x,offset):
        a=x.detach().numpy();width,channels=a.shape
        assert channels%8==0 and np.isfinite(a).all()
        packed=a.reshape(width,channels//8,8).transpose(1,0,2).copy().astype('<f2')
        assert np.isfinite(packed).all()
        self.encoder.buffers[3].write(packed.tobytes(),offset)
    def ffn_first(self, index, norm, layer):
        b = 436*index
        self.write(norm, 0)
        self.run(446+b, 449+b)
        linear = self.read(0x232800, channels=1536)
        affine = norm.half().float()*layer.final_layer_norm.weight+layer.final_layer_norm.bias
        return linear, compare(layer.fc1(affine).numpy(), linear.numpy())
    def __call__(self,mel,model):
        self.count=0;self.dispatch_ms=0;self.command_upload_bytes=0;start=time.perf_counter()
        e=self.encoder;e.buffers[5].write(mel.astype('<f2').tobytes());e.buffers[3].write(bytes(e.buffers[3].size))
        self.run(0,6);self.write(torch.nn.functional.gelu(self.read(0,width=3000)),0)
        self.run(7,16)
        state=torch.nn.functional.gelu(self.read(0))+model.embed_positions.weight
        errors=[]
        for index,layer in enumerate(model.layers):
            b=436*index
            norm=torch.nn.functional.layer_norm(state,(384,),eps=1e-5)
            self.write(norm,0x232800 if index==3 else 0x119400)
            self.write(torch.zeros_like(state),0)
            if self.attention_mode=="fp32":
                if self.qkv_mode=="fp32":
                    affine=norm*layer.self_attn_layer_norm.weight+layer.self_attn_layer_norm.bias
                    q=layer.self_attn.q_proj(affine);k=layer.self_attn.k_proj(affine);v=layer.self_attn.v_proj(affine)
                    qkv_errors={}
                else:
                    self.run(36+b,46+b if index<3 else 1351)
                    q=self.read(0x232800)
                    key_offset=0x34bc00 if index<3 else 0x465000
                    raw=np.frombuffer(e.buffers[3].map[key_offset:key_offset+1152000],'<f2').astype('f4')
                    k=torch.from_numpy(raw.reshape(375,384,4).transpose(0,2,1).copy().reshape(1500,384))
                    v=self.read(0x119400)
                    qkv_errors={}
                    affine=norm.half().float()*layer.self_attn_layer_norm.weight+layer.self_attn_layer_norm.bias
                    for name,actual,proj in [("q",q,layer.self_attn.q_proj),("k",k,layer.self_attn.k_proj),("v",v,layer.self_attn.v_proj)]:
                        qkv_errors[name]=compare(proj(affine).numpy(),actual.numpy())
                Q=q.reshape(1500,6,64).permute(1,0,2);K=k.reshape(1500,6,64).permute(1,0,2);V=v.reshape(1500,6,64).permute(1,0,2)
                attention=((Q@K.transpose(1,2)/8).softmax(-1)@V).permute(1,0,2).reshape(1500,384)
                self.write(attention,self.tasks[425+b]['registers'][0x13808])
                self.write(torch.zeros_like(state),0)
                if self.projection_mode=="ane":self.run(425+b,426+b)
            else:
                qkv_errors={}
                self.run(36+b,426+b)
            if self.projection_mode=="fp32":
                projection=layer.self_attn.out_proj(attention)
            else:
                projection=self.read(0x119400)
                attention=self.read(self.tasks[425+b]['registers'][0x13808])
            err=compare(layer.self_attn.out_proj(attention).numpy(),projection.numpy())
            assert err['nrmse']<.003,('raw attention projection',index,err)
            state=state+projection
            norm=torch.nn.functional.layer_norm(state,(384,),eps=1e-5)
            linear,fc1_error=self.ffn_first(index,norm,layer)
            assert fc1_error['nrmse']<.005,('raw FFN first projection',index,fc1_error)
            gelu=torch.nn.functional.gelu(linear);self.write(gelu,0x232800)
            self.write(torch.zeros_like(state),0x119400)
            if self.ffn_output_mode=="fp32":
                projection=layer.fc2(gelu)
                rounded_gelu=gelu
            else:
                self.run(450+b,451+b if index==3 else 452+b)
                projection=self.read(0)
                rounded_gelu=gelu.half().float()
            ferr=compare(layer.fc2(rounded_gelu).numpy(),projection.numpy())
            assert ferr['nrmse']<.003,('raw FFN projection',index,ferr)
            state=state+projection
            errors.append(dict(layer=index,attention_projection=err,ffn_first_projection=fc1_error,ffn_projection=ferr,qkv=qkv_errors))
            print(json.dumps(errors[-1]),flush=True)
        features=torch.nn.functional.layer_norm(state,(384,),model.layer_norm.weight,model.layer_norm.bias,1e-5)
        return features.half().float()[None],dict(wall_ms=(time.perf_counter()-start)*1000,dispatch_ms=self.dispatch_ms,submissions=self.count,command_upload_bytes=self.command_upload_bytes,raw_projection_checks=errors)
    def close(self):self.encoder.close()


class PairedQKV(Segments):
    def __init__(self, model, checkpoint, kernels, limbs=4, paired_outputs=False, output_replicas=1, output_partitions=1, output_feedback=False, paired_fc1=False):
        super().__init__(checkpoint, kernels, attention_mode='fp32', projection_mode='fp32', ffn_output_mode='fp32', qkv_mode='fp32')
        self.model, self.limbs, self.paired_outputs = model, limbs, paired_outputs
        self.output_replicas = output_replicas
        self.output_partitions = output_partitions
        self.output_feedback = output_feedback
        self.paired_fc1 = paired_fc1
        self.recipe = json.loads(zlib.decompress((kernels / 'packing.json.zlib').read_bytes()))
        self.commands = bytearray(self.commands)
        constants = bytearray(self.encoder.payloads["constants"])
        # QKV receives the full FP32 layer-norm affine input from the host.
        # Disable its fused gamma/beta to avoid applying the affine twice.
        for name, data in [('commands', self.commands), ('constants', constants)]:
            for op in self.recipe[name]['operations']:
                if '.self_attn_layer_norm.' in op['tensor'] or (paired_fc1 and '.final_layer_norm.' in op['tensor']):
                    values = np.ones(op['bytes'] // 2, '<f2') if op['kind'] == 'tensor' else np.zeros(op['bytes'] // 2, '<f2')
                    data[op['offset']:op['offset'] + op['bytes']] = values.tobytes()
        self.encoder.buffers[2].write(constants)
        self.weights = {}
        self.active = {}
        self.projection_checks = []
        self.output_checks = []
        for index, layer in enumerate(model.layers):
            for name in ['q', 'k', 'v']:
                weight = getattr(layer.self_attn, name + '_proj').weight.detach().numpy()
                high = weight.astype('<f2')
                low = ((weight - high.astype('f4')) * 1024).astype('<f2')
                assert np.isfinite(high).all() and np.isfinite(low).all()
                self.weights[index, name] = [high, low]
            prefix = f'model.encoder.layers.{index}.self_attn.'
            for op in self.recipe['coefficients']['operations']:
                if ((op['tensor'].startswith(prefix) and op['tensor'].endswith('_proj.bias')
                     and (paired_outputs or '.out_proj.' not in op['tensor']))
                    or (paired_outputs and op['tensor'] == f'model.encoder.layers.{index}.fc2.bias')):
                    self.encoder.buffers[0].write(bytes(op['bytes']), 1277952 + op['offset'])
                if paired_fc1 and op['tensor'] == f'model.encoder.layers.{index}.fc1.bias':
                    self.encoder.buffers[0].write(bytes(op['bytes']), 1277952 + op['offset'])

    def ffn_first(self, index, norm, layer):
        if not self.paired_fc1:
            return super().ffn_first(index, norm, layer)
        assert torch.equal(layer.fc1.weight, layer.fc1.weight.half().float()), 'FC1 checkpoint weights must fit FP16 exactly'
        affine = norm*layer.final_layer_norm.weight+layer.final_layer_norm.bias
        high = affine.half().float()
        low = ((affine-high)*1024).half().float()
        linear = None
        for source, factor in [(high, 1.), (low, 1/1024)]:
            self.write(source, 0)
            self.run(446+436*index, 449+436*index)
            partial = self.read(0x232800, channels=1536)
            linear = partial*factor if linear is None else linear+partial*factor
        linear += layer.fc1.bias
        expected = torch.nn.functional.linear(affine, layer.fc1.weight, layer.fc1.bias)
        return linear, compare(expected.numpy(), linear.numpy())

    def select_weights(self, index, limb):
        if self.active.get(index) == limb: return
        prefix = f'model.encoder.layers.{index}.self_attn.'
        for op in self.recipe['coefficients']['operations']:
            if op['tensor'].startswith(prefix) and op['tensor'].endswith('_proj.weight') and '.out_proj.' not in op['tensor']:
                assert op['kind'] == 'tile'
                name = op['tensor'].split('.')[-2][0]
                source = self.weights[index, name][limb]
                tile = source[op['first']:op['first'] + op['count']].T.copy()
                assert tile.nbytes == op['bytes']
                self.encoder.buffers[0].write(tile.tobytes(), 1277952 + op['offset'])
        self.active[index] = limb

    def matmul(self, index, source, limb):
        self.select_weights(index, limb)
        self.write(source, 0x232800 if index == 3 else 0x119400)
        self.write(torch.zeros_like(source), 0)
        b = 436 * index
        self.run(36 + b, 46 + b if index < 3 else 1351)
        q = self.read(0x232800)
        key_offset = 0x34bc00 if index < 3 else 0x465000
        raw = np.frombuffer(self.encoder.buffers[3].map[key_offset:key_offset + 1152000], '<f2').astype('f4')
        assert np.isfinite(raw).all()
        k = torch.from_numpy(raw.reshape(375, 384, 4).transpose(0, 2, 1).copy().reshape(1500, 384))
        v = self.read(0x119400)
        result = dict(q=q, k=k, v=v)
        checks = {}
        for name, actual in result.items():
            weight = torch.from_numpy(self.weights[index, name][limb].astype('f4'))
            expected = (source.half().float() @ weight.T).numpy()
            if not np.any(expected):
                assert np.array_equal(actual.numpy(), expected), ('zero limb produced nonzero output', index, name, limb)
                error = dict(nrmse=0., cosine=1., max_abs=0., exactly_zero=True)
            else:
                error = compare(expected, actual.numpy())
            assert error['nrmse'] < .003, ('raw paired limb', index, name, limb, error)
            checks[name] = error
        return result, checks

    def qkv(self, index, affine):
        begin = time.perf_counter()
        high = affine.half().float()
        low = ((affine - high) * 1024).half().float()
        result, checks = self.matmul(index, high, 0)
        raw_checks = [dict(input_limb=0, weight_limb=0, errors=checks)]
        if self.limbs > 1:
            for source, weight_limb, factor, input_limb in [(low, 0, 1/1024, 1), (high, 1, 1/1024, 0), (low, 1, 1/1048576, 1)]:
                if weight_limb and not any(np.any(self.weights[index, name][1]) for name in ['q', 'k', 'v']):
                    raw_checks.append(dict(input_limb=input_limb, weight_limb=weight_limb, skipped_exactly_zero=True))
                    continue
                partial, checks = self.matmul(index, source, weight_limb)
                for name in result: result[name] += partial[name] * factor
                raw_checks.append(dict(input_limb=input_limb, weight_limb=weight_limb, errors=checks))
        layer = self.model.layers[index]
        errors = {}
        for name in result:
            projection = getattr(layer.self_attn, name + '_proj')
            expected = torch.nn.functional.linear(affine, projection.weight, projection.bias)
            if projection.bias is not None: result[name] += projection.bias
            errors[name] = compare(expected.numpy(), result[name].numpy())
        self.projection_checks.append(dict(layer=index, errors=errors, raw_checks=raw_checks, wall_ms=(time.perf_counter() - begin)*1000))
        print(json.dumps(dict(layer=index, qkv_precision_errors=errors)), flush=True)
        return result

    def output_projection(self, index, kind, source):
        layer = self.model.layers[index]
        projection = layer.self_attn.out_proj if kind == 'attention' else layer.fc2
        assert np.array_equal(projection.weight.numpy(), projection.weight.half().float().numpy()), 'paired output path requires exact FP16 checkpoint weights'
        begin = time.perf_counter()
        result = None
        b = 436 * index
        if kind == 'attention':
            first, last = 425 + b, 426 + b
            input_offset = self.tasks[first]['registers'][0x13808]
            output_offset, residual_offset = 0x119400, 0
        else:
            first, last = 450 + b, 451 + b if index == 3 else 452 + b
            input_offset, output_offset, residual_offset = 0x232800, 0, 0x119400
        checks = []
        gains = [1.] if self.output_replicas == 1 else [1., 1.5]
        assert source.shape[1] % self.output_partitions == 0
        part_width = source.shape[1] // self.output_partitions
        for part in range(self.output_partitions):
            partial_source = torch.zeros_like(source)
            partial_source[:, part*part_width:(part+1)*part_width] = source[:, part*part_width:(part+1)*part_width]
            for gain in gains:
                scaled = partial_source * gain
                high = scaled.half().float()
                residual = scaled - high
                peak = float(residual.abs().max())
                low_scale = 2. ** int(np.floor(np.log2(64 / peak))) if peak else 1.
                low = (residual * low_scale).half().float()
                for limb, factor in [(high, 1.), (low, 1/low_scale)]:
                    self.write(limb, input_offset)
                    self.write(torch.zeros((1500, 384)), residual_offset)
                    self.run(first, last)
                    actual = self.read(output_offset)
                    expected = torch.nn.functional.linear(limb, projection.weight)
                    feedback = None
                    if self.output_feedback and self.tasks[last]['registers'].get(0x8800, 0) & (1 << 19):
                        # If the PE residual add precedes output rounding, a
                        # second evaluation with -Y_hi can expose the lost bits.
                        # If it follows rounding, this returns zero. Measure it.
                        first_output = actual
                        self.write(limb, input_offset)
                        self.write(-first_output, residual_offset)
                        self.run(first, last)
                        correction = self.read(output_offset)
                        residual = expected-first_output
                        feedback = dict(nonzero_fraction=float(torch.count_nonzero(correction))/correction.numel(),
                            max_abs=float(correction.abs().max()),
                            residual_error=compare(residual.numpy(), correction.numpy())
                                if torch.any(residual) and torch.any(correction) else None)
                        actual = first_output+correction
                    elif self.output_feedback:
                        feedback = dict(skipped="segment has no fused residual add")
                    if not torch.any(expected):
                        assert torch.equal(actual, expected)
                        check = dict(nrmse=0., cosine=1., max_abs=0., exactly_zero=True)
                    else:
                        check = compare(expected.numpy(), actual.numpy())
                    assert check['nrmse'] < .003, ('raw paired output projection', index, kind, check)
                    checks.append(dict(partition=part, gain=gain, low_scale=low_scale, error=check, feedback=feedback))
                    contribution = actual * (factor / gain / len(gains))
                    result = contribution if result is None else result + contribution
        if projection.bias is not None: result += projection.bias
        expected = torch.nn.functional.linear(source, projection.weight, projection.bias)
        error = compare(expected.numpy(), result.numpy())
        self.output_checks.append(dict(layer=index, kind=kind, gains=gains, partitions=self.output_partitions, error=error, raw_checks=checks, wall_ms=(time.perf_counter()-begin)*1000))
        print(json.dumps(dict(layer=index, kind=kind, paired_output_error=error)), flush=True)
        return result

    def __call__(self, mel, model):
        self.projection_checks = []
        self.output_checks = []
        originals = []
        caches = {}
        output_caches = {}
        def forward(index, name):
            def call(affine):
                if name == 'q': caches[index] = (affine, self.qkv(index, affine))
                else: assert torch.equal(affine, caches[index][0])
                return caches[index][1][name]
            return call
        def output_forward(index, kind):
            def call(source):
                key = (index, kind)
                if key not in output_caches or not torch.equal(source, output_caches[key][0]):
                    output_caches[key] = (source, self.output_projection(index, kind, source))
                return output_caches[key][1]
            return call
        try:
            for index, layer in enumerate(model.layers):
                for name in ['q', 'k', 'v']:
                    module = getattr(layer.self_attn, name + '_proj')
                    originals.append((module, module.forward))
                    module.forward = forward(index, name)
                if self.paired_outputs:
                    for kind, module in [('attention', layer.self_attn.out_proj), ('ffn', layer.fc2)]:
                        originals.append((module, module.forward))
                        module.forward = output_forward(index, kind)
            features, info = super().__call__(mel, model)
            info['paired_qkv_checks'] = self.projection_checks
            info['paired_output_checks'] = self.output_checks
            return features, info
        finally:
            for module, original in originals: module.forward = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-model", type=Path, default=ROOT / "whisper/models/hf-tiny.en")
    p.add_argument("--kernels", type=Path, default=ROOT / "whisper/kernels/tiny-en-encoder")
    p.add_argument("--traces", type=Path, required=True, help="Native trace directory containing CLIP-cpu/mel.f32 and logits.bin")
    p.add_argument("--clips", nargs="+", choices=("jfk", "jfk-first-5s", "jfk-repeat"), default=["jfk", "jfk-first-5s", "jfk-repeat"])
    p.add_argument("--limbs", type=int, choices=(1, 4), default=4)
    p.add_argument("--paired-outputs", action="store_true")
    p.add_argument("--output-replicas", type=int, choices=(1, 2), default=1)
    p.add_argument("--output-partitions", type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument("--output-feedback", action="store_true", help="Test whether fused residual feedback recovers projection rounding error")
    p.add_argument("--paired-fc1", action="store_true", help="Apply FP32 FFN affine on the host and split FC1 input into two FP16 planes")
    p.add_argument("--exported-paired", action="store_true", help="Replay all 24 exported paired programs with shared Mac/Asahi Python arithmetic")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output already exists")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = WhisperForConditionalGeneration.from_pretrained(a.hf_model, local_files_only=True, attn_implementation="eager").eval()
    report = dict(status="RUNNING", scope="Precision diagnostic only; CPU reference checks are included in time. CPU FP32 normalization, attention, GELU and residuals retained.",
        checkpoint_sha256=digest(a.hf_model / "model.safetensors"), script_sha256=digest(Path(__file__)),
        limbs=a.limbs, paired_outputs=a.paired_outputs, output_replicas=a.output_replicas,
        output_partitions=a.output_partitions, gate_nrmse=.005, records=[])
    report["output_feedback"] = a.output_feedback
    report["paired_fc1"] = a.paired_fc1
    if a.exported_paired:
        report.update(scope="Shared paired Python encoder: 24 ANE submissions / 32 hardware tasks per encode, FP32 HF host arithmetic and shared HF decoder. Native integration/timings and exact Mac speech-array comparison remain unverified.",
                      mode="exported-paired", kernel_manifest_sha256=digest(ROOT / "whisper/kernels/tiny-en-paired/manifest.json"))
    runner = None
    try:
        with hardware_locks(), torch.inference_mode():
            if a.exported_paired:
                from whisper.precision_encoder import PairedEncoder
                runner = PairedEncoder(model.model.encoder, a.hf_model/"model.safetensors")
                report["programs"] = runner.receipts
                report["runtime_source_sha256"] = digest(ROOT / "whisper/paired_replay.py")
                report["shared_arithmetic_source_sha256"] = digest(ROOT / "whisper/precision_encoder.py")
            else:
                runner = PairedQKV(model.model.encoder, a.hf_model/"model.safetensors", a.kernels,
                    a.limbs, a.paired_outputs, a.output_replicas, a.output_partitions,
                    a.output_feedback, a.paired_fc1)
            for clip in a.clips:
                trace = a.traces / f"{clip}-cpu"
                mel = np.fromfile(trace/"mel.f32", "<f4").reshape(80, 3000)
                reference = model.model.encoder(torch.from_numpy(mel[None])).last_hidden_state
                if a.exported_paired:
                    before = runner.submissions
                    began = time.perf_counter()
                    actual = runner(mel)
                    elapsed = (time.perf_counter()-began)*1000
                    repeated = runner(mel)
                    if not np.array_equal(actual, repeated) or runner.submissions-before != 48:
                        raise ValueError("exported paired output/submission repeatability failed")
                    features = torch.from_numpy(actual)
                    info = dict(first_encode_ms=elapsed, submissions_per_encode=24,
                                hardware_tasks_per_encode=sum(p.meta["td_count"] for p in runner.programs),
                                read_workers=sorted({p.last_read_workers for p in runner.programs}),
                                repeat_bitwise_equal=True, input_sha256=digest(trace/"mel.f32"),
                                frozen_native_logit_trace_sha256=digest(trace/"logits.bin"))
                    if info["hardware_tasks_per_encode"] != 32:
                        raise ValueError("exported paired task count changed")
                else:
                    features, info = runner(mel, model.model.encoder)
                np.save(a.output.with_name(a.output.stem+"-"+clip+".npy"), features.numpy())
                checks, history = [], []
                for tokens, _ in logits_records(trace/"logits.bin"):
                    history.extend(tokens.tolist())
                    params = dict(input_ids=torch.tensor([history]), use_cache=False)
                    left = model.proj_out(model.model.decoder(encoder_hidden_states=reference, **params).last_hidden_state[:, -1]).numpy().ravel()
                    right = model.proj_out(model.model.decoder(encoder_hidden_states=features, **params).last_hidden_state[:, -1]).numpy().ravel()
                    checks.append(dict(**compare(left, right), argmax_match=int(left.argmax()) == int(right.argmax())))
                if a.exported_paired and len(checks) != {"jfk":25, "jfk-first-5s":8, "jfk-repeat":47}[clip]:
                    raise ValueError("exported paired full-vector coverage changed")
                row = dict(audio=clip, encoder=compare(reference.numpy(), features.numpy()), timings=info,
                    checks=checks, max_logit_nrmse=max(c["nrmse"] for c in checks), all_argmax_match=all(c["argmax_match"] for c in checks))
                report["records"].append(row)
                a.output.write_text(json.dumps(report, indent=2)+"\n")
                print(json.dumps({k:v for k,v in row.items() if k not in ("checks", "timings")}), flush=True)
        report["status"] = "PASS_DIAGNOSTIC_ONLY" if all(r["max_logit_nrmse"] < .005 and r["all_argmax_match"] for r in report["records"]) else "FAIL"
    except BaseException as error:
        report.update(status="ERROR", error=str(error))
        raise
    finally:
        if runner is not None:
            runner.close()
        a.output.write_text(json.dumps(report, indent=2)+"\n")
    if report["status"] != "PASS_DIAGNOSTIC_ONLY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
