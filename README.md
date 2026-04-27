# Apple ANE running on Asahi

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/allbilly/ane)

Tested on 
- Asahi Linux fedora 6.14.8-400.asahi.fc42.aarch64+16k
- Asahi Linux fedora 6.19.11+ built from https://github.com/allbilly/linux/commit/52d22304e89d2995bfa2e678153feffba5dff23a

## 1. Generate hwx

### on MacOS 12.4

sum.hwx is from https://github.com/tinygrad/tinygrad/tree/v0.10.3/extra/accel/ane/ops

mul.hwx if from MacOS Monterey VM (v12.4 21F79) running on M4 macbook air 

```bash
python gen_mlmodel.py
git clone https://github.com/freedomtan/coreml_to_ane_hwx && cd coreml_to_ane_hwx && make && mv ./coreml2hwx ../ && cd ../
./coreml2hwx ./test.mlmodel
cp /tmp/hwx_output/test/model.hwx ./mul.hwx
```

more ops on https://github.com/eiln/ane-ex/blob/main/sources.md

### on Github action (Not working now)
sadly macos14 is the oldest macos version on gh action, which is not yet supported by anecc 
https://docs.github.com/en/actions/reference/runners/github-hosted-runners

Check the gihub action config at .github/workflows/ane-generation.yml
- go to branch "macos_buildhwx" and modify mode in builder.add_elementwise
- go to actions/runs/_runid_/ -> Artifacts -> download and unzip

## 2. Parse hwx

python parse.py hwx/sum.hwx 

Add vs Mul (verified empirically, see examples/min_add.py → min_mul.py)
| Constant | CMD_BUF offset | sum | mul | HW Register | Field | Description |
|----------|----------------|-----|-----|------------|-------|-------------|
| `MACCfg` | 0x244 | 0x00 | 0x30 | NE MACCfg (0xC804) | KernelMode, BiasMode | **ADD→MUL** |
| `PECfg`  | 0x22c | 0x00 | 0x04 | PE Cfg (0x8800) | OpMode bit 2 | Enables multiply mode |

**Key changes** (only 2 bytes needed in both CMD_BUF and BTSP_BUF):
- `CMD_BUF[MACCfg] = 0x30` — NE KernelMode=1, BiasMode=1
- `CMD_BUF[PECfg] = (CMD_BUF[PECfg] & ~0x04) | 0x04` — PE OpMode=1

Other byte differences (0x20, 0x184, 0x198, 0x218, 0x22c/MSB, 0x26d) exist in the .hwx files but are **not required** for the operation to work.

Add vs Relu (verified empirically, see examples/add.py → examples/relu_from_add.py)

### Data path difference
| | add | relu |
|---|---|---|
| Source | TileDMA (two inputs via SrcDMAConfig=0x33881) | L2 cache (single input via L2Cfg/ SourceCfg) |
| Compute | PE/NE ALU (PECfg=0x80000, scales non-zero) | Conv pipeline only (PECfg=0, all scales=0) |
| Destination | TileDMA | TileDMA (same) |

### Registers that changed (BTSP_BUF offsets, little-endian u32)

**Common section** (all must change together for relu C=1 single-channel config):
| Offset | Register | add | relu | Notes |
|--------|----------|-----|------|-------|
| 0x128 | InDim | 0x00010001 | 0x0001004d | Input dimensions |
| 0x130 | ChCfg | 0x2a | 0x22 | Channel config |
| 0x134 | Cin | 0x40 (64) | 0x01 (1) | Input channels |
| 0x138 | Cout | 0x40 (64) | 0x01 (1) | Output channels |
| 0x13c | OutDim | 0x00010001 | 0x0001004d | Output dimensions |
| 0x14c | GroupConvCfg | 0x10001 | 0x14001 | Group conv |
| 0x154 | pad3 | 4 | 0 | Mode select |
| 0x15c | Cfg | 0x33 | 0x04010101 | Config flags |
| 0x160 | TaskInfo | 0 | 0x00100000 | Task info |

**Data source** (switch from TileDMA → L2 path):
| Offset | Register | add | relu | Notes |
|--------|----------|-----|------|-------|
| 0x16c | SrcDMAConfig | 0x33881 | 0 | **Disable TileDMA source** |
| 0x170 | Srcpad0 | 0x33880 | 0x00500172 | Repurposed as L2 source cfg |
| 0x178-0x184 | SrcRow/Plane/Depth/GroupStride | 0x40/0x40/0x1000/0 | 0xa0/0xa0/0xa0/0xa0 | Stale orphaned values |
| 0x18c-0x1a8 | Srcpad2-4/Fmt/pad8 | varied | 0 | Cleared |
| 0x1e0 | L2Cfg | 0 | **0x6c013800** | **Enable L2 source path** |
| 0x1e4 | SourceCfg | 0x01500172 | **0x33881** | L2 source config |
| 0x1e8 | SourceBase | 0 | **0x8880** | L2 source base addr |
| 0x1f0 | SourceRowStride | 0x420 | **0xc0** | L2 row stride (192) |
| 0x1f4-0x1f8 | L2pad0/1 | 0x400 | **0xc0** | L2 sizes (192) |
| 0x210 | ResultCfg | 0x0050017a | 0 | Disable result path |
| 0x214 | ResultBase | 0x860 | 0 | Clear result base |
| 0x21c | ConvResultRowStride | 0 | 0x01002031 | Repurposed |

**PE → disabled** (relu runs in conv pipeline, not PE):
| Offset | Register | add | relu | Notes |
|--------|----------|-----|------|-------|
| 0x22c | PECfg | 0x80000 | **0** | **Disable PE** |
| 0x230 | BiasScale | 0x3c000000 | **0** | |
| 0x234 | PreScale | 0x3c000000 | **0** | |
| 0x238 | FinalScale | 0x3f800000 | **0** | |

**Destination** (TileDMA, stride changed for C=1):
| Offset | Register | add | relu | Notes |
|--------|----------|-----|------|-------|
| 0x260 | DstRowStride | 0x40 | **0xc0** | 192 bytes |
| 0x264 | DstPlaneStride | 0x40 | **0xc0** | |
| 0x268 | DstDepthStride | 0x1000 | **0xc0** | |
| 0x270 | DstFmt | 0x01002031 | **0x01302031** | Bit 20 set |

**BTSP program code** (instruction bytes changed):
| Offset | add | relu | Notes |
|--------|-----|------|-------|
| 0x01b | 66 49 02 00 | 25 40 02 01 | Program header entry point |
| 0x1b5 | 00 | 88 | Program instruction |
| 0x1b7 | 00 | 0c | |
| 0x1c9-0x1cc | 00 | c8 10 80 | |
| 0x1d0 | 00 | 0c | |
| 0x1d2 | 00 | 11 | |
| 0x1dd | 48 | 3c | |
| 0x225 | 00 | 01 | |

**Key insight**: Unlike add→mul (same BTSP program, 2 register changes), **add→relu requires changing the BTSP firmware program itself + ~25 critical registers**. Relu uses a fundamentally different data path (L2→Conv pipeline instead of TileDMA→PE).

### Experimental results (one-register-at-a-time revert from working relu config)

Each test: start from full relu config, revert ONE register group to add value, check if relu still works.

**Don't-care registers** (relu still works with add's value):
| Register | offset | add | relu | Verdict |
|----------|--------|-----|------|---------|
| `ChCfg`  | 0x130  | 0x2a | 0x22 | **relu works either way** |
| `Cin`    | 0x134  | 64   | 1    | **relu works either way** |

**Wrong output but no ANE crash** (size/dimension mismatch):
| Register | offset | add → relu | Verdict |
|----------|--------|------------|---------|
| `InDim`  | 0x128  | 0x10001 → 0x1004d | FAIL (zeros) |
| `OutDim` | 0x13c  | 0x10001 → 0x1004d | FAIL (zeros) |

**Critical registers** (ANE HANGs when reverted to add value):
`Cout` (0x138), `GroupConvCfg` (0x14c), `pad3` (0x154), `Cfg` (0x15c), `TaskInfo` (0x160),
`L2Cfg` (0x1e0), `SourceCfg` (0x1e4), `SourceBase` (0x1e8), `SourceChStride` (0x1ec), `SourceRowStride` (0x1f0),
`L2pad0-6` (0x1f4-0x20c), `ResultCfg` (0x210), `ResultBase` (0x214), `ConvResultRowStride` (0x21c),
`PECfg` (0x22c), `BiasScale` (0x230), `PreScale` (0x234), `FinalScale` (0x238),
`SrcDMAConfig` (0x16c), `Srcpad0` (0x170), `SrcRow/Plane/Depth/GroupStride` (0x178-0x184),
`Srcpad2-4` (0x18c-0x194), `SrcFmt` (0x1a4), `Srcpad8` (0x1a8),
`DstRowStride` (0x260), `DstPlaneStride` (0x264), `DstDepthStride` (0x268), `DstFmt` (0x270)

### Why add→relu is different from add→mul

| Aspect | add→mul | add→relu |
|--------|---------|----------|
| BTSP program | **Same** firmware | **Different** firmware (program code bytes changed) |
| Data path | TileDMA→PE→TileDMA | **L2→Conv→TileDMA** |
| PE used? | Yes (bit 2 toggles add↔mul) | **No (PECfg=0 disables PE entirely)** |
| Number of inputs | 2 | 1 |
| Minimum register changes | **2** (PECfg, MACCfg) | **~25** plus BTSP program code |

## 3. Convert and run ane 

```bash
compile https://github.com/eiln/ane/blob/main/bindings/python/dylib/Makefile and the cp libane_python.so to /usr/lib/

uv venv --python=3.11 && source .venv/bin/activate
uv pip install https://github.com/eiln/anecc.git#subdirectory=anecc https://github.com/eiln/ane.git#subdirectory=bindings/python/python
anecc hwx/sum.hwx -o hwx/sum.ane
python run.py ./hwx/sum.ane 
```

## 4. Dump IOCTL and BO

### Dump IOCTL
```bash
sudo bpftrace -e '
tracepoint:syscalls:sys_enter_ioctl /args->cmd == 0xc0186441/ { printf("ANE BO_INIT\n"); }
tracepoint:syscalls:sys_enter_ioctl /args->cmd == 0xc0086442/ { printf("ANE BO_FREE\n"); }
tracepoint:syscalls:sys_enter_ioctl /args->cmd == 0xc0986443/ { printf("ANE SUBMIT\n"); }'

Attaching 3 probes...
ANE BO_INIT
ANE BO_INIT
ANE BO_INIT
ANE BO_INIT
ANE BO_INIT
ANE SUBMIT
ANE BO_FREE
ANE BO_FREE
ANE BO_FREE
ANE BO_FREE
ANE BO_FREE
```

### Dump IOCTL submit

You can use bpftrace to dump ioctl submit content

```bash
sudo bpftrace -e '
struct drm_ane_submit {
    unsigned long long tsk_size;  // 對應 __u64 / uint64
    unsigned int td_count;        // 對應 __u32 / uint32
    unsigned int td_size;
    unsigned int handles[1];      // 先用 1 或具體數字，bpftrace 不支援 ANE_TILE_COUNT 變數
    unsigned int btsp_handle;
    unsigned int pad;
};

tracepoint:syscalls:sys_enter_ioctl 
/args->cmd == 0xc0986443/ 
{
    $s = (struct drm_ane_submit *)args->arg;
    printf("ANE SUBMIT: tsk_size=%llu, td_count=%u, td_size=%u, btsp_handle=%u\n", 
           $s->tsk_size, $s->td_count, $s->td_size, $s->btsp_handle);
}'

ANE SUBMIT: tsk_size=628, td_count=1, td_size=628, btsp_handle=0
```

but i modified the kernel driver directly to print the dump.
https://github.com/allbilly/libane

```bash
ANE NN {
  fd=3
  data=0xaaab63898000
  anec={size=17024 td_size=628 td_count=1 tsk_size=628 krn_size=16384 src_count=2 dst_count=1}
  btsp_chan={map=0xfffece008000 size=16384 handle=5 offset=4295049216}
  chans=[
    00: {map=0xfffece7d0000 size=32768 handle=1 offset=4294967296},
    01: {map=(nil) size=0 handle=0 offset=0},
    02: {map=(nil) size=0 handle=0 offset=0},
    03: {map=(nil) size=0 handle=0 offset=0},
    04: {map=0xfffece7cc000 size=16384 handle=2 offset=4295000064},
    05: {map=0xfffece1dc000 size=16384 handle=3 offset=4295016448},
    06: {map=0xfffece00c000 size=16384 handle=4 offset=4295032832},
    07: {map=(nil) size=0 handle=0 offset=0},
    08: {map=(nil) size=0 handle=0 offset=0},
    09: {map=(nil) size=0 handle=0 offset=0},
    10: {map=(nil) size=0 handle=0 offset=0},
    11: {map=(nil) size=0 handle=0 offset=0},
    12: {map=(nil) size=0 handle=0 offset=0},
    13: {map=(nil) size=0 handle=0 offset=0},
    14: {map=(nil) size=0 handle=0 offset=0},
    15: {map=(nil) size=0 handle=0 offset=0},
    16: {map=(nil) size=0 handle=0 offset=0},
    17: {map=(nil) size=0 handle=0 offset=0},
    18: {map=(nil) size=0 handle=0 offset=0},
    19: {map=(nil) size=0 handle=0 offset=0},
    20: {map=(nil) size=0 handle=0 offset=0},
    21: {map=(nil) size=0 handle=0 offset=0},
    22: {map=(nil) size=0 handle=0 offset=0},
    23: {map=(nil) size=0 handle=0 offset=0},
    24: {map=(nil) size=0 handle=0 offset=0},
    25: {map=(nil) size=0 handle=0 offset=0},
    26: {map=(nil) size=0 handle=0 offset=0},
    27: {map=(nil) size=0 handle=0 offset=0},
    28: {map=(nil) size=0 handle=0 offset=0},
    29: {map=(nil) size=0 handle=0 offset=0},
    30: {map=(nil) size=0 handle=0 offset=0},
    31: {map=(nil) size=0 handle=0 offset=0}
  ]
}
ANE SUBMIT {
  tsk_size=628
  td_count=1
  td_size=628
  btsp_handle=5
  pad=0
  handles=[1, 0, 0, 0, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
CMD_BUF (handle[0]) (size=628):
  0000: 00 00 00 02 00 00 00 00 22 04 00 00 00 00 00 00
  0010: 6a f8 ff 00 00 00 00 00 00 98 00 30 00 00 00 00
  0020: 66 49 02 00 00 00 00 00 00 f8 01 f4 00 00 00 00
  0030: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0040: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0050: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0060: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0070: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0080: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0090: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  00a0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  00b0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  00c0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  00d0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  00e0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  00f0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0100: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0110: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0120: 00 00 00 00 00 00 00 3c 01 00 01 00 01 00 00 00
  0130: 2a 00 00 00 40 00 00 00 40 00 00 00 01 00 01 00
  0140: 01 00 00 00 21 a0 00 50 41 20 00 00 01 00 01 00
  0150: 01 00 00 00 04 00 00 00 00 00 00 00 33 00 00 00
  0160: 00 00 00 00 00 00 00 00 00 38 01 6c 81 38 03 00
  0170: 80 38 03 00 00 00 00 00 40 00 00 00 40 00 00 00
  0180: 00 10 00 00 00 00 00 00 00 00 00 00 40 00 00 00
  0190: 40 00 00 00 00 10 00 00 00 00 00 00 00 00 00 00
  01a0: 00 00 00 00 31 20 00 01 30 20 00 00 00 00 00 00
  01b0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  01c0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  01d0: 00 00 00 00 00 00 00 00 00 00 00 00 00 48 00 44
  01e0: 00 00 00 00 72 01 50 01 00 00 00 00 10 00 00 00
  01f0: 20 04 00 00 00 04 00 00 00 04 00 00 40 04 00 00
  0200: 10 00 00 00 20 04 00 00 00 04 00 00 00 04 00 00
  0210: 7a 01 50 00 60 08 00 00 00 00 00 00 00 00 00 00
  0220: 00 00 00 00 00 00 00 00 00 88 00 0c 00 00 08 00
  0230: 00 00 00 3c 00 00 00 3c 00 00 80 3f 00 c8 00 10
  0240: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0250: 00 00 00 00 00 78 01 18 c1 00 00 04 00 00 00 00
  0260: 40 00 00 00 40 00 00 00 00 10 00 00 00 00 00 00
  0270: 31 20 00 01
(1, 64, 1, 1) float16
[[[[5.]]
    ...
  [[5.]]]] 
```

### Dump BO
```bash
handle[0]=1 → BO for tile/buffer index 0 (where the command/weights live)
handle[4]=2 → BO for output tile (dst 0)
handle[5]=3 → BO for input 0
handle[6]=4 → BO for input 1

python3 /home/asahi/ane-ex/dump.py /tmp/sum_cmd.bin \
  --decode-cmd --cmd-sbs-compact-grouped

python3 /home/asahi/ane-ex/dump.py /tmp/sum_weights.bin --dtype fp16 --count 64


python3 /home/asahi/ane-ex/dump.py /tmp/ane_bo_04_post.bin --dtype fp16 --tile 1,64,1,1,64,64 --count 8
[3. 3. 3. 3. 3. 3. 3. 3.]

python3 /home/asahi/ane-ex/dump.py /tmp/ane_bo_05.bin --dtype fp16 --tile 1,64,1,1,64,64 --count 8
[1. 1. 1. 1. 1. 1. 1. 1.]

python3 /home/asahi/ane-ex/dump.py /tmp/ane_bo_06.bin --dtype fp16 --tile 1,64,1,1,64,64 --count 8
[2. 2. 2. 2. 2. 2. 2. 2.]
```

# Reference
- https://github.com/eiln/linux
- https://github.com/eiln/ane
- https://github.com/eiln/anecc
- https://github.com/freedomtan/coreml_to_ane_hwx
- https://github.com/tinygrad/tinygrad/tree/v0.10.3/extra/accel/ane/