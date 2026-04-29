# CPU Low-Level Programming from Python

This guide shows how to interact with CPU internals from Python in the same bare-metal spirit as `elementwise.py` does for the ANE — raw machine code, register access, performance counters, and instruction encoding — with no frameworks, no compilers, just bytes.

---

## 1. Execute Raw x86 Machine Code (the "CPU Task Descriptor")

Analogous to building `BTSP_BUF` and submitting it via ioctl, you can craft raw x86 opcodes, mmap executable memory, and call it via a function pointer.

```python
import mmap, ctypes

# Raw x86-64 machine code: add two floats via SSE
#   movss  xmm0, [rdi]     F3 0F 10 07
#   addss  xmm0, [rsi]     F3 0F 58 06
#   movss  [rdx], xmm0     F3 0F 11 02
#   ret                     C3
code = bytes([
    0xF3, 0x0F, 0x10, 0x07,  # movss xmm0, [rdi]
    0xF3, 0x0F, 0x58, 0x06,  # addss xmm0, [rsi]
    0xF3, 0x0F, 0x11, 0x02,  # movss [rdx], xmm0
    0xC3,                     # ret
])

buf = mmap.mmap(-1, len(code), mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
buf.write(code)

func = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
    ctypes.addressof(ctypes.c_char.from_buffer(buf)))

a, b = (ctypes.c_float * 1)(3.0), (ctypes.c_float * 1)(2.0)
out = (ctypes.c_float * 1)()
func(a, b, out)
print(out[0])  # 5.0
```

## 2. Read CPU Registers

### MSRs (Model-Specific Registers) — like reading ANE register space

```python
import os, struct

msr_fd = os.open("/dev/cpu/0/msr", os.O_RDONLY)
data = os.pread(msr_fd, 8, 0x10)  # 0x10 = Time-Stamp Counter
tsc = struct.unpack("<Q", data)[0]
os.close(msr_fd)
```

Requires `sudo modprobe msr`. Intel SDM Volume 4 lists all MSR addresses.

### CPUID — query CPU features/capabilities

```python
import subprocess
info = subprocess.check_output("cpuid -1", shell=True, text=True)
```

## 3. Performance Counters — like ANE `debug_log_events` / `exe_cycles`

```python
import ctypes, os

libc = ctypes.CDLL("libc.so.6")
PERF_TYPE_HARDWARE = 0
PERF_COUNT_HW_CPU_CYCLES = 0

attr = (ctypes.c_ulong * 10)(
    (PERF_TYPE_HARDWARE | (PERF_COUNT_HW_CPU_CYCLES << 8)), 0, 0, 0, 0, 0, 0, 0, 0, 0
)
fd = libc.syscall(298, ctypes.byref(attr), 0, -1, -1, 0)  # __NR_perf_event_open
data = os.read(fd, 24)
cycles = struct.unpack("<Q", data[8:16])[0]
```

Available hardware events (analogous to ANE event mask bits):
- `PERF_COUNT_HW_CPU_CYCLES` — like ANE `exe_cycles`
- `PERF_COUNT_HW_INSTRUCTIONS` — retired instructions
- `PERF_COUNT_HW_CACHE_MISSES` — cache misses (L1, L2, LLC)
- `PERF_COUNT_HW_BRANCH_MISSES` — branch mispredictions

## 4. Read RDTSC (Cycle Counter) from Raw Opcodes

```python
# RDTSC instruction: 0F 31  (returns EDX:EAX)
code = bytes([
    0x0F, 0x31,              # rdtsc
    0x48, 0xC1, 0xE2, 0x20,  # shl rdx, 32
    0x48, 0x09, 0xD0,        # or  rax, rdx
    0xC3,                     # ret
])
buf = mmap.mmap(-1, len(code), mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
buf.write(code)
rdtsc = ctypes.CFUNCTYPE(ctypes.c_uint64)(ctypes.addressof(ctypes.c_char.from_buffer(buf)))
start = rdtsc()
# ... do work ...
end = rdtsc()
print(f"took {end - start} cycles")
```

## 5. Encode x86 Instructions at Runtime (like `build_seg`/`pack_reg`)

Using [keystone-engine](https://github.com/keystone-engine/keystone):

```python
from keystone import *

ks = Ks(KS_ARCH_X86, KS_MODE_64)
encoding, count = ks.asm("addss xmm0, xmm1; movss [rdx], xmm0")
code = bytes(encoding)  # [0xF3, 0x0F, 0x58, 0xC1, 0xF3, 0x0F, 0x11, 0x02]
```

This is the direct analog of `pack_reg(BTSP_BUF, reg.PECfg, value)` — you build instruction bytes at runtime from textual assembly.

## 6. Set SSE Control Register (MXCSR) — like PE/NE config

```python
# LDMXCSR instruction: 0F AE /2
# Set rounding mode, denormals-are-zero, etc.
# MXCSR value: bits [15:13] = rounding (00=nearest, 01=down, 10=up, 11=truncate)
DAZ = 1 << 6   # denormals-are-zero
FTZ = 1 << 15  # flush-to-zero
mxcsr = DAZ | FTZ  # or 0x1F80 for default

code = bytes([0x0F, 0xAE, 0x17]) + struct.pack("<I", mxcsr)  # ldmxcsr [rdi]
buf = mmap.mmap(...)
```

---

## ANE → CPU Concept Map

| ANE (`elementwise.py`) | CPU Analog | Python Approach |
|---|---|---|
| `reg` class (register offsets) | MSR addresses, CPUID leafs | `struct.pack` + `/dev/cpu/*/msr` |
| `build_seg` / `pack_reg` | x86 instruction encoding | `keystone-engine` |
| `BTSP_BUF` (task desc bytes) | x86 machine code bytes | `bytes([...])` or `ks.asm(...)` |
| `mmap(allocate_buffer)` | `mmap(PROT_EXEC)` | same call |
| `submit_task` ioctl | call function pointer | `ctypes.CFUNCTYPE` |
| `exe_cycles` | RDTSC | raw `\x0F\x31` opcodes |
| `debug_log_events` | `perf_event_open` | libc syscall wrapper |
| `ChCfg`, `ConvCfg` | MXCSR (SSE control reg) | `ldmxcsr` instruction bytes |
| `PECfg` op_mode | SSE opcode selection | `addss` vs `mulss` vs `maxss` |
| `Src/Dst DMA streams` | memory operands (rdi/rsi/rdx) | direct pointer args to asm |
| `fp16` format | `VADDPH` (AVX-512 FP16) or f16→f32 conversion | manual convert + `addss` |

---

## Full Example: CPU `elementwise_add.py`

```python
import mmap, ctypes, struct
from keystone import *

# Assemble: add two float vectors with SSE
# void add_f32(float *a, float *b, float *out, int n)
asm = """
    xor rcx, rcx
loop:
    movss xmm0, [rdi + rcx*4]
    addss xmm0, [rsi + rcx*4]
    movss [rdx + rcx*4], xmm0
    inc rcx
    cmp rcx, r8d
    jl loop
    ret
"""
ks = Ks(KS_ARCH_X86, KS_MODE_64)
code, count = ks.asm(asm)

buf = mmap.mmap(-1, len(code), mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
buf.write(bytes(code))

add_f32 = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)(
    ctypes.addressof(ctypes.c_char.from_buffer(buf)))

N = 1024
a = (ctypes.c_float * N)(*range(N))
b = (ctypes.c_float * N)(*range(N))
out = (ctypes.c_float * N)()

add_f32(a, b, out, N)
print(out[0], out[1], out[999])  # 0.0, 2.0, 1998.0
```

---

## Learning Path

1. **Read the Intel SDM** (Software Developer's Manual) — Volumes 1-3 are free. Focus on Vol. 2A/2B (instruction set reference) and Vol. 3 (MSRs, system programming).
2. **Explore `/dev/cpu/*/msr`** — read TSC, APERF, MPERF, thermals.
3. **Use `perf stat` on trivial programs** to understand what counters mean.
4. **Write raw bytes** for simple ops (add, mov, ret), then wrap them in keystone.
5. **Disassemble C compiler output** (`gcc -S -O2 -masm=intel`) to see what real code looks like.
6. **Build a minimal JIT** — the pattern above (`ks.asm` → `mmap(PROT_EXEC)` → `CFUNCTYPE`) is literally how LuaJIT and PyPy work, minus the optimization passes.
