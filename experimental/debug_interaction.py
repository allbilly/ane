"""Debug: find which combination of UNNEEDED registers causes zero output."""
import os, sys, struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from experimental.ane_helpers import (
    allocate_buffer, submit_task, pack_reg,
)

# Build BTSP_BUF from the builder functions (avoids importing module-level side effects)
from experimental.test_min_regs import make_relu_buf, make_sigmoid_buf

RELU_ORIG_BUF = make_relu_buf()
RELU_EXPT_BUF = make_relu_buf()

# All UNNEEDED register offsets for relu (from expt3)
UNNEEDED_RELU = {
    0x15c: "Cfg",
    0x160: "TaskInfo",
    0x170: "Srcpad0",
    0x178: "SrcRowStride",
    0x17c: "SrcPlaneStride",
    0x180: "SrcDepthStride",
    0x1ac: "SrcPadStream",
    0x1e4: "SourceCfg",
    0x1f0: "SourceRowStride",
    0x1f4: "L2pad0",
    0x1f8: "L2pad1",
    0x214: "ResultBase",
    0x240: "KernelCfg",
    0x260: "DstRowStride",
    0x264: "DstPlaneStride",
    0x268: "DstDepthStride",
}

# Original values for UNNEEDED registers (from original BTSP_BUF)
ORIG_VALUES = {off: struct.unpack_from('<I', RELU_ORIG_BUF, off)[0] for off in UNNEEDED_RELU}

# Zero out UNNEEDED registers in the EXPT buffer
for off in UNNEEDED_RELU:
    pack_reg(RELU_EXPT_BUF, off, 0)

# All UNNEEDED register offsets for relu (from expt3)
UNNEEDED_RELU = {
    0x15c: "Cfg",
    0x160: "TaskInfo",
    0x170: "Srcpad0",
    0x178: "SrcRowStride",
    0x17c: "SrcPlaneStride",
    0x180: "SrcDepthStride",
    0x1ac: "SrcPadStream",
    0x1e4: "SourceCfg",
    0x1f0: "SourceRowStride",
    0x1f4: "L2pad0",
    0x1f8: "L2pad1",
    0x214: "ResultBase",
    0x240: "KernelCfg",
    0x260: "DstRowStride",
    0x264: "DstPlaneStride",
    0x268: "DstDepthStride",
}

# Original values for UNNEEDED registers (from original BTSP_BUF)
ORIG_VALUES = {off: struct.unpack_from('<I', RELU_ORIG_BUF, off)[0] for off in UNNEEDED_RELU}

def run_relu(buf, label):
    """Run relu and check if output is correct."""
    input_a = np.tile(np.array([-3.0, 5.0, -3.0, 5.0], dtype=np.float16), 2048)
    fd = os.open("/dev/accel/accel0", os.O_RDWR)
    try:
        out_handle, out_map = allocate_buffer(fd, 0x4000)
        src1_handle, src1_map = allocate_buffer(fd, 0x4000)
        btsp_handle, btsp_map = allocate_buffer(fd, 0x4000)
        src1_map.write(input_a.tobytes())
        btsp_map.write(bytes(buf))
        submit_task(fd, 0x274, 1, 0x274,
                    [btsp_handle, 0, 0, 0, out_handle, src1_handle, 0] + [0]*25,
                    btsp_handle)
        output = np.frombuffer(out_map, dtype=np.float16, count=4).copy()
    except OSError as e:
        print(f"  {label}: HANG ({e})")
        return None
    finally:
        os.close(fd)
    expected = np.maximum(0, input_a[:4])
    ok = all(abs(output[i] - expected[i]) < 0.01 for i in range(4))
    print(f"  {label}: {list(output)} {'OK' if ok else 'FAIL'} (expected {list(expected)})")
    return ok

# Test 1: original BTSP_BUF (all registers present)
print("=== Test 1: Original relu BTSP_BUF ===")
run_relu(RELU_ORIG_BUF, "ORIG")

# Test 2: expt BTSP_BUF (all UNNEEDED commented out)
print("\n=== Test 2: Expt BTSP_BUF (all UNNEEDED zeroed) ===")
run_relu(RELU_EXPT_BUF, "EXPT")

# Test 3: Enable one group at a time, find the culprit
# Grouping: DstStrides, SrcStrides, L2Strides, ConfigRegisters, General
groups = {
    "DstStrides": [0x260, 0x264, 0x268],
    "SrcStrides": [0x178, 0x17c, 0x180],
    "L2Strides": [0x1f0, 0x1f4, 0x1f8],
    "ConfigRegs": [0x15c, 0x1e4, 0x170, 0x1ac, 0x214],
    "TaskKernel": [0x160, 0x240],
}

for group_name, offsets in groups.items():
    buf = bytearray(RELU_EXPT_BUF)
    for off in offsets:
        pack_reg(buf, off, ORIG_VALUES[off])
    print(f"\n=== Test 3: +{group_name} restored ===")
    run_relu(buf, f"+{group_name}")

# Test 4: Enable ALL groups = should be identical to original
buf = bytearray(RELU_EXPT_BUF)
for off in UNNEEDED_RELU:
    pack_reg(buf, off, ORIG_VALUES[off])
print(f"\n=== Test 4: ALL UNNEEDED restored ===")
run_relu(buf, "ALL_RESTORED")

# Test 5: Binary search — test subgroups independently
# First find which SINGLE group fixes the problem
print(f"\n=== Test 5: Individual groups (from original bas, enable just one) ===")
for group_name, offsets in groups.items():
    buf = bytearray(RELU_ORIG_BUF)  # start from original
    # Zero all OTHER UNNEEDED registers
    other_offsets = [off for off in UNNEEDED_RELU if off not in offsets]
    for off in other_offsets:
        pack_reg(buf, off, 0)
    run_relu(buf, f"ZERO-all-except-{group_name}")
