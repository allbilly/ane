#!/usr/bin/env python3
"""
Fixed multi-task atan2 runner using hwx2py-cleaned CMD_BUF + anecc buffer layout.
tsk_size=0x1774, td_count=8, td_size=0x274 (from anecc -p)
7 src buffers: src0/src1 = user inputs, src2-src6 = scratch
"""
from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np
import sys, re

# Load CMD_BUF and BTSP_BUF hex from hwx2py-generated script
_hex_data = open(os.path.join(os.path.dirname(__file__), 'atan2_from_hwx.py')).read()
CMD_BUF = bytes.fromhex(re.search(r"CMD_BUF = bytes\.fromhex\('(.+?)'\)", _hex_data).group(1))
BTSP_BUF = bytes.fromhex(re.search(r"BTSP_BUF = bytes\.fromhex\('(.+?)'\)", _hex_data).group(1))

BUF_SIZE = 0x4000        # cmd/btsp
ITM_SIZE = 0x1000000     # 16MB scratch buffer
SRC_SIZE = 0x400000      # 4MB per src buffer
DST_SIZE = 0x400000      # 4MB output
N_FP16 = SRC_SIZE // 2   # 2097152 fp16 per src
ANE_TILE_COUNT = 0x20

fd = os.open("/dev/accel/accel0", os.O_RDWR)

class drm_ane_bo_init(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32), ("size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]

class drm_ane_submit(ctypes.Structure):
    _fields_ = [("tsk_size", ctypes.c_uint64), ("td_count", ctypes.c_uint32), ("td_size", ctypes.c_uint32), ("handles", ctypes.c_uint32 * ANE_TILE_COUNT), ("btsp_handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]

def _IOWR(nr, size):
    return (3 << 30) | (0x64 << 8) | (size << 16) | nr

DRM_IOCTL_ANE_BO_INIT = _IOWR(0x41, ctypes.sizeof(drm_ane_bo_init))
DRM_IOCTL_ANE_SUBMIT = _IOWR(0x43, ctypes.sizeof(drm_ane_submit))

def bo_alloc(fd, size):
    bo = drm_ane_bo_init(handle=0, pad=0, size=size, offset=0)
    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo)
    buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    return bo.handle, buf

def submit(fd, tsk_size, td_count, td_size, handles, btsp_handle):
    s = drm_ane_submit(tsk_size=tsk_size, td_count=td_count, td_size=td_size, btsp_handle=btsp_handle, pad=0)
    for i in range(ANE_TILE_COUNT):
        s.handles[i] = handles[i] if i < len(handles) else 0
    return ioctl(fd, DRM_IOCTL_ANE_SUBMIT, s)

# Allocate buffers per anecc tile layout
cmd_h, cmd_buf = bo_alloc(fd, BUF_SIZE)
cmd_buf.write(CMD_BUF[:BUF_SIZE])

itm_h, itm_buf = bo_alloc(fd, ITM_SIZE)   # tile[3] = scratch
dst_h, dst_buf = bo_alloc(fd, DST_SIZE)    # tile[4] = output

src0_h, src0_buf = bo_alloc(fd, SRC_SIZE)  # tile[5] = input x
src1_h, src1_buf = bo_alloc(fd, SRC_SIZE)  # tile[6] = input y

# tiles[7..11] = scratch buffers (src2-src6)
scratch_h = []
for _ in range(5):
    h, b = bo_alloc(fd, SRC_SIZE)
    b.write(b'\x00' * SRC_SIZE)
    b.close()
    scratch_h.append(h)

btsp_h, btsp_buf = bo_alloc(fd, BUF_SIZE)
btsp_buf.write(BTSP_BUF[:BUF_SIZE])

# Initialize inputs: x=3.0, y=2.0
src0 = np.zeros(N_FP16, dtype=np.float16)
src0[:N_FP16:1] = np.float16(3.0)  # simple contiguous fill
src0_buf.write(src0.tobytes())
src0_buf.close()

src1 = np.zeros(N_FP16, dtype=np.float16)
src1[:N_FP16:1] = np.float16(2.0)
src1_buf.write(src1.tobytes())
src1_buf.close()

# Handle layout: [cmd, 0, 0, itm, dst, src0, src1, src2, src3, src4, src5, src6]
handles = [cmd_h, 0, 0, itm_h, dst_h, src0_h, src1_h] + scratch_h + [0] * 14

ret = submit(fd, 0x1774, 8, 0x274, handles, btsp_h)
print(f"SUBMIT ret={ret}")

out = np.frombuffer(dst_buf, dtype=np.float16, count=1024*2048).reshape(1024, 2048)
print(f"output shape: {out.shape}, dtype={out.dtype}")
print(f"first 8 values: {out.flatten()[:8]}")
print(f"sum: {out.sum()}")
print(f"min: {out.min()}, max: {out.max()}")
print(f"non-zero count: {np.count_nonzero(out)}")
os.close(fd)
