#!/usr/bin/env python3
from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np

ANE_TILE_COUNT = 0x20

class drm_ane_bo_init(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32),
                ("size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]

class drm_ane_submit(ctypes.Structure):
    _fields_ = [("tsk_size", ctypes.c_uint64), ("td_count", ctypes.c_uint32),
                ("td_size", ctypes.c_uint32), ("handles", ctypes.c_uint32 * ANE_TILE_COUNT),
                ("btsp_handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]

def _IOWR(nr, size):
    return (3 << 30) | (0x64 << 8) | (size << 16) | nr

DRM_IOCTL_ANE_BO_INIT = _IOWR(0x41, ctypes.sizeof(drm_ane_bo_init))
DRM_IOCTL_ANE_SUBMIT = _IOWR(0x43, ctypes.sizeof(drm_ane_submit))

# Load from compiled .ane (anecc handles KDMA kernel data properly)
with open('hwx/conv.ane', 'rb') as f:
    ane = f.read()
hdr = struct.unpack_from('<8I', ane, 0)
td_size = hdr[2]; krn_size = hdr[6]
ane_data = ane[0x1000:0x1000 + td_size]
kernel = ane[0x1000 + td_size:0x1000 + td_size + krn_size]

CMD_BUF = bytearray(ane_data + kernel)
CMD_BUF += b'\x00' * (32768 - len(CMD_BUF))
BTSP_BUF = bytearray(CMD_BUF[:0x4000])
BTSP_BUF[2] = 0x40

BUF_SIZE = 16384
STRIDE = 32
C = 3

def allocate_buffer(fd, size):
    bo = drm_ane_bo_init(handle=0, pad=0, size=size, offset=0)
    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo)
    buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    return bo.handle, buf

def submit_task(fd, tsk_size, td_count, td_size, handles, btsp_handle):
    req = drm_ane_submit(
        tsk_size=tsk_size, td_count=td_count, td_size=td_size,
        btsp_handle=btsp_handle, pad=0,
    )
    for i in range(ANE_TILE_COUNT):
        req.handles[i] = handles[i] if i < len(handles) else 0
    return ioctl(fd, DRM_IOCTL_ANE_SUBMIT, req)

fd = os.open("/dev/accel/accel0", os.O_RDWR)
cmd_handle, cmd_map = allocate_buffer(fd, 32768)
cmd_map.write(bytes(CMD_BUF)); cmd_map.close()
out_handle, out_map = allocate_buffer(fd, BUF_SIZE)
src1_handle, src1_map = allocate_buffer(fd, BUF_SIZE)
src1 = np.zeros(BUF_SIZE // 2, dtype=np.float16)
# Try different inputs here:
vals = [np.float16(1.0), np.float16(2.0), np.float16(3.0)]
for i in range(C):
    src1[i * STRIDE] = vals[i]
src1_map.write(src1.tobytes()); src1_map.close()
if False:
    src2_handle, src2_map = allocate_buffer(fd, BUF_SIZE)
    src2 = np.zeros(BUF_SIZE // 2, dtype=np.float16)
    src2[:C * STRIDE:STRIDE] = np.float16(2.0)
    src2_map.write(src2.tobytes()); src2_map.close()
btsp_handle, btsp_map = allocate_buffer(fd, BUF_SIZE)
btsp_map.write(bytes(BTSP_BUF)); btsp_map.close()

handles = [cmd_handle, 0, 0, 0, out_handle, src1_handle, src2_handle if False else 0] + [0] * 25
ret = submit_task(
    fd=fd,
    tsk_size=0x274,
    td_count=1,
    td_size=0x274,
    handles=handles,
    btsp_handle=btsp_handle,
)
print(f"submit returned: {ret}")
output_all = np.frombuffer(out_map, dtype=np.float16).copy(); out_map.close()
# Check if ANY non-zero values exist in output
non_zero = np.where(output_all != 0)[0]
print(f"Total output values: {len(output_all)}, non-zero count: {len(non_zero)}")
if len(non_zero) > 0:
    print(f"Non-zero indices: {non_zero[:20]}")
    print(f"Non-zero values: {output_all[non_zero[:20]]}")
# Print first 64 as before
print("output[:64] =", output_all[:64])
os.close(fd)
