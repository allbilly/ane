from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np

ANE_TILE_COUNT = 0x20

class drm_ane_bo_init(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32), ("size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]
class drm_ane_submit(ctypes.Structure):
    _fields_ = [("tsk_size", ctypes.c_uint64), ("td_count", ctypes.c_uint32), ("td_size", ctypes.c_uint32), ("handles", ctypes.c_uint32 * ANE_TILE_COUNT), ("btsp_handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]

def _IOWR(nr, size): return (3 << 30) | (0x64 << 8) | (size << 16) | nr
DRM_IOCTL_ANE_BO_INIT = _IOWR(0x41, ctypes.sizeof(drm_ane_bo_init))
DRM_IOCTL_ANE_SUBMIT = _IOWR(0x43, ctypes.sizeof(drm_ane_submit))

def allocate_buffer(fd, size):
    bo = drm_ane_bo_init(handle=0, pad=0, size=size, offset=0)
    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo)
    buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    return bo.handle, buf

def submit_task(fd, tsk_size, td_count, td_size, handles, btsp_handle):
    req = drm_ane_submit(tsk_size=tsk_size, td_count=td_count, td_size=td_size, btsp_handle=btsp_handle, pad=0)
    for i in range(ANE_TILE_COUNT):
        req.handles[i] = handles[i] if i < len(handles) else 0
    return ioctl(fd, DRM_IOCTL_ANE_SUBMIT, req)

class reg:
    InDim,pad0,ChCfg,Cin,Cout=0x128,0x12c,0x130,0x134,0x138; OutDim,pad1,ConvCfg,pad2=0x13c,0x140,0x144,0x148
    GroupConvCfg,TileCfg,pad3,pad4,Cfg=0x14c,0x150,0x154,0x158,0x15c; TaskInfo,DPE=0x160,0x164
    L2Cfg,SourceCfg,SourceBase=0x1e0,0x1e4,0x1e8; SourceChStride,SourceRowStride=0x1ec,0x1f0
    L2pad0,L2pad1,L2pad2=0x1f4,0x1f8,0x1fc; L2pad3,L2pad4,L2pad5,L2pad6=0x200,0x204,0x208,0x20c
    ResultCfg,ResultBase=0x210,0x214; ConvResultChStride,ConvResultRowStride=0x218,0x21c
    PECfg,BiasScale,PreScale,FinalScale=0x22c,0x230,0x234,0x238
    KernelCfg,MACCfg,MatrixVecBias,AccBias,PostScale=0x240,0x244,0x248,0x24c,0x250
    SrcDMAConfig,Srcpad0,SrcBaseAddr=0x16c,0x170,0x174; SrcRowStride,SrcPlaneStride,SrcDepthStride=0x178,0x17c,0x180
    SrcGroupStride,Srcpad2,Srcpad3,Srcpad4=0x184,0x18c,0x190,0x194; SrcFmt,Srcpad8=0x1a4,0x1a8
    DstDMAConfig,DstBaseAddr,DstRowStride=0x258,0x25c,0x260; DstPlaneStride,DstDepthStride,DstGroupStride,DstFmt=0x264,0x268,0x26c,0x270

commands = {
    reg.InDim: 0x00010001,
    reg.pad0: 0x00000001,
    reg.ChCfg: 0x00000022,
    reg.Cin: 0x00000200,
    reg.Cout: 0x00000200,
    reg.OutDim: 0x00010001,
    reg.pad1: 0x00000001,
    reg.ConvCfg: 0x5000b421,
    reg.pad2: 0x00002041,
    reg.GroupConvCfg: 0x00010001,
    reg.TileCfg: 0x00000001,
    reg.pad3: 0x00000000,
    reg.pad4: 0x00000000,
    reg.Cfg: 0x00244405,
    reg.TaskInfo: 0x00100000,
    reg.DPE: 0x00000000,
    reg.L2Cfg: 0x00000000,
    reg.SourceCfg: 0x00500172,
    reg.SourceBase: 0x00000000,
    reg.SourceChStride: 0x00000010,
    reg.SourceRowStride: 0x00002030,
    reg.L2pad0: 0x00002000,
    reg.L2pad1: 0x00002000,
    reg.L2pad2: 0x00000000,
    reg.L2pad3: 0x00000000,
    reg.L2pad4: 0x00000000,
    reg.L2pad5: 0x00000000,
    reg.L2pad6: 0x00000000,
    reg.ResultCfg: 0x00500172,
    reg.ResultBase: 0x00002030,
    reg.ConvResultChStride: 0x00000010,
    reg.ConvResultRowStride: 0x00002020,
    reg.PECfg: 0x00000000,
    reg.BiasScale: 0x00000000,
    reg.PreScale: 0x00000000,
    reg.FinalScale: 0x00000000,
    reg.KernelCfg: 0x00000082,
    reg.MACCfg: 0x00101c00,
    reg.MatrixVecBias: 0x00000000,
    reg.AccBias: 0x00000000,
    reg.PostScale: 0x00003c00,
    reg.SrcDMAConfig: 0x00033881,
    reg.Srcpad0: 0x00008880,
    reg.SrcBaseAddr: 0x00000000,
    reg.SrcRowStride: 0x00000040,
    reg.SrcPlaneStride: 0x00000040,
    reg.SrcDepthStride: 0x00008000,
    reg.SrcGroupStride: 0x00000000,
    reg.Srcpad2: 0x00000000,
    reg.Srcpad3: 0x00000000,
    reg.Srcpad4: 0x00000000,
    reg.SrcFmt: 0x01002031,
    reg.Srcpad8: 0x00000000,
    reg.DstDMAConfig: 0x000000c1,
    reg.DstBaseAddr: 0x00000000,
    reg.DstRowStride: 0x00000040,
    reg.DstPlaneStride: 0x00000040,
    reg.DstDepthStride: 0x00008000,
    reg.DstGroupStride: 0x00000000,
    reg.DstFmt: 0x01302031,
}

ane_data = bytes.fromhex(
    '000000020000000022040000000000006af8ff00000000000098003000000000'
    '254002052100000000f801f44000000000000000810000008100000081000000'
    '8100000081000000810000008100000081000000810000008100000081000000'
    '8100000081000000810000008100000081000000000000000080000000000100'
    '0080010000000200008002000000030000800300000004000080040000000500'
    '0080050000000600008006000000070000800700008000000080000000800000'
    '0080000000800000008000000080000000800000008000000080000000800000'
    '0080000000800000008000000080000000800000800000008000000080000000'
    '8000000000000000000000000000000000000000000000000000000000000000'
    '000000000000003c010001000100000022000000000200000002000001000100'
    '0100000021b40050412000000100010001000000000000000000000005442400'
    '00001000000000000038016c8138030080880000000000004000000040000000'
    '0080000000000000000000000000000000000000000000000000000000000000'
    '0000000031200001000000000001000000000000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000480044'
    '0000000072015000000000001000000030200000002000000020000000000000'
    '0000000000000000000000000000000072015000302000001000000020200000'
    '00200000002000000088000c0000000000000000000000000000000000c80010'
    '82000000001c10000000000000000000003c000000780118c100000000000000'
    '4000000040000000008000000000000031203001'
)
# GEMM kernel is 524288 bytes of zeros (tinygrad model has uninitialized weights)
# Inject 0.5 weights at runtime for non-zero output
kernel = b'\x00' * 524288

CMD_BUF = bytearray(ane_data + kernel)
CMD_BUF += b'\x00' * (32768 - len(CMD_BUF))
BTSP_BUF = bytearray(CMD_BUF[:0x4000])
BTSP_BUF[2] = 0x40

for offset, value in commands.items():
    struct.pack_into('<I', CMD_BUF, offset, value)
    struct.pack_into('<I', BTSP_BUF, offset, value)

# Inject non-zero weights
nz_off = len(ane_data)
w = np.full(262144, np.float16(0.5), dtype=np.float16)
CMD_BUF[nz_off:nz_off + len(w.tobytes())] = w.tobytes()

BUF_SIZE = 0x4000
C = 512
STRIDE = 32

fd = os.open("/dev/accel/accel0", os.O_RDWR)
cmd_handle, cmd_map = allocate_buffer(fd, len(CMD_BUF))
cmd_map.write(bytes(CMD_BUF)); cmd_map.close()
out_handle, out_map = allocate_buffer(fd, BUF_SIZE)
src1_handle, src1_map = allocate_buffer(fd, BUF_SIZE)
src1 = np.zeros(BUF_SIZE // 2, dtype=np.float16)
src1[:C * STRIDE:STRIDE] = np.float16(1.0)
src1_map.write(src1.tobytes()); src1_map.close()
btsp_handle, btsp_map = allocate_buffer(fd, 0x4000)
btsp_map.write(bytes(BTSP_BUF)); btsp_map.close()

handles = [cmd_handle, 0, 0, 0, out_handle, src1_handle, 0] + [0] * 25
ret = submit_task(fd, 0x274, 1, 0x274, handles, btsp_handle)
print(f"submit returned: {ret}")
out = np.frombuffer(out_map, dtype=np.float16, count=64).copy(); out_map.close()
print(f"output[:64] = {out}")
os.close(fd)
