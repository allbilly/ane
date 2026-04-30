# python uops emulator for Apple ANE
# works to test the ANE tensor/PE cores
# this is the (living) definition of uops for ANE
from typing import Any, TYPE_CHECKING, cast
import pickle, base64, itertools, time, struct, sys, functools, array, ctypes, mmap, os, math, numpy as np
from tinygrad.dtype import DType, dtypes, ImageDType, PtrDType, truncate
from tinygrad.helpers import all_same, getenv, flatten, get_single_element, EMULATE, mv_address, to_mv, DEBUG
from tinygrad.device import Compiled, Compiler, Allocator, CompilerSet, CompilerPair, BufferSpec
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import exec_alu, python_alu, Ops, UOp, GroupOp, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad.runtime.support.hcq import FileIOInterface
from tinygrad.runtime.ops_python import storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar, _load, load, _store, generic_wmma_helper
from experimental.ane_autogen import *

HALF_ONE = 0x3C00

class ANEBuffer:
  def __init__(self, va_addr, size, handle, offset):
    self.va_addr = va_addr
    self.size = size
    self.handle = handle
    self.offset = offset

class ANEProgram:
  def __init__(self, dev:'ANEDevice', name:str, lib:bytes):
    self.uops: list[tuple[Ops, DType, list[int], Any]] = pickle.loads(lib)
    self.device = dev
    self.fd = dev.fd_ctl
    self.pe_op_modes = {Ops.ADD: 0}
    self.hardware_ops = {Ops.ADD}

    self._btsp_template:bytearray|None = None
    self._cached_num_elements:int = -1

  def _build_btsp(self, num_elements:int) -> bytearray:
    if self._btsp_template is not None and self._cached_num_elements == num_elements:
      return self._btsp_template

    STRIDE = 32
    CHANNELS = num_elements
    ST = STRIDE * 2
    total_bytes = CHANNELS * STRIDE * 2
    depth_stride = max(0x1000, CHANNELS * STRIDE * 2)

    buf = bytearray(0x4000)

    # ── Task Descriptor ──
    pack_reg(buf, R.W0, (0 << 0) | (0x40 << 16) | (1 << 25))
    pack_reg(buf, R.W1, 0)
    pack_reg(buf, R.W2, 1058)
    pack_reg(buf, R.W3, 0)
    pack_reg(buf, R.W4, 0xFFF86A)
    pack_reg(buf, R.W5, 0)
    pack_reg(buf, R.W6, (38 << 10) | (3 << 28))
    pack_reg(buf, R.W7, 0)
    pack_reg(buf, R.W8,
      (6) | (1 << 5) | (5 << 6) | (1 << 11) | (4 << 12) | (1 << 17))
    pack_reg(buf, R.W9, 0)
    pack_reg(buf, R.KernelDMA, stream_header(0x1F800, 62))

    # ── Common + TileDMA Src ──
    pack_reg(buf, R.CommonStream, stream_header(0x00000, 16))
    pack_reg(buf, R.InDim, (1 << 16) | 1)
    pack_reg(buf, R.OutDim, (1 << 16) | 1)
    pack_reg(buf, R.ChCfg, (2) | (2 << 2) | (2 << 4))
    pack_reg(buf, R.Cin, CHANNELS)
    pack_reg(buf, R.Cout, CHANNELS)
    pack_reg(buf, R.pad0, 1)
    pack_reg(buf, R.pad1, 1)
    pack_reg(buf, R.pad2, 0x2041)
    pack_reg(buf, R.pad3, 4)
    pack_reg(buf, R.pad4, 0)
    pack_reg(buf, R.ConvCfg,
      (1) | (1 << 5) | (1 << 13) | (1 << 15) | (1 << 28) | (1 << 30))
    pack_reg(buf, R.GroupConvCfg, (1) | (1 << 16))
    pack_reg(buf, R.TileCfg, 1)
    pack_reg(buf, R.Cfg, (3 << 0) | (0 << 2) | (6 << 3))
    pack_reg(buf, R.TaskInfo, 0)
    pack_reg(buf, R.DPE, 0)

    # TileDMA Src
    pack_reg(buf, R.SrcStream, stream_header(0x13800, 28))
    pack_reg(buf, R.SrcDMAConfig, (1) | (8 << 4) | (8 << 8) | (3 << 12) | (3 << 16))
    pack_reg(buf, R.Srcpad0, 0x33880)
    pack_reg(buf, R.SrcBaseAddr, 0)
    pack_reg(buf, R.SrcRowStride, ST)
    pack_reg(buf, R.SrcPlaneStride, ST)
    pack_reg(buf, R.SrcDepthStride, depth_stride)
    pack_reg(buf, R.SrcGroupStride, 0)
    pack_reg(buf, R.Srcpad1, 0)
    pack_reg(buf, R.Srcpad2, ST)
    pack_reg(buf, R.Srcpad3, ST)
    pack_reg(buf, R.Srcpad4, depth_stride)
    pack_reg(buf, R.Srcpad5, 0)
    pack_reg(buf, R.Srcpad6, 0)
    pack_reg(buf, R.Srcpad7, 0)
    pack_reg(buf, R.Srcpad8, 0x2030)
    pack_reg(buf, R.SrcFmt, (1) | (3 << 4) | (2 << 12) | (1 << 24))

    # ── L2 ──
    pack_reg(buf, R.L2Stream, stream_header(0x04800, 18))
    pack_reg(buf, R.L2Cfg, 0)
    pack_reg(buf, R.SourceCfg,
      (2) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 8) | (1 << 20) | (1 << 22))
    pack_reg(buf, R.SourceBase, 0)
    pack_reg(buf, R.SourceChannelStride, 0x10)
    pack_reg(buf, R.SourceRowStride, 0x420)
    pack_reg(buf, R.L2pad0, 0x400)
    pack_reg(buf, R.L2pad1, 0x400)
    pack_reg(buf, R.L2pad2, 0x440)
    pack_reg(buf, R.L2pad3, 0x10)
    pack_reg(buf, R.L2pad4, 0x420)
    pack_reg(buf, R.L2pad5, 0x400)
    pack_reg(buf, R.L2pad6, 0x400)
    pack_reg(buf, R.ResultCfg,
      (2) | (2 << 2) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 8) | (1 << 20) | (1 << 22))
    pack_reg(buf, R.ResultBase, 0x860)

    # ── PE + NE ──
    pack_reg(buf, R.PEStream, stream_header(0x08800, 4))
    pack_reg(buf, R.PECfg, (2 << 18))
    pack_reg(buf, R.BiasScale, (HALF_ONE << 16))
    pack_reg(buf, R.PreScale, (HALF_ONE << 16))
    pack_reg(buf, R.FinalScale, 0x3f800000)

    pack_reg(buf, R.NEStream, stream_header(0x0C800, 5))
    pack_reg(buf, R.KernelCfg, 0)
    pack_reg(buf, R.MACCfg, 0)
    pack_reg(buf, R.MatrixVectorBias, 0)
    pack_reg(buf, R.AccBias, 0)
    pack_reg(buf, R.PostScale, 0)

    # ── TileDMA Dst ──
    pack_reg(buf, R.DstStream, stream_header(0x17800, 7))
    pack_reg(buf, R.DstDMAConfig, (1) | (12 << 4) | (1 << 26))
    pack_reg(buf, R.DstBaseAddr, 0)
    pack_reg(buf, R.DstRowStride, ST)
    pack_reg(buf, R.DstPlaneStride, ST)
    pack_reg(buf, R.DstDepthStride, depth_stride)
    pack_reg(buf, R.DstGroupStride, 0)
    pack_reg(buf, R.DstFmt, (1) | (3 << 4) | (2 << 12) | (1 << 24))

    self._btsp_template = buf
    self._cached_num_elements = num_elements
    return buf

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    void_ops = {Ops.END, Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.GROUP, Ops.STORE}
    loop_ends: dict[int, int] = {srcs[1]:i for i, (uop, _, srcs, _) in enumerate(self.uops) if uop == Ops.END}
    has_control_flow = any(op in (Ops.RANGE, Ops.IF, Ops.ENDIF) for op,_,_,_ in self.uops)
    vectorize_global = False
    global_iters = itertools.product(*[range(x) for x in global_size[::-1]])
    if not has_control_flow and all(x == 1 for x in local_size):
      total_elems = math.prod(global_size)
      if 1 < total_elems <= 16384:
        warp = list(itertools.product(*[range(x) for x in global_size[::-1]]))
        warp_size = len(warp)
        global_iters = [tuple(0 for _ in global_size)]
        vectorize_global = True

    for idxs in global_iters:
      values: dict[int, Any] = {}
      pbufs: list[memoryview] = list(bufs)
      pvals: list[int] = list(vals)
      i = 0
      while i < len(self.uops):
        uop, dtype, srcs, arg = self.uops[i]
        src_values = [values[v] for v in srcs if self.uops[v][0] not in void_ops]
        src_dtypes = [self.uops[v][1] for v in srcs if self.uops[v][0] not in void_ops]
        if uop is Ops.END:
          i = srcs[1]
          continue
        if uop in (Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.GROUP):
          i += 1
          continue
        assert dtype is not None, f"{uop} is missing a dtype"
        if uop is Ops.STORE:
          for j,val in enumerate(src_values[1] if src_dtypes[1].count > 1 else [src_values[1]]):
            for (m,o,g),v in zip(src_values[0], val):
              if g: _store(m, o+j, v, src_dtypes[1].scalar())
          i += 1
          continue
        if uop is Ops.AFTER: values[i] = src_values[0]
        elif uop in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}:
          assert isinstance(dtype, PtrDType), dtype
          storage_fmt = storage_fmt_for_dtype(dtype.base.scalar())
          if storage_fmt is None: raise RuntimeError(f"{dtype=} is not supported")
          if TYPE_CHECKING or sys.version_info < (3, 12): assert storage_fmt != "e"
          if uop is Ops.DEFINE_REG:
            values[i] = [memoryview(bytearray(dtype.size*dtype.itemsize)).cast(storage_fmt) for _ in range(warp_size)]
          else:
            buf = memoryview(bytearray(dtype.size*dtype.itemsize)) if uop is not Ops.DEFINE_GLOBAL else pbufs.pop(0)
            values[i] = [buf.cast(storage_fmt)] * warp_size
        elif uop is Ops.DEFINE_VAR:
          values[i] = [pvals.pop(0)] * warp_size
        elif uop is Ops.SPECIAL:
          if arg[0] == 'g': values[i] = [x[2-int(arg[-1])] for x in warp] if vectorize_global else [idxs[2-int(arg[-1])]] * warp_size
          elif arg[0] == 'l': values[i] = [0] * warp_size if vectorize_global else [x[2-int(arg[-1])] for x in warp]
        elif uop is Ops.CONST: values[i] = [arg] * warp_size
        elif uop is Ops.INDEX:
          ret:list = []
          if isinstance(src_dtypes[0], ImageDType):
            for m,ox,oy in zip(src_values[0], src_values[1][0], src_values[1][1]):
              if ox < 0 or ox >= src_dtypes[0].shape[1] or oy < 0 or oy >= src_dtypes[0].shape[0]: ret.append((m, None))
              else: ret.append((m, ox*4 + oy*src_dtypes[0].shape[1]*4))
          else:
            for m,o in zip(src_values[0], src_values[1]): ret.append((m,o))
          values[i] = [(m,o,g) for (m,o),g in zip(ret, src_values[2] if len(src_values) == 3 else [True]*len(ret))]
        elif uop is Ops.CAST and isinstance(dtype, PtrDType):
          values[i] = src_values[0]
        elif uop is Ops.RANGE:
          if i not in values: values[i] = [0] * warp_size
          else:
            for j in range(len(values[i])):
              values[i][j] += 1
          if values[i][0] == src_values[0][0]:
            del values[i]
            i = loop_ends[i] + 1
            continue
        elif uop is Ops.VECTORIZE: values[i] = src_values
        elif uop is Ops.BITCAST:
          packed = struct.pack(str(warp_size) + storage_fmt_for_dtype(src_dtypes[0].scalar()),
                               *[to_storage_scalar(x, src_dtypes[0].scalar()) for x in src_values[0]])
          values[i] = list(struct.unpack(str(warp_size) +  storage_fmt_for_dtype(dtype.scalar()), packed))
          values[i] = [from_storage_scalar(x, dtype.scalar()) for x in values[i]]
        elif uop is Ops.CAST:
          values[i] = [truncate.get(dtype, lambda dt: dt)(dtypes.as_const(x, dtype)) for x in src_values[0]]
        elif uop is Ops.LOAD:
          if dtype.count > 1:
            values[i] = [load([src_values[i][j] if i != 0 and src_dtypes[i].count > 1 else src_values[i] \
                               for i in range(len(src_values))], j, dtype.scalar()) for j in range(dtype.count)]
          else:
            values[i] = load(src_values, 0, dtype)
        elif uop is Ops.GEP:
          v = src_values[0][get_single_element(arg)]
          values[i] = v if isinstance(v, (list, tuple)) else [v]
        elif uop is Ops.ADD and dtype.scalar() == dtypes.float16 and uop in self.hardware_ops:
          # ── ANE hardware path for elementwise ADD ──
          self.device.reset_npu()
          btsp = self._build_btsp(len(src_values[0]))

          src0_data = np.asarray(src_values[0], dtype=np.float16).tobytes()
          src1_data = np.asarray(src_values[1], dtype=np.float16).tobytes()
          nbytes = len(src0_data)

          src1_buf = self.device._ane_alloc(nbytes)
          src2_buf = self.device._ane_alloc(nbytes)
          out_buf = self.device._ane_alloc(nbytes)
          btsp_buf = self.device._ane_alloc(0x4000)

          try:
            ctypes.memmove(src1_buf.va_addr, mv_address(src0_data), nbytes)
            ctypes.memmove(src2_buf.va_addr, mv_address(src1_data), nbytes)
            ctypes.memmove(btsp_buf.va_addr, btsp, 0x4000)

            self.device._gpu_sync(src1_buf)
            self.device._gpu_sync(src2_buf)

            handles = ([btsp_buf.handle, 0, 0, 0, out_buf.handle, src1_buf.handle, src2_buf.handle] + [0] * 25)[:0x20]
            self.device._ane_submit(0x274, 1, 0x274, handles, btsp_buf.handle)

            self.device._gpu_sync(out_buf)

            dst = memoryview(bytearray(nbytes))
            ctypes.memmove(mv_address(dst), out_buf.va_addr, nbytes)
            result = struct.unpack(f'<{len(src_values[0])}e', dst.tobytes())
            values[i] = list(result)
          finally:
            self.device._ane_free_multiple([src1_buf, src2_buf, out_buf, btsp_buf])
        elif uop in GroupOp.ALU:
          allow_fallback = uop in (Ops.XOR, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR)
          if allow_fallback:
            if DEBUG >= 1: print('ANE FALLBACK TO CPU', uop, dtype)
            values[i] = [exec_alu(uop, dtype, p) for p in zip(*src_values)]
          else:
            if DEBUG >= 1: print('<!> ANE EXIT UNSUPPORTED', uop, dtype, src_values)
        assert i in values, (uop, dtype, srcs, arg)
        i += 1
    return time.perf_counter() - st

def _ane_trunc_fix(x):
  if x.tag == "ane_trunc": return None
  xh = x.src[0].cast(dtypes.half)
  zero = UOp.const(dtypes.half, 0)
  neg = xh.alu(Ops.CMPLT, zero)
  shifted = xh.alu(Ops.SUB, UOp.const(dtypes.half, 0.49951171875))
  absx = UOp(Ops.WHERE, dtypes.half, src=(shifted.alu(Ops.CMPLT, zero), shifted.alu(Ops.NEG), shifted))
  mag = absx.alu(Ops.TRUNC).rtag("ane_trunc")
  signed = UOp(Ops.WHERE, dtypes.half, src=(neg, mag.alu(Ops.NEG).alu(Ops.ADD, UOp.const(dtypes.half, 1)), mag))
  return signed.cast(x.dtype)

class ANERenderer(Renderer):
  device = "ANE"
  has_threads = False
  tensor_cores = tc.rockchip
  code_for_op = {k:v for k,v in python_alu.items() if k not in [Ops.MULACC, Ops.RECIPROCAL, Ops.CMPNE]} | {Ops.FDIV: 0}

  extra_matcher = PatternMatcher([
    (UPat(Ops.ADD, dtypes.float, name="x"),
     lambda x: x.src[0].cast(dtypes.half).alu(Ops.ADD, x.src[1].cast(dtypes.half))),
    (UPat(Ops.ADD, dtypes.int, name="x"),
     lambda x: x.src[0].cast(dtypes.float16).alu(Ops.ADD, x.src[1].cast(dtypes.float16)).cast(dtypes.int)),
  ])

  def render(self, uops:list[UOp]) -> str:
    uop_to_idx = {u:i for i,u in enumerate(uops)}
    lops = [(u.op, u.dtype, ([] if u.op is Ops.SPECIAL else [uop_to_idx[v] for v in u.src]), u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()

class ANECompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class ANEAllocator(Allocator['ANEDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> memoryview:
    return memoryview(bytearray(size))
  def _copyin(self, dest:memoryview, src:memoryview):
    dest[:] = src
  def _copyout(self, dest:memoryview, src:memoryview):
    dest[:] = src

class ANEDevice(Compiled):
  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface("/dev/accel/accel0", os.O_RDWR)
    compilers = CompilerSet([CompilerPair(ANERenderer, ANECompiler)])
    super().__init__(device, ANEAllocator(self), compilers, functools.partial(ANEProgram, self))

  def _ane_alloc(self, size:int) -> ANEBuffer:
    bo = drm_ane_bo_init(handle=0, pad=0, size=size, offset=0)
    ioctl(self.fd_ctl, DRM_IOCTL_ANE_BO_INIT, bo)
    va_addr = self.fd_ctl.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, bo.offset)
    return ANEBuffer(va_addr=va_addr, size=size, handle=bo.handle, offset=bo.offset)

  def _ane_free(self, buf:ANEBuffer) -> None:
    self.fd_ctl.munmap(buf.va_addr, buf.size)

  def _ane_free_multiple(self, bufs:list[ANEBuffer]) -> None:
    for b in bufs: self._ane_free(b)

  def _ane_submit(self, tsk_size:int, td_count:int, td_size:int, handles:list[int], btsp_handle:int) -> int:
    req = drm_ane_submit(tsk_size=tsk_size, td_count=td_count, td_size=td_size,
                         btsp_handle=btsp_handle, pad=0)
    for i in range(ANE_TILE_COUNT):
      req.handles[i] = handles[i] if i < len(handles) else 0
    return ioctl(self.fd_ctl, DRM_IOCTL_ANE_SUBMIT, req)

  def _gpu_sync(self, buf:ANEBuffer) -> None:
    pass

  def reset_npu(self):
    pass
