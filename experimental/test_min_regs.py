"""expt3: Minimal Register Set per Op.

Phase 1: Eliminate zero-valued registers (confirmed don't-care by expt1/expt2).
Phase 2: Nullify each non-zero register one at a time to find which are truly essential.

Each op gets a classification per register:
  ESSENTIAL    — zeroing breaks output or causes HANG
  SKIP         — already validated as don't-care by expt1/expt2
  CONDITIONAL  — zeroing changes behavior but op still runs
  FORMAT       — must be valid but not a specific value
  UNNEEDED     — zeroing produces identical correct output

Usage:
  python experimental/test_min_regs.py           # full experiment (all ops, phase 2)
  python experimental/test_min_regs.py phase1    # phase 1 only (list zero regs to skip)
  python experimental/test_min_regs.py --op relu  # single op
"""

import os, sys, struct, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from experimental.ane_helpers import (
    allocate_buffer, submit_task, make_from_segments, stream_header,
    build_seg, pack_reg, run_one_raw, reset_ane, is_wedged, set_wedged,
    try_baseline
)

HALF_ONE = 0x3C00
DMA_EOL = 0x80000000
DMA_ACTIVE = 0x40000000


class reg:
    W0=0x00;W1=0x04;W2=0x08;W3=0x0c;W4=0x10;W5=0x14;W6=0x18;W7=0x1c;W8=0x20;W9=0x24;KernelDMA=0x28
    CommonStream=0x124;SrcStream=0x168;L2Stream=0x1DC;PEStream=0x228;NEStream=0x23C;DstStream=0x254
    InDim=0x128;pad0=0x12c;ChCfg=0x130;Cin=0x134;Cout=0x138
    OutDim=0x13c;pad1=0x140;ConvCfg=0x144;pad2=0x148
    GroupConvCfg=0x14c;TileCfg=0x150;pad3=0x154;pad4=0x158;Cfg=0x15c
    TaskInfo=0x160;DPE=0x164
    L2Cfg=0x1e0;SourceCfg=0x1e4;SourceBase=0x1e8
    SourceChannelStride=0x1ec;SourceRowStride=0x1f0
    L2pad0=0x1f4;L2pad1=0x1f8;L2pad2=0x1fc
    L2pad3=0x200;L2pad4=0x204;L2pad5=0x208;L2pad6=0x20c
    ResultCfg=0x210;ResultBase=0x214
    ConvResultChannelStride=0x218;ConvResultRowStride=0x21c
    L2pad7=0x220;L2pad8=0x224
    PECfg=0x22c;BiasScale=0x230;PreScale=0x234;FinalScale=0x238
    KernelCfg=0x240;MACCfg=0x244;MatrixVectorBias=0x248;AccBias=0x24c;PostScale=0x250
    SrcDMAConfig=0x16c;Srcpad0=0x170;SrcBaseAddr=0x174
    SrcRowStride=0x178;SrcPlaneStride=0x17c;SrcDepthStride=0x180
    SrcGroupStride=0x184;Srcpad1=0x188;Srcpad2=0x18c;Srcpad3=0x190;Srcpad4=0x194
    Srcpad5=0x198;Srcpad6=0x19c;Srcpad7=0x1a0;SrcFmt=0x1a4;Srcpad8=0x1a8;SrcPadStream=0x1AC
    DstDMAConfig=0x258;DstBaseAddr=0x25c;DstRowStride=0x260
    DstPlaneStride=0x264;DstDepthStride=0x268;DstGroupStride=0x26c;DstFmt=0x270


def build_td(w8_val, w9_val=0, w2_val=1058, w4_val=0xFFF86A, w6_val=(38<<10)|(3<<28)):
    return build_seg(0, 44, [
        (reg.W0, (0<<0)|(0x40<<16)|(1<<25)),
        (reg.W1, 0), (reg.W2, w2_val), (reg.W3, 0),
        (reg.W4, w4_val), (reg.W5, 0),
        (reg.W6, w6_val), (reg.W7, 0),
        (reg.W8, w8_val), (reg.W9, w9_val),
        (reg.KernelDMA, stream_header(0x1F800, 62)),
    ])


def build_fw_dma(dma_words):
    return struct.pack('>'+'I'*62, *dma_words)


def build_common(seg_off, seg_len, word_packs):
    return build_seg(seg_off, seg_len, word_packs)


# === BTSP_BUF builders per op ===

def make_relu_buf():
    ST = 0xc0
    return make_from_segments(0x4000, [
        (0, 44, build_td((5)|(1<<5)|(4<<12)|(1<<17)|(1<<24))),
        (0x2C, 0xF8, build_fw_dma([0]*2 + [DMA_EOL]*16 + [0]*16 + [DMA_ACTIVE]*16 + [DMA_EOL]*4 + [0]*8)),
        (292, 184, build_common(0x124, 184, [
            (reg.CommonStream, stream_header(0x00000, 16)),
            (reg.InDim, (1<<16)|77), (reg.OutDim, (1<<16)|77),
            (reg.ChCfg, (2)|(2<<4)),
            (reg.Cin, 1), (reg.Cout, 1),
            (reg.pad0, 1), (reg.pad1, 1), (reg.pad2, 0x2041),
            (reg.pad3, 0), (reg.pad4, 0),
            (reg.ConvCfg, (1)|(1<<5)|(1<<13)|(1<<15)|(1<<28)|(1<<30)),
            (reg.GroupConvCfg, (1)|(1<<14)|(1<<16)),
            (reg.TileCfg, 1),
            (reg.Cfg, (1<<0)|(1<<8)|(1<<16)|(1<<26)),
            (reg.TaskInfo, (1<<20)), (reg.DPE, 0),
            (reg.SrcStream, stream_header(0x13800, 28)),
            (reg.SrcDMAConfig, (1)|(8<<4)|(8<<8)|(3<<12)|(3<<16)),
            (reg.Srcpad0, 0x8880), (reg.SrcBaseAddr, 0),
            (reg.SrcRowStride, ST), (reg.SrcPlaneStride, ST),
            (reg.SrcDepthStride, ST), (reg.SrcGroupStride, 0),
            (reg.Srcpad1, 0), (reg.Srcpad2, 0), (reg.Srcpad3, 0),
            (reg.Srcpad4, 0), (reg.Srcpad5, 0), (reg.Srcpad6, 0),
            (reg.Srcpad7, 0), (reg.Srcpad8, 0),
            (reg.SrcFmt, (1)|(3<<4)|(2<<12)|(1<<24)),
            (reg.SrcPadStream, 0x00000100),
        ])),
        (476, 68, build_common(0x1DC, 68, [
            (reg.L2Stream, stream_header(0x04800, 18)), (reg.L2Cfg, 0),
            (reg.SourceCfg, (2)|(1<<4)|(1<<5)|(1<<6)|(1<<8)|(1<<20)|(1<<22)),
            (reg.SourceBase, 0),
            (reg.SourceChannelStride, 0xa0), (reg.SourceRowStride, 0xa0),
            (reg.L2pad0, 0xa0), (reg.L2pad1, 0xa0), (reg.L2pad2, 0),
            (reg.L2pad3, 0), (reg.L2pad4, 0), (reg.L2pad5, 0), (reg.L2pad6, 0),
            (reg.ResultCfg, (2)|(2<<2)|(1<<4)|(1<<5)|(1<<6)|(1<<8)|(1<<20)|(1<<22)),
            (reg.ResultBase, 0xa0),
            (reg.ConvResultChannelStride, 0), (reg.ConvResultRowStride, 0),
        ])),
        (552, 44, build_common(0x228, 44, [
            (reg.PEStream, stream_header(0x08800, 4)),
            (reg.NEStream, stream_header(0x0C800, 5)),
            (reg.KernelCfg, (1<<7)),
            (reg.MACCfg, (12)|(1<<16)|(1<<20)),
            (reg.MatrixVectorBias, 0), (reg.AccBias, 0),
            (reg.PostScale, HALF_ONE),
        ])),
        (596, 32, build_common(0x254, 32, [
            (reg.DstStream, stream_header(0x17800, 7)),
            (reg.DstDMAConfig, (1)|(12<<4)|(1<<26)),
            (reg.DstBaseAddr, 0),
            (reg.DstRowStride, ST), (reg.DstPlaneStride, ST),
            (reg.DstDepthStride, ST), (reg.DstGroupStride, 0),
            (reg.DstFmt, (1)|(3<<4)|(2<<12)|(1<<13)|(3<<20)|(1<<24)),
        ])),
    ])


def make_sigmoid_buf():
    buf = make_relu_buf()
    pack_reg(buf, reg.MACCfg, (12)|(1<<17)|(1<<20))  # non_linear_mode=2
    pack_reg(buf, reg.W9, 0x21)
    pack_reg(buf, reg.W8, (5)|(1<<5)|(36<<12)|(1<<24))
    pack_reg(buf, reg.ConvCfg, (1)|(1<<5)|(5<<13)|(0<<17)|(1<<28)|(1<<30))
    # Firmware DMA context: load LUT via KDMA then write output
    fw_dma = build_fw_dma([DMA_ACTIVE, 0, 0x81000000] + [DMA_EOL]*15 + [0]*16
                          + [DMA_EOL] + [DMA_ACTIVE]*15 + [DMA_EOL]*4 + [0]*8)
    buf[0x2C:0x2C+len(fw_dma)] = fw_dma
    sig_lut = struct.pack('<' + 'I' * 57,
        0x00000000, 0x00000000, 0x00000000,
        0x4829c8f8, 0x3c000000,
        0x10870d7f, 0x16261377, 0x1c2b1910, 0x21a01eda,
        0x2781249b, 0x2cdb2a12, 0x31d62fa1, 0x360a344e,
        0x38fb3800, 0x3a8a39d9, 0x3b653b0c, 0x3bc43b9f,
        0x3be93bdb, 0x3bf83bf2, 0x3bfd3bfb, 0x3bff3bfe,
        0x09ac3bff, 0x163e170c, 0x00013be6,
        0x00000001,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000,
        0x4829c8f8, 0x3c000000,
        0x10870d7f, 0x16261377, 0x1c2b1910, 0x21a01eda,
        0x2781249b, 0x2cdb2a12, 0x31d62fa1, 0x360a344e,
        0x38fb3800, 0x3a8a39d9, 0x3b653b0c, 0x3bc43b9f,
        0x3be93bdb, 0x3bf83bf2, 0x3bfd3bfb, 0x3bff3bfe,
        0x09ac3bff, 0x163e170c, 0x00013be6,
        0x00000001,
    )
    buf[0x274:0x274+len(sig_lut)] = sig_lut
    return buf


def make_add_buf():
    CHANNELS = 64
    STRIDE = 32
    return make_from_segments(0x4000, [
        (0, 44, build_td((6)|(1<<5)|(5<<6)|(1<<11)|(4<<12)|(1<<17),
                         w2_val=1058, w4_val=0xFFF86A)),
        (0x2C, 0xF8, build_fw_dma([0]*2 + [DMA_EOL]*16 + [0]*16 + [DMA_ACTIVE]*16 + [DMA_EOL]*4 + [0]*8)),
        (292, 184, build_common(0x124, 184, [
            (reg.CommonStream, stream_header(0x00000, 16)),
            (reg.InDim, (1<<16)|1), (reg.OutDim, (1<<16)|1),
            (reg.ChCfg, (2)|(2<<2)|(2<<4)),
            (reg.Cin, CHANNELS), (reg.Cout, CHANNELS),
            (reg.pad0, 1), (reg.pad1, 1), (reg.pad2, 0x2041),
            (reg.pad3, 4), (reg.pad4, 0),
            (reg.ConvCfg, (1)|(1<<5)|(1<<13)|(1<<15)|(1<<28)|(1<<30)),
            (reg.GroupConvCfg, (1)|(1<<16)),
            (reg.TileCfg, 1),
            (reg.Cfg, (3)|(0<<2)|(6<<3)),
            (reg.TaskInfo, 0), (reg.DPE, 0),
            (reg.SrcStream, stream_header(0x13800, 28)),
            (reg.SrcDMAConfig, (1)|(8<<4)|(8<<8)|(3<<12)|(3<<16)),
            (reg.Srcpad0, 0x33880), (reg.SrcBaseAddr, 0),
            (reg.SrcRowStride, STRIDE*2), (reg.SrcPlaneStride, STRIDE*2),
            (reg.SrcDepthStride, CHANNELS*STRIDE*2), (reg.SrcGroupStride, 0),
            (reg.Srcpad1, 0), (reg.Srcpad2, STRIDE*2),
            (reg.Srcpad3, STRIDE*2), (reg.Srcpad4, CHANNELS*STRIDE*2),
            (reg.Srcpad5, 0), (reg.Srcpad6, 0), (reg.Srcpad7, 0),
            (reg.Srcpad8, 0x2030),
            (reg.SrcFmt, (1)|(3<<4)|(2<<12)|(1<<24)),
        ])),
        (476, 68, build_common(0x1DC, 68, [
            (reg.L2Stream, stream_header(0x04800, 18)), (reg.L2Cfg, 0),
            (reg.SourceCfg, (2)|(1<<4)|(1<<5)|(1<<6)|(1<<8)|(1<<20)|(1<<22)|(1<<24)),
            (reg.SourceBase, 0),
            (reg.SourceChannelStride, 0x10), (reg.SourceRowStride, 0x420),
            (reg.L2pad0, 0x400), (reg.L2pad1, 0x400), (reg.L2pad2, 0x440),
            (reg.L2pad3, 0x10), (reg.L2pad4, 0x420), (reg.L2pad5, 0x400),
            (reg.L2pad6, 0x400),
            (reg.ResultCfg, (2)|(2<<2)|(1<<4)|(1<<5)|(1<<6)|(1<<8)|(1<<20)|(1<<22)),
            (reg.ResultBase, 0x860),
        ])),
        (552, 44, build_common(0x228, 44, [
            (reg.PEStream, stream_header(0x08800, 4)),
            (reg.PECfg, (2<<18)),
            (reg.BiasScale, (HALF_ONE<<16)),
            (reg.PreScale, (HALF_ONE<<16)),
            (reg.FinalScale, 0x3f800000),
            (reg.NEStream, stream_header(0x0C800, 5)),
            (reg.KernelCfg, 0), (reg.MACCfg, 0),
            (reg.MatrixVectorBias, 0), (reg.AccBias, 0),
            (reg.PostScale, 0),
        ])),
        (596, 32, build_common(0x254, 32, [
            (reg.DstStream, stream_header(0x17800, 7)),
            (reg.DstDMAConfig, (1)|(12<<4)|(1<<26)),
            (reg.DstBaseAddr, 0),
            (reg.DstRowStride, STRIDE*2), (reg.DstPlaneStride, STRIDE*2),
            (reg.DstDepthStride, CHANNELS*STRIDE*2), (reg.DstGroupStride, 0),
            (reg.DstFmt, (1)|(3<<4)|(2<<12)|(1<<24)),
        ])),
    ])


def make_mul_buf():
    buf = make_add_buf()
    pack_reg(buf, reg.PECfg, (2<<18)|(1<<2))
    pack_reg(buf, reg.MACCfg, (1<<4)|(1<<5))
    return buf


def make_conv_buf():
    C = 3
    KERNEL_STRIDE = 0x40
    kernel = bytearray(C * KERNEL_STRIDE)
    w = np.float16(2.0).view(np.uint16).item()
    for oc in range(C):
        off = oc * KERNEL_STRIDE + 12
        struct.pack_into('<HHH', kernel, off, w, w, w)
    kernel = bytes(kernel)

    return make_from_segments(0x4000, [
        (0, 44, build_td((5)|(1<<5)|(36<<12)|(1<<24)|(1<<25), w9_val=0x21)),
        (0x2C, 0xF8, build_fw_dma([DMA_ACTIVE, 0, 0x81000000, 0x81000000, 0x81000000] +
                                  [DMA_EOL]*13 + [0, DMA_ACTIVE, DMA_EOL] + [0]*13 +
                                  [DMA_ACTIVE]*16 + [DMA_EOL]*4 + [0]*8)),
        (292, 184, build_common(0x124, 184, [
            (reg.CommonStream, stream_header(0x00000, 16)),
            (reg.InDim, (1<<16)|1), (reg.OutDim, (1<<16)|1),
            (reg.ChCfg, (2)|(2<<4)),
            (reg.Cin, C), (reg.Cout, C),
            (reg.pad0, 1), (reg.pad1, 1), (reg.pad2, 0x2041),
            (reg.pad3, 0), (reg.pad4, 0),
            (reg.ConvCfg, (1)|(1<<5)|(1<<13)|(1<<15)|(1<<28)|(1<<30)),
            (reg.GroupConvCfg, (1)|(1<<16)),
            (reg.TileCfg, 1),
            (reg.Cfg, (1<<0)|(1<<2)|(1<<10)|(1<<14)|(1<<18)|(1<<20)|(1<<26)),
            (reg.TaskInfo, (1<<20)), (reg.DPE, 0),
            (reg.SrcStream, stream_header(0x13800, 28)),
            (reg.SrcDMAConfig, (1)|(8<<4)|(8<<8)|(3<<12)|(3<<16)),
            (reg.Srcpad0, 0x8880), (reg.SrcBaseAddr, 0),
            (reg.SrcRowStride, 0x40), (reg.SrcPlaneStride, 0x40),
            (reg.SrcDepthStride, 0xc0), (reg.SrcGroupStride, 0),
            (reg.Srcpad1, 0), (reg.Srcpad2, 0), (reg.Srcpad3, 0),
            (reg.Srcpad4, 0), (reg.Srcpad5, 0), (reg.Srcpad6, 0),
            (reg.Srcpad7, 0), (reg.Srcpad8, 0),
            (reg.SrcFmt, (1)|(3<<4)|(2<<12)|(1<<24)),
            (reg.SrcPadStream, 0x00000100),
        ])),
        (476, 76, build_common(0x1DC, 76, [
            (reg.L2Stream, stream_header(0x04800, 18)), (reg.L2Cfg, 0),
            (reg.SourceCfg, (2)|(1<<4)|(1<<5)|(1<<6)|(1<<8)|(1<<20)|(1<<22)),
            (reg.SourceBase, 0),
            (reg.SourceChannelStride, 0x10), (reg.SourceRowStride, 0x30),
            (reg.L2pad0, 0x30), (reg.L2pad1, 0x30), (reg.L2pad2, 0),
            (reg.L2pad3, 0), (reg.L2pad4, 0), (reg.L2pad5, 0), (reg.L2pad6, 0),
            (reg.ResultCfg, (2)|(1<<4)|(1<<5)|(1<<6)|(1<<8)|(1<<20)|(1<<22)),
            (reg.ResultBase, 0x30),
            (reg.ConvResultChannelStride, 0x10), (reg.ConvResultRowStride, 0x30),
            (reg.L2pad7, 0x30), (reg.L2pad8, 0x30),
        ])),
        (552, 44, build_common(0x228, 44, [
            (reg.PEStream, stream_header(0x08800, 4)),
            (reg.PECfg, 0), (reg.BiasScale, 0), (reg.PreScale, 0),
            (reg.FinalScale, 0),
            (reg.NEStream, stream_header(0x0C800, 5)),
            (reg.KernelCfg, (1<<7)|(1<<1)),
            (reg.MACCfg, (1<<10)|(1<<11)|(1<<12)|(1<<20)),
            (reg.MatrixVectorBias, 0), (reg.AccBias, 0),
            (reg.PostScale, HALF_ONE),
        ])),
        (596, 32, build_common(0x254, 32, [
            (reg.DstStream, stream_header(0x17800, 7)),
            (reg.DstDMAConfig, (1)|(12<<4)),
            (reg.DstBaseAddr, 0),
            (reg.DstRowStride, 0x40), (reg.DstPlaneStride, 0x40),
            (reg.DstDepthStride, 0xc0), (reg.DstGroupStride, 0),
            (reg.DstFmt, (1)|(3<<4)|(2<<12)|(3<<20)|(1<<24)),
        ])),
        (0x274, len(kernel), kernel),
    ])


def make_gemm_buf():
    C = 512

    def _be32(v):
        return struct.unpack('>I', struct.pack('<I', v))[0]

    return make_from_segments(0x4000, [
        (0, 44, build_td((5)|(1<<5)|(36<<12)|(1<<24)|(1<<26), w9_val=0x21)),
        (0x2C, 0xF8, build_fw_dma([DMA_ACTIVE, 0] + [0x81000000]*16 +
                                   [_be32(i * 0x8000) for i in range(16)] +
                                   [_be32(0x8000)]*16 +
                                   [DMA_EOL]*4 + [0]*8)),
        (292, 184, build_common(0x124, 184, [
            (reg.CommonStream, stream_header(0x00000, 16)),
            (reg.InDim, (1<<16)|1), (reg.OutDim, (1<<16)|1),
            (reg.ChCfg, (2)|(2<<4)),
            (reg.Cin, C), (reg.Cout, C),
            (reg.pad0, 1), (reg.pad1, 1), (reg.pad2, 0x2041),
            (reg.pad3, 0), (reg.pad4, 0),
            (reg.ConvCfg, 0x5000b421),
            (reg.GroupConvCfg, 0x00010001),
            (reg.TileCfg, 1),
            (reg.Cfg, 0x00244405),
            (reg.TaskInfo, (1<<20)), (reg.DPE, 0),
            (reg.SrcStream, stream_header(0x13800, 28)),
            (reg.SrcDMAConfig, (1)|(8<<4)|(8<<8)|(3<<12)|(3<<16)),
            (reg.Srcpad0, 0x8880), (reg.SrcBaseAddr, 0),
            (reg.SrcRowStride, 0x40), (reg.SrcPlaneStride, 0x40),
            (reg.SrcDepthStride, 0x8000), (reg.SrcGroupStride, 0),
            (reg.Srcpad1, 0), (reg.Srcpad2, 0), (reg.Srcpad3, 0),
            (reg.Srcpad4, 0), (reg.Srcpad5, 0), (reg.Srcpad6, 0),
            (reg.Srcpad7, 0), (reg.Srcpad8, 0),
            (reg.SrcFmt, (1)|(3<<4)|(2<<12)|(1<<24)),
            (reg.SrcPadStream, 0x00000100),
        ])),
        (476, 76, build_common(0x1DC, 76, [
            (reg.L2Stream, stream_header(0x04800, 18)), (reg.L2Cfg, 0),
            (reg.SourceCfg, 0x00500172),
            (reg.SourceBase, 0),
            (reg.SourceChannelStride, 0x10), (reg.SourceRowStride, 0x2030),
            (reg.L2pad0, 0x2000), (reg.L2pad1, 0x2000), (reg.L2pad2, 0),
            (reg.L2pad3, 0), (reg.L2pad4, 0), (reg.L2pad5, 0), (reg.L2pad6, 0),
            (reg.ResultCfg, 0x00500172),
            (reg.ResultBase, 0x2030),
            (reg.ConvResultChannelStride, 0x10), (reg.ConvResultRowStride, 0x2020),
            (0x220, 0x2000), (0x224, 0x2000),
        ])),
        (552, 44, build_common(0x228, 44, [
            (reg.PEStream, stream_header(0x08800, 4)),
            (reg.PECfg, 0), (reg.BiasScale, 0), (reg.PreScale, 0),
            (reg.FinalScale, 0),
            (reg.NEStream, stream_header(0x0C800, 5)),
            (reg.KernelCfg, 0x82),
            (reg.MACCfg, 0x00101c00),
            (reg.MatrixVectorBias, 0), (reg.AccBias, 0),
            (reg.PostScale, HALF_ONE),
        ])),
        (596, 32, build_common(0x254, 32, [
            (reg.DstStream, stream_header(0x17800, 7)),
            (reg.DstDMAConfig, (1)|(12<<4)),
            (reg.DstBaseAddr, 0),
            (reg.DstRowStride, 0x40), (reg.DstPlaneStride, 0x40),
            (reg.DstDepthStride, 0x8000), (reg.DstGroupStride, 0),
            (reg.DstFmt, (1)|(3<<4)|(2<<12)|(3<<20)|(1<<24)),
        ])),
    ])


# === Register extraction from BTSP_BUF ===

def extract_nonzero_regs(btsp_buf):
    """Read all non-zero 32-bit values from a BTSP_BUF at known register offsets.
    Returns dict: offset -> (name, value)
    """
    reg_offsets = {
        0x00: "W0", 0x04: "W1", 0x08: "W2", 0x0c: "W3",
        0x10: "W4", 0x14: "W5", 0x18: "W6", 0x1c: "W7",
        0x20: "W8", 0x24: "W9", 0x28: "KernelDMA",
        0x124: "CommonStream",
        0x128: "InDim", 0x12c: "pad0", 0x130: "ChCfg", 0x134: "Cin", 0x138: "Cout",
        0x13c: "OutDim", 0x140: "pad1", 0x144: "ConvCfg", 0x148: "pad2",
        0x14c: "GroupConvCfg", 0x150: "TileCfg", 0x154: "pad3", 0x158: "pad4", 0x15c: "Cfg",
        0x160: "TaskInfo", 0x164: "DPE",
        0x168: "SrcStream",
        0x16c: "SrcDMAConfig", 0x170: "Srcpad0", 0x174: "SrcBaseAddr",
        0x178: "SrcRowStride", 0x17c: "SrcPlaneStride", 0x180: "SrcDepthStride",
        0x184: "SrcGroupStride", 0x188: "Srcpad1",
        0x18c: "Srcpad2", 0x190: "Srcpad3", 0x194: "Srcpad4",
        0x198: "Srcpad5", 0x19c: "Srcpad6", 0x1a0: "Srcpad7",
        0x1a4: "SrcFmt", 0x1a8: "Srcpad8", 0x1AC: "SrcPadStream",
        0x1DC: "L2Stream",
        0x1e0: "L2Cfg", 0x1e4: "SourceCfg", 0x1e8: "SourceBase",
        0x1ec: "SourceChannelStride", 0x1f0: "SourceRowStride",
        0x1f4: "L2pad0", 0x1f8: "L2pad1", 0x1fc: "L2pad2",
        0x200: "L2pad3", 0x204: "L2pad4", 0x208: "L2pad5", 0x20c: "L2pad6",
        0x210: "ResultCfg", 0x214: "ResultBase",
        0x218: "ConvResultChannelStride", 0x21c: "ConvResultRowStride",
        0x220: "L2pad7", 0x224: "L2pad8",
        0x228: "PEStream",
        0x22c: "PECfg", 0x230: "BiasScale", 0x234: "PreScale", 0x238: "FinalScale",
        0x23C: "NEStream",
        0x240: "KernelCfg", 0x244: "MACCfg", 0x248: "MatrixVectorBias",
        0x24c: "AccBias", 0x250: "PostScale",
        0x254: "DstStream",
        0x258: "DstDMAConfig", 0x25c: "DstBaseAddr",
        0x260: "DstRowStride", 0x264: "DstPlaneStride", 0x268: "DstDepthStride",
        0x26c: "DstGroupStride", 0x270: "DstFmt",
    }
    result = {}
    for off, name in reg_offsets.items():
        val = struct.unpack_from('<I', btsp_buf, off)[0]
        if val != 0:
            result[off] = (name, val)
    return result


# Phase 1 skip list: registers known to be don't-care (confirmed by expt1/expt2)
# These are zero-valued in baseline and truly unused
PHASE1_AUTO_SKIP = {
    # Confirmed dead by expt2
    0x164,   # DPE
    0x188,   # Srcpad1
    0x198,   # Srcpad5
    0x19c,   # Srcpad6
    0x1a0,   # Srcpad7
    0x1fc,   # L2pad2
    0x200,   # L2pad3
    0x204,   # L2pad4
    0x208,   # L2pad5
    0x20c,   # L2pad6
    0x220,   # L2pad7
    0x224,   # L2pad8
    0x248,   # MatrixVectorBias
    0x24c,   # AccBias
    0x154,   # pad3
    0x158,   # pad4
    0x218,   # ConvResultChannelStride
    0x21c,   # ConvResultRowStride
    0x1e8,   # SourceBase
    0x1e0,   # L2Cfg
    0x174,   # SrcBaseAddr
    0x25c,   # DstBaseAddr
    0x184,   # SrcGroupStride
    0x26c,   # DstGroupStride
}

# Special: stream headers must be preserved (they're not "registers" in the usual sense)
STREAM_HEADERS = {0x124, 0x168, 0x1DC, 0x228, 0x23C, 0x254, 0x28}

# Task descriptor words that are always needed
TD_ALWAYS = {0x00, 0x08, 0x10, 0x18, 0x20, 0x28}


def run_op(op_buf, inputs, n_src_bufs=1, channels=1, gemm_mode=False,
           input_stride=1, read_stride=1):
    """Run an op and return output values.

    Args:
        op_buf: BTSP_BUF bytearray
        inputs: list of input values
        n_src_bufs: 1 or 2
        channels: number of channels (Cin=Cout)
        gemm_mode: use separate cmd_buf + BTSP_BUF
        input_stride: element stride for input placement (1=contiguous, 32=per-channel)
        read_stride: stride for reading outputs (1=contiguous, 32=per-channel)
    """
    try:
        fd = os.open("/dev/accel/accel0", os.O_RDWR)

        if gemm_mode:
            kernel_size = 524288
            weights = np.full(kernel_size // 2, np.float16(0.5), dtype=np.float16)
            cmd_buf = bytearray(op_buf)
            cmd_buf.extend(b'\x00' * (32768 - len(cmd_buf)))
            cmd_buf[0x274:0x274 + len(weights.tobytes())] = weights.tobytes()
            cmd_h, cmd_m = allocate_buffer(fd, len(cmd_buf))
            cmd_m.write(bytes(cmd_buf))
            cmd_m.close()
            btsp_h, btsp_m = allocate_buffer(fd, 0x4000)
            btsp_m.write(bytes(op_buf))
            btsp_m.close()
            buf_handle = cmd_h
        else:
            btsp_h, btsp_m = allocate_buffer(fd, 0x4000)
            btsp_m.write(bytes(op_buf))
            buf_handle = btsp_h

        out_size = 0x8000 if gemm_mode else 0x4000
        out_h, out_m = allocate_buffer(fd, out_size)

        # Build input buffer: for n_src_bufs==1, place inputs at their positions.
        # For n_src_bufs==2 (elementwise), broadcast: all channels in buf A get inputs[0],
        # all channels in buf B get inputs[1].
        src_h, src_m = allocate_buffer(fd, 0x4000)
        inp = np.zeros(8192, dtype=np.float16)
        if n_src_bufs == 2:
            for ch in range(channels):
                inp[ch * input_stride] = np.float16(inputs[0] if len(inputs) > 0 else 0)
        else:
            for i, v in enumerate(inputs):
                inp[i * input_stride] = np.float16(v)
        src_m.write(inp.tobytes())

        if n_src_bufs == 2:
            src2_h, src2_m = allocate_buffer(fd, 0x4000)
            src2 = np.zeros(8192, dtype=np.float16)
            for ch in range(channels):
                src2[ch * input_stride] = np.float16(inputs[1] if len(inputs) > 1 else 0)
            src2_m.write(src2.tobytes())
            handles = [buf_handle, 0, 0, 0, out_h, src_h, src2_h] + [0] * 25
        else:
            handles = [buf_handle, 0, 0, 0, out_h, src_h, 0] + [0] * 25

        submit_task(fd, 0x274, 1, 0x274, handles, btsp_h)
        n_read = max(channels * read_stride, len(inputs) * read_stride, 8)
        out = np.frombuffer(out_m, dtype=np.float16, count=n_read).copy()
        os.close(fd)
        return [float(out[i * read_stride]) for i in range(len(inputs))]
    except Exception as e:
        try:
            os.close(fd)
        except:
            pass
        return None


# === Op definitions ===

EXAMPLES = []

# relu
_relu_buf = make_relu_buf()
_relu_regs = extract_nonzero_regs(_relu_buf)
EXAMPLES.append({
    "name": "relu",
    "buf": _relu_buf,
    "inputs": [-3.0, 5.0, -1.0, 2.0],
    "check": lambda out: (
        out is not None and len(out) >= 2 and
        abs(out[0]) < 0.01 and abs(out[1] - 5.0) < 0.01
    ),
    "check_desc": "relu([-3,5,-1,2]) -> [0,5,0,2]",
    "regs": _relu_regs,
    "n_src_bufs": 1,
    "channels": 1,
    "input_stride": 1,
    "read_stride": 1,
    "gemm_mode": False,
})

# sigmoid
_sig_buf = make_sigmoid_buf()
_sig_regs = extract_nonzero_regs(_sig_buf)
EXAMPLES.append({
    "name": "sigmoid",
    "buf": _sig_buf,
    "inputs": [-5.0, 0.0, 5.0],
    "check": lambda out: (
        out is not None and len(out) >= 3 and
        abs(out[0] - 0.007) < 0.01 and
        abs(out[1] - 0.5) < 0.02 and
        abs(out[2] - 0.993) < 0.01
    ),
    "check_desc": "sigmoid([-5,0,5]) -> [~0.007, ~0.5, ~0.993]",
    "regs": _sig_regs,
    "n_src_bufs": 1,
    "channels": 1,
    "input_stride": 1,
    "read_stride": 1,
    "gemm_mode": False,
})

# add
_add_buf = make_add_buf()
_add_regs = extract_nonzero_regs(_add_buf)
EXAMPLES.append({
    "name": "add",
    "buf": _add_buf,
    "inputs": [3.0, 2.0],
    "check": lambda out: (
        out is not None and len(out) >= 2 and
        abs(out[0] - 5.0) < 0.01
    ),
    "check_desc": "add(3,2) -> 5",
    "regs": _add_regs,
    "n_src_bufs": 2,
    "channels": 64,
    "input_stride": 32,
    "read_stride": 32,
    "gemm_mode": False,
})

# mul
_mul_buf = make_mul_buf()
_mul_regs = extract_nonzero_regs(_mul_buf)
EXAMPLES.append({
    "name": "mul",
    "buf": _mul_buf,
    "inputs": [3.0, 2.0],
    "check": lambda out: (
        out is not None and len(out) >= 2 and
        abs(out[0] - 6.0) < 0.01
    ),
    "check_desc": "mul(3,2) -> 6",
    "regs": _mul_regs,
    "n_src_bufs": 2,
    "channels": 64,
    "input_stride": 32,
    "read_stride": 32,
    "gemm_mode": False,
})

# conv
_conv_buf = make_conv_buf()
_conv_regs = extract_nonzero_regs(_conv_buf)
EXAMPLES.append({
    "name": "conv",
    "buf": _conv_buf,
    "inputs": [1.0, 2.0, 3.0],  # 3 channels, each multiplied by weight=2.0
    "check": lambda out: (
        out is not None and len(out) >= 3 and
        all(abs(out[i] - 12.0) < 0.1 for i in range(3))
    ),
    "check_desc": "conv(1,2,3) w=2.0 -> [12,12,12]",
    "regs": _conv_regs,
    "n_src_bufs": 1,
    "channels": 3,
    "input_stride": 32,
    "read_stride": 32,
    "gemm_mode": False,
})

# gemm
_gemm_buf = make_gemm_buf()
_gemm_regs = extract_nonzero_regs(_gemm_buf)
EXAMPLES.append({
    "name": "gemm",
    "buf": _gemm_buf,
    "inputs": [1.0],  # all 512 channels get 1.0 * 256 * 0.5 = 128
    "check": lambda out: (
        out is not None and len(out) >= 1 and
        abs(out[0] - 128.0) < 1.0
    ),
    "check_desc": "gemm(1.0) w=0.5 -> 128.0",
    "regs": _gemm_regs,
    "n_src_bufs": 1,
    "channels": 512,
    "input_stride": 32,
    "read_stride": 32,
    "gemm_mode": True,
})


def run_baseline(op):
    """Run baseline for a single op and report result."""
    name = op["name"]
    print(f"\n--- {name} baseline ---")
    out = run_op(op["buf"], op["inputs"], op["n_src_bufs"],
                 op["channels"], op["gemm_mode"],
                 input_stride=op.get("input_stride", 1),
                 read_stride=op.get("read_stride", 1))
    if out is None:
        print(f"  FAIL: HANG")
        return False
    ok = op["check"](out)
    print(f"  Output: {[f'{v:.4f}' for v in out[:4]]}")
    print(f"  Check: {'OK' if ok else 'FAIL'} ({op['check_desc']})")
    return ok


def verify_all_baselines():
    """Verify all op baselines work before running experiment."""
    all_ok = True
    relu_buf = EXAMPLES[0]["buf"]
    for i, op in enumerate(EXAMPLES):
        if i > 0:
            reset_ane(relu_buf)
        if op["name"] == "gemm":
            ok = run_baseline_gemm_standalone(op)
        else:
            ok = run_baseline(op)
        if not ok:
            all_ok = False
    return all_ok


# === Phase 2: nullify each non-zero register ===

# Known poison-pill registers: zeroing causes unrecoverable HANG (expt1 confirmed)
# These need SOME valid value, so nullify test is not useful
NO_NULLIFY = {
    0x24,   # W9 — needed for KDMA kernel DMA config
    0x12c,  # pad0 — must be 1
    0x130,  # ChCfg — must have valid format
    0x134,  # Cin — zero channels HANGs
    0x138,  # Cout — mismatch wedses
    0x13c,  # OutDim — must match InDim
    0x140,  # pad1 — must be 1
    0x144,  # ConvCfg — must be valid config
    0x148,  # pad2 — unknown, zero HANGs
    0x14c,  # GroupConvCfg — must be valid
    0x150,  # TileCfg — must be valid
    0x16c,  # SrcDMAConfig — en=0 stops DMA
    0x1a4,  # SrcFmt — must be valid format
    0x1ec,  # SourceChannelStride — zero HANGs hard
    0x210,  # ResultCfg — must be valid
    0x218,  # ConvResultChannelStride — zero HANGs on conv pipeline
    0x244,  # MACCfg — op_mode=0 HANGs
    0x258,  # DstDMAConfig — en=0 stops DMA
    0x270,  # DstFmt — must be valid format
}


def op_supports_nullify(op, off, name):
    """Check if a register should be tested by nullifying.
    
    Skip: stream headers, TD always-needed, phase 1 auto-skip, poison-pill.
    """
    if off in STREAM_HEADERS:
        return False
    if off in TD_ALWAYS:
        return False
    if off in NO_NULLIFY:
        return False
    return True


def run_phase2_for_op(op):
    """For a single op, test each non-zero register by setting it to 0."""
    name = op["name"]
    regs = op["regs"]

    results = {}
    total = len(regs)
    tested = 0
    skipped = 0

    print(f"\n{'='*80}")
    print(f"PHASE 2: {name}")
    print(f"{'='*80}")
    print(f"{'Register':<25} {'Value':>10} {'Result':<20} {'Output':<30}")
    print("-" * 85)

    # Test in REVERSE offset order (safe stride/pad regs first, dangerous config last)
    for off in sorted(regs.keys(), reverse=True):
        rname, val = regs[off]

        if not op_supports_nullify(op, off, rname):
            results[off] = ("SKIP", f"always-needed ({rname})", None)
            skipped += 1
            continue

        tested += 1
        if is_wedged():
            results[off] = ("SKIP", "ANE_WEDGED", None)
            print(f"{rname:<25} {val:>10x} {'SKIP (WEDGED)':<20}")
            continue

        buf = bytearray(op["buf"])
        pack_reg(buf, off, 0)
        out = run_op(buf, op["inputs"], op["n_src_bufs"],
                     op["channels"], op["gemm_mode"],
                     input_stride=op.get("input_stride", 1),
                     read_stride=op.get("read_stride", 1))

        if out is None:
            result = "HANG"
            out_str = "HANG"
        elif op["check"](out):
            result = "UNNEEDED"
            out_str = f"[{', '.join(f'{v:.4f}' for v in out[:4])}]"
        else:
            result = "ESSENTIAL"
            out_str = f"[{', '.join(f'{v:.4f}' for v in out[:4])}]"

        results[off] = (result, rname, out)
        print(f"{rname:<25} {val:>10x} {result:<20} {out_str:<30}")

        if out is None:
            if not reset_ane(op["buf"]):
                print(f"  >>> HANG — reset_ane FAILED for {name}, aborting <<<")
                break

    print(f"\n{name} summary: {tested} tested, {skipped} skipped")

    essential = {off: regs[off] for off, (r, _, _) in results.items() if r == "ESSENTIAL"}
    unneeded = {off: regs[off] for off, (r, _, _) in results.items() if r == "UNNEEDED"}
    hang = {off: regs[off] for off, (r, _, _) in results.items() if r == "HANG"}

    if essential:
        print(f"\nESSENTIAL ({len(essential)}):")
        for off in sorted(essential):
            rname, val = essential[off]
            print(f"  {rname:<25} (0x{off:04x}) = 0x{val:x}")

    if unneeded:
        print(f"\nUNNEEDED ({len(unneeded)}):")
        for off in sorted(unneeded):
            rname, val = unneeded[off]
            print(f"  {rname:<25} (0x{off:04x}) = 0x{val:x}")

    if hang:
        print(f"\nHANG ({len(hang)}):")
        for off in sorted(hang):
            rname, val = hang[off]
            print(f"  {rname:<25} (0x{off:04x}) = 0x{val:x}")

    return results


def run_baseline_gemm_standalone(op):
    """Run gemm baseline in a separate process to avoid device state pollution."""
    import subprocess, json, tempfile
    script = '''
import sys, os; sys.path.insert(0, '.')
import numpy as np
from experimental.ane_helpers import allocate_buffer, submit_task
from experimental.test_min_regs import make_gemm_buf, reg

buf = make_gemm_buf()
weights = np.full(524288 // 2, np.float16(0.5), dtype=np.float16)
cmd = bytearray(buf)
cmd.extend(b'\\x00' * (32768 - len(cmd)))
cmd[0x274:0x274 + len(weights.tobytes())] = weights.tobytes()

fd = os.open('/dev/accel/accel0', os.O_RDWR)
cmd_h, cmd_m = allocate_buffer(fd, len(cmd))
cmd_m.write(bytes(cmd)); cmd_m.close()
out_h, out_m = allocate_buffer(fd, 0x8000)
src_h, src_m = allocate_buffer(fd, 0x4000)
src = np.zeros(0x4000 // 2, dtype=np.float16)
src[:512 * 32:32] = np.float16(1.0)
src_m.write(src.tobytes()); src_m.close()
btsp_h, btsp_m = allocate_buffer(fd, 0x4000)
btsp_m.write(bytes(buf)); btsp_m.close()

handles = [cmd_h, 0, 0, 0, out_h, src_h, 0] + [0] * 25
submit_task(fd, 0x274, 1, 0x274, handles, btsp_h)
out = np.frombuffer(out_m, dtype=np.float16, count=512*32).reshape(512, 32)[:, 0].copy()
print(float(out[0]))
os.close(fd)
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f'\n--- gemm baseline (standalone) ---')
        print(f'  FAIL: {result.stderr.strip()}')
        return False
    out_val = float(result.stdout.strip())
    ok = abs(out_val - 128.0) < 1.0
    print(f'\n--- gemm baseline (standalone) ---')
    print(f'  Output: [{out_val:.4f}]')
    status_str = "OK" if ok else "FAIL"
    print(f'  Check: {status_str} (gemm(1.0) w=0.5 -> 128.0)')
    return ok


def run_phase2():
    """Run phase 2 for all ops."""
    all_results = {}
    relu_buf = EXAMPLES[0]["buf"]
    for i, op in enumerate(EXAMPLES):
        if i > 0:
            reset_ane(relu_buf)
        if is_wedged():
            print(f"\nWEDGED — skipping remaining ops")
            break
        if op["name"] == "gemm":
            print(f"\nSkipping gemm phase 2 (requires standalone process)")
            all_results["gemm"] = {}
            continue
        results = run_phase2_for_op(op)
        all_results[op["name"]] = results
    return all_results


def main():
    phase1_only = len(sys.argv) > 1 and sys.argv[1] == "phase1"
    single_op = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--op" and i + 2 < len(sys.argv):
            single_op = sys.argv[i + 2]

    print("=" * 80)
    print("EXPT3: MINIMAL REGISTER SET PER OP")
    print("=" * 80)

    if not phase1_only:
        print("\nPhase 0: Verify baselines...")
        ok = verify_all_baselines()
        if not ok:
            print("\nWARNING: Some baselines failed. Phase 2 results may be unreliable.")
        else:
            print("All baselines OK.")

    if phase1_only:
        print("\nPHASE 1: Zero-valued registers (auto-skip)")
        print("=" * 60)
        print("\nThe following registers have baseline value 0 in every example")
        print("and can be omitted immediately (confirmed by expt1/expt2):")
        print()
        for off in sorted(PHASE1_AUTO_SKIP):
            name = {
                0x164: "DPE", 0x188: "Srcpad1", 0x198: "Srcpad5",
                0x19c: "Srcpad6", 0x1a0: "Srcpad7",
                0x1fc: "L2pad2", 0x200: "L2pad3", 0x204: "L2pad4",
                0x208: "L2pad5", 0x20c: "L2pad6",
                0x220: "L2pad7", 0x224: "L2pad8",
                0x248: "MatrixVectorBias", 0x24c: "AccBias",
                0x154: "pad3", 0x158: "pad4",
                0x218: "ConvResultChannelStride", 0x21c: "ConvResultRowStride",
                0x1e8: "SourceBase", 0x1e0: "L2Cfg",
                0x174: "SrcBaseAddr", 0x25c: "DstBaseAddr",
                0x184: "SrcGroupStride", 0x26c: "DstGroupStride",
            }.get(off, f"0x{off:x}")
            print(f"  {name:<30} (0x{off:04x})")
        print()
        print("Also skip: all stream headers, TD words (always needed)")
        print("These are the stream/register section markers, not operation config.")
        print()
        print("Phase 2 will test each remaining non-zero register by setting to 0.")
        return

    print("\nPhase 2: Nullifying non-zero registers...")
    if single_op:
        targets = [op for op in EXAMPLES if op["name"] == single_op]
        if not targets:
            print(f"Unknown op: {single_op}")
            print(f"Available: {[op['name'] for op in EXAMPLES]}")
            return
        run_phase2_for_op(targets[0])
    else:
        run_phase2()


if __name__ == "__main__":
    main()
