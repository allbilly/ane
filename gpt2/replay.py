"""Direct DRM replay, matching allbilly/libane's ane_accel.h ABI."""
import ctypes
from fcntl import ioctl
import json
import mmap
import os
from pathlib import Path
import platform
import stat
import struct
import numpy as np
from hwx import parse_tasks, require


class BOInit(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32),
                ("size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]


class BOFree(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]


class Submit(ctypes.Structure):
    _fields_ = [("tsk_size", ctypes.c_uint64), ("td_count", ctypes.c_uint32),
                ("td_size", ctypes.c_uint32), ("handles", ctypes.c_uint32 * 32),
                ("btsp_handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]


def iowr(number, cls):
    return (3 << 30) | (0x64 << 8) | (ctypes.sizeof(cls) << 16) | number


INIT, FREE, SUBMIT = iowr(0x41, BOInit), iowr(0x42, BOFree), iowr(0x43, Submit)


def device_path(requested=None):
    require(platform.system() == "Linux" and platform.machine() in ("aarch64", "arm64"),
            "ANE replay requires M1 Asahi Linux; use --backend cpu on this host")
    compatible = Path("/proc/device-tree/compatible")
    require(compatible.is_file() and b"apple,t8103\0" in compatible.read_bytes(),
            "this dump targets base Apple M1 (t8103); other chips need their own verified dump")
    paths = [Path(requested)] if requested else sorted(Path("/dev/accel").glob("accel*"))
    for path in paths:
        try:
            if not stat.S_ISCHR(path.stat().st_mode):
                continue
            driver = Path("/sys/class/accel") / path.name / "device/driver"
            if driver.resolve().name != "ane":
                continue
            require(os.access(path, os.R_OK | os.W_OK), f"no read/write permission for {path}")
            return path
        except FileNotFoundError:
            continue
    raise ValueError("no accessible ANE accel device; install/boot the ANE driver and device tree described in ~/ane/README.md")


class Buffer:
    def __init__(self, fd, size):
        self.fd, self.handle, self.map = fd, 0, None
        self.size = (size + 0x3FFF) & ~0x3FFF
        request = BOInit(size=self.size)
        ioctl(fd, INIT, request)
        self.handle = request.handle
        require(bool(self.handle), "driver returned a zero BO handle")
        try:
            self.map = mmap.mmap(fd, self.size, mmap.MAP_SHARED,
                                 mmap.PROT_READ | mmap.PROT_WRITE, offset=request.offset)
        except BaseException:
            self.close()
            raise

    def write(self, data, offset=0):
        require(offset >= 0 and offset + len(data) <= self.size, "buffer overflow")
        self.map[offset:offset + len(data)] = data

    def close(self):
        if self.map is not None:
            self.map.close()
            self.map = None
        if self.handle:
            try:
                ioctl(self.fd, FREE, BOFree(handle=self.handle))
            finally:
                self.handle = 0


class Kernel:
    def __init__(self, device, name):
        self.device, self.buffers, self.bootstrap = device, {}, None
        self.meta = json.loads((device.root / "kernels" / name / "meta.json").read_text())
        require(device.assets is not None, "external weights must be configured before ANE replay")
        program = device.assets.payload(self.meta, "program")
        parse_tasks(program, self.meta["td_size"], self.meta["td_count"])
        weights = device.assets.payload(self.meta, "weights")
        constants = device.assets.payload(self.meta, "constants")
        require(len(program) == self.meta["tsk_size"] and len(program) % 16 == 0, "invalid command boundary")
        try:
            self.buffers[0] = Buffer(device.fd, len(program) + len(weights) + 1)
            self.buffers[0].write(program)
            self.buffers[0].write(weights, len(program))
            self.buffers[2] = Buffer(device.fd, len(constants))
            self.buffers[2].write(constants)
            for item in self.meta["buffers"]:
                self.buffers[item["bank"]] = Buffer(device.fd, item["size"])
            self.bootstrap = Buffer(device.fd, self.meta["td_size"])
            bootstrap = bytearray(program[:self.meta["td_size"]])
            header = struct.unpack_from("<I", bootstrap)[0]
            struct.pack_into("<I", bootstrap, 0, (header & ~(0xFF << 16)) | (0x40 << 16))
            self.bootstrap.write(bootstrap)
        except BaseException:
            self.close()
            raise

    def run(self, x):
        x = np.asarray(x, dtype="<f2")
        require(x.shape == (768, 32) and bool(np.isfinite(x).all()), "expected finite fp16 input [768,32]")
        for item in self.meta["buffers"]:
            buffer = self.buffers[item["bank"]]
            if item["role"] == "input":
                buffer.write(x.tobytes())
            elif item["role"] == "scratch":
                buffer.write(bytes(buffer.size))
            else:
                buffer.write(np.full(768 * 32, np.nan, dtype="<f2").tobytes())
        request = Submit(tsk_size=self.meta["tsk_size"], td_count=self.meta["td_count"],
                         td_size=self.meta["td_size"], btsp_handle=self.bootstrap.handle)
        for bank, buffer in self.buffers.items():
            request.handles[bank] = buffer.handle
        require(request.handles[1] == 0, "kernel BAR is synthesized by the driver")
        ioctl(self.device.fd, SUBMIT, request)
        result = {}
        for item in self.meta["io"]:
            if item["role"] == "output":
                value = np.frombuffer(self.buffers[item["bank"]].map, dtype="<f2", count=768 * 32).reshape(768, 32).copy()
                require(bool(np.isfinite(value).all()), f"nonfinite/unwritten output: {self.meta['kernel']}/{item['name']}")
                result[item["name"]] = value
        return result

    def vector(self, x):
        tile = np.zeros((768, 32), dtype="<f2")
        tile[:, 0] = x
        return self.run(tile)

    def close(self):
        buffers = list(self.buffers.values()) + ([self.bootstrap] if self.bootstrap else [])
        self.buffers = {}
        self.bootstrap = None
        for buffer in reversed(buffers):
            try:
                buffer.close()
            except OSError:
                pass  # closing the device FD also releases remaining BOs


class Device:
    def __init__(self, root, requested=None):
        self.root, self.kernels, self.assets = root, {}, None
        self.path = device_path(requested)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)

    def kernel(self, name):
        if name not in self.kernels:
            self.kernels[name] = Kernel(self, name)
        return self.kernels[name]

    def close(self):
        for kernel in self.kernels.values():
            kernel.close()
        self.kernels.clear()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
