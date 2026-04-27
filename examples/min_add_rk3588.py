from fcntl import ioctl
import os, mmap
import ctypes
RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0

fd = os.open(f"/dev/dri/card1", os.O_RDWR)

class rknpu_mem_create(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
        ("obj_addr", ctypes.c_uint64),
        ("dma_addr", ctypes.c_uint64),
        ("sram_size", ctypes.c_uint64),
    ]

class rknpu_mem_map(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
    ]

class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
    ]

class rknpu_submit(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("timeout", ctypes.c_uint32),
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
        ("task_counter", ctypes.c_uint32),
        ("priority", ctypes.c_int32),
        ("task_obj_addr", ctypes.c_uint64),
        ("regcfg_obj_addr", ctypes.c_uint64),
        ("task_base_addr", ctypes.c_uint64),
        ("user_data", ctypes.c_uint64),
        ("core_mask", ctypes.c_uint32),
        ("fence_fd", ctypes.c_int32),
        ("subcore_task", rknpu_subcore_task * 5),
    ]

def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | (nr << 0) | (size << 16)
DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR('d', 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR('d', 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit))

class struct_rknpu_task(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("op_idx", ctypes.c_uint32),
        ("enable_mask", ctypes.c_uint32),
        ("int_mask", ctypes.c_uint32),
        ("int_clear", ctypes.c_uint32),
        ("int_status", ctypes.c_uint32),
        ("regcfg_amount", ctypes.c_uint32),
        ("regcfg_offset", ctypes.c_uint32),
        ("regcmd_addr", ctypes.c_uint64),
    ]

class rknpu_mem_create(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
        ("obj_addr", ctypes.c_uint64),
        ("dma_addr", ctypes.c_uint64),
        ("sram_size", ctypes.c_uint64),
    ]

class rknpu_mem_map(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
    ]

class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
    ]

class rknpu_submit(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("timeout", ctypes.c_uint32),
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
        ("task_counter", ctypes.c_uint32),
        ("priority", ctypes.c_int32),
        ("task_obj_addr", ctypes.c_uint64),
        ("regcfg_obj_addr", ctypes.c_uint64),
        ("task_base_addr", ctypes.c_uint64),
        ("user_data", ctypes.c_uint64),
        ("core_mask", ctypes.c_uint32),
        ("fence_fd", ctypes.c_int32),
        ("subcore_task", rknpu_subcore_task * 5),
    ]


def mem_allocate(fd, size, flags=0):
    mem_create = rknpu_mem_create(
        flags=flags, #0x10 | 0x2,  # KERNEL_MAPPING | NON_CACHEABLE
        size=size
    )
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    print(f"ret={ret}, handle={mem_create.handle}, obj_addr={mem_create.obj_addr:#x}, dma_addr={mem_create.dma_addr:#x}")
    
    # Map memory to access from userspace
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    print(f"Memory mapped at offset={mem_map.offset:#x}")
    return buf, mem_create

def submit(task_obj_addr):
    submit_struct = rknpu_submit(
        flags=0x1 | 0x2 | 0x4,  # RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG
        timeout=6000,
        task_start=0,
        task_number=1,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=1,
        fence_fd=-1,
    )
    # struct len is 5 but only 3 NPU core
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=1, task_number=0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=2, task_number=0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)
    
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

task_map, tasks_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)

tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
inputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), ctypes.POINTER(ctypes.c_uint16))
weights = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), ctypes.POINTER(ctypes.c_uint16))
outputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), ctypes.POINTER(ctypes.c_uint16))

npu_regs = [
    # DPU configuration registers
    (0x1001 << 48) | (0x000001e5 << 16) | 0x400c,  # REG_DPU_FEATURE_MODE_CFG: burst_len=15, output_mode=2, flying_mode=1
    (0x1001 << 48) | (0x48000002 << 16) | 0x4010,  # REG_DPU_DATA_FORMAT: precision=float16
    (0x1001 << 48) | (0x00070007 << 16) | 0x403c,  # REG_DPU_DATA_CUBE_CHANNEL: channel=7
    (0x1001 << 48) | (0x00000000 << 16) | 0x4030,  # REG_DPU_DATA_CUBE_WIDTH: width=0
    (0x1001 << 48) | (0x108202c0 << 16) | 0x4070,  # REG_DPU_EW_CFG: ew_cvt_type, ew_data_mode, ew_alu_algo
    
    # RDMA configuration registers
    (0x2001 << 48) | (0x00000000 << 16) | 0x500c,  # REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH: width=0
    (0x2001 << 48) | (0x00000000 << 16) | 0x5010,  # REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT: height=0
    (0x2001 << 48) | (0x00000007 << 16) | 0x5014,  # REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL: channel=7
    (0x2001 << 48) | (0x40000008 << 16) | 0x5034,  # REG_DPU_RDMA_RDMA_ERDMA_CFG: erdma_data_mode=1, erdma_data_size=2
    
    # DMA address registers (dynamic)
    (0x1001 << 48) | ((output_mem_create.dma_addr & 0xFFFFFFFF) << 16) | 0x4020,  # REG_DPU_DST_BASE_ADDR
    (0x2001 << 48) | ((input_mem_create.dma_addr & 0xFFFFFFFF) << 16)  | 0x5018,  # REG_DPU_RDMA_RDMA_SRC_BASE_ADDR
    (0x2001 << 48) | ((weight_mem_create.dma_addr & 0xFFFFFFFF) << 16) | 0x5038,  # REG_DPU_RDMA_RDMA_EW_BASE_ADDR
    
    # Task completion and padding
    (0x2001 << 48) | (0x00017849 << 16) | 0x5044,  # REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG
    (0x0081 << 48) | (0x00000018 << 16) | 0x0008,  # REG_PC_OPERATION_ENABLE
]

for i in range(len(npu_regs)):
    regcmd[i] = npu_regs[i]
for i in range(8):
    inputs[i] = 3 
    weights[i] = 5

tasks[0].flags  = 0;
tasks[0].op_idx = 4;
tasks[0].enable_mask = 0x18;
tasks[0].int_mask = 0x300;
tasks[0].int_clear = 0x1ffff;
tasks[0].int_status = 0;
tasks[0].regcfg_amount = len(npu_regs)
tasks[0].regcfg_offset = 0;
tasks[0].regcmd_addr = regcmd_mem_create.dma_addr

ret = submit(tasks_mem_create.obj_addr)
print(f"SUBMIT ret={ret}")

for i in range(8):
    print(outputs[i])