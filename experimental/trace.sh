sudo bpftrace -e '
struct drm_ane_bo_init {
    unsigned int handle;
    unsigned int pad;
    unsigned long long size;
    unsigned long long offset;
};

struct drm_ane_bo_free {
    unsigned int handle;
    unsigned int pad;
};

struct drm_ane_submit {
    unsigned long long tsk_size;
    unsigned int td_count;
    unsigned int td_size;
    unsigned int handles[32];
    unsigned int btsp_handle;
    unsigned int pad;
};

// Trace ANE BO_INIT operations
tracepoint:syscalls:sys_enter_ioctl 
/args->cmd == 0xc0186441/ 
{
    $b = (struct drm_ane_bo_init *)args->arg;
    printf("ANE BO_INIT: fd=%d, handle=%u, size=0x%llx, offset=0x%llx\n", 
           args->fd, $b->handle, $b->size, $b->offset);
}

// Trace ANE BO_FREE operations
tracepoint:syscalls:sys_enter_ioctl 
/args->cmd == 0xc0086442/ 
{
    $b = (struct drm_ane_bo_free *)args->arg;
    printf("ANE BO_FREE: fd=%d, handle=%u\n", args->fd, $b->handle);
}

// Trace ANE SUBMIT operations
tracepoint:syscalls:sys_enter_ioctl 
/args->cmd == 0xc0986443/ 
{
    $s = (struct drm_ane_submit *)args->arg;
    printf("ANE SUBMIT: fd=%d, tsk_size=%llu, td_count=%u, td_size=%u, btsp_handle=%u\n", 
           args->fd, $s->tsk_size, $s->td_count, $s->td_size, $s->btsp_handle);
}
' &
sleep 2
python run.py hwx/sum.ane

sudo pkill bpftrace
