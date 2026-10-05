// Adapted from qwen3.c; see LICENSE-Qwen3C and provenance/ane-template.json.
// Direct M1 ANE register programming. ABI and GEMV stream from ~/ane.
#define _GNU_SOURCE
#include "ane_matmul.h"
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linear_template.h"

enum { BATCH = 32, TD_SIZE = 504, CMD_SIZE = 2304 };
// MIT-compatible UAPI: ~/ane/kmod/uapi/drm/ane_accel.h (Eileen Yoon).
struct bo_init { uint32_t handle, pad; uint64_t size, offset; };
struct bo_free { uint32_t handle, pad; };
struct submit {
    uint64_t tsk_size;
    uint32_t td_count, td_size, handles[32], btsp_handle, pad;
};
#define BO_INIT _IOWR('d', 0x41, struct bo_init)
#define BO_FREE _IOWR('d', 0x42, struct bo_free)
#define SUBMIT _IOWR('d', 0x43, struct submit)
_Static_assert(sizeof(struct bo_init) == 24, "ANE BO ABI");
_Static_assert(sizeof(struct submit) == 152, "ANE submission ABI");

typedef struct { uint32_t handle; size_t size; void *map; } Buffer;
struct AneDevice {
    int fd;
    Buffer input, output, bootstrap, scratch, bias, constants;
    __fp16 *host_input, *host_output;
    size_t host_input_size, host_output_size;
    unsigned long long submissions;
};
struct AnePlan {
    AneDevice *device;
    int inputs, outputs, k, n;
    Buffer command, weights;
};

static void buffer_free(AneDevice *d, Buffer *b) {
    if (b->map) munmap(b->map, b->size);
    if (b->handle) {
        struct bo_free request = {.handle = b->handle};
        ioctl(d->fd, BO_FREE, &request);
    }
    memset(b, 0, sizeof(*b));
}

static int buffer_alloc(AneDevice *d, Buffer *b, size_t bytes) {
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    b->size = (bytes + page - 1) & ~(page - 1);
    struct bo_init request = {.size = b->size};
    if (ioctl(d->fd, BO_INIT, &request) || !request.handle) {
        memset(b,0,sizeof(*b));
        return 0;
    }
    b->handle = request.handle;
    void *map = mmap(NULL, b->size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     d->fd, request.offset);
    if (map == MAP_FAILED) { buffer_free(d, b); return 0; }
    b->map = map;
    memset(map,0,b->size);
    return 1;
}

static int m1_host(void) {
    char data[4096];
    int fd = open("/proc/device-tree/compatible", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    ssize_t bytes = read(fd, data, sizeof(data));
    close(fd);
    static const char compatible[] = "apple,t8103";
    return bytes > 0 && memmem(data, bytes, compatible, sizeof(compatible));
}

static int open_ane(void) {
    glob_t paths = {0};
    int fd = -1;
    const char *requested = getenv("ANE_DEVICE");
    if (glob(requested ? requested : "/dev/accel/accel*", 0, NULL, &paths))
        return -1;
    for (size_t i = 0; i < paths.gl_pathc; i++) {
        const char *name = strrchr(paths.gl_pathv[i], '/');
        char driver[PATH_MAX], resolved[PATH_MAX];
        snprintf(driver, sizeof(driver), "/sys/class/accel/%s/device/driver", name + 1);
        if (!realpath(driver, resolved)) continue;
        name = strrchr(resolved, '/');
        if (!name || strcmp(name + 1, "ane")) continue;
        fd = open(paths.gl_pathv[i], O_RDWR | O_CLOEXEC);
        if (fd >= 0) break;
    }
    globfree(&paths);
    return fd;
}

AneDevice *ane_device_open(void) {
    if (!m1_host()) {
        fprintf(stderr, "ANE: this register stream requires base M1 (T8103) Asahi Linux.\n");
        return NULL;
    }
    AneDevice *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->fd = open_ane();
    if (d->fd < 0) {
        fprintf(stderr, "ANE: no accessible ane driver device; see ~/ane/kmod/README.md.\n");
        free(d); return NULL;
    }
    if (!buffer_alloc(d, &d->input, 768 * BATCH * sizeof(__fp16)) ||
        !buffer_alloc(d, &d->output, 768 * BATCH * sizeof(__fp16)) ||
        !buffer_alloc(d, &d->bootstrap, TD_SIZE) ||
        !buffer_alloc(d, &d->scratch, 768 * BATCH * sizeof(__fp16)) ||
        !buffer_alloc(d, &d->bias, 768 * sizeof(__fp16)) ||
        !buffer_alloc(d, &d->constants, 16384)) {
        ane_device_close(d); return NULL;
    }
    return d;
}

void ane_device_close(AneDevice *d) {
    if (!d) return;
    buffer_free(d, &d->constants);
    buffer_free(d, &d->bias);
    buffer_free(d, &d->scratch);
    buffer_free(d, &d->bootstrap);
    buffer_free(d, &d->output);
    buffer_free(d, &d->input);
    free(d->host_output);
    free(d->host_input);
    close(d->fd);
    free(d);
}

void ane_plan_free(AnePlan *p) {
    if (!p) return;
    buffer_free(p->device, &p->weights);
    buffer_free(p->device, &p->command);
    free(p);
}

static int ensure_buffer(AneDevice *d, Buffer *b, size_t bytes) {
    if (b->size >= bytes) return 1;
    buffer_free(d, b);
    return buffer_alloc(d, b, bytes);
}

static int ensure_output(AneDevice *d, size_t bytes) {
    if (!ensure_buffer(d,&d->output,bytes)) return 0;
    if (d->host_output_size>=bytes) return 1;
    void *p=realloc(d->host_output,bytes);
    if (!p) return 0;
    d->host_output=p; d->host_output_size=bytes;
    return 1;
}

static int ensure_input(AneDevice *d, size_t bytes) {
    if (!ensure_buffer(d,&d->input,bytes)) return 0;
    if (d->host_input_size>=bytes) return 1;
    void *p=realloc(d->host_input,bytes);
    if (!p) return 0;
    d->host_input=p; d->host_input_size=bytes;
    return 1;
}

// Read whole cache lines from the driver's uncached output mappings.
static void read_output(void *dst, const void *src, size_t bytes) {
    uint8_t *out=dst;
    const uint8_t *in=src;
    for (size_t i=0;i<bytes;i+=64) {
        // Keep the compiler from splitting this into individual load pairs.
        asm volatile("ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%0]\n\t"
                     "st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%1]"
                     : : "r"(in+i), "r"(out+i) : "v0","v1","v2","v3","memory");
    }
}

AnePlan *ane_plan_create_f16(AneDevice *d, const uint16_t *w,
                             int inputs, int outputs) {
    if (!d || !w || inputs <= 0 || outputs <= 0 ||
        inputs > 32736 || outputs > 32736 ||
        (int64_t)inputs * outputs > INT_MAX) return NULL;
    AnePlan *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->device = d; p->inputs = inputs; p->outputs = outputs;
    p->k = (inputs + 31) & ~31; p->n = (outputs + 31) & ~31;
    if (!ensure_input(d, (size_t)p->k * BATCH * 2) ||
        !ensure_buffer(d, &d->scratch, (size_t)p->k * BATCH * 2) ||
        !ensure_output(d, (size_t)p->n * BATCH * 2) ||
        !ensure_buffer(d, &d->bias, (size_t)p->n * 2) ||
        !buffer_alloc(d, &p->command, CMD_SIZE + 1) ||
        !buffer_alloc(d, &p->weights, (size_t)p->k * p->n * 2)) {
        ane_plan_free(p); return NULL;
    }
    uint32_t *program = p->command.map;
    memcpy(program, ane_linear_template, CMD_SIZE);
    for (size_t i=0;i<sizeof(ane_k_patches)/sizeof(ane_k_patches[0]);i++) {
        AneDimensionPatch a=ane_k_patches[i];
        program[a.offset/4] += (p->k - 768) * a.scale;
    }
    for (size_t i=0;i<sizeof(ane_n_patches)/sizeof(ane_n_patches[0]);i++) {
        AneDimensionPatch a=ane_n_patches[i];
        program[a.offset/4] += (p->n - 768) * a.scale;
    }
    // The runtime matrix occupies K*N FP16 values in the TileDMA source.
    program[0x37c/4] = program[0x384/4] = program[0x388/4] = p->k*p->n*2;
    __fp16 *packed = p->weights.map;
    // The dynamic graph transposes the input activations into the convolution
    // coefficient stream. The model matrix stays resident in [K,N] order.
    int valid=1;
    #pragma omp parallel for num_threads(4) reduction(&:valid)
    for (int k=0;k<inputs;k++) {
        for (int n=0;n<outputs;n++) {
            size_t index=(size_t)n*inputs+k;
            float value=((const __fp16 *)w)[index];
            valid &= isfinite(value) && fabsf(value)<=65504;
            packed[(size_t)k*p->n+n]=(__fp16)value;
        }
    }
    if (!valid) { ane_plan_free(p); return NULL; }
    return p;
}

int ane_plan_run_batch(AnePlan *p, const float *input, float *output, int rows) {
    if (!p || !input || !output || rows<1 || rows>BATCH) return 0;
    AneDevice *d=p->device;
    __fp16 *source=d->host_input, *result=d->host_output;
    memset(source,0,(size_t)p->k*BATCH*2);
    for (int row=0;row<rows;row++) {
        for (int k=0;k<p->inputs;k++) {
            float x=input[(size_t)row*p->inputs+k];
            if (!isfinite(x) || fabsf(x)>65504) return 0;
            source[(size_t)row*p->k+k]=(__fp16)x;
        }
    }
    memcpy(d->input.map,source,(size_t)p->k*BATCH*2);
    // Detect a successful ioctl that did not write its advertised output.
    uint16_t *bits=(uint16_t *)result;
    for (int i=0;i<rows*p->n;i++) bits[i]=0x7e00;
    memcpy(d->output.map,result,(size_t)rows*p->n*2);
    memcpy(d->bootstrap.map,p->command.map,TD_SIZE);
    uint32_t *header=d->bootstrap.map;
    header[0]=(header[0]&~(0xffu<<16))|(0x40u<<16);
    struct submit request={.tsk_size=CMD_SIZE,.td_count=4,.td_size=TD_SIZE,
                           .btsp_handle=d->bootstrap.handle};
    request.handles[0]=p->command.handle;
    request.handles[2]=d->constants.handle;
    request.handles[3]=d->scratch.handle;
    request.handles[4]=d->input.handle;
    request.handles[5]=p->weights.handle;
    request.handles[6]=d->bias.handle;
    request.handles[7]=d->output.handle;
    if (ioctl(d->fd,SUBMIT,&request)) {
        fprintf(stderr,"ANE submission failed: %s\n",strerror(errno));
        return 0;
    }
    d->submissions++;
    read_output(result,d->output.map,(size_t)rows*p->n*2);
    // All input rows are in the device buffer before writing output: alias safe.
    for (int row=0;row<rows;row++) {
        for (int n=0;n<p->outputs;n++) {
            float value=result[(size_t)row*p->n+n];
            if (!isfinite(value)) return 0;
            output[(size_t)row*p->outputs+n]=value;
        }
    }
    return 1;
}

int ane_plan_run(AnePlan *p, const float *input, float *output) {
    return ane_plan_run_batch(p,input,output,1);
}

unsigned long long ane_device_submissions(const AneDevice *d) {
    return d ? d->submissions : 0;
}
