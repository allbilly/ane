// Adapted from qwen3.c; see LICENSE-Qwen3C and provenance/ane-template.json.
// Direct M1 ANE register programming. ABI and GEMV stream from ~/ane.
#define _GNU_SOURCE
#include "ane_matmul.h"
#include <arm_neon.h>
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
#include <time.h>
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
    int profiling;
    AneTimings timings;
};

static uint64_t profile_ns(const AneDevice *d) {
    if (!d->profiling) return 0;
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec*1000000000ull + (uint64_t)t.tv_nsec;
}

void ane_device_profile(AneDevice *d, int enabled) { if (d) d->profiling = !!enabled; }
AneTimings ane_device_timings(const AneDevice *d) {
    return d ? d->timings : (AneTimings){0};
}
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

static int submit_rows(AnePlan *p, int rows) {
    AneDevice *d=p->device;
    __fp16 *source=d->host_input, *result=d->host_output;
    uint64_t start=profile_ns(d);
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
    d->timings.write_ns += profile_ns(d)-start;
    start=profile_ns(d);
    if (ioctl(d->fd,SUBMIT,&request)) {
        fprintf(stderr,"ANE submission failed: %s\n",strerror(errno));
        return 0;
    }
    d->submissions++;
    d->timings.ioctl_ns += profile_ns(d)-start;
    start=profile_ns(d);
    read_output(result,d->output.map,(size_t)rows*p->n*2);
    d->timings.read_ns += profile_ns(d)-start;
    return 1;
}

int ane_plan_run_batch(AnePlan *p, const float *input, float *output, int rows) {
    if (!p || !input || !output || rows<1 || rows>BATCH) return 0;
    AneDevice *d=p->device;
    __fp16 *source=d->host_input, *result=d->host_output;
    uint64_t start=profile_ns(d);
    memset(source,0,(size_t)p->k*BATCH*2);
    for (int row=0;row<rows;row++) {
        int k=0;
        for (;k+8<=p->inputs;k+=8) {
            float32x4_t a=vld1q_f32(input+(size_t)row*p->inputs+k);
            float32x4_t b=vld1q_f32(input+(size_t)row*p->inputs+k+4);
            uint32x4_t valid=vandq_u32(vcleq_f32(vabsq_f32(a),vdupq_n_f32(65504)),
                                     vcleq_f32(vabsq_f32(b),vdupq_n_f32(65504)));
            if (!vminvq_u32(valid)) return 0;
            vst1q_f16(source+(size_t)row*p->k+k,vcombine_f16(vcvt_f16_f32(a),vcvt_f16_f32(b)));
        }
        for (;k<p->inputs;k++) {
            float x=input[(size_t)row*p->inputs+k];
            if (!isfinite(x) || fabsf(x)>65504) return 0;
            source[(size_t)row*p->k+k]=(__fp16)x;
        }
    }
    d->timings.pack_ns += profile_ns(d)-start;
    if (!submit_rows(p,rows)) return 0;
    start=profile_ns(d);
    // All input rows are in the device buffer before writing output: alias safe.
    for (int row=0;row<rows;row++) {
        int n=0;
        for (;n+8<=p->outputs;n+=8) {
            float16x8_t h=vld1q_f16(result+(size_t)row*p->n+n);
            uint16x8_t exponent=vandq_u16(vreinterpretq_u16_f16(h),vdupq_n_u16(0x7c00));
            if (vmaxvq_u16(vceqq_u16(exponent,vdupq_n_u16(0x7c00)))) return 0;
            vst1q_f32(output+(size_t)row*p->outputs+n,vcvt_f32_f16(vget_low_f16(h)));
            vst1q_f32(output+(size_t)row*p->outputs+n+4,vcvt_f32_f16(vget_high_f16(h)));
        }
        for (;n<p->outputs;n++) {
            float value=result[(size_t)row*p->n+n];
            if (!isfinite(value)) return 0;
            output[(size_t)row*p->outputs+n]=value;
        }
    }
    d->timings.unpack_ns += profile_ns(d)-start;
    return 1;
}

// Use spare batch rows to compute two FP16 activation planes per K partition.
// Replicas with distinct gains average different FP16 output rounding grids.
// Every matrix product still runs on ANE, once; CPU combines partial outputs.
int ane_plan_run_compensated(AnePlan *p, const float *input, float *output,
                             int partitions, const float *gains, int replicas,
                             float input_limit) {
    if (!p || !input || !output || !gains || partitions<1 || replicas<1 ||
        partitions>BATCH/2 || replicas>BATCH/(2*partitions) ||
        p->inputs%(32*partitions) || !isfinite(input_limit) ||
        input_limit<=0 || input_limit>65504) return 0;
    int width=p->inputs/partitions, rows=2*partitions*replicas;
    AneDevice *d=p->device;
    __fp16 *source=d->host_input, *result=d->host_output;
    float factors[BATCH]={0}, residual[width];
    int restore_shifts[BATCH]={0}, active[BATCH]={0}, slow_restore=0;
    memset(source,0,(size_t)p->k*BATCH*2);
    for (int replica=0;replica<replicas;replica++) {
        float gain=gains[replica];
        if (!isfinite(gain) || gain<1 || gain>=2) return 0;
        for (int part=0;part<partitions;part++) {
            int start=part*width, row=2*(replica*partitions+part);
            float peak=0;
            for (int k=0;k<width;k++) {
                float x=input[start+k];
                if (!isfinite(x)) return 0;
                peak=fmaxf(peak,fabsf(x));
            }
            if (!peak) continue;
            int shift=(int)floor(log2((double)input_limit)-log2((double)peak*gain));
            float low_peak=0;
            float scale=scalbnf(1.,shift);
            int k=0;
            if (isfinite(scale) && scale>0) {
                float32x4_t maximum=vdupq_n_f32(0);
                for (;k+8<=width;k+=8) {
                    float32x4_t a=vmulq_n_f32(vmulq_n_f32(vld1q_f32(input+start+k),scale),gain);
                    float32x4_t b=vmulq_n_f32(vmulq_n_f32(vld1q_f32(input+start+k+4),scale),gain);
                    float16x4_t ha=vcvt_f16_f32(a), hb=vcvt_f16_f32(b);
                    vst1q_f16(source+(size_t)row*p->k+start+k,vcombine_f16(ha,hb));
                    float32x4_t ra=vsubq_f32(a,vcvt_f32_f16(ha)), rb=vsubq_f32(b,vcvt_f32_f16(hb));
                    vst1q_f32(residual+k,ra); vst1q_f32(residual+k+4,rb);
                    maximum=vmaxq_f32(maximum,vmaxq_f32(vabsq_f32(ra),vabsq_f32(rb)));
                }
                low_peak=vmaxvq_f32(maximum);
            }
            for (;k<width;k++) {
                float x=scalbnf(input[start+k],shift)*gain;
                __fp16 high=(__fp16)x;
                source[(size_t)row*p->k+start+k]=high;
                residual[k]=x-(float)high;
                low_peak=fmaxf(low_peak,fabsf(residual[k]));
            }
            factors[row]=(float)ldexp(1.,-shift);
            restore_shifts[row]=-shift;
            active[row]=1;
            slow_restore |= factors[row]==0 || !isfinite(factors[row]);
            if (low_peak) {
                int low_shift=(int)floor(log2((double)input_limit)-log2((double)low_peak));
                float low_scale=scalbnf(1.,low_shift);
                int k=0;
                if (isfinite(low_scale) && low_scale>0) {
                    for (;k+8<=width;k+=8) {
                        float32x4_t a=vmulq_n_f32(vld1q_f32(residual+k),low_scale);
                        float32x4_t b=vmulq_n_f32(vld1q_f32(residual+k+4),low_scale);
                        vst1q_f16(source+(size_t)(row+1)*p->k+start+k,
                                  vcombine_f16(vcvt_f16_f32(a),vcvt_f16_f32(b)));
                    }
                }
                for (;k<width;k++)
                    source[(size_t)(row+1)*p->k+start+k]=(__fp16)scalbnf(residual[k],low_shift);
                factors[row+1]=(float)ldexp(1.,-shift-low_shift);
                restore_shifts[row+1]=-shift-low_shift;
                active[row+1]=1;
                slow_restore |= factors[row+1]==0 || !isfinite(factors[row+1]);
            }
        }
    }
    if (!submit_rows(p,rows)) return 0;
    // For extreme FP32 values, materializing 2^-shift itself can underflow
    // although the restored dot product is representable. Scale each result
    // directly in that rare case, as the public batch wrapper does with ldexp.
    if (slow_restore) {
        for (int n=0;n<p->outputs;n++) {
            float sum=0;
            for (int replica=0;replica<replicas;replica++) {
                float partial=0;
                for (int row=replica*2*partitions;row<(replica+1)*2*partitions;row++) {
                    float value=result[(size_t)row*p->n+n];
                    if (!isfinite(value)) return 0;
                    if (active[row]) partial+=scalbnf(value,restore_shifts[row]);
                }
                sum+=partial/gains[replica];
            }
            output[n]=sum/replicas;
            if (!isfinite(output[n])) return 0;
        }
        return 1;
    }
    // Convert and reduce eight outputs at a time, avoiding a rows*N FP32 copy.
    for (int n=0;n<p->outputs;n+=8) {
        float32x4_t lo=vdupq_n_f32(0), hi=vdupq_n_f32(0);
        for (int replica=0;replica<replicas;replica++) {
            float32x4_t part_lo=vdupq_n_f32(0), part_hi=vdupq_n_f32(0);
            for (int row=replica*2*partitions;row<(replica+1)*2*partitions;row++) {
                float16x8_t h=vld1q_f16(result+(size_t)row*p->n+n);
                uint16x8_t exponent=vandq_u16(vreinterpretq_u16_f16(h),vdupq_n_u16(0x7c00));
                if (vmaxvq_u16(vceqq_u16(exponent,vdupq_n_u16(0x7c00)))) return 0;
                part_lo=vaddq_f32(part_lo,vmulq_n_f32(vcvt_f32_f16(vget_low_f16(h)),factors[row]));
                part_hi=vaddq_f32(part_hi,vmulq_n_f32(vcvt_f32_f16(vget_high_f16(h)),factors[row]));
            }
            lo=vaddq_f32(lo,vdivq_f32(part_lo,vdupq_n_f32(gains[replica])));
            hi=vaddq_f32(hi,vdivq_f32(part_hi,vdupq_n_f32(gains[replica])));
        }
        lo=vdivq_f32(lo,vdupq_n_f32(replicas));
        hi=vdivq_f32(hi,vdupq_n_f32(replicas));
        if (n+8<=p->outputs) {
            vst1q_f32(output+n,lo);
            vst1q_f32(output+n+4,hi);
        } else {
            float tail[8];
            vst1q_f32(tail,lo); vst1q_f32(tail+4,hi);
            for (int j=n;j<p->outputs;j++) output[j]=tail[j-n];
        }
    }
    for (int n=0;n<p->outputs;n++) if (!isfinite(output[n])) return 0;
    return 1;
}

int ane_plan_run(AnePlan *p, const float *input, float *output) {
    return ane_plan_run_batch(p,input,output,1);
}

unsigned long long ane_device_submissions(const AneDevice *d) {
    return d ? d->submissions : 0;
}
