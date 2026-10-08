#ifndef QWEN35_ANE_MATMUL_H
#define QWEN35_ANE_MATMUL_H
#include <stdint.h>
#include <stddef.h>
typedef struct AneDevice AneDevice;
typedef struct AnePlan AnePlan;
typedef struct {
    uint64_t pack_ns, write_ns, ioctl_ns, read_ns, unpack_ns;
    uint32_t read_threads_max;
} AneTimings;
AneDevice *ane_device_open(void);
// Borrow the guarded M1 ANE fd; the device retains ownership.
int ane_device_fd(const AneDevice *device);
void ane_device_profile(AneDevice *device, int enabled);
void ane_device_read_threads(AneDevice *device, int threads);
AneTimings ane_device_timings(const AneDevice *device);
// Copy whole cache lines; returns the actual number of read workers, or 0.
int ane_copy_uncached(void *dst, const void *src, size_t bytes, int threads);
void ane_device_close(AneDevice *device);
AnePlan *ane_plan_create_f16(AneDevice *device, const uint16_t *weights,
                            int inputs, int outputs);
int ane_plan_run_batch(AnePlan *plan, const float *input, float *output, int rows);
int ane_plan_run_compensated(AnePlan *plan, const float *input, float *output,
                             int partitions, const float *gains, int replicas,
                             float input_limit);
int ane_plan_run(AnePlan *plan, const float *input, float *output);
void ane_plan_free(AnePlan *plan);
unsigned long long ane_device_submissions(const AneDevice *device);
#endif
