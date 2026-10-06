#ifndef QWEN35_ANE_MATMUL_H
#define QWEN35_ANE_MATMUL_H
#include <stdint.h>
typedef struct AneDevice AneDevice;
typedef struct AnePlan AnePlan;
typedef struct {
    uint64_t pack_ns, write_ns, ioctl_ns, read_ns, unpack_ns;
} AneTimings;
AneDevice *ane_device_open(void);
void ane_device_profile(AneDevice *device, int enabled);
AneTimings ane_device_timings(const AneDevice *device);
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
