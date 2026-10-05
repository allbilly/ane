#ifndef QWEN35_ANE_MATMUL_H
#define QWEN35_ANE_MATMUL_H
#include <stdint.h>
typedef struct AneDevice AneDevice;
typedef struct AnePlan AnePlan;
AneDevice *ane_device_open(void);
void ane_device_close(AneDevice *device);
AnePlan *ane_plan_create_f16(AneDevice *device, const uint16_t *weights,
                            int inputs, int outputs);
int ane_plan_run_batch(AnePlan *plan, const float *input, float *output, int rows);
int ane_plan_run(AnePlan *plan, const float *input, float *output);
void ane_plan_free(AnePlan *plan);
unsigned long long ane_device_submissions(const AneDevice *device);
#endif
