/* Mirai asymmetric W4, group 32. FP32 scales/activations/accumulation.
 * Packed nibble bytes stay row-major; no dense weight expansion at inference.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

int qwen35_abi(void) { return 1; }

void qwen35_w4(const uint8_t *weights, const float *scales,
               const uint8_t *zeros, const float *x, float *out,
               int rows, int cols, int threads) {
    const int groups = cols / 32;
    #pragma omp parallel for num_threads(threads) if(rows >= 1024 && threads > 1)
    for (int row = 0; row < rows; ++row) {
#ifdef __aarch64__
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        for (int g = 0; g < groups; ++g) {
            size_t parameter = (size_t)row * groups + g;
            float scale = scales[parameter], bias = -scale * zeros[parameter];
            uint8x16_t p = vld1q_u8(weights + (size_t)row * (cols / 2) + g * 16);
            uint8x16_t lo = vandq_u8(p, vdupq_n_u8(15)), hi = vshrq_n_u8(p, 4);
            uint8x16_t codes[2] = {vzip1q_u8(lo, hi), vzip2q_u8(lo, hi)};
            for (int half = 0; half < 2; ++half) {
                uint16x8_t a = vmovl_u8(vget_low_u8(codes[half]));
                uint16x8_t b = vmovl_u8(vget_high_u8(codes[half]));
                float32x4_t w[4] = {
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(a))),
                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(a))),
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(b))),
                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(b)))};
                for (int v = 0; v < 4; ++v) {
                    /* Separate multiply/add matches the stored IntSpec. */
                    w[v] = vaddq_f32(vmulq_n_f32(w[v], scale), vdupq_n_f32(bias));
                    float32x4_t activation = vld1q_f32(x + g * 32 + half * 16 + v * 4);
                    if (v & 1) acc1 = vfmaq_f32(acc1, w[v], activation);
                    else acc0 = vfmaq_f32(acc0, w[v], activation);
                }
            }
        }
        out[row] = vaddvq_f32(vaddq_f32(acc0, acc1));
#else
        float acc = 0;
        for (int k = 0; k < cols; ++k) {
            size_t p = (size_t)row * groups + k / 32;
            uint8_t packed = weights[(size_t)row * (cols / 2) + k / 2];
            int code = (packed >> ((k & 1) * 4)) & 15;
            acc += (scales[p] * code - scales[p] * zeros[p]) * x[k];
        }
        out[row] = acc;
#endif
    }
}

void qwen35_gdn(float *state, const float *projection, const float *a_log,
                const float *dt_bias, const float *norm, float *out) {
    /* [q2048, k2048, v2048, z2048, beta16, a16]; state [head,v,k]. */
    for (int h = 0; h < 16; ++h) {
        float q[128], k[128], o[128], qs = 1e-6f, ks = 1e-6f;
        for (int j = 0; j < 128; ++j) {
            q[j] = projection[h * 128 + j];
            k[j] = projection[2048 + h * 128 + j];
            qs += q[j] * q[j]; ks += k[j] * k[j];
        }
        float qi = 1.f / sqrtf(qs) / sqrtf(128.f), ki = 1.f / sqrtf(ks), kq = 0;
        for (int j = 0; j < 128; ++j) { q[j] *= qi; k[j] *= ki; kq += k[j] * q[j]; }
        float beta = 1.f / (1.f + expf(-projection[8192 + h]));
        float a = projection[8208 + h] + dt_bias[h];
        float sp = a > 20.f ? a : log1pf(expf(a));
        float decay = expf(-expf(a_log[h]) * sp), sumsq = 0;
        for (int i = 0; i < 128; ++i) {
            float *s = state + ((size_t)h * 128 + i) * 128;
            float sk = 0, sq = 0;
            for (int j = 0; j < 128; ++j) { sk += s[j] * k[j]; sq += s[j] * q[j]; }
            float delta = beta * (projection[4096 + h * 128 + i] - decay * sk);
            o[i] = decay * sq + delta * kq;
            for (int j = 0; j < 128; ++j) s[j] = decay * s[j] + delta * k[j];
            sumsq += o[i] * o[i];
        }
        float inv = 1.f / sqrtf(sumsq / 128.f + 1e-6f);
        for (int i = 0; i < 128; ++i) {
            float z = projection[6144 + h * 128 + i];
            out[h * 128 + i] = o[i] * inv * norm[i] * z / (1.f + expf(-z));
        }
    }
}
