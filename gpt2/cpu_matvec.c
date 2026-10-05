/* Packed GPT-2 CPU matrix-vector kernels for ARMv8.2 FP16 + NEON.
 * Weights are rounded to FP16, matching the ANE/NumPy weight contract;
 * activations and accumulation remain FP32. No activation quantization.
 */
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

int gpt2_matvec_abi(void) { return 2; }
int gpt2_matvec_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

static inline void table16(uint8x16_t q, float16_t scale,
                           const float32x4_t *x, float32x4_t *s) {
    /* All integer values -8..7 have zero low bytes in FP16. One 16-byte
     * constant decodes both nibbles without int-to-float conversion or
     * dependent reads from a scale-indexed memory table. */
    const uint8x16_t codes = {0xc8, 0xc7, 0xc6, 0xc5, 0xc4, 0xc2, 0xc0, 0xbc,
                             0, 0x3c, 0x40, 0x42, 0x44, 0x45, 0x46, 0x47};
    uint8x16_t high = vqtbl1q_u8(codes, q), zero = vdupq_n_u8(0);
    float16x8_t w0 = vmulq_n_f16(vreinterpretq_f16_u8(vzip1q_u8(zero, high)), scale);
    float16x8_t w1 = vmulq_n_f16(vreinterpretq_f16_u8(vzip2q_u8(zero, high)), scale);
    s[0] = vfmaq_f32(s[0], vcvt_f32_f16(vget_low_f16(w0)), x[0]);
    s[1] = vfmaq_f32(s[1], vcvt_f32_f16(vget_high_f16(w0)), x[1]);
    s[0] = vfmaq_f32(s[0], vcvt_f32_f16(vget_low_f16(w1)), x[2]);
    s[1] = vfmaq_f32(s[1], vcvt_f32_f16(vget_high_f16(w1)), x[3]);
}

/* Four output rows are interleaved at the 32-weight block level. Reuse
 * each activation load across rows, while retaining the packed byte size.
 * Q4 values are decoded in registers and multiplied by their FP16 scale.
 */
void gpt2_q4(const uint8_t *w, const float *x, float *out, int rows, int cols,
             int threads) {
    int blocks = cols / 32;
    #pragma omp parallel for num_threads(threads) if(rows >= 2048 && threads > 1)
    for (int i = 0; i < rows; i += 4) {
        float32x4_t s[8];
        for (int k = 0; k < 8; ++k) s[k] = vdupq_n_f32(0);
        const uint8_t *group = w + (size_t)(i / 4) * blocks * 72;
        for (int b = 0; b < blocks; ++b) {
            float32x4_t xx[8];
            for (int k = 0; k < 8; ++k) xx[k] = vld1q_f32(x + b * 32 + k * 4);
            #pragma GCC unroll 4
            for (int r = 0; r < 4; ++r) {
                const uint8_t *p = group + b * 72 + r * 18;
                float16_t d; memcpy(&d, p, sizeof(d));
                uint8x16_t q = vld1q_u8(p + 2);
                table16(vandq_u8(q, vdupq_n_u8(15)), d, xx, s + r * 2);
                table16(vshrq_n_u8(q, 4), d, xx + 4, s + r * 2);
            }
        }
        for (int r = 0; r < 4 && i + r < rows; ++r)
            out[i + r] = vaddvq_f32(vaddq_f32(s[r * 2], s[r * 2 + 1]));
    }
}

static inline void half16(int8x16_t q, const float32x4_t *x, float16_t d, float32x4_t *s) {
    float16x8_t w0 = vmulq_n_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(q))), d);
    float16x8_t w1 = vmulq_n_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(q))), d);
    s[0] = vfmaq_f32(s[0], vcvt_f32_f16(vget_low_f16(w0)), x[0]);
    s[1] = vfmaq_f32(s[1], vcvt_f32_f16(vget_high_f16(w0)), x[1]);
    s[0] = vfmaq_f32(s[0], vcvt_f32_f16(vget_low_f16(w1)), x[2]);
    s[1] = vfmaq_f32(s[1], vcvt_f32_f16(vget_high_f16(w1)), x[3]);
}

void gpt2_q8(const uint8_t *w, const float *x, float *out, int rows, int cols, int threads) {
    int blocks = cols / 32;
    #pragma omp parallel for num_threads(threads) if(rows >= 2048 && threads > 1)
    for (int i = 0; i < rows; i += 4) {
        float32x4_t s[8];
        for (int k = 0; k < 8; ++k) s[k] = vdupq_n_f32(0);
        const uint8_t *group = w + (size_t)(i / 4) * blocks * 136;
        for (int b = 0; b < blocks; ++b) {
            float32x4_t xx[8];
            for (int k = 0; k < 8; ++k) xx[k] = vld1q_f32(x + b * 32 + k * 4);
            #pragma GCC unroll 4
            for (int r = 0; r < 4; ++r) {
                const uint8_t *p = group + b * 136 + r * 34;
                float16_t d; memcpy(&d, p, sizeof(d));
                half16(vld1q_s8((const int8_t *)p + 2), xx, d, s + r * 2);
                half16(vld1q_s8((const int8_t *)p + 18), xx + 4, d, s + r * 2);
            }
        }
        for (int r = 0; r < 4 && i + r < rows; ++r)
            out[i + r] = vaddvq_f32(vaddq_f32(s[r * 2], s[r * 2 + 1]));
    }
}

void gpt2_f16(const uint8_t *weights, const float *x, float *out, int rows, int cols, int threads) {
    const float16_t *w = (const float16_t *)weights;
    #pragma omp parallel for num_threads(threads) if(rows >= 2048 && threads > 1)
    for (int i = 0; i < rows; ++i) {
        float32x4_t s[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
        for (int j = 0; j < cols; j += 16) {
            float16x8_t lo = vld1q_f16(w + (size_t)i * cols + j);
            float16x8_t hi = vld1q_f16(w + (size_t)i * cols + j + 8);
            s[0] = vfmaq_f32(s[0], vcvt_f32_f16(vget_low_f16(lo)), vld1q_f32(x + j));
            s[1] = vfmaq_f32(s[1], vcvt_f32_f16(vget_high_f16(lo)), vld1q_f32(x + j + 4));
            s[2] = vfmaq_f32(s[2], vcvt_f32_f16(vget_low_f16(hi)), vld1q_f32(x + j + 8));
            s[3] = vfmaq_f32(s[3], vcvt_f32_f16(vget_high_f16(hi)), vld1q_f32(x + j + 12));
        }
        out[i] = vaddvq_f32(vaddq_f32(vaddq_f32(s[0], s[1]), vaddq_f32(s[2], s[3])));
    }
}
