/* Mirai asymmetric W4, group 32. FP32 scales/activations/accumulation.
 * Packed nibble bytes stay row-major; no dense weight expansion at inference.
 */
#include <math.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

int qwen35_abi(void) { return 1; }

void qwen35_w4(const uint8_t *, const float *, const uint8_t *, const float *,
               float *, int, int, int);

void qwen35_w4_serial(const uint8_t *weights, const float *scales,
                      const uint8_t *zeros, const float *x, float *out,
                      int rows, int cols, int threads) {
    int groups = cols / 32;
    #pragma omp parallel for num_threads(threads) if(rows >= 1024 && threads > 1)
    for (int row = 0; row < rows; ++row) {
        float sum = 0;
        for (int k = 0; k < cols; ++k) {
            size_t p = (size_t)row * groups + k / 32;
            uint8_t packed = weights[(size_t)row * (cols / 2) + k / 2];
            int code = (packed >> ((k & 1) * 4)) & 15;
            float value = scales[p] * code - scales[p] * zeros[p];
            sum += value * x[k];
        }
        out[row] = sum;
    }
}

static float bfloat(float x) {
    uint32_t bits; memcpy(&bits, &x, 4);
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
    memcpy(&x, &bits, 4); return x;
}

void qwen35_rms_bf16(const float *x, const float *scale, float *out,
                     int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        float sum = 0;
        for (int j = 0; j < cols; ++j) { float v = x[row * cols + j]; sum += v * v; }
        float inv = 1.f / sqrtf(sum / cols + 1e-6f);
        for (int j = 0; j < cols; ++j)
            out[row * cols + j] = bfloat(bfloat(x[row * cols + j] * inv) * bfloat(scale[j] + 1.f));
    }
}

void qwen35_rope_bf16(float *x, int heads, int position) {
    for (int j = 0; j < 32; ++j) {
        float inverse = 1.f / powf(1e7f, (float)(2 * j) / 64.f);
        float angle = position * inverse, c = cosf(angle), s = sinf(angle);
        for (int h = 0; h < heads; ++h) {
            float a = x[h * 256 + j], b = x[h * 256 + j + 32];
            x[h * 256 + j] = bfloat(a * c - b * s);
            x[h * 256 + j + 32] = bfloat(b * c + a * s);
        }
    }
}

void qwen35_conv_bf16(float *projection, float *state, const float *weights) {
    for (int channel = 0; channel < 6144; ++channel) {
        float x = projection[channel], acc = 0;
        for (int tap = 0; tap < 3; ++tap) acc += state[channel * 3 + tap] * weights[channel * 4 + tap];
        acc += x * weights[channel * 4 + 3];
        projection[channel] = bfloat(acc / (1.f + expf(-acc)));
        state[channel * 3] = state[channel * 3 + 1];
        state[channel * 3 + 1] = state[channel * 3 + 2];
        state[channel * 3 + 2] = x;
    }
}

void qwen35_mlp_bf16(const float *input, float *out) {
    for (int j = 0; j < 3584; ++j) {
        float gate = input[j + 3584];
        gate = bfloat(gate / (1.f + expf(-gate)));
        out[j] = bfloat(input[j] * gate);
    }
}

void qwen35_attention_bf16(const float *q, const float *keys, const float *values,
                           const float *gate, float *out, int length) {
    for (int h = 0; h < 8; ++h) {
        float query[256], acc[256] = {0}, maximum = -INFINITY, total = 0;
        for (int j = 0; j < 256; ++j) query[j] = q[h * 256 + j] / 16.f;
        for (int t = 0; t < length; ++t) {
            const float *k = keys + (size_t)t * 512 + (h / 4) * 256;
            const float *v = values + (size_t)t * 512 + (h / 4) * 256;
            float score = 0;
            for (int j = 0; j < 256; ++j) score += query[j] * k[j];
            float next = fmaxf(maximum, score), factor = expf(maximum - next), weight = expf(score - next);
            total = total * factor + weight;
            for (int j = 0; j < 256; ++j) acc[j] = acc[j] * factor + weight * v[j];
            maximum = next;
        }
        for (int j = 0; j < 256; ++j) {
            float output = bfloat(acc[j] / total);
            float sigmoid = 1.f / (1.f + expf(-gate[h * 256 + j]));
            out[h * 256 + j] = bfloat(output * sigmoid);
        }
    }
}

void qwen35_w4_dot(const uint8_t *weights, const float *scales,
                   const uint8_t *zeros, const float *x, float *out,
                   int rows, int cols, int threads) {
#ifdef __ARM_FEATURE_DOTPROD
    int groups = cols / 32;
    int8_t hi[cols], lo[cols];
    float divisors[groups];
    for (int g = 0; g < groups; ++g) {
        float maximum = 0;
        for (int j = 0; j < 32; ++j) maximum = fmaxf(maximum, fabsf(x[g * 32 + j]));
        float divisor = maximum / 32639.f;
        divisors[g] = divisor;
        float inverse = maximum > 0 ? 32639.f / maximum : 0;
        if (!isfinite(inverse) || (maximum > 0 && divisor == 0)) {
            qwen35_w4(weights, scales, zeros, x, out, rows, cols, threads);
            return;
        }
        for (int j = 0; j < 32; ++j) {
            int value = (int)nearbyintf(x[g * 32 + j] * inverse);
            int upper = (int)floorf((value + 128.f) / 256.f);
            hi[g * 32 + j] = (int8_t)upper;
            lo[g * 32 + j] = (int8_t)(value - upper * 256);
        }
    }
    #pragma omp parallel for num_threads(threads) if(rows >= 1024 && threads > 1)
    for (int row = 0; row < rows; ++row) {
        float sum = 0;
        for (int g = 0; g < groups; ++g) {
            size_t p = (size_t)row * groups + g;
            uint8x16_t packed = vld1q_u8(weights + (size_t)row * (cols / 2) + g * 16);
            uint8x16_t a = vandq_u8(packed, vdupq_n_u8(15)), b = vshrq_n_u8(packed, 4);
            uint8x16_t zp = vdupq_n_u8(zeros[p]);
            int8x16_t w0 = vreinterpretq_s8_u8(vsubq_u8(vzip1q_u8(a, b), zp));
            int8x16_t w1 = vreinterpretq_s8_u8(vsubq_u8(vzip2q_u8(a, b), zp));
            int32x4_t dh = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(hi + g * 32));
            dh = vdotq_s32(dh, w1, vld1q_s8(hi + g * 32 + 16));
            int32x4_t dl = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(lo + g * 32));
            dl = vdotq_s32(dl, w1, vld1q_s8(lo + g * 32 + 16));
            int dot = vaddvq_s32(dh) * 256 + vaddvq_s32(dl);
            sum += (float)dot * (divisors[g] * scales[p]);
        }
        out[row] = sum;
    }
#else
    qwen35_w4(weights, scales, zeros, x, out, rows, cols, threads);
#endif
}

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
        float q[128], k[128], o[128], qs = 0, ks = 0;
        for (int j = 0; j < 128; ++j) {
            q[j] = projection[h * 128 + j];
            k[j] = projection[2048 + h * 128 + j];
            qs += q[j] * q[j]; ks += k[j] * k[j];
        }
        float qi = 1.f / sqrtf(qs + 1e-6f), qscale = 1.f / sqrtf(128.f);
        float ki = 1.f / sqrtf(ks + 1e-6f), kq = 0;
        for (int j = 0; j < 128; ++j) { q[j] *= qi; q[j] *= qscale; k[j] *= ki; kq += k[j] * q[j]; }
        float beta = 1.f / (1.f + expf(-projection[8192 + h]));
        float a = projection[8208 + h] + dt_bias[h];
        float sp = a > 20.f ? a : logf(1.f + expf(a));
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
            float gate = z / (1.f + expf(-z));
            out[h * 128 + i] = o[i] * inv * norm[i] * gate;
        }
    }
}
