#include "asahi_encoder.h"
#include "ggml-backend.h"
extern "C" {
#include "ane_matmul.h"
}
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct WhisperAsahi::Impl {
    struct Plan { AnePlan * handle; int k, n; float limit; };
    AneDevice * device = nullptr;
    std::map<const ggml_tensor *, Plan> plans;
    unsigned long long projections = 0, previous_submissions = 0;
    std::vector<float> input, output;
    bool profiling = false;
    int64_t plan_us = 0, scale_us = 0, run_us = 0, restore_us = 0;
    AneTimings previous_timings{};
};

bool whisper_asahi_enabled() {
    const char * flag = std::getenv("WHISPER_ASAHI_ANE");
    if (!flag || !std::strcmp(flag, "0")) return false;
    if (std::strcmp(flag, "1")) GGML_ABORT("WHISPER_ASAHI_ANE must be 0 or 1");
    return true;
}

WhisperAsahi::WhisperAsahi() : impl(new Impl) {
    impl->device = ane_device_open();
    if (!impl->device) GGML_ABORT("Asahi ANE requested but device initialization failed");
    impl->profiling = std::getenv("WHISPER_ASAHI_PROFILE") != nullptr;
    ane_device_profile(impl->device, impl->profiling);
    std::fprintf(stderr, "ASAHI_ANE ready: encoder dense projections; CPU conv/attention/decoder\n");
}

WhisperAsahi::~WhisperAsahi() {
    for (auto & item : impl->plans) ane_plan_free(item.second.handle);
    ane_device_close(impl->device);
}

ggml_tensor * WhisperAsahi::project(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input) {
    if (weight->type != GGML_TYPE_F16 || !ggml_is_contiguous(weight) ||
        input->type != GGML_TYPE_F32 || input->nb[0] != sizeof(float) ||
        weight->ne[0] != input->ne[0] ||
        weight->ne[2] != 1 || weight->ne[3] != 1 ||
        input->ne[2] != 1 || input->ne[3] != 1 ||
        weight->ne[0] > 32736 || weight->ne[1] > 32736) {
        GGML_ABORT("Asahi encoder requires finite F16 matrices and 2D F32 activations");
    }
    ggml_tensor * sources[] = {weight, input};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, weight->ne[1], input->ne[1], 1, 1,
                          sources, 2, compute, 1, this);
}

void WhisperAsahi::compute(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith) return;
    auto & state = *static_cast<WhisperAsahi *>(userdata)->impl;
    const ggml_tensor * weight = dst->src[0], * activation = dst->src[1];
    const auto plan_start = state.profiling ? ggml_time_us() : 0;
    auto found = state.plans.find(weight);
    if (found == state.plans.end()) {
        Impl::Plan plan{};
        plan.k = int(weight->ne[0]); plan.n = int(weight->ne[1]);
        // ggml stores F16 weight rows in [output, input] order, as the existing
        // matrix API expects. Its packer preserves the checkpoint values.
        plan.handle = ane_plan_create_f16(state.device,
                    static_cast<const uint16_t *>(weight->data), plan.k, plan.n);
        if (!plan.handle) GGML_ABORT("Asahi encoder matrix preparation failed");
        std::vector<float> row(plan.k);
        double bound = 1.;
        for (int n = 0; n < plan.n; ++n) {
            ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(weight->data) + n*plan.k,
                                 row.data(), plan.k);
            double total = 0.;
            for (float value : row) total += std::fabs(value);
            bound = std::max(bound, total);
        }
        plan.limit = float(std::min(64., 32768. / bound));
        found = state.plans.emplace(weight, plan).first;
    }
    const auto & plan = found->second;
    if (state.profiling) state.plan_us += ggml_time_us() - plan_start;
    state.input.resize(32*plan.k); state.output.resize(32*plan.n);
    int shifts[32];
    constexpr int positions_per_batch = 32;
    for (int position = 0; position < activation->ne[1]; position += positions_per_batch) {
        const int rows = int(std::min<int64_t>(positions_per_batch, activation->ne[1] - position));
        const auto scale_start = state.profiling ? ggml_time_us() : 0;
        for (int row = 0; row < rows; ++row) {
            const float * source = reinterpret_cast<const float *>(
                static_cast<const char *>(activation->data) + (position + row)*activation->nb[1]);
            float peak = 0.f;
            for (int k = 0; k < plan.k; ++k) {
                if (!std::isfinite(source[k])) GGML_ABORT("nonfinite Asahi encoder input");
                peak = std::max(peak, std::fabs(source[k]));
            }
            shifts[row] = peak ? int(std::floor(std::log2(double(plan.limit)) - std::log2(double(peak)))) : 0;
            for (int k = 0; k < plan.k; ++k) {
                state.input[row*plan.k + k] = std::ldexp(source[k], shifts[row]);
            }
        }
        if (state.profiling) state.scale_us += ggml_time_us() - scale_start;
        const auto run_start = state.profiling ? ggml_time_us() : 0;
        if (!ane_plan_run_batch(plan.handle, state.input.data(), state.output.data(), rows))
            GGML_ABORT("Asahi encoder submission failed; no CPU fallback");
        if (state.profiling) state.run_us += ggml_time_us() - run_start;
        const auto restore_start = state.profiling ? ggml_time_us() : 0;
        for (int row = 0; row < rows; ++row) {
            float * target = reinterpret_cast<float *>(
                static_cast<char *>(dst->data) + (position + row)*dst->nb[1]);
            for (int n = 0; n < plan.n; ++n) {
                target[n] = std::ldexp(state.output[row*plan.n + n], -shifts[row]);
                if (!std::isfinite(target[n])) GGML_ABORT("nonfinite Asahi encoder output");
            }
        }
        if (state.profiling) state.restore_us += ggml_time_us() - restore_start;
    }
    ++state.projections;
}

void WhisperAsahi::finish_encoder(int layers, int positions) {
    const auto total = ane_device_submissions(impl->device);
    const auto submissions = total - impl->previous_submissions;
    const auto expected = static_cast<unsigned long long>(6*layers);
    if (impl->projections != expected || submissions != expected*((positions + 31)/32))
        GGML_ABORT("Asahi encoder projection/submission count mismatch");
    std::fprintf(stderr, "ASAHI_ANE encoder: projections=%llu submissions=%llu plans=%zu replicas=1\n",
                 impl->projections, submissions, impl->plans.size());
    if (impl->profiling) {
        const auto t = ane_device_timings(impl->device), p = impl->previous_timings;
        std::fprintf(stderr, "ASAHI_PROFILE projections: plan=%.3f scale=%.3f run=%.3f restore=%.3f ms\n",
                     impl->plan_us/1000., impl->scale_us/1000., impl->run_us/1000., impl->restore_us/1000.);
        std::fprintf(stderr, "ASAHI_PROFILE device: pack=%.3f write=%.3f ioctl=%.3f read=%.3f unpack=%.3f ms\n",
                     (t.pack_ns-p.pack_ns)/1e6, (t.write_ns-p.write_ns)/1e6,
                     (t.ioctl_ns-p.ioctl_ns)/1e6, (t.read_ns-p.read_ns)/1e6,
                     (t.unpack_ns-p.unpack_ns)/1e6);
        impl->previous_timings = t;
        impl->plan_us = impl->scale_us = impl->run_us = impl->restore_us = 0;
    }
    impl->projections = 0;
    impl->previous_submissions = total;
}

static FILE * trace_file(const char * name, const char * mode) {
    const char * directory = std::getenv("WHISPER_ASAHI_TRACE");
    if (!directory || !*directory) return nullptr;
    FILE * file = std::fopen((std::string(directory) + "/" + name).c_str(), mode);
    if (!file) GGML_ABORT("could not open requested Asahi trace");
    return file;
}

static void write_checked(FILE * file, const void * data, size_t bytes) {
    if (std::fwrite(data, 1, bytes, file) != bytes) GGML_ABORT("Asahi trace write failed");
}

void whisper_asahi_trace_tensor(const char * name, const ggml_tensor * tensor) {
    FILE * file = trace_file(name, "wb");
    if (!file) return;
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor));
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    write_checked(file, values.data(), values.size()*sizeof(float));
    if (std::fclose(file)) GGML_ABORT("Asahi trace close failed");
}

void whisper_asahi_trace_logits(const float * logits, int vocabulary, const int32_t * tokens, int count) {
    FILE * file = trace_file("logits.bin", "ab");
    if (!file) return;
    const int32_t header[] = {count, vocabulary};
    write_checked(file, header, sizeof(header));
    write_checked(file, tokens, count*sizeof(int32_t));
    write_checked(file, logits, vocabulary*sizeof(float));
    if (std::fclose(file)) GGML_ABORT("Asahi trace close failed");
}
