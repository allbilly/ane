#include "macos_precision.h"
#include "ggml-backend.h"
#include <cstdlib>
#include <cstring>

bool whisper_macos_precision_enabled() {
    const char * path = std::getenv("WHISPER_MACOS_PRECISION");
    return path && *path;
}

#if defined(__APPLE__) && defined(__aarch64__)
#include <arm_neon.h>
#include <cfloat>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct ane_e5rt_program;
using compile_fn = ane_e5rt_program * (*)(const char *, const char *, uint64_t,
    const char * const *, const size_t *, size_t, const char * const *, const size_t *, size_t);
using buffer_fn = int (*)(ane_e5rt_program *, const char *, void **, size_t *);
using execute_fn = int (*)(ane_e5rt_program *);
using release_fn = void (*)(ane_e5rt_program *);

struct WhisperMacPrecision::Impl {
    struct Plan {
        Impl * owner;
        ane_e5rt_program * program;
        float16_t * feed;
        const float16_t * output;
        int k, n;
        std::string name;
    };
    void * library = nullptr;
    compile_fn compile = nullptr;
    buffer_fn input_buffer = nullptr, output_buffer = nullptr;
    execute_fn execute = nullptr;
    release_fn release = nullptr;
    std::string root;
    std::map<const ggml_tensor *, Plan> plans;
    int submissions = 0;
    int64_t pack_us = 0, execute_us = 0, combine_us = 0;
    bool profiling = false;
};

static std::vector<char> read_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0) GGML_ABORT("precision program file missing");
    std::vector<char> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    if (!file.read(bytes.data(), bytes.size())) GGML_ABORT("precision program file read failed");
    return bytes;
}

// Convert four position rows into four channel rows, or back again.
static void transpose4(float32x4_t & a, float32x4_t & b, float32x4_t & c, float32x4_t & d) {
    const auto ab0 = vtrn1q_f32(a, b), ab1 = vtrn2q_f32(a, b);
    const auto cd0 = vtrn1q_f32(c, d), cd1 = vtrn2q_f32(c, d);
    a = vcombine_f32(vget_low_f32(ab0), vget_low_f32(cd0));
    b = vcombine_f32(vget_low_f32(ab1), vget_low_f32(cd1));
    c = vcombine_f32(vget_high_f32(ab0), vget_high_f32(cd0));
    d = vcombine_f32(vget_high_f32(ab1), vget_high_f32(cd1));
}

WhisperMacPrecision::WhisperMacPrecision() : impl(new Impl) {
    const char * library = std::getenv("ANEFORGE_DYLIB");
    if (!library || std::getenv("ANEFORGE_ENCODER"))
        GGML_ABORT("precision projections require ANEFORGE_DYLIB and exclude ANEFORGE_ENCODER");
    impl->root = std::getenv("WHISPER_MACOS_PRECISION");
    impl->profiling = std::getenv("WHISPER_PROFILE");
    impl->library = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!impl->library) GGML_ABORT("precision E5RT runtime dlopen failed");
    impl->compile = reinterpret_cast<compile_fn>(dlsym(impl->library, "ane_e5rt_program_compile"));
    impl->input_buffer = reinterpret_cast<buffer_fn>(dlsym(impl->library, "ane_e5rt_program_input_buffer"));
    impl->output_buffer = reinterpret_cast<buffer_fn>(dlsym(impl->library, "ane_e5rt_program_output_buffer"));
    impl->execute = reinterpret_cast<execute_fn>(dlsym(impl->library, "ane_e5rt_program_execute"));
    impl->release = reinterpret_cast<release_fn>(dlsym(impl->library, "ane_e5rt_program_release"));
    if (!impl->compile || !impl->input_buffer || !impl->output_buffer || !impl->execute || !impl->release)
        GGML_ABORT("precision E5RT runtime symbols missing");
    std::fprintf(stderr, "MACOS_PRECISION ready: FP32 host arithmetic and paired ANE projections\n");
}

WhisperMacPrecision::~WhisperMacPrecision() {
    for (auto & item : impl->plans) impl->release(item.second.program);
    if (impl->library) dlclose(impl->library);
}

ggml_tensor * WhisperMacPrecision::project(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input,
                                           int layer, const char * name) {
    GGML_ASSERT(layer >= 0 && layer < 4 && weight->type == GGML_TYPE_F16 && ggml_is_contiguous(weight));
    GGML_ASSERT(input->type == GGML_TYPE_F32 && input->nb[0] == sizeof(float));
    GGML_ASSERT(input->ne[1] == 1500 && weight->ne[0] == input->ne[0]);
    GGML_ASSERT(weight->ne[2] == 1 && weight->ne[3] == 1 && input->ne[2] == 1 && input->ne[3] == 1);
    auto found = impl->plans.find(weight);
    if (found == impl->plans.end()) {
        Impl::Plan plan{};
        plan.owner = impl.get();
        plan.k = int(weight->ne[0]); plan.n = int(weight->ne[1]);
        GGML_ASSERT((plan.k == 384 || plan.k == 1536) && (plan.n == 384 || plan.n == 1536));
        plan.name = "layer" + std::to_string(layer) + "-" + name;
        const std::string directory = impl->root + "/" + plan.name;
        const auto blob = read_file(directory + "/weights.bin");
        // The validated single FP16 BLOBFILE has its descriptor at byte 64 and
        // grouped checkpoint payload at byte 128. Check it against native weights.
        uint32_t count, version, magic, type;
        uint64_t size, offset;
        GGML_ASSERT(blob.size() >= 128);
        std::memcpy(&count, blob.data(), 4); std::memcpy(&version, blob.data()+4, 4);
        std::memcpy(&magic, blob.data()+64, 4); std::memcpy(&type, blob.data()+68, 4);
        std::memcpy(&size, blob.data()+72, 8); std::memcpy(&offset, blob.data()+80, 8);
        GGML_ASSERT(count == 1 && version == 2 && magic == 0xdeadbeef && type == 1);
        GGML_ASSERT(offset == 128 && size == size_t(plan.k)*plan.n*2 && blob.size() == offset+size);
        std::vector<ggml_fp16_t> original(size/2);
        ggml_backend_tensor_get(weight, original.data(), 0, size);
        const int half = plan.k/2;
        for (int part = 0; part < 2; ++part)
            for (int n = 0; n < plan.n; ++n)
                if (std::memcmp(blob.data()+128 + (part*plan.n+n)*half*2,
                                original.data()+n*plan.k+part*half, half*2))
                    GGML_ABORT("precision ANE weights differ from native checkpoint");

        const char * in = "t0", * out = "t1";
        const size_t input_bytes = size_t(plan.k)*6000*2, output_bytes = size_t(2*plan.n)*6000*2;
        plan.program = impl->compile((directory+"/model.mil").c_str(), (directory+"/native-cache").c_str(),
            4, &in, &input_bytes, 1, &out, &output_bytes, 1);
        if (!plan.program) GGML_ABORT("precision ANE projection compilation failed");
        void * feed = nullptr, * output = nullptr;
        size_t feed_size = 0, output_size = 0;
        if (impl->input_buffer(plan.program, in, &feed, &feed_size) || feed_size != input_bytes || !feed ||
            impl->output_buffer(plan.program, out, &output, &output_size) || output_size != output_bytes || !output)
            GGML_ABORT("precision ANE port layout mismatch");
        plan.feed = static_cast<float16_t *>(feed);
        plan.output = static_cast<const float16_t *>(output);
        found = impl->plans.emplace(weight, plan).first;
    }
    ggml_tensor * sources[] = {weight, input};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, weight->ne[1], 1500, 1, 1,
                          sources, 2, compute, 1, &found->second);
}

void WhisperMacPrecision::compute(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith) return;
    auto & plan = *static_cast<Impl::Plan *>(userdata);
    auto & state = *plan.owner;
    const auto * input = dst->src[1];
    const auto t0 = ggml_time_us();
    const float gains[] = {1.f, 1.375f};
    for (int position = 0; position < 1500; position += 4) {
        for (int channel = 0; channel < plan.k; channel += 4) {
            float32x4_t values[4];
            for (int row = 0; row < 4; ++row)
                values[row] = vld1q_f32(reinterpret_cast<const float *>(
                    static_cast<const char *>(input->data)+(position+row)*input->nb[1])+channel);
            transpose4(values[0], values[1], values[2], values[3]);
            for (int grid = 0; grid < 2; ++grid) {
                for (int row = 0; row < 4; ++row) {
                    const auto scaled = vmulq_n_f32(values[row], gains[grid]);
                    const auto high = vcvt_f16_f32(scaled);
                    const auto low = vcvt_f16_f32(vmulq_n_f32(vsubq_f32(scaled, vcvt_f32_f16(high)), 4096.f));
                    if (!vminvq_u32(vcleq_f32(vabsq_f32(vcvt_f32_f16(high)), vdupq_n_f32(FLT_MAX))) ||
                        !vminvq_u32(vcleq_f32(vabsq_f32(vcvt_f32_f16(low)), vdupq_n_f32(FLT_MAX))))
                        GGML_ABORT("nonfinite precision projection input");
                    auto * target = plan.feed+(channel+row)*6000+grid*3000+position;
                    vst1_f16(target, high); vst1_f16(target+1500, low);
                }
            }
        }
    }
    const auto t1 = ggml_time_us();
    if (state.execute(plan.program)) GGML_ABORT("precision ANE execute failed; no CPU fallback");
    ++state.submissions;
    const auto t2 = ggml_time_us();
    for (int channel = 0; channel < plan.n; channel += 4) {
        for (int position = 0; position < 1500; position += 4) {
            float32x4_t values[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
            for (int grid = 0; grid < 2; ++grid) {
                for (int part = 0; part < 2; ++part) {
                    for (int row = 0; row < 4; ++row) {
                        const auto * source = plan.output+(part*plan.n+channel+row)*6000+grid*3000+position;
                        const auto high = vcvt_f32_f16(vld1_f16(source));
                        const auto low = vcvt_f32_f16(vld1_f16(source+1500));
                        const auto numerator = vaddq_f32(high, vmulq_n_f32(low, 1.f/4096));
                        values[row] = vaddq_f32(values[row], vdivq_f32(numerator, vdupq_n_f32(gains[grid]*2)));
                    }
                }
            }
            transpose4(values[0], values[1], values[2], values[3]);
            for (int row = 0; row < 4; ++row) {
                if (!vminvq_u32(vcleq_f32(vabsq_f32(values[row]), vdupq_n_f32(FLT_MAX))))
                    GGML_ABORT("nonfinite precision ANE projection output");
                auto * target = reinterpret_cast<float *>(static_cast<char *>(dst->data)+(position+row)*dst->nb[1]);
                vst1q_f32(target+channel, values[row]);
            }
        }
    }
    const auto t3 = ggml_time_us();
    state.pack_us += t1-t0; state.execute_us += t2-t1; state.combine_us += t3-t2;
}

void WhisperMacPrecision::finish_encoder() {
    if (impl->submissions != 24 || impl->plans.size() != 24)
        GGML_ABORT("precision encoder did not execute all 24 projections");
    std::fprintf(stderr, "MACOS_PRECISION encoder: projections=24 submissions=24\n");
    if (impl->profiling)
        std::fprintf(stderr, "WHISPER_PROFILE precision: pack=%.3f dispatch=%.3f combine=%.3f ms\n",
            impl->pack_us/1000., impl->execute_us/1000., impl->combine_us/1000.);
    impl->submissions = 0;
    impl->pack_us = impl->execute_us = impl->combine_us = 0;
}

#else
struct WhisperMacPrecision::Impl {};
WhisperMacPrecision::WhisperMacPrecision() { GGML_ABORT("paired Mac projections require Apple Silicon macOS"); }
WhisperMacPrecision::~WhisperMacPrecision() = default;
ggml_tensor * WhisperMacPrecision::project(ggml_context *, ggml_tensor *, ggml_tensor *, int, const char *) { return nullptr; }
void WhisperMacPrecision::finish_encoder() {}
#endif
