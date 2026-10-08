// Opt-in cross-attention projection fusion for the pinned tiny.en CPU graph.
#pragma once
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

inline bool whisper_fused_cross_kv_enabled() {
    const char * value = std::getenv("WHISPER_FUSED_CROSS_KV");
    return value && std::strcmp(value, "1") == 0;
}

class WhisperCrossKV {
    ggml_context * context_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_tensor * weights_ = nullptr;

public:
    explicit WhisperCrossKV(const std::vector<ggml_tensor *> & weights) {
        // Each source is [input channels, output channels], with output rows
        // stored consecutively. Concatenate K0,V0,K1,V1,... without transposing.
        GGML_ASSERT(weights.size() == 8);
        for (const auto * weight : weights) {
            GGML_ASSERT(weight->type == GGML_TYPE_F16);
            GGML_ASSERT(weight->ne[0] == 384 && weight->ne[1] == 384);
            GGML_ASSERT(weight->ne[2] == 1 && weight->ne[3] == 1);
            GGML_ASSERT(ggml_is_contiguous(weight));
        }
        const ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
        context_ = ggml_init(params);
        GGML_ASSERT(context_);
        weights_ = ggml_new_tensor_2d(context_, GGML_TYPE_F32, 384, 3072);
        ggml_set_name(weights_, "whisper.cross_kv.cached_weights");
        buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(context_, ggml_backend_cpu_buffer_type());
        GGML_ASSERT(buffer_);
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // Conversion occurs while preparing the persistent decoder state, not
        // during any warm encode. F16->F32 widening preserves weights exactly.
        std::vector<ggml_fp16_t> half(384*384);
        std::vector<float> full(half.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            ggml_backend_tensor_get(weights[i], half.data(), 0, half.size()*sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(half.data(), full.data(), full.size());
            ggml_backend_tensor_set(weights_, full.data(), i*full.size()*sizeof(float), full.size()*sizeof(float));
        }
    }

    ~WhisperCrossKV() {
        ggml_backend_buffer_free(buffer_);
        ggml_free(context_);
    }
    WhisperCrossKV(const WhisperCrossKV &) = delete;
    WhisperCrossKV & operator=(const WhisperCrossKV &) = delete;

    ggml_tensor * project(ggml_context * graph_context, ggml_tensor * input) const {
        GGML_ASSERT(input->type == GGML_TYPE_F32);
        GGML_ASSERT(input->ne[0] == 384 && input->ne[1] == 1500);
        GGML_ASSERT(input->ne[2] == 1 && input->ne[3] == 1);
        ggml_tensor * result = ggml_mul_mat(graph_context, weights_, input);
        ggml_set_name(result, "whisper.cross_kv.fused");
        return result;
    }

    static ggml_tensor * view(ggml_context * graph_context, ggml_tensor * result, int index) {
        GGML_ASSERT(index >= 0 && index < 8);
        // ggml's CPU scaling kernel requires contiguous input. Materialize
        // each slice so downstream scaling and bias retain their usual layout.
        return ggml_cont(graph_context, ggml_view_2d(graph_context, result, 384, 1500,
                            3072*sizeof(float), index*384*sizeof(float)));
    }
};
