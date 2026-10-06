#pragma once

#include "ggml.h"
#include <memory>

// The encoder's dense projections use ANE. All other Whisper operations use CPU.
class WhisperAsahi {
public:
    WhisperAsahi();
    ~WhisperAsahi();
    WhisperAsahi(const WhisperAsahi &) = delete;
    WhisperAsahi & operator=(const WhisperAsahi &) = delete;
    ggml_tensor * project(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input);
    void finish_encoder(int layers, int positions);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    static void compute(ggml_tensor * dst, int ith, int nth, void * userdata);
};

bool whisper_asahi_enabled();
void whisper_asahi_profile_stage(const char * name, int64_t start_us);
void whisper_asahi_trace_tensor(const char * name, const ggml_tensor * tensor);
void whisper_asahi_trace_logits(const float * logits, int vocabulary,
                               const int32_t * tokens, int count);
