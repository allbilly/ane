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
