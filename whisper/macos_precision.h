#pragma once
#include "ggml.h"
#include <memory>

bool whisper_macos_precision_enabled();

class WhisperMacPrecision {
    struct Impl;
    std::unique_ptr<Impl> impl;
    static void compute(ggml_tensor * dst, int ith, int nth, void * userdata);
public:
    WhisperMacPrecision();
    ~WhisperMacPrecision();
    ggml_tensor * project(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input,
                          int layer, const char * name);
    void finish_encoder();
};
