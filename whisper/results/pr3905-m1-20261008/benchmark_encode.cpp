#include "whisper.h"
#include <cstdio>
#include <cstdlib>

// Only the timing harness differs; both libraries are unmodified whisper.cpp.
int main(int argc, char ** argv) {
    if (argc != 4) return 1;
    const int warmups = std::atoi(argv[2]), runs = std::atoi(argv[3]);
    if (warmups < 1 || runs < 1) return 1;
    ggml_backend_load_all();
    auto params = whisper_context_default_params();
    params.use_gpu = false;
    params.flash_attn = true;
    auto * ctx = whisper_init_from_file_with_params(argv[1], params);
    if (!ctx) return 2;
    // Same empty mel, padded with zeroes to the full context, as whisper-bench.
    if (whisper_set_mel(ctx, nullptr, 0, whisper_model_n_mels(ctx))) return 3;
    for (int i = 0; i < warmups + runs; ++i) {
        whisper_reset_timings(ctx);
        if (whisper_encode(ctx, 0, 4)) return 4;
        auto * timings = whisper_get_timings(ctx);
        if (!timings) return 5;
        std::printf("BENCH_ENCODE\t%s\t%d\t%.6f\n",
                    i < warmups ? "warmup" : "measure", i, timings->encode_ms);
        delete timings;
    }
    whisper_free(ctx);
    return 0;
}
