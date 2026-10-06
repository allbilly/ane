// Warm, in-process transcription benchmark using the public whisper.cpp API.
#include "whisper.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s MODEL PCM_F32 USE_GPU WARMUPS RUNS\n", argv[0]);
        return 1;
    }
    try {
        std::ifstream file(argv[2], std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("could not open PCM input");
        const auto size = file.tellg();
        if (size <= 0 || size % sizeof(float)) throw std::runtime_error("invalid PCM length");
        std::vector<float> audio(static_cast<size_t>(size) / sizeof(float));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char *>(audio.data()), size)) {
            throw std::runtime_error("could not read PCM input");
        }
        const int warmups = std::stoi(argv[4]), runs = std::stoi(argv[5]);
        if (warmups < 1 || runs < 1) throw std::runtime_error("warmups and runs must be positive");
        ggml_backend_load_all();
        auto cparams = whisper_context_default_params();
        cparams.use_gpu = std::stoi(argv[3]) != 0;
        cparams.flash_attn = true;
        auto * ctx = whisper_init_from_file_with_params(argv[1], cparams);
        if (!ctx) throw std::runtime_error("model initialization failed");
        auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.n_threads = 4;
        params.language = "en";
        params.no_context = true;
        params.no_timestamps = true;
        params.print_realtime = false;
        params.print_progress = false;
        params.print_timestamps = false;
        params.greedy.best_of = 1;
        params.temperature = 0.0f;
        params.temperature_inc = 0.0f;
        params.suppress_nst = false;
        for (int i = 0; i < warmups + runs; ++i) {
            const char * phase = i < warmups ? "warmup" : "measure";
            const int index = i < warmups ? i + 1 : i - warmups + 1;
            std::fprintf(stderr, "BENCH_BEGIN\t%s\t%d\n", phase, index);
            whisper_reset_timings(ctx);
            const auto start = std::chrono::steady_clock::now();
            const int ret = whisper_full(ctx, params, audio.data(), static_cast<int>(audio.size()));
            const double wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (ret) {
                whisper_free(ctx);
                throw std::runtime_error("whisper_full failed: " + std::to_string(ret));
            }
            whisper_print_timings(ctx);
            std::fprintf(stderr, "BENCH_END\t%s\t%d\n", phase, index);
            std::string text;
            for (int s = 0; s < whisper_full_n_segments(ctx); ++s) {
                text += whisper_full_get_segment_text(ctx, s);
            }
            for (auto & ch : text) if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
            std::printf("BENCH_RESULT\t%s\t%d\t%.6f\t%s\n", phase, index, wall_ms, text.c_str());
            std::fflush(stdout);
        }
        whisper_free(ctx);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "benchmark failed: %s\n", error.what());
        return 2;
    }
    return 0;
}
