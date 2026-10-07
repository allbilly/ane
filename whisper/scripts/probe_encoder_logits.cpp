// Capture complete raw logits on identical, externally supplied token histories.
#include "whisper.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

template<class T> std::vector<T> read(const char * path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("could not open probe input");
    const auto size = file.tellg();
    if (size <= 0 || size % sizeof(T)) throw std::runtime_error("invalid input length");
    std::vector<T> data(static_cast<size_t>(size) / sizeof(T));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(data.data()), size)) throw std::runtime_error("input read failed");
    return data;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s MODEL MEL_F32 TOKENS_I32 OUTPUT\n", argv[0]);
        return 1;
    }
    try {
        const auto mel = read<float>(argv[2]);
        const auto tokens = read<whisper_token>(argv[3]);
        if (mel.size() != 80 * 3000 || tokens.size() < 2 || tokens.size() > 448)
            throw std::runtime_error("expected tiny.en mel and bounded token history");
        ggml_backend_load_all();
        auto params = whisper_context_default_params();
        params.use_gpu = false;
        params.flash_attn = true;
        std::unique_ptr<whisper_context, decltype(&whisper_free)> ctx(
            whisper_init_from_file_with_params(argv[1], params), whisper_free);
        if (!ctx || whisper_n_vocab(ctx.get()) != 51864 ||
            tokens[0] != whisper_token_sot(ctx.get()) || tokens[1] != whisper_token_not(ctx.get()))
            throw std::runtime_error("expected tiny.en and its no-timestamps prompt");
        if (whisper_set_mel(ctx.get(), mel.data(), 3000, 80)) throw std::runtime_error("mel setup failed");
        const auto start = std::chrono::steady_clock::now();
        if (whisper_encode(ctx.get(), 0, 4)) throw std::runtime_error("encoder failed");
        const double encode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::ofstream out(argv[4], std::ios::binary);
        if (!out) throw std::runtime_error("output open failed");
        int past = 0;
        for (size_t offset = 0; offset < tokens.size();) {
            const int count = offset ? 1 : 2;
            if (whisper_decode(ctx.get(), tokens.data() + offset, count, past, 4))
                throw std::runtime_error("decoder failed");
            const int header[] = {count, 51864};
            out.write(reinterpret_cast<const char *>(header), sizeof(header));
            out.write(reinterpret_cast<const char *>(tokens.data() + offset), count * sizeof(whisper_token));
            const float * logits = whisper_get_logits(ctx.get()) + (count - 1) * 51864;
            out.write(reinterpret_cast<const char *>(logits), 51864 * sizeof(float));
            offset += count;
            past += count;
        }
        if (!out) throw std::runtime_error("logit write failed");
        std::printf("PROBE_ENCODE_MS\t%.6f\n", encode_ms);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "probe failed: %s\n", error.what());
        return 2;
    }
    return 0;
}
