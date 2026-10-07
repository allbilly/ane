// The same native trace format and stage boundaries on macOS and Asahi.
#pragma once
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

inline void whisper_profile_stage(const char * name, int64_t start_us) {
    if (std::getenv("WHISPER_PROFILE") || std::getenv("WHISPER_ASAHI_PROFILE"))
        std::fprintf(stderr, "WHISPER_PROFILE stage: %s=%.3f ms\n", name, (ggml_time_us()-start_us)/1000.);
}

inline FILE * whisper_trace_file(const char * name, const char * mode) {
    const char * directory = std::getenv("WHISPER_TRACE");
    if (!directory) directory = std::getenv("WHISPER_ASAHI_TRACE");
    if (!directory || !*directory) return nullptr;
    FILE * file = std::fopen((std::string(directory) + "/" + name).c_str(), mode);
    if (!file) GGML_ABORT("could not open requested Whisper trace");
    return file;
}

inline void whisper_trace_write(FILE * file, const void * data, size_t bytes) {
    if (std::fwrite(data, 1, bytes, file) != bytes) GGML_ABORT("Whisper trace write failed");
}

inline void whisper_trace_tensor(const char * name, const ggml_tensor * tensor) {
    FILE * file = whisper_trace_file(name, "wb");
    if (!file) return;
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor));
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    whisper_trace_write(file, values.data(), values.size()*sizeof(float));
    if (std::fclose(file)) GGML_ABORT("Whisper trace close failed");
}

inline void whisper_trace_logits(const float * logits, int vocabulary, const int32_t * tokens, int count) {
    FILE * file = whisper_trace_file("logits.bin", "ab");
    if (!file) return;
    const int32_t header[] = {count, vocabulary};
    whisper_trace_write(file, header, sizeof(header));
    whisper_trace_write(file, tokens, count*sizeof(int32_t));
    whisper_trace_write(file, logits, vocabulary*sizeof(float));
    if (std::fclose(file)) GGML_ABORT("Whisper trace close failed");
}

inline bool whisper_profile_matrix(const ggml_tensor * tensor) {
    return std::getenv("WHISPER_PROFILE_MATMUL") &&
           std::string(tensor->name).rfind("whisper.cross_kv.", 0) == 0;
}

inline void whisper_profile_matrix_input(const ggml_tensor * tensor) {
    const char * path = std::getenv("WHISPER_PROFILE_INPUT");
    static bool saved = false;
    if (!path || !*path || saved) return;
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor));
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) GGML_ABORT("could not create requested cross-K/V input");
    FILE * file = fdopen(fd, "wb");
    if (!file) { close(fd); GGML_ABORT("could not open cross-K/V input stream"); }
    whisper_trace_write(file, tensor->data, ggml_nbytes(tensor));
    if (std::fclose(file)) GGML_ABORT("cross-K/V input close failed");
    saved = true;
}

inline void whisper_profile_matrix_result(const ggml_tensor * dst, const char * backend,
        const char * conversion, int requested_threads, int blas_threads, void * routine,
        int64_t allocate_us, int64_t convert_us, int64_t thread_setup_us, int64_t gemm_us, int64_t total_us) {
    const auto * weight = dst->src[0];
    const auto * input = dst->src[1];
    Dl_info info{};
    dladdr(routine, &info);
    // Matrix inputs and node names come from the pinned tiny.en checkpoint.
    std::fprintf(stderr, "MATRIX_PROFILE\t{\"name\":\"%s\",\"weight\":\"%s\",\"backend\":\"%s\","
        "\"routine\":\"cblas_sgemm\",\"library\":\"%s\",\"symbol\":\"%s\",\"address\":\"%p\","
        "\"m\":%lld,\"n\":%lld,\"k\":%lld,\"order\":\"row_major\",\"transpose_a\":false,\"transpose_b\":true,"
        "\"lda\":%lld,\"ldb\":%lld,\"ldc\":%lld,\"weight_type\":\"%s\",\"input_type\":\"%s\","
        "\"output_type\":\"%s\",\"gemm_type\":\"f32\",\"weight_strides\":[%zu,%zu,%zu,%zu],"
        "\"input_strides\":[%zu,%zu,%zu,%zu],\"output_strides\":[%zu,%zu,%zu,%zu],"
        "\"requested_threads\":%d,\"blas_threads\":%d,\"conversion\":\"%s\","
        "\"allocate_us\":%lld,\"convert_us\":%lld,\"thread_setup_us\":%lld,\"gemm_us\":%lld,\"total_us\":%lld}\n",
        dst->name, weight->name, backend, info.dli_fname ? info.dli_fname : "unknown",
        info.dli_sname ? info.dli_sname : "unknown", routine,
        (long long)input->ne[1], (long long)weight->ne[1], (long long)input->ne[0],
        (long long)input->ne[0], (long long)weight->ne[0], (long long)weight->ne[1],
        ggml_type_name(weight->type), ggml_type_name(input->type), ggml_type_name(dst->type),
        weight->nb[0], weight->nb[1], weight->nb[2], weight->nb[3],
        input->nb[0], input->nb[1], input->nb[2], input->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        requested_threads, blas_threads, conversion,
        (long long)allocate_us, (long long)convert_us, (long long)thread_setup_us, (long long)gemm_us, (long long)total_us);
}
