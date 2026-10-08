// Complete tiny.en replay behind whisper.cpp's existing external-encoder API.
// Payloads are generated and SHA-256 checked by whisper.encoder_kernel.
#include "aneforge/whisper-aneforge.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <glob.h>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#define DRM_COMMAND_BASE 0x40
#define DRM_IOWR(nr, type) _IOWR('d', nr, type)
#include "../kmod/uapi/drm/ane_accel.h"
static_assert(sizeof(drm_ane_bo_init) == 24 && sizeof(drm_ane_submit) == 152, "ANE driver ABI");

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void read_output(void * target, const void * source, size_t bytes) {
    // BO mappings are uncached. Load complete cache lines, as in ane_matmul.c,
    // using disjoint ranges so the shared payload retains exactly its bits.
    check(!(bytes % 64), "unaligned complete encoder output");
    const size_t lines = bytes / 64;
    const char * workers = std::getenv("WHISPER_ASAHI_READ_THREADS");
    const int threads = workers ? std::atoi(workers) : 4;
    check(threads == 1 || threads == 4, "WHISPER_ASAHI_READ_THREADS must be 1 or 4");
    #pragma omp parallel num_threads(threads)
    {
        const size_t first = lines * omp_get_thread_num() / omp_get_num_threads();
        const size_t last = lines * (omp_get_thread_num()+1) / omp_get_num_threads();
        for (size_t line = first; line < last; ++line) {
            const auto * in = static_cast<const char *>(source) + line*64;
            auto * out = static_cast<char *>(target) + line*64;
            asm volatile("ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%0]\n\t"
                         "st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%1]"
                         : : "r"(in), "r"(out) : "v0", "v1", "v2", "v3", "memory");
        }
    }
}

static std::vector<char> load(const std::string & path, size_t bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    check(bool(file) && file.tellg() == std::streamoff(bytes), "incorrect encoder payload size");
    std::vector<char> data(bytes);
    file.seekg(0);
    check(bool(file.read(data.data(), bytes)), "could not read encoder payload");
    return data;
}

struct EncoderBuffer {
    int fd = -1;
    uint32_t handle = 0;
    size_t size = 0;
    void * map = nullptr;
    void allocate(int device, size_t bytes) {
        fd = device;
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
        size = (bytes + page - 1) & ~(page - 1);
        drm_ane_bo_init request{};
        request.size = size;
        check(!ioctl(fd, DRM_IOCTL_ANE_BO_INIT, &request) && request.handle, "encoder BO allocation failed");
        handle = request.handle;
        map = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, request.offset);
        if (map == MAP_FAILED) { map = nullptr; throw std::runtime_error("encoder mmap failed"); }
        std::memset(map, 0, size);
    }
    void close() {
        if (map) munmap(map, size);
        if (handle) { drm_ane_bo_free request{}; request.handle = handle; ioctl(fd, DRM_IOCTL_ANE_BO_FREE, &request); }
        map = nullptr; handle = 0;
    }
};

struct whisper_aneforge_context {
    int fd = -1;
    EncoderBuffer banks[7], bootstrap;
    drm_ane_submit request{};
    std::vector<ggml_fp16_t> mel = std::vector<ggml_fp16_t>(80*3000);
    std::vector<ggml_fp16_t> output = std::vector<ggml_fp16_t>(1500*384);
    size_t mel_stride = 0;
    ~whisper_aneforge_context() {
        bootstrap.close();
        for (auto & bank : banks) bank.close();
        if (fd >= 0) ::close(fd);
    }
};

static int open_device() {
    std::ifstream file("/proc/device-tree/compatible", std::ios::binary);
    std::string compatible((std::istreambuf_iterator<char>(file)), {});
    check(compatible.find(std::string("apple,t8103\0", 12)) != std::string::npos, "encoder targets base M1/T8103");
    glob_t paths{};
    glob(std::getenv("ANE_DEVICE") ? std::getenv("ANE_DEVICE") : "/dev/accel/accel*", 0, nullptr, &paths);
    int fd = -1;
    for (size_t i = 0; i < paths.gl_pathc; ++i) {
        const std::string path = paths.gl_pathv[i];
        const std::string sys = "/sys/class/accel/" + path.substr(path.rfind('/')+1) + "/device/driver";
        char * driver = realpath(sys.c_str(), nullptr);
        if (driver) {
            const char * name = std::strrchr(driver, '/');
            if (name && !std::strcmp(name+1, "ane")) fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
            std::free(driver);
        }
        if (fd >= 0) break;
    }
    globfree(&paths);
    check(fd >= 0, "no accessible ANE accel device");
    return fd;
}

whisper_aneforge_context * whisper_aneforge_init(const char * directory) {
    try {
        const std::string dir = directory;
        std::ifstream layout(dir+"/layout.txt");
        std::string magic;
        size_t tasks, bootstrap_size, command_size, coefficient_size, constant_size, sizes[4], mel_stride, pos_stride;
        check(bool(layout >> magic >> tasks >> bootstrap_size >> command_size >> coefficient_size >> constant_size
                          >> sizes[0] >> sizes[1] >> sizes[2] >> sizes[3] >> mel_stride >> pos_stride)
              && magic == "ANE_WHISPER_V1", "invalid native encoder layout; regenerate payloads");
        check((tasks == 1779 || tasks == 1783) && bootstrap_size == 504
              && command_size == 1277952 && coefficient_size == 15450112 && constant_size == 26112,
              "unsupported native encoder graph");
        check((tasks == 1779 ? mel_stride == 6016 && pos_stride == 3008 && sizes[0] == 8093696
                            : mel_stride == 6000 && pos_stride == 3000 && sizes[0] == 10371072)
              && sizes[1] == 1163264
              && sizes[2] == 491520 && sizes[3] == 1163264,
              "invalid native encoder input strides/buffers");
        auto commands = load(dir+"/commands.bin", command_size);
        auto coefficients = load(dir+"/coefficients.bin", coefficient_size);
        auto constants = load(dir+"/constants.bin", constant_size);
        auto positions = load(dir+"/pos.f16", 1152000);
        // Follow the complete chain before allocating hardware buffers.
        size_t offset = 0, size = bootstrap_size;
        for (unsigned id = 0; id < tasks; ++id) {
            check(size >= 40 && offset+size <= commands.size(), "invalid encoder descriptor bounds");
            uint32_t header[10]; std::memcpy(header, commands.data()+offset, sizeof(header));
            check((header[0]&65535) == id, "nonsequential encoder task ID");
            if (id == tasks-1) check(!header[7] && (header[0]&(1u<<25)), "encoder missing end-of-network");
            else check(header[7] >= offset+size && !(header[7]&3), "invalid encoder NextPtr");
            offset = header[7]; size = (((header[1]>>16)&511)+1)*4;
        }
        auto ctx = std::make_unique<whisper_aneforge_context>();
        ctx->mel_stride = mel_stride;
        ctx->fd = open_device();
        ctx->banks[0].allocate(ctx->fd, commands.size()+coefficients.size()+1);
        std::memcpy(ctx->banks[0].map, commands.data(), commands.size());
        std::memcpy(static_cast<char *>(ctx->banks[0].map)+commands.size(), coefficients.data(), coefficients.size());
        ctx->banks[2].allocate(ctx->fd, constants.size());
        std::memcpy(ctx->banks[2].map, constants.data(), constants.size());
        for (int bank = 3; bank <= 6; ++bank) ctx->banks[bank].allocate(ctx->fd, sizes[bank-3]);
        for (size_t channel = 0; channel < 384; ++channel)
            std::memcpy(static_cast<char *>(ctx->banks[4].map)+channel*pos_stride, positions.data()+channel*3000, 3000);
        ctx->bootstrap.allocate(ctx->fd, bootstrap_size);
        uint32_t first; std::memcpy(&first, commands.data(), 4);
        first = (first & ~(255u<<16)) | (64u<<16);
        std::memcpy(ctx->bootstrap.map, commands.data(), bootstrap_size);
        std::memcpy(ctx->bootstrap.map, &first, 4);
        ctx->request.tsk_size = commands.size(); ctx->request.td_count = tasks; ctx->request.td_size = bootstrap_size;
        ctx->request.btsp_handle = ctx->bootstrap.handle;
        for (int bank = 0; bank <= 6; ++bank) ctx->request.handles[bank] = ctx->banks[bank].handle;
        std::fprintf(stderr, "ASAHI_FULL_ANE ready: tasks=%zu; CPU cross-K/V and decoder\n", tasks);
        return ctx.release();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "ASAHI_FULL_ANE initialization failed: %s\n", error.what());
        return nullptr;
    }
}

void whisper_aneforge_encode(whisper_aneforge_context * ctx, int64_t n_mel, int64_t n_len,
                             const float * mel, float * output) {
    const bool profile = std::getenv("WHISPER_ASAHI_PROFILE") != nullptr;
    auto start = std::chrono::steady_clock::now();
    auto stage = [&](const char * name) {
        if (!profile) return;
        auto end = std::chrono::steady_clock::now();
        std::fprintf(stderr, "ASAHI_PROFILE %s %.3f ms\n", name,
                     std::chrono::duration<double, std::milli>(end-start).count());
        start = end;
    };
    if (!ctx || !((n_mel == 80 && n_len == 3000) || (n_mel == 3000 && n_len == 80)))
        GGML_ABORT("complete ANE encoder requires tiny.en mel [80,3000]");
    for (size_t i = 0; i < ctx->mel.size(); ++i) if (!std::isfinite(mel[i])) GGML_ABORT("nonfinite encoder mel");
    ggml_fp32_to_fp16_row(mel, ctx->mel.data(), ctx->mel.size());
    for (auto value : ctx->mel) if ((value & 0x7c00u) == 0x7c00u) GGML_ABORT("mel exceeds finite FP16 range");
    for (size_t channel = 0; channel < 80; ++channel)
        std::memcpy(static_cast<char *>(ctx->banks[5].map)+channel*ctx->mel_stride, ctx->mel.data()+channel*3000, 6000);
    stage("full_input");
    std::memset(ctx->banks[3].map, 0, ctx->banks[3].size);
    std::fill(ctx->output.begin(), ctx->output.end(), ggml_fp32_to_fp16(NAN));
    std::memcpy(ctx->banks[6].map, ctx->output.data(), ctx->output.size()*2);
    asm volatile("dsb sy" ::: "memory");
    stage("full_clear");
    if (ioctl(ctx->fd, DRM_IOCTL_ANE_SUBMIT, &ctx->request)) GGML_ABORT("complete ANE encoder submission failed");
    stage("full_dispatch");
    read_output(ctx->output.data(), ctx->banks[6].map, ctx->output.size()*2);
    stage("full_read");
    ggml_fp16_to_fp32_row(ctx->output.data(), output, ctx->output.size());
    for (size_t i = 0; i < ctx->output.size(); ++i) if (!std::isfinite(output[i])) GGML_ABORT("nonfinite/unwritten encoder output");
    stage("full_output");
    std::fprintf(stderr, "ASAHI_FULL_ANE encoder: tasks=%u submissions=1\n", ctx->request.td_count);
}

void whisper_aneforge_free(whisper_aneforge_context * ctx) { delete ctx; }
