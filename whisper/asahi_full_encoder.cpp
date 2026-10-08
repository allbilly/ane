// Native full-encoder replay of checkpoint-regenerated PR 3905 payloads.
// UAPI matches kmod/uapi/drm/ane_accel.h; no Apple runtime or Python at inference.
#include "asahi_full_encoder.h"
#include "aneforge/whisper-aneforge.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <glob.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace whisper_pr {

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct Submit {
    uint64_t tsk_size;
    uint32_t td_count, td_size, handles[32], btsp_handle, pad;
};
struct BOInit { uint32_t handle, pad; uint64_t size, offset; };
struct BOFree { uint32_t handle, pad; };
static_assert(sizeof(Submit) == 152 && sizeof(BOInit) == 24 && sizeof(BOFree) == 8, "ANE DRM ABI");
// Linux _IOWR encoding, independent of the host used for structural tests.
constexpr unsigned long bo_init_op = 0xc0186441, bo_free_op = 0xc0086442, submit_op = 0xc0986443;

struct Port {
    int bank = 0;
    size_t offset = 0, channels = 0, height = 0, width = 0, plane = 0, row = 0;
    size_t count() const { return channels * height * width; }
    size_t end() const { return offset + (channels - 1) * plane + (height - 1) * row + width * 2; }
};
struct Payload {
    std::string name;
    size_t bytes = 0, offset = 0;
    uint32_t crc = 0;
    int bank = 0;
};
struct Plan {
    int state = 0, layers = 0;
    uint32_t td_count = 0, td_size = 0;
    size_t command_bytes = 0;
    std::map<int, size_t> buffers;
    std::array<Port, 3> ports;
    std::vector<Payload> payloads;
};

static Plan parse_plan(std::istream & input) {
    Plan p;
    std::string magic;
    size_t nb = 0, np = 0;
    input >> magic >> p.state >> p.layers >> p.td_count >> p.td_size >> p.command_bytes >> nb >> np;
    require(bool(input) && magic == "ANE_WHISPER_PR_V1", "invalid native PR layout header");
    require((p.state == 384 && p.layers == 4 && p.td_count == 1779) ||
            (p.state == 512 && p.layers == 6 && p.td_count == 3434) ||
            (p.state == 768 && p.layers == 12 && p.td_count == 10015), "unsupported PR model/task count");
    require(p.td_size >= 40 && p.td_size <= 65536 && p.command_bytes >= p.td_size &&
            p.command_bytes % 16 == 0 && p.command_bytes <= 16 * 1024 * 1024 &&
            nb == size_t(p.state == 768 ? 7 : 6) && np == size_t(p.state == 768 ? 6 : 5), "invalid PR command/buffer counts");
    for (size_t i = 0; i < nb; ++i) {
        int bank = -1;
        size_t bytes = 0;
        input >> bank >> bytes;
        require(bool(input) && bank >= 0 && bank < 32 && bank != 1 && bytes > 0 && bytes <= 256 * 1024 * 1024 &&
                p.buffers.emplace(bank, bytes).second, "invalid/duplicate native PR buffer");
    }
    std::set<int> actual;
    for (const auto & b : p.buffers) actual.insert(b.first);
    auto expected = std::set<int>{0, 2, 3, 4, 5, 6};
    if (p.state == 768) expected.insert(8);
    require(actual == expected, "unsupported native PR BAR set");
    require(p.buffers.at(0) > p.command_bytes, "native PR coefficients must follow commands");
    for (size_t i = 0; i < p.ports.size(); ++i) {
        auto & port = p.ports[i];
        input >> port.bank >> port.offset >> port.channels >> port.height >> port.width >> port.plane >> port.row;
        require(bool(input) && port.bank == int(i == 0 ? 5 : i == 1 ? 4 : 6) && port.offset == 0 &&
                port.channels == size_t(i == 0 ? 80 : i == 1 ? p.state : 1) &&
                port.height == size_t(i == 2 ? 1500 : 1) && port.width == size_t(i == 0 ? 3000 : i == 1 ? 1500 : p.state) &&
                port.row >= port.width * 2 && port.row <= 16 * 1024 * 1024 && port.row % 2 == 0 &&
                port.plane >= port.height * port.row && port.plane <= 16 * 1024 * 1024 && port.plane % 2 == 0 &&
                port.end() <= p.buffers.at(port.bank), "invalid native PR padded port");
    }
    require(p.ports[2].row == size_t(p.state) * 2, "native PR output rows must be tight");
    std::set<std::string> names;
    for (size_t i = 0; i < np; ++i) {
        Payload a;
        input >> a.name >> a.bytes >> a.crc >> a.bank >> a.offset;
        require(bool(input) && a.bytes > 0 && a.bytes <= 256 * 1024 * 1024 && names.insert(a.name).second,
                "invalid/duplicate native PR payload");
        bool ok = false;
        if (a.name == "commands.bin") ok = a.bank == 0 && a.offset == 0 && a.bytes == p.command_bytes;
        if (a.name == "coefficients-1.bin") ok = a.bank == 0 && a.offset == p.command_bytes && a.bytes < p.buffers.at(0) - p.command_bytes;
        if (a.name == "constants.bin") ok = a.bank == 2 && a.offset == 0 && a.bytes == p.buffers.at(2);
        if (a.name == "coefficients-8.bin") ok = p.state == 768 && a.bank == 8 && a.offset == 0 && a.bytes == p.buffers.at(8);
        if (a.name == "positions.bin") ok = a.bank == -1 && a.offset == 0 && a.bytes == size_t(p.state) * 1500 * 2;
        if (a.name == "bootstrap.bin") ok = a.bank == -2 && a.offset == 0 && a.bytes == p.td_size;
        require(ok && (a.bank < 0 || (a.offset <= p.buffers.at(a.bank) && a.bytes <= p.buffers.at(a.bank) - a.offset)),
                "invalid native PR payload destination");
        p.payloads.push_back(a);
    }
    auto wanted = std::set<std::string>{"commands.bin", "coefficients-1.bin", "constants.bin", "positions.bin", "bootstrap.bin"};
    // There are five common payloads; the two-bank small adds one.
    require(np == wanted.size() + size_t(p.state == 768), "native PR payload count mismatch");
    if (p.state == 768) wanted.insert("coefficients-8.bin");
    require(names == wanted, "native PR payload set mismatch");
    input >> std::ws;
    require(input.eof(), "trailing native PR layout data");
    return p;
}

static uint32_t crc32(const uint8_t * data, size_t bytes) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> result{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t n = i;
            for (int bit = 0; bit < 8; ++bit) n = (n >> 1) ^ ((n & 1) ? 0xedb88320u : 0u);
            result[i] = n;
        }
        return result;
    }();
    uint32_t value = ~0u;
    for (size_t i = 0; i < bytes; ++i) value = table[(value ^ data[i]) & 255] ^ (value >> 8);
    return ~value;
}

using Payloads = std::map<std::string, std::vector<uint8_t>>;

static Payloads load_payloads(const std::string & directory, const Plan & plan) {
    Payloads result;
    for (const auto & p : plan.payloads) {
        std::ifstream file(directory + "/" + p.name, std::ios::binary | std::ios::ate);
        require(bool(file) && file.tellg() == std::streamoff(p.bytes), "native PR payload length mismatch");
        auto & bytes = result[p.name];
        bytes.resize(p.bytes);
        file.seekg(0);
        file.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
        require(bool(file) && crc32(bytes.data(), bytes.size()) == p.crc, "native PR payload checksum mismatch");
    }
    auto bootstrap = std::vector<uint8_t>(result.at("commands.bin").begin(), result.at("commands.bin").begin() + plan.td_size);
    bootstrap[2] = 0x40;
    require(bootstrap == result.at("bootstrap.bin"), "native PR bootstrap differs from first descriptor");
    const auto & positions = result.at("positions.bin");
    for (size_t i = 0; i < positions.size(); i += 2)
        require((uint16_t(positions[i]) | uint16_t(positions[i + 1]) << 8) % 0x8000 < 0x7c00, "nonfinite native PR positions");
    return result;
}

struct Buffer { uint32_t handle = 0; size_t size = 0; uint8_t * map = nullptr; };
struct Transport {
    virtual ~Transport() = default;
    virtual Buffer allocate(size_t bytes) = 0;
    virtual void release(Buffer & b) = 0;
    virtual void submit(Submit & request) = 0;
};

static void pack(const uint16_t * source, uint8_t * target, size_t size, const Port & p) {
    std::memset(target, 0, size);
    for (size_t channel = 0; channel < p.channels; ++channel)
        for (size_t row = 0; row < p.height; ++row)
            std::memcpy(target + p.offset + channel * p.plane + row * p.row,
                        source + (channel * p.height + row) * p.width, p.width * 2);
}

static void read_lines(uint8_t * output, const uint8_t * input, size_t bytes) {
#if defined(__aarch64__)
    for (size_t i = 0; i < bytes; i += 64) {
        asm volatile("ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%0]\n\t"
                     "st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%1]"
                     : : "r"(input + i), "r"(output + i) : "v0", "v1", "v2", "v3", "memory");
    }
#else
    std::memcpy(output, input, bytes);
#endif
}

struct Timing { double convert_ms, pack_ms, clear_ms, dispatch_ms, read_ms, widen_ms, total_ms; };

class Runtime {
public:
    Plan plan;
    std::unique_ptr<Transport> transport;
    std::array<Buffer, 32> buffers;
    Buffer bootstrap;
    Submit request{};
    unsigned long long submissions = 0;
    int read_workers = 1;
    Timing timing{};
    std::vector<uint16_t> mel16, output16;

    Runtime(Plan p, const Payloads & payloads, std::unique_ptr<Transport> t) : plan(std::move(p)), transport(std::move(t)) {
        try {
            for (const auto & item : plan.buffers) buffers[item.first] = transport->allocate(item.second);
            bootstrap = transport->allocate(plan.td_size);
            for (const auto & p : plan.payloads) {
                const auto & data = payloads.at(p.name);
                if (p.bank >= 0) std::memcpy(buffers[p.bank].map + p.offset, data.data(), data.size());
                if (p.bank == -2) std::memcpy(bootstrap.map, data.data(), data.size());
                if (p.bank == -1) {
                    const auto & port = plan.ports[1];
                    pack(reinterpret_cast<const uint16_t *>(data.data()), buffers[port.bank].map, buffers[port.bank].size, port);
                }
            }
            request.tsk_size = plan.command_bytes;
            request.td_count = plan.td_count;
            request.td_size = plan.td_size;
            request.btsp_handle = bootstrap.handle;
            for (const auto & b : plan.buffers) request.handles[b.first] = buffers[b.first].handle;
            require(request.handles[1] == 0, "driver BAR 1 must remain synthesized");
            mel16.resize(plan.ports[0].count());
            output16.resize(plan.ports[2].count());
        } catch (...) { clear(); throw; }
    }
    ~Runtime() { clear(); }
    void clear() {
        transport->release(bootstrap);
        for (auto & b : buffers) transport->release(b);
    }

    void encode(int64_t n_mel, int64_t n_len, const float * mel, float * output) {
        require(n_mel == 80 && n_len == 3000 && mel && output, "native PR encoder requires full-context mel [80,3000]");
        using Clock = std::chrono::steady_clock;
        const auto begin = Clock::now();
        auto * half = reinterpret_cast<__fp16 *>(mel16.data());
        for (size_t i = 0; i < mel16.size(); ++i) {
            require(std::isfinite(mel[i]) && std::fabs(mel[i]) <= 65504.f, "nonfinite/out-of-range native PR mel");
            half[i] = __fp16(mel[i]);
        }
        const auto converted = Clock::now();
        const auto & mel_port = plan.ports[0];
        pack(mel16.data(), buffers[mel_port.bank].map, buffers[mel_port.bank].size, mel_port);
        const auto packed = Clock::now();
        std::memset(buffers[3].map, 0, buffers[3].size);
        const auto & out_port = plan.ports[2];
        auto * out = reinterpret_cast<uint16_t *>(buffers[out_port.bank].map + out_port.offset);
        std::fill(out, out + output16.size(), uint16_t(0x7e00));
        const auto prepared = Clock::now();
        transport->submit(request);
        const auto completed = Clock::now();
        ++submissions;
        const auto * source = reinterpret_cast<const uint8_t *>(out);
        auto * target = reinterpret_cast<uint8_t *>(output16.data());
        const size_t lines = output16.size() * 2 / 64;
#ifdef _OPENMP
        int actual_workers = 1;
#pragma omp parallel num_threads(4)
        {
#pragma omp single
            actual_workers = omp_get_num_threads();
            const size_t first = lines * omp_get_thread_num() / omp_get_num_threads();
            const size_t last = lines * (omp_get_thread_num() + 1) / omp_get_num_threads();
            read_lines(target + first * 64, source + first * 64, (last - first) * 64);
        }
        read_workers = actual_workers;
#else
        read_lines(target, source, lines * 64);
#endif
        const auto read = Clock::now();
        const auto * encoded = reinterpret_cast<const __fp16 *>(output16.data());
        for (size_t i = 0; i < output16.size(); ++i) {
            require(std::isfinite(float(encoded[i])), "native PR encoder produced nonfinite/unwritten output");
            output[i] = float(encoded[i]);
        }
        const auto end = Clock::now();
        auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        timing = {ms(begin, converted), ms(converted, packed), ms(packed, prepared), ms(prepared, completed),
                  ms(completed, read), ms(read, end), ms(begin, end)};
    }
};

static bool native_m1() {
#if defined(__linux__) && defined(__aarch64__)
    std::ifstream file("/proc/device-tree/compatible", std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return data.find(std::string("apple,t8103\0", 12)) != std::string::npos;
#else
    return false;
#endif
}

class LinuxTransport final : public Transport {
    int fd = -1;
public:
    LinuxTransport() {
        require(native_m1(), "native PR replay requires base-M1 Asahi Linux");
        const char * requested = std::getenv("ANE_DEVICE");
        glob_t paths{};
        require(!glob(requested ? requested : "/dev/accel/accel*", 0, nullptr, &paths), "no ANE accel device");
        for (size_t i = 0; i < paths.gl_pathc; ++i) {
            const char * path = paths.gl_pathv[i];
            const char * name = std::strrchr(path, '/');
            struct stat info{};
            if (!name || stat(path, &info) || !S_ISCHR(info.st_mode)) continue;
            std::string driver = std::string("/sys/class/accel/") + (name + 1) + "/device/driver";
            char * resolved = realpath(driver.c_str(), nullptr);
            if (!resolved) continue;
            const char * suffix = std::strrchr(resolved, '/');
            const bool is_ane = suffix && !std::strcmp(suffix + 1, "ane");
            std::free(resolved);
            if (is_ane) fd = open(path, O_RDWR | O_CLOEXEC);
            if (fd >= 0) break;
        }
        globfree(&paths);
        require(fd >= 0, "no accessible ane driver device");
    }
    ~LinuxTransport() override { if (fd >= 0) close(fd); }
    Buffer allocate(size_t bytes) override {
        const long page = sysconf(_SC_PAGESIZE);
        require(page > 0 && (page & (page - 1)) == 0, "invalid host page size");
        Buffer b;
        b.size = (bytes + size_t(page) - 1) & ~(size_t(page) - 1);
        BOInit request{};
        request.size = b.size;
        require(!ioctl(fd, bo_init_op, &request) && request.handle, "ANE buffer allocation failed");
        b.handle = request.handle;
        void * mapping = mmap(nullptr, b.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, request.offset);
        if (mapping == MAP_FAILED) { release(b); throw std::runtime_error("ANE buffer mapping failed"); }
        b.map = static_cast<uint8_t *>(mapping);
        std::memset(b.map, 0, b.size);
        return b;
    }
    void release(Buffer & b) override {
        if (b.map) munmap(b.map, b.size);
        if (b.handle) { BOFree request{b.handle, 0}; ioctl(fd, bo_free_op, &request); }
        b = {};
    }
    void submit(Submit & request) override {
        require(!ioctl(fd, submit_op, &request), "ANE full-encoder submission failed");
    }
};
} // namespace whisper_pr

struct whisper_aneforge_context { std::unique_ptr<whisper_pr::Runtime> runtime; };

struct whisper_aneforge_context * whisper_aneforge_init(const char * directory) {
    try {
        whisper_pr::require(whisper_pr::native_m1(), "native PR replay requires base-M1 Asahi Linux; no device access attempted");
        whisper_pr::require(directory && *directory, "native PR payload directory required");
        std::ifstream layout(std::string(directory) + "/native-layout.txt");
        auto plan = whisper_pr::parse_plan(layout);
        auto payloads = whisper_pr::load_payloads(directory, plan);
        auto transport = std::unique_ptr<whisper_pr::Transport>(new whisper_pr::LinuxTransport);
        auto ctx = std::unique_ptr<whisper_aneforge_context>(new whisper_aneforge_context);
        ctx->runtime.reset(new whisper_pr::Runtime(std::move(plan), payloads, std::move(transport)));
        std::fprintf(stderr, "ASAHI_PR_ANE ready: state=%d layers=%d tasks=%u coefficient_banks=%d\n",
                     ctx->runtime->plan.state, ctx->runtime->plan.layers, ctx->runtime->plan.td_count,
                     ctx->runtime->plan.state == 768 ? 2 : 1);
        return ctx.release();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "ASAHI_PR_ANE init failed: %s\n", error.what());
        return nullptr;
    }
}

bool whisper_asahi_full_model_matches(const whisper_aneforge_context * ctx, int mels, int context, int state, int layers) {
    return ctx && ctx->runtime && mels == 80 && context == 1500 &&
           state == ctx->runtime->plan.state && layers == ctx->runtime->plan.layers;
}

void whisper_aneforge_encode(struct whisper_aneforge_context * ctx, int64_t n_mel, int64_t n_len, const float * mel, float * out) {
    try {
        whisper_pr::require(ctx && ctx->runtime, "uninitialized native PR encoder");
        // whisper.cpp passes GGML dimensions: frames first, mel channels second.
        ctx->runtime->encode(n_len, n_mel, mel, out);
        if (std::getenv("WHISPER_PROFILE")) {
            const auto & t = ctx->runtime->timing;
            std::fprintf(stderr, "ASAHI_PR_PROFILE encoder: convert=%.3f pack=%.3f clear=%.3f dispatch=%.3f read=%.3f widen=%.3f total=%.3f ms submissions=1 read_workers=%d\n",
                         t.convert_ms, t.pack_ms, t.clear_ms, t.dispatch_ms, t.read_ms, t.widen_ms, t.total_ms, ctx->runtime->read_workers);
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "ASAHI_PR_ANE encode failed: %s\n", error.what());
        std::abort();
    }
}

void whisper_aneforge_free(struct whisper_aneforge_context * ctx) { delete ctx; }
