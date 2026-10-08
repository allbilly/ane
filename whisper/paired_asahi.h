#pragma once
// Linux transport for macos_precision.cpp's unchanged paired arithmetic.
// The input/output allocations and cached views are shared by all 24 plans.
extern "C" {
#include "ane_matmul.h"
}
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <omp.h>

namespace whisper_paired {
struct Init { uint32_t handle, pad; uint64_t size, offset; };
struct Free { uint32_t handle, pad; };
struct Submit { uint64_t size; uint32_t count, descriptor, handles[32], bootstrap, pad; };
static_assert(sizeof(Init) == 24 && sizeof(Free) == 8 && sizeof(Submit) == 152, "ANE DRM ABI");
constexpr unsigned long init_op = 0xc0186441, free_op = 0xc0086442, submit_op = 0xc0986443;

struct Buffer {
    int fd;
    uint32_t handle = 0;
    size_t bytes = 0;
    void * data = nullptr;
    explicit Buffer(int device) : fd(device) {}
    Buffer(const Buffer &) = delete;
    ~Buffer() {
        if (data) munmap(data, bytes);
        if (handle) { Free request{handle, 0}; ioctl(fd, free_op, &request); }
    }
    void allocate(size_t size) {
        GGML_ASSERT(!handle && size);
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
        bytes = (size + page - 1) & ~(page - 1);
        Init request{0, 0, bytes, 0};
        if (ioctl(fd, init_op, &request) || !request.handle) GGML_ABORT("paired ANE buffer allocation failed");
        handle = request.handle;
        data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, request.offset);
        if (data == MAP_FAILED) { data = nullptr; GGML_ABORT("paired ANE buffer mapping failed"); }
        std::memset(data, 0, bytes);
    }
};

inline uint32_t checksum(const std::vector<char> & bytes) {
    static const auto table = [] {
        std::array<uint32_t, 256> values{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t n = i;
            for (int bit = 0; bit < 8; ++bit) n = (n >> 1) ^ ((n & 1) ? 0xedb88320u : 0u);
            values[i] = n;
        }
        return values;
    }();
    uint32_t crc = ~0u;
    for (unsigned char b : bytes) crc = table[(crc ^ b) & 255] ^ (crc >> 8);
    return ~crc;
}

inline std::vector<char> payload(const std::string & root, const char * name, size_t size, uint32_t crc) {
    std::ifstream file(root + "/" + name + ".bin", std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(size)) GGML_ABORT("paired payload size differs from descriptor");
    std::vector<char> bytes(size);
    file.seekg(0);
    if (!file.read(bytes.data(), bytes.size()) || checksum(bytes) != crc)
        GGML_ABORT("paired payload checksum differs from descriptor");
    return bytes;
}

class Device {
    std::unique_ptr<AneDevice, decltype(&ane_device_close)> device;
public:
    int fd;
    Buffer input, output;
    std::vector<float16_t> feed, result;
    std::vector<char> padded;
    int tasks = 0, read_workers = 0;
    int64_t write_us = 0, ioctl_us = 0, read_us = 0, copy_us = 0;
    Device() : device(ane_device_open(), ane_device_close), fd(ane_device_fd(device.get())), input(fd), output(fd),
               feed(1536*6000), result(3072*6000), padded(3072*12032) {
        if (fd < 0) GGML_ABORT("paired projections require accessible base-M1 ANE");
        ane_device_read_threads(device.get(), 4);
        input.allocate(1536*12032);
        output.allocate(3072*12032);
    }
    void reset_profile() {
        tasks = read_workers = 0;
        write_us = ioctl_us = read_us = copy_us = 0;
    }
};

class Program {
    Device & device;
    Buffer command, constants, bootstrap;
    Submit request{};
    int k, n;
public:
    Program(Device & d, const std::string & root, int inputs, int outputs)
        : device(d), command(d.fd), constants(d.fd), bootstrap(d.fd), k(inputs), n(outputs) {
        std::ifstream layout(root + "/native-layout.txt");
        std::string magic;
        size_t commands_size, coefficients_size, constants_size, stride, input_size, output_size;
        int actual_k, actual_n;
        std::array<uint32_t, 5> crc{};
        layout >> magic >> request.count >> request.descriptor >> commands_size >> coefficients_size
               >> constants_size >> actual_k >> actual_n >> stride >> input_size >> output_size;
        for (auto & value : crc) layout >> value;
        if (!layout || magic != "ANE_WHISPER_PAIRED_V1" || actual_k != k || actual_n != n ||
            !((k == 384 && n == 384) || (k == 384 && n == 1536) || (k == 1536 && n == 384)) ||
            request.count != uint32_t(k == 384 && n == 384 ? 1 : 2) || request.descriptor != 628 ||
            commands_size != 32768 || coefficients_size != size_t(k)*n*2 || constants_size != 16384 ||
            stride != 12032 || input_size != size_t(k)*stride || output_size != size_t(2*n)*stride)
            GGML_ABORT("paired native port/task contract changed");
        layout >> std::ws;
        if (!layout.eof()) GGML_ABORT("trailing paired native descriptor data");
        const auto commands = payload(root, "commands", commands_size, crc[0]);
        const auto constant_bytes = payload(root, "constants", constants_size, crc[1]);
        const auto coefficients = payload(root, "coefficients", coefficients_size, crc[2]);
        const auto first = payload(root, "bootstrap", 628, crc[3]);
        payload(root, "weights", 128+coefficients_size, crc[4]);
        // Follow the complete captured chain without changing its registers.
        size_t offset = 0, size = 628;
        for (uint32_t task = 0; task < request.count; ++task) {
            if (size < 40 || size % 4 || offset + size > commands.size()) GGML_ABORT("paired task bounds changed");
            uint32_t header[10];
            std::memcpy(header, commands.data()+offset, sizeof(header));
            if ((header[0] & 0xffff) != task) GGML_ABORT("paired task ID changed");
            if (task + 1 == request.count) {
                if (header[7] || !(header[0] & (1u << 25))) GGML_ABORT("paired task chain is incomplete");
            } else {
                if (header[7] < offset+size || header[7] % 4) GGML_ABORT("paired task chain overlaps");
                offset = header[7]; size = (((header[1] >> 16) & 511) + 1)*4;
            }
        }
        auto expected = std::vector<char>(commands.begin(), commands.begin()+628);
        expected[2] = 0x40;
        if (first != expected) GGML_ABORT("paired bootstrap differs from first task");
        command.allocate(commands_size+coefficients_size+1);
        constants.allocate(constants_size); bootstrap.allocate(628);
        std::memcpy(command.data, commands.data(), commands.size());
        std::memcpy(static_cast<char *>(command.data)+commands.size(), coefficients.data(), coefficients.size());
        std::memcpy(constants.data, constant_bytes.data(), constant_bytes.size());
        std::memcpy(bootstrap.data, first.data(), first.size());
        request.size = commands_size;
        request.handles[0] = command.handle; request.handles[2] = constants.handle;
        request.handles[4] = device.input.handle; request.handles[5] = device.output.handle;
        request.bootstrap = bootstrap.handle;
    }
    void execute(int ith, int nth) {
        // Use the existing ggml workers. A nested readback team competes with
        // the enclosing graph's workers and needlessly oversubscribes the CPU.
        if (nth != 4 || !omp_in_parallel() || omp_get_num_threads() != nth)
            GGML_ABORT("paired Linux encoder requires four OpenMP graph workers");
        const auto t0 = ggml_time_us();
        for (int row = k*ith/nth; row < k*(ith+1)/nth; ++row)
            std::memcpy(static_cast<char *>(device.input.data)+row*12032, device.feed.data()+row*6000, 12000);
        // Poison every logical output; the shared combine checks finiteness.
        const int begin = 2*n*ith/nth, end = 2*n*(ith+1)/nth;
        std::memset(device.padded.data()+begin*12032, 0, size_t(end-begin)*12032);
        auto * poison = reinterpret_cast<uint16_t *>(device.padded.data());
        for (int row = begin; row < end; ++row)
            std::fill_n(poison+row*6016, 6000, uint16_t(0x7e00));
        std::memcpy(static_cast<char *>(device.output.data)+begin*12032,
                    device.padded.data()+begin*12032, size_t(end-begin)*12032);
        #pragma omp barrier
        const auto t1 = ggml_time_us();
        if (!ith && ioctl(device.fd, submit_op, &request)) GGML_ABORT("paired ANE submission failed; no CPU fallback");
        #pragma omp barrier
        const auto t2 = ggml_time_us();
        if (ane_copy_uncached(device.padded.data()+begin*12032,
                             static_cast<const char *>(device.output.data)+begin*12032,
                             size_t(end-begin)*12032, 1) != 1)
            GGML_ABORT("paired native readback failed");
        #pragma omp barrier
        const auto t3 = ggml_time_us();
        for (int row = begin; row < end; ++row)
            std::memcpy(device.result.data()+row*6000, device.padded.data()+row*12032, 12000);
        #pragma omp barrier
        const auto t4 = ggml_time_us();
        if (!ith) {
            device.tasks += request.count; device.read_workers = nth;
            device.write_us += t1-t0; device.ioctl_us += t2-t1;
            device.read_us += t3-t2; device.copy_us += t4-t3;
        }
    }
};
}
