// Host verification with captured inputs/outputs; the transport does not run ANE.
#include "whisper/asahi_full_encoder.cpp"
#include <limits>

using namespace whisper_pr;

struct State {
    unsigned created = 0, released = 0, calls = 0;
    int fail_at = -1;
    bool fail_submit = false, leave_unwritten = false;
    std::map<uint32_t, std::vector<uint8_t>> memory;
};

class FakeTransport final : public Transport {
public:
    std::shared_ptr<State> state;
    Plan plan;
    const Payloads & payloads;
    std::vector<uint16_t> expected, mel;
    std::vector<uint8_t> padded_mel, padded_positions;
    FakeTransport(std::shared_ptr<State> s, const Plan & p, const Payloads & data) : state(std::move(s)), plan(p), payloads(data) {}
    Buffer allocate(size_t bytes) override {
        if (int(state->created) == state->fail_at) throw std::runtime_error("test allocation failed");
        const uint32_t handle = ++state->created;
        auto & data = state->memory[handle];
        data.resize(bytes);
        Buffer b;
        b.handle = handle; b.size = bytes; b.map = data.data();
        return b;
    }
    void release(Buffer & b) override {
        if (b.handle) { state->memory.erase(b.handle); ++state->released; }
        b = {};
    }
    void submit(Submit & request) override {
        ++state->calls;
        require(request.handles[1] == 0 && request.tsk_size == plan.command_bytes && request.td_size == plan.td_size &&
                request.td_count == plan.td_count && request.pad == 0, "host test submission ABI mismatch");
        for (const auto & b : plan.buffers) require(request.handles[b.first] != 0, "host test missing BAR");
        for (const auto & p : plan.payloads) {
            if (p.bank >= 0) {
                const auto & actual = state->memory.at(request.handles[p.bank]);
                const auto & expected_bytes = payloads.at(p.name);
                require(std::equal(expected_bytes.begin(), expected_bytes.end(), actual.begin() + p.offset),
                        "host test coefficients/commands changed");
            }
        }
        require(state->memory.at(request.btsp_handle) == payloads.at("bootstrap.bin"), "host test bootstrap mismatch");
        const auto & scratch = state->memory.at(request.handles[3]);
        require(std::all_of(scratch.begin(), scratch.end(), [](uint8_t b) { return b == 0; }), "host test scratch not cleared");
        require(state->memory.at(request.handles[5]) == padded_mel, "host test padded mel mismatch");
        require(state->memory.at(request.handles[4]) == padded_positions, "host test padded positions mismatch");
        auto & output = state->memory.at(request.handles[6]);
        const auto * half = reinterpret_cast<const uint16_t *>(output.data());
        require(std::all_of(half, half + expected.size(), [](uint16_t b) { return b == 0x7e00; }), "host test missing output sentinel");
        if (state->fail_submit) throw std::runtime_error("test ioctl failed");
        if (!state->leave_unwritten) std::memcpy(output.data(), expected.data(), expected.size() * 2);
    }
};

template<typename T> static std::vector<T> read_array(const std::string & path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file) && file.tellg() == std::streamoff(count * sizeof(T)), "host fixture size mismatch");
    std::vector<T> result(count);
    file.seekg(0);
    file.read(reinterpret_cast<char *>(result.data()), result.size() * sizeof(T));
    require(bool(file), "host fixture read failed");
    return result;
}

template<typename F> static void rejects(F action, const char * reason) {
    bool failed = false;
    try { action(); } catch (const std::runtime_error & e) { failed = std::strstr(e.what(), reason) != nullptr; }
    require(failed, "host rejection test failed");
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3 || (argc == 4 && !std::strcmp(argv[3], "zero-mel")),
                "usage: test_pr_native_encoder PAYLOAD_DIRECTORY FIXTURE_DIRECTORY [zero-mel]");
        const std::string directory = argv[1], fixtures = argv[2];
        std::ifstream file(directory + "/native-layout.txt");
        const std::string descriptor((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        std::istringstream stream(descriptor);
        auto plan = parse_plan(stream);
        auto payloads = load_payloads(directory, plan);
        const std::string crc_test = "123456789";
        require(crc32(reinterpret_cast<const uint8_t *>(crc_test.data()), crc_test.size()) == 0xcbf43926, "CRC32 standard vector failed");
        for (const auto & variant : {std::string("invalid\n") + descriptor, descriptor + "extra\n"}) {
            rejects([&] { std::istringstream bad(variant); parse_plan(bad); }, variant == descriptor + "extra\n" ? "trailing" : "header");
        }
        auto corrupted = plan;
        corrupted.payloads[0].crc ^= 1;
        rejects([&] { load_payloads(directory, corrupted); }, "checksum");
        auto state = std::make_shared<State>();
        {
            auto fake = std::unique_ptr<FakeTransport>(new FakeTransport(state, plan, payloads));
            auto * transport = fake.get();
            whisper_aneforge_context context;
            context.runtime.reset(new Runtime(plan, payloads, std::move(fake)));
            auto & runtime = *context.runtime;
            transport->padded_positions = read_array<uint8_t>(fixtures + "/positions.padded.bin", plan.buffers.at(4));
            require(whisper_asahi_full_model_matches(&context, 80, 1500, plan.state, plan.layers), "model dimensions rejected");
            require(!whisper_asahi_full_model_matches(&context, 80, 1500, plan.state + 1, plan.layers) &&
                    !whisper_asahi_full_model_matches(&context, 80, 1499, plan.state, plan.layers), "wrong model dimensions accepted");
            const auto cases = argc == 4 ? std::vector<const char *>{"zero-mel"}
                                        : std::vector<const char *>{"zero-mel", "jfk", "jfk-first-5s", "jfk-repeat"};
            for (const char * name : cases) {
                auto mel = read_array<float>(fixtures + "/" + name + ".mel.f32", 80 * 3000);
                transport->mel = read_array<uint16_t>(fixtures + "/" + name + ".mel.f16", 80 * 3000);
                transport->expected = read_array<uint16_t>(fixtures + "/" + name + ".out.f16", size_t(plan.state) * 1500);
                const auto expected_f32 = read_array<float>(fixtures + "/" + name + ".out.f32", size_t(plan.state) * 1500);
                transport->padded_mel = read_array<uint8_t>(fixtures + "/" + name + ".mel.padded.bin", plan.buffers.at(5));
                std::vector<float> output(size_t(plan.state) * 1500);
                const auto before = runtime.submissions;
                whisper_aneforge_encode(&context, 3000, 80, mel.data(), output.data());
                require(runtime.submissions == before + 1 && runtime.mel16 == transport->mel, "wrong submission count/conversion");
                require(!std::memcmp(output.data(), expected_f32.data(), output.size() * sizeof(float)), "wrong native readback/widening");
#ifdef _OPENMP
                require(runtime.read_workers == 4, "four-worker native readback not exercised");
#else
                require(runtime.read_workers == 1, "serial native readback worker mismatch");
#endif
                const auto & t = runtime.timing;
                require(std::fabs(t.total_ms - t.convert_ms - t.pack_ms - t.clear_ms - t.dispatch_ms - t.read_ms - t.widen_ms) < 1e-8,
                        "native timing boundaries do not add up");
            }
            auto mel = std::vector<float>(80 * 3000, 0.f), output = std::vector<float>(size_t(plan.state) * 1500);
            mel[0] = std::numeric_limits<float>::quiet_NaN();
            const auto calls = state->calls;
            rejects([&] { runtime.encode(80, 3000, mel.data(), output.data()); }, "nonfinite");
            rejects([&] { runtime.encode(80, 2999, mel.data(), output.data()); }, "full-context");
            require(state->calls == calls, "invalid input reached submission");
            mel[0] = 0;
            transport->mel.assign(80 * 3000, 0);
            transport->padded_mel.assign(plan.buffers.at(5), 0);
            state->fail_submit = true;
            const auto completed = runtime.submissions;
            rejects([&] { runtime.encode(80, 3000, mel.data(), output.data()); }, "ioctl failed");
            require(runtime.submissions == completed, "failed submission counted as completed");
            state->fail_submit = false;
            state->leave_unwritten = true;
            rejects([&] { runtime.encode(80, 3000, mel.data(), output.data()); }, "unwritten");
        }
        require(state->memory.empty() && state->released == state->created, "host resources not released");
        auto failed = std::make_shared<State>();
        failed->fail_at = 3;
        rejects([&] { Runtime runtime(plan, payloads, std::unique_ptr<Transport>(new FakeTransport(failed, plan, payloads))); }, "allocation failed");
        require(failed->created == 3 && failed->released == 3 && failed->memory.empty(), "partial allocation leaked");
        if (!native_m1()) require(whisper_aneforge_init("/missing") == nullptr, "native host guard failed");
        std::printf("PASS_HOST_NATIVE_TRANSFER state=%d tasks=%u cases=%d coefficient_banks=%d hardware_execution=unverified\n",
                    plan.state, plan.td_count, argc == 4 ? 1 : 4, plan.state == 768 ? 2 : 1);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "host native test failed: %s\n", error.what());
        return 1;
    }
}
