#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/scheduler_cost.hpp"
#include "kora/execution_plan.hpp"
#include "kora/runtime.hpp"
#include "kora/task_graph.hpp"
#include "memory/suballocator.hpp"

namespace {
using Clock = std::chrono::steady_clock;

struct Options {
    std::size_t iterations = 100;
    std::size_t warmup = 10;
    bool gpu = false;
};

struct Result {
    std::string name;
    std::vector<std::uint64_t> samples_ns;
    std::uint64_t median_ns = 0;
    std::uint64_t p95_ns = 0;
};

bool parse_count(std::string_view text, std::size_t& value) {
    if (text.empty()) return false;
    std::size_t parsed = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size() || parsed == 0) return false;
    value = parsed;
    return true;
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: kora-bench [--iterations N] [--warmup N] [--smoke] [--gpu]\n"
                         "Runs CPU microbenchmarks; --gpu adds Vulkan tiny-dispatch and vector-add cases.\n";
            return false;
        }
        if (argument == "--smoke") {
            options.iterations = 3;
            options.warmup = 1;
            continue;
        }
        if (argument == "--gpu") {
            options.gpu = true;
            continue;
        }
        if (argument == "--iterations" || argument == "--warmup") {
            if (++index >= argc) return false;
            auto& destination = argument == "--iterations" ? options.iterations : options.warmup;
            if (!parse_count(argv[index], destination)) return false;
            continue;
        }
        return false;
    }
    if (options.iterations > 1'000'000 || options.warmup > 1'000'000) return false;
    return true;
}

std::uint64_t percentile(const std::vector<std::uint64_t>& values, std::size_t percentile_percent) {
    auto ordered = values;
    std::sort(ordered.begin(), ordered.end());
    const auto rank = (percentile_percent * ordered.size() + 99) / 100;
    return ordered[std::max<std::size_t>(rank, 1) - 1];
}

Result run(std::string name, const std::function<void()>& operation, const Options& options) {
    for (std::size_t index = 0; index < options.warmup; ++index) operation();
    Result result{std::move(name), {}, 0, 0};
    result.samples_ns.reserve(options.iterations);
    for (std::size_t index = 0; index < options.iterations; ++index) {
        const auto begin = Clock::now();
        operation();
        const auto end = Clock::now();
        result.samples_ns.push_back(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
    }
    result.median_ns = percentile(result.samples_ns, 50);
    result.p95_ns = percentile(result.samples_ns, 95);
    return result;
}

bool scheduler_operation() {
    kora::detail::SchedulerCostInput input;
    input.dependency_ready_ns = 100;
    input.backend_available_ns = 450;
    input.input_transfer_ns = 120;
    input.execute_ns = 800;
    input.required_output_ns = 90;
    input.synchronization_ns = 30;
    input.predicted_usage_bytes = 128;
    input.effective_budget_bytes = 1024;
    input.memory_penalty_scale_ns = 500;
    input.uncertainty_sigma_ns = 20;
    input.locality_saved_ns = 10;
    input.locality_cap_ns = 50;
    return static_cast<bool>(kora::detail::evaluate_scheduler_cost(input));
}

bool task_graph_operation() {
    kora::TaskGraph graph;
    kora::TaskId previous = 0;
    for (std::uint32_t index = 0; index < 64; ++index) {
        kora::TaskMetadata metadata;
        metadata.type = kora::TaskType::Dispatch;
        auto task = graph.add_task(std::move(metadata));
        if (!task) return false;
        if (previous != 0 && !graph.add_dependency(previous, task.value())) return false;
        previous = task.value();
    }
    return static_cast<bool>(graph.finalize());
}

kora::Status count_task(const kora::TaskNode&, void* state) {
    ++*static_cast<std::uint64_t*>(state);
    return {};
}

bool graph_replay_operation() {
    kora::TaskGraph graph;
    for (std::uint32_t index = 0; index < 64; ++index) {
        kora::TaskMetadata metadata;
        metadata.type = kora::TaskType::HostCallback;
        if (!graph.add_task(std::move(metadata))) return false;
    }
    auto plan = kora::ExecutionPlan::create(graph);
    if (!plan) return false;
    std::uint64_t invoked = 0;
    return static_cast<bool>(plan.value().replay(count_task, &invoked)) && invoked == 64;
}

bool allocation_operation() {
    std::vector<std::byte> bytes(4096);
    std::fill(bytes.begin(), bytes.end(), std::byte{0});
    return bytes.front() == std::byte{0};
}

bool cpu_vector_add_operation() {
    std::array<std::uint32_t, 1024> left{};
    std::array<std::uint32_t, 1024> right{};
    std::array<std::uint32_t, 1024> output{};
    left.fill(3);
    right.fill(7);
    for (std::size_t index = 0; index < output.size(); ++index) output[index] = left[index] + right[index];
    static std::atomic<std::uint32_t> sink{0};
    sink.store(output.back(), std::memory_order_relaxed);
    return sink.load(std::memory_order_relaxed) == 10;
}

bool allocator_throughput_operation() {
    auto created = kora::detail::Suballocator::create(1U << 20U);
    if (!created) return false;
    auto allocator = std::move(created).value();
    std::vector<kora::detail::Suballocation> allocations;
    allocations.reserve(256);
    for (std::uint64_t index = 0; index < 256; ++index) {
        auto allocation = allocator.allocate(32 + (index % 7) * 16, 16);
        if (!allocation) return false;
        allocations.push_back(allocation.value());
    }
    for (const auto& allocation : allocations)
        if (!allocator.deallocate(allocation)) return false;
    return allocator.validate_invariants();
}

bool allocator_fragmentation_operation() {
    auto created = kora::detail::Suballocator::create(1U << 20U);
    if (!created) return false;
    auto allocator = std::move(created).value();
    std::vector<kora::detail::Suballocation> allocations;
    allocations.reserve(128);
    for (std::uint64_t index = 0; index < 128; ++index) {
        auto allocation = allocator.allocate(64 + (index % 3) * 16, 16);
        if (!allocation) return false;
        allocations.push_back(allocation.value());
    }
    for (std::size_t index = 0; index < allocations.size(); index += 2)
        if (!allocator.deallocate(allocations[index])) return false;
    const auto stats = allocator.stats();
    return allocator.validate_invariants() && stats.live_allocations == 64 &&
           stats.external_fragmentation_bytes > 0;
}

#ifdef KORA_BENCH_VECTOR_ADD_FIXTURE_NAME
bool add_gpu_benchmarks(std::vector<Result>& results, const Options& options, const char* executable_path) {
    kora::RuntimeOptions runtime_options;
    runtime_options.telemetry = kora::TelemetryLevel::Off;
    runtime_options.enable_startup_calibration = false;
    runtime_options.disable_autotune = true;
    auto runtime_result = kora::Runtime::create(runtime_options);
    if (!runtime_result) {
        std::cerr << "GPU benchmarks unavailable: " << runtime_result.error().describe() << '\n';
        return false;
    }
    auto runtime = std::move(runtime_result).value();
    kora::KernelMetadata metadata;
    metadata.kernel_id = "kora_bench.vector_add";
    metadata.local_size = {64, 1, 1};
    metadata.push_constant_size = sizeof(std::uint32_t);
    metadata.push_constants = {{"count", "u32", 0, sizeof(std::uint32_t)}};
    metadata.bindings = {
        {0, 0, "input_a", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Read},
        {0, 1, "input_b", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Read},
        {0, 2, "output_data", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Write},
    };
    auto fixture =
        std::filesystem::absolute(executable_path).parent_path() / KORA_BENCH_VECTOR_ADD_FIXTURE_NAME;
    if (!std::filesystem::exists(fixture))
        fixture = std::filesystem::absolute(executable_path).parent_path().parent_path() / "share" / "kora" /
                  "benchmarks" / KORA_BENCH_VECTOR_ADD_FIXTURE_NAME;
    auto kernel = runtime.load_kernel(fixture, "main", std::move(metadata));
    if (!kernel) {
        std::cerr << "GPU benchmarks unavailable: " << kernel.error().describe() << '\n';
        return false;
    }
    constexpr std::uint32_t element_count = 1024;
    std::vector<std::uint32_t> input_a(element_count, 3), input_b(element_count, 7);
    auto a = runtime.create_buffer<std::uint32_t>(element_count);
    auto b = runtime.create_buffer<std::uint32_t>(element_count);
    auto output = runtime.create_buffer<std::uint32_t>(element_count);
    if (!a || !b || !output || !a.value().write(input_a) || !b.value().write(input_b)) return false;
    const auto dispatch = [&](std::uint32_t count) {
        auto event = runtime.dispatch(kernel.value())
                         .global(count)
                         .read(a.value())
                         .read(b.value())
                         .write(output.value())
                         .push(count)
                         .submit();
        if (!event || !event.value().wait()) throw std::runtime_error("Vulkan dispatch failed");
    };
    results.push_back(run("gpu_tiny_dispatch_1", [&] { dispatch(1); }, options));
    dispatch(element_count);
    auto sum_values = output.value().read();
    if (!sum_values || !std::all_of(sum_values.value().begin(), sum_values.value().end(),
                                    [](std::uint32_t value) { return value == 10; }))
        return false;
    results.push_back(run("gpu_vector_add_1024_end_to_end", [&] { dispatch(element_count); }, options));

    kora::KernelMetadata saxpy_metadata;
    saxpy_metadata.kernel_id = "kora_bench.saxpy";
    saxpy_metadata.local_size = {64, 1, 1};
    saxpy_metadata.push_constant_size = sizeof(float) + sizeof(std::uint32_t);
    saxpy_metadata.push_constants = {{"scale", "f32", 0, sizeof(float)},
                                     {"count", "u32", sizeof(float), sizeof(std::uint32_t)}};
    saxpy_metadata.bindings = {
        {0, 0, "input_x", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Read},
        {0, 1, "input_y", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::ReadWrite},
    };
    auto saxpy_fixture = fixture.parent_path() / KORA_BENCH_SAXPY_FIXTURE_NAME;
    if (!std::filesystem::exists(saxpy_fixture))
        saxpy_fixture = fixture.parent_path().parent_path() / "share" / "kora" / "benchmarks" /
                        KORA_BENCH_SAXPY_FIXTURE_NAME;
    auto saxpy = runtime.load_kernel(saxpy_fixture, "main", std::move(saxpy_metadata));
    if (!saxpy) {
        std::cerr << "GPU benchmarks unavailable: " << saxpy.error().describe() << '\n';
        return false;
    }
    std::vector<float> saxpy_x(element_count, 2.0F), saxpy_y(element_count, 1.0F);
    auto x = runtime.create_buffer<float>(element_count);
    auto y = runtime.create_buffer<float>(element_count);
    if (!x || !y || !x.value().write(saxpy_x) || !y.value().write(saxpy_y)) return false;
    struct SaxpyParams {
        float scale;
        std::uint32_t count;
    };
    static_assert(sizeof(SaxpyParams) == sizeof(float) + sizeof(std::uint32_t));
    const SaxpyParams params{2.0F, element_count};
    const auto run_saxpy = [&] {
        if (!y.value().write(saxpy_y)) throw std::runtime_error("SAXPY input reset failed");
        auto event = runtime.dispatch(saxpy.value())
                         .global(element_count)
                         .read(x.value())
                         .read_write(y.value())
                         .push(params)
                         .submit();
        if (!event || !event.value().wait()) throw std::runtime_error("Vulkan SAXPY dispatch failed");
    };
    run_saxpy();
    auto saxpy_values = y.value().read();
    if (!saxpy_values || !std::all_of(saxpy_values.value().begin(), saxpy_values.value().end(),
                                      [](float value) { return value == 5.0F; }))
        return false;
    results.push_back(run("gpu_saxpy_1024_end_to_end", [&] { run_saxpy(); }, options));
    return true;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        for (int index = 1; index < argc; ++index)
            if (std::string_view(argv[index]) == "--help") return 0;
        std::cerr << "invalid kora-bench arguments; use --help for usage\n";
        return 2;
    }
    std::vector<Result> results;
    try {
        for (auto [name, operation] : std::vector<std::pair<std::string, std::function<bool()>>>{
                 {"scheduler_cost", scheduler_operation},
                 {"task_graph_finalize_64", task_graph_operation},
                 {"task_graph_replay_64", graph_replay_operation},
                 {"allocator_allocate_free_256", allocator_throughput_operation},
                 {"allocator_fragmentation_128", allocator_fragmentation_operation},
                 {"cpu_vector_add_1024", cpu_vector_add_operation},
                 {"host_allocation_4k", allocation_operation}}) {
            auto result = run(
                std::move(name),
                [&] {
                    if (!operation()) throw std::runtime_error("benchmark workload failed");
                },
                options);
            results.push_back(std::move(result));
        }
    } catch (const std::exception& error) {
        std::cerr << "benchmark workload failed: " << error.what() << '\n';
        return 1;
    }
    if (options.gpu) {
#ifdef KORA_BENCH_VECTOR_ADD_FIXTURE_NAME
        if (!add_gpu_benchmarks(results, options, argv[0])) return 3;
#else
        std::cerr << "GPU benchmarks require glslangValidator at build time\n";
        return 3;
#endif
    }
    std::cout << "{\"schema_version\":1,\"clock\":\"steady_clock\",\"iterations\":" << options.iterations
              << ",\"warmup\":" << options.warmup << ",\"results\":[";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const auto& result = results[index];
        if (index != 0) std::cout << ',';
        std::cout << "{\"name\":\"" << result.name << "\",\"unit\":\"ns\",\"median\":" << result.median_ns
                  << ",\"p95\":" << result.p95_ns << ",\"samples\":[";
        for (std::size_t sample = 0; sample < result.samples_ns.size(); ++sample) {
            if (sample != 0) std::cout << ',';
            std::cout << result.samples_ns[sample];
        }
        std::cout << "]}";
    }
    std::cout << "]}\n";
    return std::cout ? 0 : 1;
}
