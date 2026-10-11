#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
#include "kora/task_graph.hpp"
#include "memory/suballocator.hpp"

namespace {
using Clock = std::chrono::steady_clock;

struct Options {
    std::size_t iterations = 100;
    std::size_t warmup = 10;
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
            std::cout << "Usage: kora-bench [--iterations N] [--warmup N] [--smoke]\n"
                         "Runs deterministic CPU-side microbenchmarks and writes a JSON report to stdout.\n";
            return false;
        }
        if (argument == "--smoke") {
            options.iterations = 3;
            options.warmup = 1;
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
