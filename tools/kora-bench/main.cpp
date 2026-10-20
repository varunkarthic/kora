#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#include "core/device_caps.hpp"
#include "core/scheduler_cost.hpp"
#include "kora/cluster.hpp"
#include "kora/execution_plan.hpp"
#include "kora/runtime.hpp"
#include "kora/task_graph.hpp"
#include "kora/worker.hpp"
#include "memory/memory_manager.hpp"
#include "memory/memory_pool.hpp"
#include "memory/staging_ring.hpp"
#include "memory/suballocator.hpp"
#include "memory/transfer_engine.hpp"
#include "vulkan/api.hpp"
#include "vulkan/command_pool_set.hpp"
#include "vulkan/queue_context.hpp"

namespace {
using Clock = std::chrono::steady_clock;

// Matched KORA/raw Vulkan compute tiers: dispatch-bound, intermediate, and throughput-bound sizes.
// Every value is a multiple of the 64-lane workgroup and 8×8 matmul tile used by the fixtures.
constexpr std::array<std::uint32_t, 3> kComputeSizes{4096, 65536, 1048576};
constexpr std::array<std::uint32_t, 3> kMatrixDimensions{16, 64, 128};

struct Options {
    std::size_t iterations = 100;
    std::size_t warmup = 10;
    bool gpu = false;
    bool profile_phases = false;
    bool cluster = false;
};

struct Result {
    std::string name;
    std::vector<std::uint64_t> samples_ns;
    std::uint64_t median_ns = 0;
    std::uint64_t p95_ns = 0;
    std::uint64_t bytes_transferred = 0;
    std::uint64_t throughput_bytes_per_second = 0;
    std::uint32_t thread_count = 0;
};

struct GpuDeviceReport {
    std::uint32_t index = 0;
    std::string name;
    std::string api_version;
    std::string driver_name;
    std::string device_kind;
    std::uint32_t compute_queue_family = 0;
    std::vector<std::uint32_t> kora_gpu_timestamp_sizes;
    std::vector<std::uint32_t> raw_gpu_timestamp_sizes;
};

std::string_view operating_system() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

std::string_view architecture() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#else
    return "unknown";
#endif
}

std::string compiler_name() {
#if defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
#if defined(__apple_build_version__)
    return "AppleClang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." +
           std::to_string(__clang_patchlevel__);
#else
    return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." +
           std::to_string(__clang_patchlevel__);
#endif
#elif defined(__GNUC__)
    return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
           std::to_string(__GNUC_PATCHLEVEL__);
#else
    return "unknown";
#endif
}

std::string cpu_model_name() {
#if defined(__APPLE__)
    std::size_t length = 0;
    if (sysctlbyname("machdep.cpu.brand_string", nullptr, &length, nullptr, 0) == 0 && length > 1) {
        std::string model(length, '\0');
        if (sysctlbyname("machdep.cpu.brand_string", model.data(), &length, nullptr, 0) == 0) {
            model.resize(length - 1);
            return model;
        }
    }
#elif defined(_WIN32)
    if (const auto* model = std::getenv("PROCESSOR_IDENTIFIER"); model != nullptr) return model;
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        const auto separator = line.find(':');
        if (separator == std::string::npos) continue;
        const auto key = line.substr(0, separator);
        if (key != "model name" && key != "Hardware" && key != "Processor") continue;
        const auto start = line.find_first_not_of(" \t", separator + 1);
        if (start != std::string::npos) return line.substr(start);
    }
#endif
    return "unknown";
}

std::string_view measurement_domain(std::string_view name) {
    if (name.starts_with("profile_kora_")) return "runtime_dispatch_phase";
    if (name.ends_with("_gpu_execution")) return "gpu_execution_timestamp";
    if (name.starts_with("cpu_") || name.starts_with("staging_ring_reserve_")) return "host_cpu_execution";
    if (name.starts_with("gpu_") || name.starts_with("raw_vulkan_") || name.starts_with("cluster_"))
        return "host_submit_completion";
    if (name.starts_with("transfer_") || name.starts_with("raw_vulkan_copy_") ||
        name.starts_with("kora_transfer_copy_"))
        return "host_transfer_completion";
    if (name.starts_with("command_pool_")) return "host_api_only";
    return "host_operation";
}

void json_string(std::ostream& out, std::string_view value) {
    out << '"';
    for (const unsigned char c : value) {
        switch (c) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (c < 0x20U) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    out << '"';
}

std::string version_string(std::uint32_t version) {
    return std::to_string(VK_VERSION_MAJOR(version)) + "." + std::to_string(VK_VERSION_MINOR(version)) + "." +
           std::to_string(VK_VERSION_PATCH(version));
}

std::string_view gpu_path(std::string_view name) {
    if (name.starts_with("gpu_")) return "kora_runtime";
    if (name.starts_with("kora_transfer_copy_") || name.starts_with("transfer_"))
        return "kora_transfer_engine";
    if (name.starts_with("raw_vulkan_") || name.starts_with("command_pool_")) return "raw_vulkan";
    return {};
}

bool selected_device_matches(const kora::DeviceId& selected, const kora::vk::PhysicalDeviceInfo& physical) {
    if (selected.uuid.empty()) return true;
    static constexpr char hex[] = "0123456789abcdef";
    std::string uuid;
    uuid.reserve(sizeof(physical.id_properties.deviceUUID) * 2);
    for (const auto byte : physical.id_properties.deviceUUID) {
        uuid.push_back(hex[byte >> 4U]);
        uuid.push_back(hex[byte & 0x0fU]);
    }
    return uuid == selected.uuid;
}

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
            std::cout
                << "Usage: kora-bench [--iterations N] [--warmup N] [--smoke] [--gpu] [--profile-phases] "
                   "[--cluster]\n"
                   "Runs CPU microbenchmarks; --gpu adds Vulkan workloads, --profile-phases adds\n"
                   "telemetry-instrumented dispatch phase diagnostics, and --cluster adds kora::Cluster\n"
                   "dispatches run locally and on a loopback kora-worker on the same device.\n";
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
        if (argument == "--cluster") {
            options.gpu = true;
            options.cluster = true;
            continue;
        }
        if (argument == "--profile-phases") {
            options.gpu = true;
            options.profile_phases = true;
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

Result run_checked(std::string name, const std::function<void()>& operation,
                   const std::function<bool()>& validate, const Options& options) {
    for (std::size_t index = 0; index < options.warmup; ++index) {
        operation();
        if (!validate()) throw std::runtime_error("benchmark correctness check failed: " + name);
    }
    Result result{std::move(name), {}, 0, 0};
    result.samples_ns.reserve(options.iterations);
    for (std::size_t index = 0; index < options.iterations; ++index) {
        const auto begin = Clock::now();
        operation();
        const auto end = Clock::now();
        result.samples_ns.push_back(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
        if (!validate()) throw std::runtime_error("benchmark correctness check failed: " + result.name);
    }
    result.median_ns = percentile(result.samples_ns, 50);
    result.p95_ns = percentile(result.samples_ns, 95);
    return result;
}

std::optional<Result> run_timestamps(std::string name,
                                     const std::function<std::optional<std::uint64_t>()>& operation,
                                     const Options& options) {
    for (std::size_t index = 0; index < options.warmup; ++index)
        if (!operation()) return std::nullopt;
    Result result{std::move(name), {}, 0, 0};
    result.samples_ns.reserve(options.iterations);
    for (std::size_t index = 0; index < options.iterations; ++index) {
        const auto sample = operation();
        if (!sample) return std::nullopt;
        result.samples_ns.push_back(*sample);
    }
    result.median_ns = percentile(result.samples_ns, 50);
    result.p95_ns = percentile(result.samples_ns, 95);
    return result;
}

Result run_transfer(std::string name, std::uint64_t bytes, const std::function<void()>& operation,
                    const Options& options) {
    auto result = run(std::move(name), operation, options);
    result.bytes_transferred = bytes;
    result.throughput_bytes_per_second =
        result.median_ns == 0 ? 0 : bytes * 1'000'000'000ULL / result.median_ns;
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

template <typename T, typename Operation, typename Expected>
Result run_cpu_compute(std::string name, std::vector<T>& output, Operation operation, Expected expected,
                       const Options& options) {
    return run_checked(
        std::move(name), operation,
        [&] {
            for (std::size_t index = 0; index < output.size(); ++index)
                if (!expected(index, output[index])) return false;
            return true;
        },
        options);
}

template <typename Operation>
void parallel_for(kora::CpuExecutor& executor, std::size_t count, Operation operation) {
    const auto workers = executor.worker_count();
    std::vector<kora::Event> events;
    events.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        const auto begin = count * worker / workers;
        const auto end = count * (worker + 1) / workers;
        auto event = executor.submit([&, worker, begin, end] {
            operation(worker, begin, end);
            return kora::Status{};
        });
        if (!event) throw std::runtime_error("CPU executor rejected benchmark work");
        events.push_back(std::move(event).value());
    }
    for (const auto& event : events)
        if (!executor.wait(event)) throw std::runtime_error("CPU executor benchmark task failed");
}

void add_cpu_compute_benchmarks(std::vector<Result>& results, const Options& options) {
    kora::CpuExecutorOptions executor_options;
    executor_options.worker_count = 4;
    auto executor_result = kora::CpuExecutor::create(executor_options);
    if (!executor_result) throw std::runtime_error("could not create CPU benchmark executor");
    auto executor = std::move(executor_result).value();
    const auto worker_count = static_cast<std::uint32_t>(executor->worker_count());
    constexpr std::array<std::size_t, 3> sizes{4096, 65536, 1048576};
    for (const auto count : sizes) {
        std::vector<std::uint32_t> int_left(count), int_right(count), int_output(count);
        for (std::size_t index = 0; index < count; ++index) {
            int_left[index] = static_cast<std::uint32_t>(index % 17);
            int_right[index] = static_cast<std::uint32_t>(index % 29);
        }
        const auto parallel_vector_add = [&] {
            parallel_for(*executor, count, [&](std::size_t, std::size_t begin, std::size_t end) {
                for (auto index = begin; index < end; ++index)
                    int_output[index] = int_left[index] + int_right[index];
            });
        };
        auto parallel_vector_result = run_cpu_compute(
            "cpu_vector_add_parallel_" + std::to_string(count), int_output, parallel_vector_add,
            [&](std::size_t index, std::uint32_t value) {
                return value == int_left[index] + int_right[index];
            },
            options);
        parallel_vector_result.thread_count = worker_count;
        results.push_back(std::move(parallel_vector_result));

        std::vector<float> x(count), y(count), output(count);
        for (std::size_t index = 0; index < count; ++index) {
            x[index] = static_cast<float>(static_cast<int>(index % 97) - 48) * 0.125F;
            y[index] = static_cast<float>(static_cast<int>(index % 53) - 26) * 0.25F;
        }
        const auto saxpy = [&] {
            for (std::size_t index = 0; index < count; ++index) output[index] = 1.5F * x[index] + y[index];
        };
        saxpy();
        const auto saxpy_expected = [&](std::size_t index, float value) {
            return std::abs(value - (1.5F * x[index] + y[index])) <= 1e-6F;
        };
        results.push_back(
            run_cpu_compute("cpu_saxpy_" + std::to_string(count), output, saxpy, saxpy_expected, options));
        const auto parallel_saxpy = [&] {
            parallel_for(*executor, count, [&](std::size_t, std::size_t begin, std::size_t end) {
                for (auto index = begin; index < end; ++index) output[index] = 1.5F * x[index] + y[index];
            });
        };
        auto parallel_saxpy_result = run_cpu_compute("cpu_saxpy_parallel_" + std::to_string(count), output,
                                                     parallel_saxpy, saxpy_expected, options);
        parallel_saxpy_result.thread_count = worker_count;
        results.push_back(std::move(parallel_saxpy_result));

        constexpr std::size_t reduction_width = 64;
        const auto group_count = count / reduction_width;
        std::vector<float> reduced(group_count);
        const auto reduction = [&] {
            for (std::size_t group = 0; group < group_count; ++group) {
                float sum = 0.0F;
                for (std::size_t lane = 0; lane < reduction_width; ++lane)
                    sum += x[group * reduction_width + lane];
                reduced[group] = sum;
            }
        };
        reduction();
        const auto reduction_expected = [&](std::size_t group, float value) {
            double reference = 0.0;
            for (std::size_t lane = 0; lane < reduction_width; ++lane)
                reference += x[group * reduction_width + lane];
            return std::abs(static_cast<double>(value) - reference) <= 1e-3;
        };
        results.push_back(run_cpu_compute("cpu_reduction_" + std::to_string(count), reduced, reduction,
                                          reduction_expected, options));
        const auto parallel_reduction = [&] {
            parallel_for(*executor, group_count, [&](std::size_t, std::size_t begin, std::size_t end) {
                for (auto group = begin; group < end; ++group) {
                    float sum = 0.0F;
                    for (std::size_t lane = 0; lane < reduction_width; ++lane)
                        sum += x[group * reduction_width + lane];
                    reduced[group] = sum;
                }
            });
        };
        auto parallel_reduction_result =
            run_cpu_compute("cpu_reduction_parallel_" + std::to_string(count), reduced, parallel_reduction,
                            reduction_expected, options);
        parallel_reduction_result.thread_count = worker_count;
        results.push_back(std::move(parallel_reduction_result));

        std::vector<float> stencil(count);
        const auto stencil_operation = [&] {
            for (std::size_t index = 0; index < count; ++index) {
                float sum = x[index];
                if (index > 0) sum += x[index - 1];
                if (index + 1 < count) sum += x[index + 1];
                stencil[index] = sum;
            }
        };
        stencil_operation();
        results.push_back(run_cpu_compute(
            "cpu_stencil_" + std::to_string(count), stencil, stencil_operation,
            [&](std::size_t index, float value) {
                auto expected = x[index];
                if (index > 0) expected += x[index - 1];
                if (index + 1 < count) expected += x[index + 1];
                return std::abs(value - expected) <= 1e-6F;
            },
            options));
        const auto parallel_stencil = [&] {
            parallel_for(*executor, count, [&](std::size_t, std::size_t begin, std::size_t end) {
                for (auto index = begin; index < end; ++index) {
                    float sum = x[index];
                    if (index > 0) sum += x[index - 1];
                    if (index + 1 < count) sum += x[index + 1];
                    stencil[index] = sum;
                }
            });
        };
        auto parallel_stencil_result = run_cpu_compute(
            "cpu_stencil_parallel_" + std::to_string(count), stencil, parallel_stencil,
            [&](std::size_t index, float value) {
                auto expected = x[index];
                if (index > 0) expected += x[index - 1];
                if (index + 1 < count) expected += x[index + 1];
                return std::abs(value - expected) <= 1e-6F;
            },
            options);
        parallel_stencil_result.thread_count = worker_count;
        results.push_back(std::move(parallel_stencil_result));

        // Inclusive scan within independent 64-element tiles, matching the GPU scan fixture.
        constexpr std::size_t scan_width = 64;
        std::vector<std::uint32_t> scanned(count);
        const auto scan_tiles = [&](std::size_t first_tile, std::size_t last_tile) {
            for (auto tile = first_tile; tile < last_tile; ++tile) {
                std::uint32_t sum = 0;
                for (auto index = tile * scan_width; index < (tile + 1) * scan_width; ++index)
                    scanned[index] = sum += int_left[index];
            }
        };
        const auto tile_count = count / scan_width;
        std::vector<std::uint32_t> scan_reference(count);
        for (std::size_t index = 0; index < count; ++index)
            scan_reference[index] =
                (index % scan_width == 0 ? 0U : scan_reference[index - 1]) + int_left[index];
        const auto scan_expected = [&](std::size_t index, std::uint32_t value) {
            return value == scan_reference[index];
        };
        results.push_back(run_cpu_compute(
            "cpu_tiled_scan_" + std::to_string(count), scanned, [&] { scan_tiles(0, tile_count); },
            scan_expected, options));
        auto parallel_scan_result = run_cpu_compute(
            "cpu_tiled_scan_parallel_" + std::to_string(count), scanned,
            [&] {
                parallel_for(*executor, tile_count, [&](std::size_t, std::size_t begin, std::size_t end) {
                    scan_tiles(begin, end);
                });
            },
            scan_expected, options);
        parallel_scan_result.thread_count = worker_count;
        results.push_back(std::move(parallel_scan_result));
    }

    for (const auto dimension : {16U, 64U, 128U}) {
        const auto count = static_cast<std::size_t>(dimension) * dimension;
        std::vector<float> a(count, 0.5F), b(count, 0.25F), output(count);
        constexpr std::size_t tile = 32;
        const auto matmul = [&] {
            std::fill(output.begin(), output.end(), 0.0F);
            for (std::size_t kk = 0; kk < dimension; kk += tile)
                for (std::size_t ii = 0; ii < dimension; ii += tile)
                    for (std::size_t jj = 0; jj < dimension; jj += tile)
                        for (std::size_t i = ii; i < std::min<std::size_t>(ii + tile, dimension); ++i)
                            for (std::size_t k = kk; k < std::min<std::size_t>(kk + tile, dimension); ++k) {
                                const auto aik = a[i * dimension + k];
                                for (std::size_t j = jj; j < std::min<std::size_t>(jj + tile, dimension); ++j)
                                    output[i * dimension + j] += aik * b[k * dimension + j];
                            }
        };
        matmul();
        results.push_back(run_cpu_compute(
            "cpu_tiled_matmul_" + std::to_string(dimension), output, matmul,
            [dimension](std::size_t, float value) {
                return std::abs(value - 0.125F * static_cast<float>(dimension)) <= 1e-5F;
            },
            options));
        const auto parallel_matmul = [&] {
            parallel_for(*executor, dimension, [&](std::size_t, std::size_t begin, std::size_t end) {
                for (auto i = begin; i < end; ++i) {
                    for (std::size_t j = 0; j < dimension; ++j) output[i * dimension + j] = 0.0F;
                }
                for (std::size_t kk = 0; kk < dimension; kk += tile)
                    for (std::size_t jj = 0; jj < dimension; jj += tile)
                        for (auto i = begin; i < end; ++i)
                            for (std::size_t k = kk; k < std::min<std::size_t>(kk + tile, dimension); ++k) {
                                const auto aik = a[i * dimension + k];
                                for (std::size_t j = jj; j < std::min<std::size_t>(jj + tile, dimension); ++j)
                                    output[i * dimension + j] += aik * b[k * dimension + j];
                            }
            });
        };
        auto parallel_matmul_result = run_cpu_compute(
            "cpu_tiled_matmul_parallel_" + std::to_string(dimension), output, parallel_matmul,
            [dimension](std::size_t, float value) {
                return std::abs(value - 0.125F * static_cast<float>(dimension)) <= 1e-5F;
            },
            options);
        parallel_matmul_result.thread_count = worker_count;
        results.push_back(std::move(parallel_matmul_result));
    }
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

struct HostTransferBuffer {
    kora::vk::Api api;
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize allocation_size = 0;
    std::byte* mapped = nullptr;
    bool coherent = false;

    ~HostTransferBuffer() {
        if (mapped != nullptr) api.unmap_memory(device, memory);
        if (buffer != VK_NULL_HANDLE) api.destroy_buffer(device, buffer, nullptr);
        if (memory != VK_NULL_HANDLE) api.free_memory(device, memory, nullptr);
    }

    kora::detail::TransferBuffer view() const {
        return {.buffer = buffer,
                .size = size,
                .memory = memory,
                .memory_allocation_size = allocation_size,
                .mapped_data = mapped,
                .host_visible = true,
                .host_coherent = coherent};
    }
};

std::unique_ptr<HostTransferBuffer> make_host_transfer_buffer(const kora::vk::Device& device,
                                                              const kora::DeviceInfo& info,
                                                              VkDeviceSize size) {
    const auto& api = device.api();
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto result = std::make_unique<HostTransferBuffer>();
    result->api = api;
    result->device = device.handle();
    result->size = size;
    if (api.create_buffer(device.handle(), &buffer_info, nullptr, &result->buffer) != VK_SUCCESS) return {};
    VkMemoryRequirements requirements{};
    api.get_buffer_memory_requirements(device.handle(), result->buffer, &requirements);
    auto type = kora::detail::MemoryManager::select_memory_type(
        info.memory_types(), info.memory_type_preferences(), kora::detail::MemoryPolicy::DevicePreferred,
        requirements.memoryTypeBits, {.device_local = true, .host_visible = true});
    if (!type) return {};
    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = type.value();
    if (api.allocate_memory(device.handle(), &allocation_info, nullptr, &result->memory) != VK_SUCCESS)
        return {};
    if (api.bind_buffer_memory(device.handle(), result->buffer, result->memory, 0) != VK_SUCCESS) return {};
    void* mapped = nullptr;
    if (api.map_memory(device.handle(), result->memory, 0, requirements.size, 0, &mapped) != VK_SUCCESS)
        return {};
    result->mapped = static_cast<std::byte*>(mapped);
    result->allocation_size = requirements.size;
    result->coherent = info.memory_types()[type.value()].host_coherent;
    return result;
}

void synchronize_mapped_buffer(const kora::vk::Device& device, const HostTransferBuffer& buffer, bool flush) {
    if (buffer.coherent) return;
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = buffer.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    const auto result = flush ? device.api().flush_mapped_memory_ranges(device.handle(), 1, &range)
                              : device.api().invalidate_mapped_memory_ranges(device.handle(), 1, &range);
    if (result != VK_SUCCESS)
        throw std::runtime_error(flush ? "raw Vulkan source flush failed"
                                       : "mapped transfer invalidate failed");
}

class RawComputeRunner {
public:
    RawComputeRunner(const kora::vk::PhysicalDeviceInfo& physical, const kora::vk::Device& device,
                     const std::filesystem::path& shader_path,
                     const std::vector<HostTransferBuffer*>& buffers, std::uint32_t push_size)
        : api_(device.api()),
          device_(device.handle()),
          queue_(device.compute_queue()),
          timestamp_period_(physical.properties.limits.timestampPeriod),
          timestamp_bits_(
              physical.queue_family_properties[device.compute_queue_family()].timestampValidBits) {
        if (buffers.empty() || push_size == 0) throw std::runtime_error("invalid raw compute layout");
        try {
            std::ifstream shader_file(shader_path, std::ios::binary | std::ios::ate);
            if (!shader_file) throw std::runtime_error("raw Vulkan shader fixture is unavailable");
            const auto shader_size = shader_file.tellg();
            if (shader_size <= 0 || static_cast<std::uint64_t>(shader_size) % sizeof(std::uint32_t) != 0)
                throw std::runtime_error("raw Vulkan shader fixture has an invalid size");
            std::vector<std::uint32_t> shader_code(static_cast<std::size_t>(shader_size) /
                                                   sizeof(std::uint32_t));
            shader_file.seekg(0);
            shader_file.read(reinterpret_cast<char*>(shader_code.data()), shader_size);
            if (!shader_file) throw std::runtime_error("raw Vulkan shader fixture could not be read");

            const auto require = [](VkResult result, const char* operation) {
                if (result != VK_SUCCESS) throw std::runtime_error(operation);
            };
            VkShaderModuleCreateInfo shader_info{};
            shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shader_info.codeSize = shader_code.size() * sizeof(std::uint32_t);
            shader_info.pCode = shader_code.data();
            require(api_.create_shader_module(device_, &shader_info, nullptr, &shader_),
                    "raw Vulkan shader-module creation failed");

            std::vector<VkDescriptorSetLayoutBinding> bindings(buffers.size());
            for (std::uint32_t index = 0; index < bindings.size(); ++index) {
                bindings[index].binding = index;
                bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[index].descriptorCount = 1;
                bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo descriptor_layout_info{};
            descriptor_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptor_layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
            descriptor_layout_info.pBindings = bindings.data();
            require(api_.create_descriptor_set_layout(device_, &descriptor_layout_info, nullptr,
                                                      &descriptor_layout_),
                    "raw Vulkan descriptor-layout creation failed");

            VkPushConstantRange push_range{};
            push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push_range.size = push_size;
            VkPipelineLayoutCreateInfo pipeline_layout_info{};
            pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipeline_layout_info.setLayoutCount = 1;
            pipeline_layout_info.pSetLayouts = &descriptor_layout_;
            pipeline_layout_info.pushConstantRangeCount = 1;
            pipeline_layout_info.pPushConstantRanges = &push_range;
            require(api_.create_pipeline_layout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_),
                    "raw Vulkan pipeline-layout creation failed");

            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader_;
            stage.pName = "main";
            VkComputePipelineCreateInfo pipeline_info{};
            pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline_layout_;
            require(api_.create_compute_pipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                                  &pipeline_),
                    "raw Vulkan compute-pipeline creation failed");

            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                           static_cast<std::uint32_t>(buffers.size())};
            VkDescriptorPoolCreateInfo descriptor_pool_info{};
            descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            descriptor_pool_info.maxSets = 1;
            descriptor_pool_info.poolSizeCount = 1;
            descriptor_pool_info.pPoolSizes = &pool_size;
            require(api_.create_descriptor_pool(device_, &descriptor_pool_info, nullptr, &descriptor_pool_),
                    "raw Vulkan descriptor-pool creation failed");
            VkDescriptorSetAllocateInfo descriptor_allocate{};
            descriptor_allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptor_allocate.descriptorPool = descriptor_pool_;
            descriptor_allocate.descriptorSetCount = 1;
            descriptor_allocate.pSetLayouts = &descriptor_layout_;
            require(api_.allocate_descriptor_sets(device_, &descriptor_allocate, &descriptor_set_),
                    "raw Vulkan descriptor-set allocation failed");
            std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
            std::vector<VkWriteDescriptorSet> writes(buffers.size());
            for (std::uint32_t index = 0; index < buffers.size(); ++index) {
                buffer_infos[index] = {buffers[index]->buffer, 0, buffers[index]->size};
                writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[index].dstSet = descriptor_set_;
                writes[index].dstBinding = index;
                writes[index].descriptorCount = 1;
                writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[index].pBufferInfo = &buffer_infos[index];
            }
            api_.update_descriptor_sets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0,
                                        nullptr);

            VkCommandPoolCreateInfo command_pool_info{};
            command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            command_pool_info.queueFamilyIndex = device.compute_queue_family();
            require(api_.create_command_pool(device_, &command_pool_info, nullptr, &command_pool_),
                    "raw Vulkan command-pool creation failed");
            VkCommandBufferAllocateInfo command_allocate{};
            command_allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            command_allocate.commandPool = command_pool_;
            command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            command_allocate.commandBufferCount = 1;
            require(api_.allocate_command_buffers(device_, &command_allocate, &command_),
                    "raw Vulkan command-buffer allocation failed");
            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            require(api_.create_fence(device_, &fence_info, nullptr, &fence_),
                    "raw Vulkan fence creation failed");

            const bool timestamp_supported = timestamp_bits_ != 0 && api_.cmd_write_timestamp != nullptr &&
                                             api_.cmd_reset_query_pool != nullptr &&
                                             api_.create_query_pool != nullptr &&
                                             static_cast<bool>(api_.get_query_pool_results);
            if (timestamp_supported) {
                VkQueryPoolCreateInfo query_info{};
                query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
                query_info.queryCount = 2;
                if (api_.create_query_pool(device_, &query_info, nullptr, &query_pool_) != VK_SUCCESS)
                    query_pool_ = VK_NULL_HANDLE;
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~RawComputeRunner() { cleanup(); }

    std::optional<std::uint64_t> dispatch(std::uint32_t groups_x, std::uint32_t groups_y,
                                          const void* push_constants, std::uint32_t push_size,
                                          bool collect_timestamp = false) {
        if (api_.reset_command_buffer(command_, 0) != VK_SUCCESS)
            throw std::runtime_error("raw Vulkan command reset failed");
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (api_.begin_command_buffer(command_, &begin) != VK_SUCCESS)
            throw std::runtime_error("raw Vulkan command begin failed");
        if (collect_timestamp && query_pool_ != VK_NULL_HANDLE) {
            api_.cmd_reset_query_pool(command_, query_pool_, 0, 2);
            api_.cmd_write_timestamp(command_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 0);
        }
        VkMemoryBarrier host_barrier{};
        host_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        host_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        host_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        api_.cmd_pipeline_barrier(command_, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 1, &host_barrier, 0, nullptr, 0, nullptr);
        api_.cmd_bind_pipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        api_.cmd_bind_descriptor_sets(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                                      &descriptor_set_, 0, nullptr);
        api_.cmd_push_constants(command_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size,
                                push_constants);
        api_.cmd_dispatch(command_, groups_x, groups_y, 1);
        if (collect_timestamp && query_pool_ != VK_NULL_HANDLE)
            api_.cmd_write_timestamp(command_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 1);
        VkMemoryBarrier host_read_barrier{};
        host_read_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        host_read_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        host_read_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        api_.cmd_pipeline_barrier(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                  0, 1, &host_read_barrier, 0, nullptr, 0, nullptr);
        if (api_.end_command_buffer(command_) != VK_SUCCESS)
            throw std::runtime_error("raw Vulkan command end failed");
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_;
        if (api_.queue_submit(queue_, 1, &submit, fence_) != VK_SUCCESS ||
            api_.wait_for_fences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            api_.reset_fences(device_, 1, &fence_) != VK_SUCCESS)
            throw std::runtime_error("raw Vulkan dispatch submit/wait failed");
        if (!collect_timestamp || query_pool_ == VK_NULL_HANDLE) return std::nullopt;
        std::array<std::uint64_t, 2> timestamps{};
        if (api_.get_query_pool_results(device_, query_pool_, 0, 2, sizeof(timestamps), timestamps.data(),
                                        sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
            return std::nullopt;
        auto ticks = timestamps[1] - timestamps[0];
        if (timestamp_bits_ < 64) ticks &= (std::uint64_t{1} << timestamp_bits_) - 1;
        return static_cast<std::uint64_t>(std::llround(static_cast<double>(ticks) * timestamp_period_));
    }

private:
    void cleanup() noexcept {
        if (query_pool_ != VK_NULL_HANDLE) api_.destroy_query_pool(device_, query_pool_, nullptr);
        if (fence_ != VK_NULL_HANDLE) api_.destroy_fence(device_, fence_, nullptr);
        if (command_pool_ != VK_NULL_HANDLE) api_.destroy_command_pool(device_, command_pool_, nullptr);
        if (descriptor_pool_ != VK_NULL_HANDLE)
            api_.destroy_descriptor_pool(device_, descriptor_pool_, nullptr);
        if (pipeline_ != VK_NULL_HANDLE) api_.destroy_pipeline(device_, pipeline_, nullptr);
        if (pipeline_layout_ != VK_NULL_HANDLE)
            api_.destroy_pipeline_layout(device_, pipeline_layout_, nullptr);
        if (descriptor_layout_ != VK_NULL_HANDLE)
            api_.destroy_descriptor_set_layout(device_, descriptor_layout_, nullptr);
        if (shader_ != VK_NULL_HANDLE) api_.destroy_shader_module(device_, shader_, nullptr);
        query_pool_ = VK_NULL_HANDLE;
        fence_ = VK_NULL_HANDLE;
        command_pool_ = VK_NULL_HANDLE;
        descriptor_pool_ = VK_NULL_HANDLE;
        pipeline_ = VK_NULL_HANDLE;
        pipeline_layout_ = VK_NULL_HANDLE;
        descriptor_layout_ = VK_NULL_HANDLE;
        shader_ = VK_NULL_HANDLE;
    }
    const kora::vk::Api& api_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    double timestamp_period_ = 0.0;
    std::uint32_t timestamp_bits_ = 0;
    VkShaderModule shader_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
};

bool add_transfer_benchmarks(std::vector<Result>& results, const Options& options,
                             const kora::vk::PhysicalDeviceInfo& physical, const kora::vk::Device& device) {
    using namespace kora::detail;
    const auto info = DeviceCaps::from_vulkan(physical).to_device_info();
    auto host_type = MemoryManager::select_memory_type(
        info.memory_types(), info.memory_type_preferences(), MemoryPolicy::HostVisible,
        std::numeric_limits<std::uint32_t>::max(), {.host_visible = true});
    if (!host_type) return false;
    const auto& selected_type = info.memory_types()[host_type.value()];
    const auto& selected_heap = info.memory_heaps()[selected_type.heap_index];
    MemoryPoolConfig pool_config;
    pool_config.memory_type_index = host_type.value();
    pool_config.heap_index = selected_type.heap_index;
    pool_config.usage = MemoryUsageClass::Upload;
    pool_config.host_visible = true;
    pool_config.host_coherent = selected_type.host_coherent;
    pool_config.heap_size = selected_heap.size_bytes;
    pool_config.budget = pool_config.heap_size;
    pool_config.block_size_hint = 4U * 1024U * 1024U;
    auto upload_pool = MemoryPool::create(device, pool_config);
    auto readback_config = pool_config;
    readback_config.usage = MemoryUsageClass::Readback;
    auto readback_pool = MemoryPool::create(device, readback_config);
    if (!upload_pool || !readback_pool) return false;

    auto queue_result =
        kora::vk::QueueContext::create(device.api(), device.handle(), device.transfer_queue());
    if (!queue_result) return false;
    auto queue = std::move(queue_result).value();
    TransferEngine::Options direct_options;
    direct_options.queue_family = device.transfer_queue_family();
    direct_options.command_buffer_count = 8;
    direct_options.topology = kora::MemoryTopology::Uma;
    direct_options.non_coherent_atom_size = physical.properties.limits.nonCoherentAtomSize;
    auto direct_engine =
        TransferEngine::create(device, *queue, *upload_pool.value(), *readback_pool.value(), direct_options);
    auto staged_options = direct_options;
    staged_options.measured_staging_benefit = true;
    staged_options.staging_initial_capacity = 4U * 1024U * 1024U;
    staged_options.staging_maximum_capacity = 8U * 1024U * 1024U;
    auto staged_engine =
        TransferEngine::create(device, *queue, *upload_pool.value(), *readback_pool.value(), staged_options);
    if (!direct_engine || !staged_engine) return false;

    // Measure only StagingRing::reserve() with a controlled number of live ranges. Setup and cancel
    // are outside each timed interval; no transfer, queue submission, or timeline wait is included.
    const auto add_reservation_benchmark = [&](std::size_t live_count, bool fragmented) {
        const VkDeviceSize atom_size = physical.properties.limits.nonCoherentAtomSize;
        const VkDeviceSize unit = std::max<VkDeviceSize>(256, atom_size);
        StagingRing::Options ring_options;
        ring_options.topology = info.memory_topology();
        ring_options.measured_staging_benefit = true;
        ring_options.non_coherent_atom_size = atom_size;
        ring_options.initial_capacity = fragmented ? (3 * live_count + 2) * unit : (live_count + 2) * unit;
        ring_options.maximum_capacity = ring_options.initial_capacity;
        auto ring_result = StagingRing::create(device, *upload_pool.value(), ring_options);
        if (!ring_result)
            throw std::runtime_error("staging-ring benchmark setup failed: " +
                                     ring_result.error().describe());
        auto& ring = *ring_result.value();
        std::vector<StagingSlice> live;
        live.reserve(live_count);
        for (std::size_t index = 0; index < live_count; ++index) {
            auto slice = ring.reserve(fragmented ? 2 * unit : unit, 1, 0, {});
            if (!slice)
                throw std::runtime_error("staging-ring live-range setup failed: " + slice.error().describe());
            live.push_back(slice.value());
            if (fragmented) {
                auto hole = ring.reserve(unit, 1, 0, {});
                if (!hole || !ring.cancel(hole.value()))
                    throw std::runtime_error("staging-ring fragmentation setup failed");
            }
        }
        const auto tail = ring.reserve(2 * unit, 1, 0, {});
        if (!tail || !ring.cancel(tail.value()))
            throw std::runtime_error("staging-ring benchmark tail setup failed");

        Result result{"staging_ring_reserve_" + std::string(fragmented ? "fragmented_" : "dense_") +
                          std::to_string(live_count),
                      {},
                      0,
                      0};
        result.samples_ns.reserve(options.iterations);
        const auto sample = [&] {
            const auto begin = Clock::now();
            auto slice = ring.reserve(2 * unit, 1, 0, {});
            const auto end = Clock::now();
            if (!slice)
                throw std::runtime_error("staging-ring timed reservation failed: " +
                                         slice.error().describe());
            result.samples_ns.push_back(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
            if (!ring.cancel(slice.value())) throw std::runtime_error("staging-ring timed cancel failed");
        };
        for (std::size_t index = 0; index < options.warmup; ++index) sample();
        result.samples_ns.clear();
        for (std::size_t index = 0; index < options.iterations; ++index) sample();
        result.median_ns = percentile(result.samples_ns, 50);
        result.p95_ns = percentile(result.samples_ns, 95);
        for (const auto& slice : live)
            if (!ring.cancel(slice)) throw std::runtime_error("staging-ring live-range cleanup failed");
        results.push_back(std::move(result));
    };
    for (const auto live_count : {0U, 1U, 8U, 32U, 128U, 512U}) {
        add_reservation_benchmark(live_count, false);
        add_reservation_benchmark(live_count, true);
    }

    constexpr VkDeviceSize maximum_size = 1024U * 1024U;
    auto source = make_host_transfer_buffer(device, info, maximum_size);
    auto destination = make_host_transfer_buffer(device, info, maximum_size);
    if (!source || !destination) return false;
    const auto source_view = source->view();
    const auto destination_view = destination->view();
    const auto sizes = {4096U, 65536U, 1024U * 1024U};
    for (const auto size : sizes) {
        std::vector<std::byte> input(size, std::byte{0x5a});
        std::vector<std::byte> output(size);
        const auto upload = [&](TransferEngine& engine) {
            auto event = engine.upload(source_view, 0, input);
            if (!event || !event.value().wait()) throw std::runtime_error("Vulkan upload benchmark failed");
        };
        const auto download = [&](TransferEngine& engine) {
            auto event = engine.download(source_view, 0, output);
            if (!event || !event.value().wait()) throw std::runtime_error("Vulkan download benchmark failed");
        };
        upload(*direct_engine.value());
        if (!std::equal(input.begin(), input.end(), source->mapped)) return false;
        results.push_back(run_transfer(
            "transfer_direct_upload_" + std::to_string(size), size, [&] { upload(*direct_engine.value()); },
            options));
        results.push_back(run_transfer(
            "transfer_staged_upload_" + std::to_string(size), size, [&] { upload(*staged_engine.value()); },
            options));
        download(*direct_engine.value());
        if (output != input) return false;
        download(*staged_engine.value());
        if (output != input) return false;
        results.push_back(run_transfer(
            "transfer_direct_download_" + std::to_string(size), size,
            [&] { download(*direct_engine.value()); }, options));
        results.push_back(run_transfer(
            "transfer_staged_download_" + std::to_string(size), size,
            [&] { download(*staged_engine.value()); }, options));
        if (output != input) return false;

        auto copy = staged_engine.value()->copy(source_view, 0, destination_view, 0, size);
        if (!copy || !copy.value().wait()) return false;
        if (!std::equal(input.begin(), input.end(), destination->mapped)) return false;
    }

    const auto& api = device.api();
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = device.transfer_queue_family();
    VkCommandPool pool = VK_NULL_HANDLE;
    if (api.create_command_pool(device.handle(), &pool_info, nullptr, &pool) != VK_SUCCESS) return false;
    struct RawPoolGuard {
        const kora::vk::Api& api;
        VkDevice device;
        VkCommandPool pool;
        VkFence fence = VK_NULL_HANDLE;
        ~RawPoolGuard() {
            if (fence != VK_NULL_HANDLE) api.destroy_fence(device, fence, nullptr);
            if (pool != VK_NULL_HANDLE) api.destroy_command_pool(device, pool, nullptr);
        }
    } raw_guard{api, device.handle(), pool};
    VkCommandBufferAllocateInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (api.allocate_command_buffers(device.handle(), &command_info, &command) != VK_SUCCESS) return false;
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (api.create_fence(device.handle(), &fence_info, nullptr, &raw_guard.fence) != VK_SUCCESS) return false;
    for (const auto size : sizes) {
        const auto raw_copy = [&] {
            synchronize_mapped_buffer(device, *source, true);
            if (api.reset_command_buffer(command, 0) != VK_SUCCESS)
                throw std::runtime_error("raw Vulkan command reset failed");
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (api.begin_command_buffer(command, &begin_info) != VK_SUCCESS)
                throw std::runtime_error("raw Vulkan command begin failed");
            VkBufferCopy region{0, 0, size};
            api.cmd_copy_buffer(command, source->buffer, destination->buffer, 1, &region);
            if (api.end_command_buffer(command) != VK_SUCCESS)
                throw std::runtime_error("raw Vulkan command end failed");
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            if (api.queue_submit(device.transfer_queue(), 1, &submit, raw_guard.fence) != VK_SUCCESS ||
                api.wait_for_fences(device.handle(), 1, &raw_guard.fence, VK_TRUE, UINT64_MAX) !=
                    VK_SUCCESS ||
                api.reset_fences(device.handle(), 1, &raw_guard.fence) != VK_SUCCESS)
                throw std::runtime_error("raw Vulkan transfer submit/wait failed");
            synchronize_mapped_buffer(device, *destination, false);
        };
        raw_copy();
        if (!std::equal(static_cast<const std::byte*>(source->mapped),
                        static_cast<const std::byte*>(source->mapped) + size,
                        static_cast<const std::byte*>(destination->mapped)))
            return false;
        results.push_back(run_transfer("raw_vulkan_copy_" + std::to_string(size), size, raw_copy, options));
        results.push_back(run_transfer(
            "kora_transfer_copy_" + std::to_string(size), size,
            [&] {
                auto event = staged_engine.value()->copy(source_view, 0, destination_view, 0, size);
                if (!event || !event.value().wait())
                    throw std::runtime_error("KORA transfer copy benchmark failed");
            },
            options));
        synchronize_mapped_buffer(device, *destination, false);
        if (!std::equal(static_cast<const std::byte*>(source->mapped),
                        static_cast<const std::byte*>(source->mapped) + size,
                        static_cast<const std::byte*>(destination->mapped)))
            return false;
    }
    return true;
}

bool add_raw_vector_add_benchmark(std::vector<Result>& results, const Options& options,
                                  const kora::vk::PhysicalDeviceInfo& physical,
                                  const kora::vk::Device& device, const std::filesystem::path& shader_path,
                                  std::uint32_t count, GpuDeviceReport& device_report) {
    const auto info = kora::detail::DeviceCaps::from_vulkan(physical).to_device_info();
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * sizeof(std::uint32_t);
    auto input_a = make_host_transfer_buffer(device, info, bytes);
    auto input_b = make_host_transfer_buffer(device, info, bytes);
    auto output = make_host_transfer_buffer(device, info, bytes);
    if (!input_a || !input_b || !output) return false;
    std::fill_n(reinterpret_cast<std::uint32_t*>(input_a->mapped), count, 3U);
    std::fill_n(reinterpret_cast<std::uint32_t*>(input_b->mapped), count, 7U);
    synchronize_mapped_buffer(device, *input_a, true);
    synchronize_mapped_buffer(device, *input_b, true);

    std::ifstream shader_file(shader_path, std::ios::binary | std::ios::ate);
    if (!shader_file) return false;
    const auto shader_size = shader_file.tellg();
    if (shader_size <= 0 || static_cast<std::uint64_t>(shader_size) % sizeof(std::uint32_t) != 0)
        return false;
    std::vector<std::uint32_t> shader_code(static_cast<std::size_t>(shader_size) / sizeof(std::uint32_t));
    shader_file.seekg(0);
    shader_file.read(reinterpret_cast<char*>(shader_code.data()), shader_size);
    if (!shader_file) return false;

    const auto& api = device.api();
    struct RawComputeGuard {
        const kora::vk::Api& api;
        VkDevice device;
        VkShaderModule shader = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkQueryPool query_pool = VK_NULL_HANDLE;
        ~RawComputeGuard() {
            if (query_pool != VK_NULL_HANDLE) api.destroy_query_pool(device, query_pool, nullptr);
            if (fence != VK_NULL_HANDLE) api.destroy_fence(device, fence, nullptr);
            if (command_pool != VK_NULL_HANDLE) api.destroy_command_pool(device, command_pool, nullptr);
            if (descriptor_pool != VK_NULL_HANDLE)
                api.destroy_descriptor_pool(device, descriptor_pool, nullptr);
            if (pipeline != VK_NULL_HANDLE) api.destroy_pipeline(device, pipeline, nullptr);
            if (pipeline_layout != VK_NULL_HANDLE)
                api.destroy_pipeline_layout(device, pipeline_layout, nullptr);
            if (descriptor_layout != VK_NULL_HANDLE)
                api.destroy_descriptor_set_layout(device, descriptor_layout, nullptr);
            if (shader != VK_NULL_HANDLE) api.destroy_shader_module(device, shader, nullptr);
        }
    } guard{api, device.handle()};

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = shader_code.size() * sizeof(std::uint32_t);
    shader_info.pCode = shader_code.data();
    if (api.create_shader_module(device.handle(), &shader_info, nullptr, &guard.shader) != VK_SUCCESS)
        return false;

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo descriptor_layout_info{};
    descriptor_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    descriptor_layout_info.pBindings = bindings.data();
    if (api.create_descriptor_set_layout(device.handle(), &descriptor_layout_info, nullptr,
                                         &guard.descriptor_layout) != VK_SUCCESS)
        return false;
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(count);
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &guard.descriptor_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    if (api.create_pipeline_layout(device.handle(), &pipeline_layout_info, nullptr, &guard.pipeline_layout) !=
        VK_SUCCESS)
        return false;
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = guard.shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = guard.pipeline_layout;
    if (api.create_compute_pipelines(device.handle(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                     &guard.pipeline) != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo descriptor_pool_info{};
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &pool_size;
    if (api.create_descriptor_pool(device.handle(), &descriptor_pool_info, nullptr, &guard.descriptor_pool) !=
        VK_SUCCESS)
        return false;
    VkDescriptorSetAllocateInfo descriptor_allocate{};
    descriptor_allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_allocate.descriptorPool = guard.descriptor_pool;
    descriptor_allocate.descriptorSetCount = 1;
    descriptor_allocate.pSetLayouts = &guard.descriptor_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    if (api.allocate_descriptor_sets(device.handle(), &descriptor_allocate, &descriptor_set) != VK_SUCCESS)
        return false;
    std::array<VkDescriptorBufferInfo, 3> buffer_infos{
        {{input_a->buffer, 0, bytes}, {input_b->buffer, 0, bytes}, {output->buffer, 0, bytes}}};
    std::array<VkWriteDescriptorSet, 3> writes{};
    for (std::uint32_t index = 0; index < writes.size(); ++index) {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = descriptor_set;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = &buffer_infos[index];
    }
    api.update_descriptor_sets(device.handle(), static_cast<std::uint32_t>(writes.size()), writes.data(), 0,
                               nullptr);

    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = device.compute_queue_family();
    if (api.create_command_pool(device.handle(), &command_pool_info, nullptr, &guard.command_pool) !=
        VK_SUCCESS)
        return false;
    VkCommandBufferAllocateInfo command_allocate{};
    command_allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_allocate.commandPool = guard.command_pool;
    command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocate.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (api.allocate_command_buffers(device.handle(), &command_allocate, &command) != VK_SUCCESS)
        return false;
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (api.create_fence(device.handle(), &fence_info, nullptr, &guard.fence) != VK_SUCCESS) return false;

    const auto timestamp_bits =
        physical.queue_family_properties[device.compute_queue_family()].timestampValidBits;
    const bool timestamp_supported =
        timestamp_bits != 0 && api.cmd_write_timestamp != nullptr && api.cmd_reset_query_pool != nullptr &&
        api.create_query_pool != nullptr && static_cast<bool>(api.get_query_pool_results);
    if (timestamp_supported) {
        VkQueryPoolCreateInfo query_info{};
        query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_info.queryCount = 2;
        if (api.create_query_pool(device.handle(), &query_info, nullptr, &guard.query_pool) != VK_SUCCESS)
            guard.query_pool = VK_NULL_HANDLE;
    }

    const auto raw_dispatch_profiled = [&]() -> std::optional<std::uint64_t> {
        if (api.reset_command_buffer(command, 0) != VK_SUCCESS)
            throw std::runtime_error("raw Vulkan dispatch command reset failed");
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (api.begin_command_buffer(command, &begin) != VK_SUCCESS)
            throw std::runtime_error("raw Vulkan dispatch command begin failed");
        if (guard.query_pool != VK_NULL_HANDLE) {
            api.cmd_reset_query_pool(command, guard.query_pool, 0, 2);
            api.cmd_write_timestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, guard.query_pool, 0);
        }
        VkMemoryBarrier host_barrier{};
        host_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        host_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        host_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        api.cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &host_barrier, 0, nullptr, 0, nullptr);
        api.cmd_bind_pipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, guard.pipeline);
        api.cmd_bind_descriptor_sets(command, VK_PIPELINE_BIND_POINT_COMPUTE, guard.pipeline_layout, 0, 1,
                                     &descriptor_set, 0, nullptr);
        api.cmd_push_constants(command, guard.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count),
                               &count);
        api.cmd_dispatch(command, (count + 63U) / 64U, 1, 1);
        if (guard.query_pool != VK_NULL_HANDLE)
            api.cmd_write_timestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, guard.query_pool, 1);
        VkMemoryBarrier host_read_barrier{};
        host_read_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        host_read_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        host_read_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        api.cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                                 1, &host_read_barrier, 0, nullptr, 0, nullptr);
        if (api.end_command_buffer(command) != VK_SUCCESS)
            throw std::runtime_error("raw Vulkan dispatch command end failed");
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        if (api.queue_submit(device.compute_queue(), 1, &submit, guard.fence) != VK_SUCCESS ||
            api.wait_for_fences(device.handle(), 1, &guard.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            api.reset_fences(device.handle(), 1, &guard.fence) != VK_SUCCESS)
            throw std::runtime_error("raw Vulkan dispatch submit/wait failed");
        if (guard.query_pool == VK_NULL_HANDLE) return std::nullopt;
        std::array<std::uint64_t, 2> timestamps{};
        if (api.get_query_pool_results(device.handle(), guard.query_pool, 0, 2, sizeof(timestamps),
                                       timestamps.data(), sizeof(std::uint64_t),
                                       VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
            return std::nullopt;
        auto ticks = timestamps[1] - timestamps[0];
        if (timestamp_bits < 64) ticks &= (std::uint64_t{1} << timestamp_bits) - 1;
        return static_cast<std::uint64_t>(
            std::llround(static_cast<double>(ticks) * physical.properties.limits.timestampPeriod));
    };
    const auto raw_dispatch = [&] { (void)raw_dispatch_profiled(); };
    raw_dispatch();
    synchronize_mapped_buffer(device, *output, false);
    const auto* output_values = reinterpret_cast<const std::uint32_t*>(output->mapped);
    if (!std::all_of(output_values, output_values + count, [](std::uint32_t value) { return value == 10; }))
        return false;
    results.push_back(
        run("raw_vulkan_vector_add_" + std::to_string(count) + "_dispatch_wait", raw_dispatch, options));
    const auto gpu_execution = run_timestamps(
        "raw_vulkan_vector_add_" + std::to_string(count) + "_gpu_execution", raw_dispatch_profiled, options);
    if (gpu_execution) {
        device_report.raw_gpu_timestamp_sizes.push_back(count);
        results.push_back(std::move(*gpu_execution));
    }
    return true;
}

#ifdef KORA_BENCH_VECTOR_ADD_FIXTURE_NAME
bool add_raw_non_vector_compute_benchmarks(std::vector<Result>& results, const Options& options,
                                           const kora::vk::PhysicalDeviceInfo& physical,
                                           const kora::vk::Device& device,
                                           const std::filesystem::path& vector_add_fixture) {
    using namespace kora::detail;
    const auto info = DeviceCaps::from_vulkan(physical).to_device_info();
    const auto fixture_path = [&](const char* name) {
        auto path = vector_add_fixture.parent_path() / name;
        if (!std::filesystem::exists(path))
            path = vector_add_fixture.parent_path().parent_path() / "share" / "kora" / "benchmarks" / name;
        return path;
    };
    constexpr std::uint32_t local_size = 64;
    const auto make_buffer = [&](VkDeviceSize bytes) {
        auto buffer = make_host_transfer_buffer(device, info, bytes);
        if (!buffer) throw std::runtime_error("raw Vulkan compute buffer allocation failed");
        return buffer;
    };
    const auto finish_samples = [&](const std::string& workload, RawComputeRunner& runner,
                                    std::uint32_t groups_x, std::uint32_t groups_y, const void* push_data,
                                    std::uint32_t push_size, const std::function<void()>& prepare,
                                    const std::function<bool()>& validate) {
        const auto operation = [&] {
            prepare();
            (void)runner.dispatch(groups_x, groups_y, push_data, push_size);
        };
        results.push_back(
            run_checked("raw_vulkan_" + workload + "_dispatch_wait", operation, validate, options));
        const auto timestamp = run_timestamps(
            "raw_vulkan_" + workload + "_gpu_execution",
            [&]() -> std::optional<std::uint64_t> {
                prepare();
                return runner.dispatch(groups_x, groups_y, push_data, push_size, true);
            },
            options);
        if (timestamp) results.push_back(std::move(*timestamp));
        operation();
        if (!validate()) throw std::runtime_error("raw Vulkan output validation failed: " + workload);
        return true;
    };
    struct SaxpyParams {
        float scale;
        std::uint32_t count;
    };
    static_assert(sizeof(SaxpyParams) == 8);
    for (const auto count : kComputeSizes) {
        const VkDeviceSize float_bytes = VkDeviceSize{count} * sizeof(float);
        const VkDeviceSize uint_bytes = VkDeviceSize{count} * sizeof(std::uint32_t);
        const auto suffix = std::to_string(count);

        const SaxpyParams saxpy_params{2.0F, count};
        auto saxpy_x = make_buffer(float_bytes);
        auto saxpy_y = make_buffer(float_bytes);
        auto* saxpy_x_values = reinterpret_cast<float*>(saxpy_x->mapped);
        auto* saxpy_y_values = reinterpret_cast<float*>(saxpy_y->mapped);
        std::fill_n(saxpy_x_values, count, 2.0F);
        const std::vector<float> saxpy_initial(count, 1.0F);
        std::copy(saxpy_initial.begin(), saxpy_initial.end(), saxpy_y_values);
        synchronize_mapped_buffer(device, *saxpy_x, true);
        synchronize_mapped_buffer(device, *saxpy_y, true);
        RawComputeRunner saxpy_runner(physical, device, fixture_path(KORA_BENCH_SAXPY_FIXTURE_NAME),
                                      {saxpy_x.get(), saxpy_y.get()}, sizeof(saxpy_params));
        finish_samples(
            "saxpy_" + suffix + "_upload", saxpy_runner, count / local_size, 1, &saxpy_params,
            sizeof(saxpy_params),
            [&] {
                std::copy(saxpy_initial.begin(), saxpy_initial.end(), saxpy_y_values);
                synchronize_mapped_buffer(device, *saxpy_y, true);
            },
            [&] {
                synchronize_mapped_buffer(device, *saxpy_y, false);
                return std::all_of(saxpy_y_values, saxpy_y_values + count,
                                   [](float value) { return value == 5.0F; });
            });

        const std::uint32_t count_push = count;
        auto reduction_input = make_buffer(float_bytes);
        auto reduction_output = make_buffer((count / local_size) * sizeof(float));
        std::fill_n(reinterpret_cast<float*>(reduction_input->mapped), count, 1.0F);
        synchronize_mapped_buffer(device, *reduction_input, true);
        RawComputeRunner reduction_runner(physical, device, fixture_path(KORA_BENCH_REDUCTION_FIXTURE_NAME),
                                          {reduction_input.get(), reduction_output.get()},
                                          sizeof(count_push));
        finish_samples(
            "reduction_" + suffix, reduction_runner, count / local_size, 1, &count_push, sizeof(count_push),
            [] {},
            [&] {
                synchronize_mapped_buffer(device, *reduction_output, false);
                const auto* values = reinterpret_cast<const float*>(reduction_output->mapped);
                return std::all_of(values, values + count / local_size,
                                   [](float value) { return value == 64.0F; });
            });

        auto scan_input = make_buffer(uint_bytes);
        auto scan_output = make_buffer(uint_bytes);
        std::fill_n(reinterpret_cast<std::uint32_t*>(scan_input->mapped), count, 1U);
        synchronize_mapped_buffer(device, *scan_input, true);
        RawComputeRunner scan_runner(physical, device, fixture_path(KORA_BENCH_SCAN_FIXTURE_NAME),
                                     {scan_input.get(), scan_output.get()}, sizeof(count_push));
        finish_samples(
            "tiled_scan_" + suffix, scan_runner, count / local_size, 1, &count_push, sizeof(count_push),
            [] {},
            [&] {
                synchronize_mapped_buffer(device, *scan_output, false);
                const auto* values = reinterpret_cast<const std::uint32_t*>(scan_output->mapped);
                return std::all_of(values, values + count,
                                   [index = std::uint32_t{0}](std::uint32_t value) mutable {
                                       return value == (index++ % local_size) + 1;
                                   });
            });

        auto stencil_input = make_buffer(float_bytes);
        auto stencil_output = make_buffer(float_bytes);
        std::fill_n(reinterpret_cast<float*>(stencil_input->mapped), count, 1.0F);
        synchronize_mapped_buffer(device, *stencil_input, true);
        RawComputeRunner stencil_runner(physical, device, fixture_path(KORA_BENCH_STENCIL_FIXTURE_NAME),
                                        {stencil_input.get(), stencil_output.get()}, sizeof(count_push));
        finish_samples(
            "stencil_" + suffix, stencil_runner, count / local_size, 1, &count_push, sizeof(count_push),
            [] {},
            [&] {
                synchronize_mapped_buffer(device, *stencil_output, false);
                const auto* values = reinterpret_cast<const float*>(stencil_output->mapped);
                return values[0] == 2.0F && values[count - 1] == 2.0F &&
                       std::all_of(values + 1, values + count - 1, [](float value) { return value == 3.0F; });
            });
    }

    for (const auto dimension : kMatrixDimensions) {
        const auto matrix_count = dimension * dimension;
        const VkDeviceSize matrix_bytes = VkDeviceSize{matrix_count} * sizeof(float);
        auto matrix_a = make_buffer(matrix_bytes);
        auto matrix_b = make_buffer(matrix_bytes);
        auto matrix_c = make_buffer(matrix_bytes);
        std::fill_n(reinterpret_cast<float*>(matrix_a->mapped), matrix_count, 1.0F);
        std::fill_n(reinterpret_cast<float*>(matrix_b->mapped), matrix_count, 1.0F);
        synchronize_mapped_buffer(device, *matrix_a, true);
        synchronize_mapped_buffer(device, *matrix_b, true);
        RawComputeRunner matmul_runner(physical, device, fixture_path(KORA_BENCH_MATMUL_FIXTURE_NAME),
                                       {matrix_a.get(), matrix_b.get(), matrix_c.get()}, sizeof(dimension));
        finish_samples(
            "tiled_matmul_" + std::to_string(dimension), matmul_runner, dimension / 8, dimension / 8,
            &dimension, sizeof(dimension), [] {},
            [&] {
                synchronize_mapped_buffer(device, *matrix_c, false);
                const auto* values = reinterpret_cast<const float*>(matrix_c->mapped);
                const auto expected = static_cast<float>(dimension);
                return std::all_of(values, values + matrix_count,
                                   [expected](float value) { return value == expected; });
            });
    }
    return true;
}
#endif

#ifdef KORA_BENCH_VECTOR_ADD_FIXTURE_NAME
// kora::Cluster SAXPY (y += 2x) run locally and on an in-process loopback worker that uses the same
// device. The remote rows measure the worker protocol's cost (request, execution on the worker's
// own Runtime, and the output returned to the host), not a second machine's speed.
bool add_cluster_benchmarks(std::vector<Result>& results, const Options& options,
                            const std::filesystem::path& saxpy_fixture,
                            const kora::KernelMetadata& metadata) {
    kora::WorkerOptions worker_options;
    worker_options.listen = "127.0.0.1:0";
    auto worker = kora::WorkerServer::start(worker_options);
    if (!worker) {
        std::cerr << "cluster benchmarks unavailable: " << worker.error().describe() << '\n';
        return false;
    }
    struct SaxpyParams {
        float scale;
        std::uint32_t count;
    };
    for (const auto& [policy, prefix] :
         {std::pair{kora::OffloadPolicy::LocalOnly, "cluster_local"},
          std::pair{kora::OffloadPolicy::PreferRemote, "cluster_remote_loopback"}}) {
        kora::ClusterOptions cluster_options;
        cluster_options.workers = {"127.0.0.1:" + std::to_string(worker.value()->port())};
        cluster_options.policy = policy;
        auto cluster = kora::Cluster::create(cluster_options);
        if (!cluster) {
            std::cerr << "cluster benchmarks unavailable: " << cluster.error().describe() << '\n';
            return false;
        }
        if (policy == kora::OffloadPolicy::PreferRemote) {
            const auto deadline = Clock::now() + std::chrono::seconds(5);
            while (Clock::now() < deadline && !cluster.value().workers().front().connected)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if (!cluster.value().workers().front().connected) {
                std::cerr << "cluster benchmarks unavailable: loopback worker did not connect\n";
                return false;
            }
        }
        auto kernel = cluster.value().load_kernel(saxpy_fixture, "main", metadata);
        if (!kernel) throw std::runtime_error("cluster kernel load failed: " + kernel.error().describe());
        for (const auto count : kComputeSizes) {
            auto x = cluster.value().create_buffer<float>(count);
            auto y = cluster.value().create_buffer<float>(count);
            if (!x || !y || !x.value().write(std::vector<float>(count, 1.0F)) ||
                !y.value().write(std::vector<float>(count, 0.0F)))
                throw std::runtime_error("cluster buffer setup failed");
            float expected = 0.0F;
            const SaxpyParams params{2.0F, count};
            const auto operation = [&] {
                auto event = cluster.value()
                                 .dispatch(kernel.value())
                                 .global(count)
                                 .read(x.value())
                                 .read_write(y.value())
                                 .push(params)
                                 .submit();
                if (!event || !event.value().wait()) throw std::runtime_error("cluster dispatch failed");
                expected += 2.0F;
            };
            const auto validate = [&] {
                auto values = y.value().read();
                return values && std::all_of(values.value().begin(), values.value().end(),
                                             [&](float value) { return value == expected; });
            };
            results.push_back(
                run_checked(std::string(prefix) + "_saxpy_" + std::to_string(count) + "_dispatch_wait",
                            operation, validate, options));
        }
        const auto stats = cluster.value().stats();
        if (policy == kora::OffloadPolicy::PreferRemote &&
            (stats.local_dispatches != 0 || stats.failovers != 0))
            throw std::runtime_error("loopback cluster benchmark did not run every dispatch remotely");
    }
    return true;
}

bool add_gpu_benchmarks(std::vector<Result>& results, const Options& options, const char* executable_path,
                        GpuDeviceReport& device_report) {
    kora::RuntimeOptions runtime_options;
    runtime_options.telemetry =
        options.profile_phases ? kora::TelemetryLevel::Profile : kora::TelemetryLevel::Basic;
    runtime_options.enable_startup_calibration = false;
    runtime_options.disable_autotune = true;
    runtime_options.debug_safety = false;
    runtime_options.log_level = kora::log::Level::Off;
    auto runtime_result = kora::Runtime::create(runtime_options);
    if (!runtime_result) {
        std::cerr << "GPU benchmarks unavailable: " << runtime_result.error().describe() << '\n';
        return false;
    }
    auto runtime = std::move(runtime_result).value();
    const auto selected_devices = runtime.devices();
    if (selected_devices.empty()) {
        std::cerr << "GPU benchmarks unavailable: Runtime selected no devices\n";
        return false;
    }
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
    constexpr std::uint32_t element_count = 4096;
    constexpr std::uint32_t vector_count = 4'194'240;
    std::vector<std::uint32_t> input_a(vector_count, 3), input_b(vector_count, 7);
    auto a = runtime.create_buffer<std::uint32_t>(vector_count);
    auto b = runtime.create_buffer<std::uint32_t>(vector_count);
    auto output = runtime.create_buffer<std::uint32_t>(vector_count);
    if (!a || !b || !output || !a.value().write(input_a) || !b.value().write(input_b)) return false;
    const auto dispatch = [&](std::uint32_t count, bool defer_readback = false) {
        auto builder = runtime.dispatch(kernel.value())
                           .global(count)
                           .read(a.value())
                           .read(b.value())
                           .write(output.value())
                           .push(count);
        if (defer_readback) builder.defer_readback();
        auto event = builder.submit();
        if (!event || !event.value().wait()) throw std::runtime_error("Vulkan dispatch failed");
    };
    // Basic telemetry timestamps one dispatch in 16 (the Runtime default), so repeat the operation
    // until a new device-time sample arrives; each sample is still a single dispatch's interval.
    const auto sample_kora_timestamp =
        [&](const std::function<void()>& operation) -> std::optional<std::uint64_t> {
        auto before = runtime.profiling_stats(selected_devices.front());
        if (!before) return std::nullopt;
        for (int attempt = 0; attempt < 32; ++attempt) {
            operation();
            auto after = runtime.profiling_stats(selected_devices.front());
            if (!after) return std::nullopt;
            if (after.value().last_submission_id != before.value().last_submission_id)
                return after.value().last_duration_nanoseconds;
        }
        return std::nullopt;
    };
    const auto add_kora_gpu_timestamp = [&](std::string name, const std::function<void()>& operation) {
        auto result =
            run_timestamps(std::move(name), [&] { return sample_kora_timestamp(operation); }, options);
        if (result) results.push_back(std::move(*result));
    };
    results.push_back(run("gpu_tiny_dispatch_1", [&] { dispatch(1); }, options));
    const auto dispatch_tiny_batch = [&] {
        for (std::size_t index = 0; index < 10; ++index) dispatch(1);
    };
    dispatch_tiny_batch();
    auto batch_values = output.value().read();
    if (!batch_values || batch_values.value().front() != 10) return false;
    results.push_back(run("gpu_tiny_dispatch_batch_10", dispatch_tiny_batch, options));
    dispatch(element_count);
    auto sum_values = output.value().read();
    if (!sum_values || !std::all_of(sum_values.value().begin(), sum_values.value().begin() + 1,
                                    [](std::uint32_t value) { return value == 10; }))
        return false;
    for (const auto count : {64U, 256U, 1024U, 4096U, 65536U, 262144U, 1048576U, 4194240U}) {
        dispatch(count);
        auto values = output.value().read();
        if (!values || !std::all_of(values.value().begin(), values.value().begin() + count,
                                    [](std::uint32_t value) { return value == 10; }))
            return false;
        const auto name = "gpu_vector_add_" + std::to_string(count) + "_dispatch_wait";
        results.push_back(run(name, [&, count] { dispatch(count); }, options));
        const auto deferred_name =
            "gpu_vector_add_" + std::to_string(count) + "_deferred_readback_dispatch_wait";
        results.push_back(run(deferred_name, [&, count] { dispatch(count, true); }, options));
        const auto gpu_execution = run_timestamps(
            "gpu_vector_add_" + std::to_string(count) + "_gpu_execution",
            [&, count] { return sample_kora_timestamp([&] { dispatch(count, true); }); }, options);
        if (gpu_execution) {
            device_report.kora_gpu_timestamp_sizes.push_back(count);
            results.push_back(std::move(*gpu_execution));
        }
    }

    if (options.profile_phases) {
        struct PhaseSample {
            std::uint64_t builder_setup = 0;
            std::uint64_t submit_call = 0;
            std::uint64_t event_wait = 0;
            kora::RuntimeProfilingStats runtime;
        };
        const auto profile_dispatch = [&]() -> std::optional<PhaseSample> {
            PhaseSample sample;
            auto before = runtime.profiling_stats(selected_devices.front());
            if (!before) return std::nullopt;
            const auto builder_started = Clock::now();
            auto builder = runtime.dispatch(kernel.value());
            builder.global(element_count)
                .read(a.value())
                .read(b.value())
                .write(output.value())
                .push(element_count)
                .defer_readback();
            const auto builder_finished = Clock::now();
            sample.builder_setup = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(builder_finished - builder_started)
                    .count());
            const auto submit_started = Clock::now();
            auto event = builder.submit();
            const auto submit_finished = Clock::now();
            if (!event) return std::nullopt;
            sample.submit_call = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(submit_finished - submit_started)
                    .count());
            const auto wait_started = Clock::now();
            if (!event.value().wait()) return std::nullopt;
            sample.event_wait = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - wait_started).count());
            auto stats = runtime.profiling_stats(selected_devices.front());
            if (!stats ||
                stats.value().last_profile_submission_id == before.value().last_profile_submission_id)
                return std::nullopt;
            sample.runtime = stats.value();
            return sample;
        };

        std::array<std::vector<std::uint64_t>, 18> phase_samples;
        const auto append_sample = [&](const PhaseSample& sample, bool retain) {
            if (!retain) return;
            phase_samples[0].push_back(sample.builder_setup);
            phase_samples[1].push_back(sample.submit_call);
            phase_samples[2].push_back(sample.event_wait);
            phase_samples[3].push_back(sample.runtime.last_dispatch_validation_nanoseconds);
            phase_samples[4].push_back(sample.runtime.last_binding_preflight_nanoseconds);
            phase_samples[5].push_back(sample.runtime.last_pipeline_lookup_nanoseconds);
            phase_samples[6].push_back(sample.runtime.last_host_submission_nanoseconds);
            phase_samples[7].push_back(sample.runtime.last_buffer_residency_nanoseconds);
            phase_samples[8].push_back(sample.runtime.last_submission_assembly_nanoseconds);
            phase_samples[9].push_back(sample.runtime.last_queue_enqueue_nanoseconds);
            phase_samples[10].push_back(sample.runtime.last_command_buffer_acquire_nanoseconds);
            phase_samples[11].push_back(sample.runtime.last_descriptor_setup_nanoseconds);
            phase_samples[12].push_back(sample.runtime.last_command_recording_nanoseconds);
            phase_samples[13].push_back(sample.runtime.last_queue_completion_nanoseconds);
            phase_samples[14].push_back(sample.runtime.last_completion_worker_delay_nanoseconds);
            phase_samples[15].push_back(sample.runtime.last_device_completion_wait_nanoseconds);
            phase_samples[16].push_back(sample.runtime.last_timestamp_collection_nanoseconds);
            phase_samples[17].push_back(sample.runtime.last_completion_callback_nanoseconds);
        };
        bool profile_available = true;
        for (std::size_t iteration = 0; iteration < options.warmup; ++iteration) {
            if (!profile_dispatch()) {
                std::cerr << "KORA dispatch phase profiling unavailable on the selected device\n";
                profile_available = false;
                break;
            }
        }
        if (profile_available) {
            for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
                auto sample = profile_dispatch();
                if (!sample) {
                    std::cerr << "KORA dispatch phase profiling stopped: timestamp sample unavailable\n";
                    profile_available = false;
                    break;
                }
                append_sample(*sample, true);
            }
        }
        constexpr std::array<std::string_view, 18> phase_names{"builder_setup",
                                                               "submit_call",
                                                               "event_wait",
                                                               "dispatch_validation",
                                                               "binding_preflight",
                                                               "pipeline_lookup",
                                                               "submission_preparation",
                                                               "buffer_residency_transfer",
                                                               "submission_assembly",
                                                               "queue_enqueue",
                                                               "command_buffer_acquire",
                                                               "descriptor_setup",
                                                               "command_recording",
                                                               "queue_completion",
                                                               "completion_worker_delay",
                                                               "device_completion_wait",
                                                               "timestamp_collection",
                                                               "completion_callback"};
        if (profile_available) {
            for (std::size_t phase = 0; phase < phase_samples.size(); ++phase) {
                auto& samples = phase_samples[phase];
                Result result{"profile_kora_4096_" + std::string(phase_names[phase]), std::move(samples), 0,
                              0};
                result.median_ns = percentile(result.samples_ns, 50);
                result.p95_ns = percentile(result.samples_ns, 95);
                results.push_back(std::move(result));
            }
        }
    }

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
    auto saxpy = runtime.load_kernel(saxpy_fixture, "main", saxpy_metadata);
    if (!saxpy) {
        std::cerr << "GPU benchmarks unavailable: " << saxpy.error().describe() << '\n';
        return false;
    }
    struct SaxpyParams {
        float scale;
        std::uint32_t count;
    };
    static_assert(sizeof(SaxpyParams) == sizeof(float) + sizeof(std::uint32_t));
    for (const auto count : kComputeSizes) {
        const std::vector<float> saxpy_x(count, 2.0F), saxpy_y(count, 1.0F);
        auto x = runtime.create_buffer<float>(count);
        auto y = runtime.create_buffer<float>(count);
        if (!x || !y || !x.value().write(saxpy_x) || !y.value().write(saxpy_y)) return false;
        const SaxpyParams params{2.0F, count};
        const auto run_saxpy = [&](bool defer_readback = false) {
            if (!y.value().write(saxpy_y)) throw std::runtime_error("SAXPY input reset failed");
            auto builder = runtime.dispatch(saxpy.value())
                               .global(count)
                               .read(x.value())
                               .read_write(y.value())
                               .push(params);
            if (defer_readback) builder.defer_readback();
            auto event = builder.submit();
            if (!event || !event.value().wait()) throw std::runtime_error("Vulkan SAXPY dispatch failed");
        };
        const auto validate_saxpy = [&] {
            auto readback = runtime.readback(y.value());
            if (!readback || !readback.value().wait()) return false;
            auto values = y.value().read();
            return values && std::all_of(values.value().begin(), values.value().end(),
                                         [](float value) { return value == 5.0F; });
        };
        run_saxpy();
        if (!validate_saxpy()) return false;
        const auto name = "gpu_saxpy_" + std::to_string(count);
        results.push_back(
            run_checked(name + "_upload_dispatch_wait", [&] { run_saxpy(); }, validate_saxpy, options));
        results.push_back(run_checked(
            name + "_upload_deferred_readback_dispatch_wait", [&] { run_saxpy(true); }, validate_saxpy,
            options));
        add_kora_gpu_timestamp(name + "_gpu_execution", [&] { run_saxpy(); });
    }

    kora::KernelMetadata reduction_metadata;
    reduction_metadata.kernel_id = "kora_bench.reduction";
    reduction_metadata.local_size = {64, 1, 1};
    reduction_metadata.push_constant_size = sizeof(std::uint32_t);
    reduction_metadata.push_constants = {{"count", "u32", 0, sizeof(std::uint32_t)}};
    reduction_metadata.bindings = {
        {0, 0, "input_data", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Read},
        {0, 1, "output_data", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Write},
    };
    auto reduction_fixture = fixture.parent_path() / KORA_BENCH_REDUCTION_FIXTURE_NAME;
    if (!std::filesystem::exists(reduction_fixture))
        reduction_fixture = fixture.parent_path().parent_path() / "share" / "kora" / "benchmarks" /
                            KORA_BENCH_REDUCTION_FIXTURE_NAME;
    auto reduction = runtime.load_kernel(reduction_fixture, "main", std::move(reduction_metadata));
    if (!reduction) {
        std::cerr << "GPU benchmarks unavailable: " << reduction.error().describe() << '\n';
        return false;
    }
    for (const auto count : kComputeSizes) {
        const std::vector<float> reduction_input(count, 1.0F);
        auto reduction_source = runtime.create_buffer<float>(count);
        auto reduction_output = runtime.create_buffer<float>(count / 64);
        if (!reduction_source || !reduction_output || !reduction_source.value().write(reduction_input))
            return false;
        const auto run_reduction = [&](bool defer_readback = false) {
            auto builder = runtime.dispatch(reduction.value())
                               .global(count)
                               .read(reduction_source.value())
                               .write(reduction_output.value())
                               .push(count);
            if (defer_readback) builder.defer_readback();
            auto event = builder.submit();
            if (!event || !event.value().wait()) throw std::runtime_error("Vulkan reduction dispatch failed");
        };
        const auto validate_reduction = [&] {
            auto readback = runtime.readback(reduction_output.value());
            if (!readback || !readback.value().wait()) return false;
            auto values = reduction_output.value().read();
            return values && std::all_of(values.value().begin(), values.value().end(),
                                         [](float value) { return value == 64.0F; });
        };
        run_reduction();
        if (!validate_reduction()) return false;
        const auto name = "gpu_reduction_" + std::to_string(count);
        results.push_back(
            run_checked(name + "_dispatch_wait", [&] { run_reduction(); }, validate_reduction, options));
        results.push_back(run_checked(
            name + "_deferred_readback_dispatch_wait", [&] { run_reduction(true); }, validate_reduction,
            options));
        add_kora_gpu_timestamp(name + "_gpu_execution", [&] { run_reduction(); });
    }

    kora::KernelMetadata scan_metadata;
    scan_metadata.kernel_id = "kora_bench.scan_tile64";
    scan_metadata.local_size = {64, 1, 1};
    scan_metadata.push_constant_size = sizeof(std::uint32_t);
    scan_metadata.push_constants = {{"count", "u32", 0, sizeof(std::uint32_t)}};
    scan_metadata.bindings = {
        {0, 0, "input_data", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Read},
        {0, 1, "output_data", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Write},
    };
    auto scan_fixture = fixture.parent_path() / KORA_BENCH_SCAN_FIXTURE_NAME;
    if (!std::filesystem::exists(scan_fixture))
        scan_fixture = fixture.parent_path().parent_path() / "share" / "kora" / "benchmarks" /
                       KORA_BENCH_SCAN_FIXTURE_NAME;
    auto scan = runtime.load_kernel(scan_fixture, "main", std::move(scan_metadata));
    if (!scan) {
        std::cerr << "GPU benchmarks unavailable: " << scan.error().describe() << '\n';
        return false;
    }
    for (const auto count : kComputeSizes) {
        const std::vector<std::uint32_t> scan_input(count, 1U);
        auto scan_source = runtime.create_buffer<std::uint32_t>(count);
        auto scan_output = runtime.create_buffer<std::uint32_t>(count);
        if (!scan_source || !scan_output || !scan_source.value().write(scan_input)) return false;
        const auto run_scan = [&](bool defer_readback = false) {
            auto builder = runtime.dispatch(scan.value())
                               .global(count)
                               .read(scan_source.value())
                               .write(scan_output.value())
                               .push(count);
            if (defer_readback) builder.defer_readback();
            auto event = builder.submit();
            if (!event || !event.value().wait()) throw std::runtime_error("Vulkan scan dispatch failed");
        };
        const auto validate_scan = [&] {
            auto readback = runtime.readback(scan_output.value());
            if (!readback || !readback.value().wait()) return false;
            auto values = scan_output.value().read();
            return values && std::all_of(values.value().begin(), values.value().end(),
                                         [index = std::uint32_t{0}](std::uint32_t value) mutable {
                                             return value == (index++ % 64) + 1;
                                         });
        };
        run_scan();
        if (!validate_scan()) return false;
        const auto name = "gpu_tiled_scan_" + std::to_string(count);
        results.push_back(run_checked(name + "_dispatch_wait", [&] { run_scan(); }, validate_scan, options));
        results.push_back(run_checked(
            name + "_deferred_readback_dispatch_wait", [&] { run_scan(true); }, validate_scan, options));
        add_kora_gpu_timestamp(name + "_gpu_execution", [&] { run_scan(); });
    }

    kora::KernelMetadata matmul_metadata;
    matmul_metadata.kernel_id = "kora_bench.tiled_matmul";
    matmul_metadata.local_size = {8, 8, 1};
    matmul_metadata.push_constant_size = sizeof(std::uint32_t);
    matmul_metadata.push_constants = {{"dimension", "u32", 0, sizeof(std::uint32_t)}};
    matmul_metadata.bindings = {
        {0, 0, "matrix_a", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Read},
        {0, 1, "matrix_b", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Read},
        {0, 2, "matrix_c", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Write},
    };
    auto matmul_fixture = fixture.parent_path() / KORA_BENCH_MATMUL_FIXTURE_NAME;
    if (!std::filesystem::exists(matmul_fixture))
        matmul_fixture = fixture.parent_path().parent_path() / "share" / "kora" / "benchmarks" /
                         KORA_BENCH_MATMUL_FIXTURE_NAME;
    auto matmul = runtime.load_kernel(matmul_fixture, "main", std::move(matmul_metadata));
    if (!matmul) {
        std::cerr << "GPU benchmarks unavailable: " << matmul.error().describe() << '\n';
        return false;
    }
    for (const auto matrix_dimension : kMatrixDimensions) {
        const auto matrix_elements = matrix_dimension * matrix_dimension;
        const std::vector<float> matrix_values(matrix_elements, 1.0F);
        auto matrix_a = runtime.create_buffer<float>(matrix_elements);
        auto matrix_b = runtime.create_buffer<float>(matrix_elements);
        auto matrix_c = runtime.create_buffer<float>(matrix_elements);
        if (!matrix_a || !matrix_b || !matrix_c || !matrix_a.value().write(matrix_values) ||
            !matrix_b.value().write(matrix_values))
            return false;
        const auto run_matmul = [&](bool defer_readback = false) {
            auto builder = runtime.dispatch(matmul.value())
                               .global(matrix_dimension, matrix_dimension, 1)
                               .read(matrix_a.value())
                               .read(matrix_b.value())
                               .write(matrix_c.value())
                               .push(matrix_dimension);
            if (defer_readback) builder.defer_readback();
            auto event = builder.submit();
            if (!event || !event.value().wait()) throw std::runtime_error("Vulkan tiled matmul failed");
        };
        const auto validate_matmul = [&] {
            auto readback = runtime.readback(matrix_c.value());
            if (!readback || !readback.value().wait()) return false;
            auto values = matrix_c.value().read();
            const auto expected = static_cast<float>(matrix_dimension);
            return values && std::all_of(values.value().begin(), values.value().end(),
                                         [expected](float value) { return value == expected; });
        };
        run_matmul();
        if (!validate_matmul()) return false;
        const auto name = "gpu_tiled_matmul_" + std::to_string(matrix_dimension);
        results.push_back(
            run_checked(name + "_dispatch_wait", [&] { run_matmul(); }, validate_matmul, options));
        results.push_back(run_checked(
            name + "_deferred_readback_dispatch_wait", [&] { run_matmul(true); }, validate_matmul, options));
        add_kora_gpu_timestamp(name + "_gpu_execution", [&] { run_matmul(); });
    }

    kora::KernelMetadata stencil_metadata;
    stencil_metadata.kernel_id = "kora_bench.stencil";
    stencil_metadata.local_size = {64, 1, 1};
    stencil_metadata.push_constant_size = sizeof(std::uint32_t);
    stencil_metadata.push_constants = {{"count", "u32", 0, sizeof(std::uint32_t)}};
    stencil_metadata.bindings = {
        {0, 0, "input_data", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Read},
        {0, 1, "output_data", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Write},
    };
    auto stencil_fixture = fixture.parent_path() / KORA_BENCH_STENCIL_FIXTURE_NAME;
    if (!std::filesystem::exists(stencil_fixture))
        stencil_fixture = fixture.parent_path().parent_path() / "share" / "kora" / "benchmarks" /
                          KORA_BENCH_STENCIL_FIXTURE_NAME;
    auto stencil = runtime.load_kernel(stencil_fixture, "main", std::move(stencil_metadata));
    if (!stencil) {
        std::cerr << "GPU benchmarks unavailable: " << stencil.error().describe() << '\n';
        return false;
    }
    for (const auto count : kComputeSizes) {
        const std::vector<float> stencil_input(count, 1.0F);
        auto stencil_source = runtime.create_buffer<float>(count);
        auto stencil_output = runtime.create_buffer<float>(count);
        if (!stencil_source || !stencil_output || !stencil_source.value().write(stencil_input)) return false;
        const auto run_stencil = [&](bool defer_readback = false) {
            auto builder = runtime.dispatch(stencil.value())
                               .global(count)
                               .read(stencil_source.value())
                               .write(stencil_output.value())
                               .push(count);
            if (defer_readback) builder.defer_readback();
            auto event = builder.submit();
            if (!event || !event.value().wait()) throw std::runtime_error("Vulkan stencil dispatch failed");
        };
        const auto validate_stencil = [&] {
            auto readback = runtime.readback(stencil_output.value());
            if (!readback || !readback.value().wait()) return false;
            auto values = stencil_output.value().read();
            return values && values.value().front() == 2.0F && values.value().back() == 2.0F &&
                   std::all_of(values.value().begin() + 1, values.value().end() - 1,
                               [](float value) { return value == 3.0F; });
        };
        run_stencil();
        if (!validate_stencil()) return false;
        const auto name = "gpu_stencil_" + std::to_string(count);
        results.push_back(
            run_checked(name + "_dispatch_wait", [&] { run_stencil(); }, validate_stencil, options));
        results.push_back(run_checked(
            name + "_deferred_readback_dispatch_wait", [&] { run_stencil(true); }, validate_stencil,
            options));
        add_kora_gpu_timestamp(name + "_gpu_execution", [&] { run_stencil(); });
    }

    auto loader = kora::vk::Loader::load();
    if (!loader) return false;
    auto instance = loader.value().create_instance();
    if (!instance) return false;
    auto physical_devices = instance.value().physical_devices();
    if (!physical_devices || physical_devices.value().empty()) return false;
    const auto selected_index = selected_devices.front().index;
    if (selected_index >= physical_devices.value().size()) {
        std::cerr << "GPU benchmarks unavailable: Runtime-selected Vulkan device was not enumerated\n";
        return false;
    }
    const auto& selected_physical = physical_devices.value()[selected_index];
    if (!selected_device_matches(selected_devices.front(), selected_physical)) {
        std::cerr << "GPU benchmarks unavailable: Runtime and raw Vulkan device identities do not match\n";
        return false;
    }
    kora::vk::DeviceOptions device_options;
    device_options.enable_dedicated_transfer_queue = false;
    device_options.enable_timeline_semaphore = false;
    device_options.enable_synchronization2 = false;
    auto device = instance.value().create_device(selected_physical, device_options);
    if (!device) return false;
    device_report.index = selected_index;
    device_report.name = selected_physical.properties.deviceName;
    switch (selected_physical.properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            device_report.device_kind = "discrete_gpu";
            break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            device_report.device_kind = "integrated_gpu";
            break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            device_report.device_kind = "virtual_gpu";
            break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            device_report.device_kind = "cpu_vulkan";
            break;
        default:
            device_report.device_kind = "other";
            break;
    }
    device_report.api_version = version_string(selected_physical.properties.apiVersion);
    device_report.driver_name = selected_physical.vulkan12_properties.driverName;
    if (device_report.driver_name.empty()) device_report.driver_name = "unreported";
    device_report.compute_queue_family = device.value().compute_queue_family();
    if (!add_transfer_benchmarks(results, options, selected_physical, device.value())) return false;
    for (const auto count : {64U, 256U, 1024U, 4096U, 65536U, 262144U, 1048576U, 4194240U})
        if (!add_raw_vector_add_benchmark(results, options, selected_physical, device.value(), fixture, count,
                                          device_report))
            return false;
    if (!add_raw_non_vector_compute_benchmarks(results, options, selected_physical, device.value(), fixture))
        return false;
    kora::vk::CommandPoolSet::Options pool_options;
    pool_options.queue_family = device.value().compute_queue_family();
    pool_options.ring_size = 16;
    auto reused_pool = std::make_unique<kora::vk::CommandPoolSet>(device.value().api(),
                                                                  device.value().handle(), pool_options);
    if (reused_pool->capacity() != pool_options.ring_size) return false;
    const auto record_and_discard = [](kora::vk::CommandPoolSet& pool) {
        auto command = pool.acquire(0);
        if (!command || !pool.end_recording(command.value()) || !pool.discard(command.value()))
            throw std::runtime_error("Vulkan command-buffer reuse failed");
    };
    results.push_back(
        run("command_pool_reuse_steady_state", [&] { record_and_discard(*reused_pool); }, options));
    auto cold_pool_options = pool_options;
    cold_pool_options.ring_size = 1;
    results.push_back(run(
        "command_pool_cold_create_record_destroy",
        [&] {
            kora::vk::CommandPoolSet pool(device.value().api(), device.value().handle(), cold_pool_options);
            if (pool.capacity() != 1) throw std::runtime_error("Vulkan command pool create failed");
            record_and_discard(pool);
        },
        options));
    if (options.cluster && !add_cluster_benchmarks(results, options, saxpy_fixture, saxpy_metadata))
        return false;
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
    GpuDeviceReport gpu_device;
    try {
        kora::TaskGraph reused_graph;
        for (std::uint32_t index = 0; index < 64; ++index) {
            kora::TaskMetadata metadata;
            metadata.type = kora::TaskType::HostCallback;
            if (!reused_graph.add_task(std::move(metadata)))
                throw std::runtime_error("C++ replay benchmark graph creation failed");
        }
        auto reused_plan = kora::ExecutionPlan::create(reused_graph);
        if (!reused_plan) throw std::runtime_error("C++ replay benchmark plan creation failed");
        std::uint64_t reused_callbacks = 0;
        results.push_back(run(
            "task_graph_replay_64_reuse_plan",
            [&] {
                if (!reused_plan.value().replay(count_task, &reused_callbacks) || reused_callbacks % 64 != 0)
                    throw std::runtime_error("C++ replay benchmark callback failed");
            },
            options));
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
        for (const auto count : {1U, 64U, 256U, 1024U, 4096U, 65536U, 262144U, 1048576U, 4194240U}) {
            const auto name = "cpu_vector_add_" + std::to_string(count);
            std::vector<std::uint32_t> left(count), right(count), output(count);
            for (std::size_t index = 0; index < count; ++index) {
                left[index] = static_cast<std::uint32_t>(index % 17);
                right[index] = static_cast<std::uint32_t>(index % 29);
            }
            const auto operation = [&] {
                for (std::size_t index = 0; index < count; ++index)
                    output[index] = left[index] + right[index];
            };
            results.push_back(run_cpu_compute(
                name, output, operation,
                [&](std::size_t index, std::uint32_t value) { return value == left[index] + right[index]; },
                options));
        }
        add_cpu_compute_benchmarks(results, options);
    } catch (const std::exception& error) {
        std::cerr << "benchmark workload failed: " << error.what() << '\n';
        return 1;
    }
    if (options.gpu) {
#ifdef KORA_BENCH_VECTOR_ADD_FIXTURE_NAME
        try {
            if (!add_gpu_benchmarks(results, options, argv[0], gpu_device)) return 3;
        } catch (const std::exception& error) {
            std::cerr << "GPU benchmark workload failed: " << error.what() << '\n';
            return 1;
        }
#else
        std::cerr << "GPU benchmarks require glslangValidator at build time\n";
        return 3;
#endif
    }
    std::cout << "{\"schema_version\":3,\"clock\":\"steady_clock\",\"iterations\":" << options.iterations
              << ",\"warmup\":" << options.warmup
              << ",\"diagnostic_profile\":" << (options.profile_phases ? "true" : "false");
    std::cout << ",\"environment\":{\"os\":";
    json_string(std::cout, operating_system());
    std::cout << ",\"architecture\":";
    json_string(std::cout, architecture());
    std::cout << ",\"compiler\":";
    json_string(std::cout, compiler_name());
    std::cout << ",\"cpu_model\":";
    json_string(std::cout, cpu_model_name());
    std::cout << ",\"build_type\":";
#ifdef KORA_BENCH_BUILD_TYPE
    json_string(std::cout, KORA_BENCH_BUILD_TYPE);
#else
    json_string(std::cout, "unknown");
#endif
    std::cout << ",\"build_flags\":";
#ifdef KORA_BENCH_BUILD_FLAGS
    json_string(std::cout, KORA_BENCH_BUILD_FLAGS);
#else
    json_string(std::cout, "unknown");
#endif
    std::cout << ",\"source_revision\":";
#ifdef KORA_BENCH_SOURCE_REVISION
    json_string(std::cout, KORA_BENCH_SOURCE_REVISION);
#else
    json_string(std::cout, "unknown");
#endif
    std::cout << ",\"logical_cpu_count\":" << std::thread::hardware_concurrency() << '}';
    if (options.gpu) {
        std::cout << ",\"gpu_measurements\":{\"backend\":\"vulkan\",\"device_index\":" << gpu_device.index
                  << ",\"device_name\":";
        json_string(std::cout, gpu_device.name);
        std::cout << ",\"api_version\":";
        json_string(std::cout, gpu_device.api_version);
        std::cout << ",\"driver_name\":";
        json_string(std::cout, gpu_device.driver_name);
        std::cout << ",\"device_kind\":";
        json_string(std::cout, gpu_device.device_kind);
        std::cout << ",\"compute_queue_family\":" << gpu_device.compute_queue_family
                  << ",\"same_physical_device_for_kora_and_raw_vulkan\":true"
                     ",\"kora_gpu_timestamps_available\":"
                  << (!gpu_device.kora_gpu_timestamp_sizes.empty() ? "true" : "false")
                  << ",\"raw_gpu_timestamps_available\":"
                  << (!gpu_device.raw_gpu_timestamp_sizes.empty() ? "true" : "false")
                  << ",\"kora_gpu_timestamp_sizes\":[";
        for (std::size_t index = 0; index < gpu_device.kora_gpu_timestamp_sizes.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << gpu_device.kora_gpu_timestamp_sizes[index];
        }
        std::cout << "],\"raw_gpu_timestamp_sizes\":[";
        for (std::size_t index = 0; index < gpu_device.raw_gpu_timestamp_sizes.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << gpu_device.raw_gpu_timestamp_sizes[index];
        }
        const auto has_timestamp = [&](std::string_view prefix) {
            return std::any_of(results.begin(), results.end(), [&](const Result& result) {
                return result.name.starts_with(prefix) && result.name.ends_with("_gpu_execution");
            });
        };
        const auto kora_compute_timestamps = has_timestamp("gpu_saxpy_") || has_timestamp("gpu_reduction_") ||
                                             has_timestamp("gpu_tiled_scan_") ||
                                             has_timestamp("gpu_tiled_matmul_") ||
                                             has_timestamp("gpu_stencil_");
        const auto raw_compute_timestamps =
            has_timestamp("raw_vulkan_saxpy_") || has_timestamp("raw_vulkan_reduction_") ||
            has_timestamp("raw_vulkan_tiled_scan_") || has_timestamp("raw_vulkan_tiled_matmul_") ||
            has_timestamp("raw_vulkan_stencil_");
        const auto emit_compute_timestamps = [&](bool raw) {
            std::cout << '[';
            bool first = true;
            for (const auto& result : results) {
                const auto has_prefix = [&](std::string_view prefix) {
                    return result.name.starts_with(prefix);
                };
                const auto prefix = raw ? "raw_vulkan_" : "gpu_";
                const bool is_compute = has_prefix(std::string(prefix) + "saxpy_") ||
                                        has_prefix(std::string(prefix) + "reduction_") ||
                                        has_prefix(std::string(prefix) + "tiled_scan_") ||
                                        has_prefix(std::string(prefix) + "tiled_matmul_") ||
                                        has_prefix(std::string(prefix) + "stencil_");
                if (is_compute && result.name.ends_with("_gpu_execution")) {
                    if (!first) std::cout << ',';
                    json_string(std::cout, result.name);
                    first = false;
                }
            }
            std::cout << ']';
        };
        std::cout << "],\"kora_gpu_compute_timestamps_available\":"
                  << (kora_compute_timestamps ? "true" : "false")
                  << ",\"kora_gpu_compute_timestamp_workloads\":";
        emit_compute_timestamps(false);
        std::cout << ",\"raw_gpu_compute_timestamps_available\":"
                  << (raw_compute_timestamps ? "true" : "false")
                  << ",\"raw_gpu_compute_timestamp_workloads\":";
        emit_compute_timestamps(true);
        std::cout << ",\"paths\":[\"kora_runtime\",\"kora_transfer_engine\",\"raw_vulkan\"]}";
    }
    std::cout << ",\"results\":[";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const auto& result = results[index];
        if (index != 0) std::cout << ',';
        std::cout << "{\"name\":\"" << result.name << "\",\"unit\":\"ns\",\"median\":" << result.median_ns
                  << ",\"p95\":" << result.p95_ns << ",\"timing_domain\":";
        json_string(std::cout, measurement_domain(result.name));
        if (options.gpu) {
            const auto path = gpu_path(result.name);
            if (!path.empty()) {
                std::cout << ",\"path\":";
                json_string(std::cout, path);
                std::cout << ",\"device_index\":" << gpu_device.index;
            }
        }
        if (result.bytes_transferred != 0)
            std::cout << ",\"bytes_transferred\":" << result.bytes_transferred
                      << ",\"throughput_bytes_per_second\":" << result.throughput_bytes_per_second;
        if (result.thread_count != 0) std::cout << ",\"thread_count\":" << result.thread_count;
        std::cout << ",\"samples\":[";
        for (std::size_t sample = 0; sample < result.samples_ns.size(); ++sample) {
            if (sample != 0) std::cout << ',';
            std::cout << result.samples_ns[sample];
        }
        std::cout << "]}";
    }
    std::cout << "]}\n";
    return std::cout ? 0 : 1;
}
