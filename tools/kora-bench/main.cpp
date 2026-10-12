#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

#include "core/device_caps.hpp"
#include "core/scheduler_cost.hpp"
#include "kora/execution_plan.hpp"
#include "kora/runtime.hpp"
#include "kora/task_graph.hpp"
#include "memory/memory_manager.hpp"
#include "memory/memory_pool.hpp"
#include "memory/suballocator.hpp"
#include "memory/transfer_engine.hpp"
#include "vulkan/api.hpp"
#include "vulkan/command_pool_set.hpp"
#include "vulkan/queue_context.hpp"

namespace {
using Clock = std::chrono::steady_clock;

struct Options {
    std::size_t iterations = 100;
    std::size_t warmup = 10;
    bool gpu = false;
    bool profile_phases = false;
};

struct Result {
    std::string name;
    std::vector<std::uint64_t> samples_ns;
    std::uint64_t median_ns = 0;
    std::uint64_t p95_ns = 0;
    std::uint64_t bytes_transferred = 0;
    std::uint64_t throughput_bytes_per_second = 0;
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

std::string_view measurement_domain(std::string_view name) {
    if (name.starts_with("profile_kora_")) return "runtime_dispatch_phase";
    if (name.ends_with("_gpu_execution")) return "gpu_execution_timestamp";
    if (name.starts_with("cpu_")) return "host_cpu_execution";
    if (name.starts_with("gpu_") || name.starts_with("raw_vulkan_vector_add_"))
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
                << "Usage: kora-bench [--iterations N] [--warmup N] [--smoke] [--gpu] [--profile-phases]\n"
                   "Runs CPU microbenchmarks; --gpu adds Vulkan workloads and --profile-phases\n"
                   "adds telemetry-instrumented dispatch phase diagnostics.\n";
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

bool cpu_vector_add_operation(std::uint32_t count) {
    static const std::vector<std::uint32_t> left(4'194'240, 3);
    static const std::vector<std::uint32_t> right(4'194'240, 7);
    static std::vector<std::uint32_t> output(4'194'240);
    for (std::size_t index = 0; index < count; ++index) output[index] = left[index] + right[index];
    static std::atomic<std::uint32_t> sink{0};
    sink.store(output[count - 1], std::memory_order_relaxed);
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
bool add_gpu_benchmarks(std::vector<Result>& results, const Options& options, const char* executable_path,
                        GpuDeviceReport& device_report) {
    kora::RuntimeOptions runtime_options;
    runtime_options.telemetry =
        options.profile_phases ? kora::TelemetryLevel::Profile : kora::TelemetryLevel::Off;
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
            [&, count]() -> std::optional<std::uint64_t> {
                auto before = runtime.profiling_stats(selected_devices.front());
                if (!before) return std::nullopt;
                dispatch(count, true);
                auto after = runtime.profiling_stats(selected_devices.front());
                if (!after || after.value().last_submission_id == before.value().last_submission_id)
                    return std::nullopt;
                return after.value().last_duration_nanoseconds;
            },
            options);
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

        std::array<std::vector<std::uint64_t>, 14> phase_samples;
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
        constexpr std::array<std::string_view, 14> phase_names{"builder_setup",
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
                                                               "queue_completion"};
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
    results.push_back(run("gpu_saxpy_4096_upload_dispatch_wait", [&] { run_saxpy(); }, options));

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
    std::vector<float> reduction_input(element_count, 1.0F);
    auto reduction_source = runtime.create_buffer<float>(element_count);
    auto reduction_output = runtime.create_buffer<float>(element_count / 64);
    if (!reduction_source || !reduction_output || !reduction_source.value().write(reduction_input))
        return false;
    const auto run_reduction = [&] {
        auto event = runtime.dispatch(reduction.value())
                         .global(element_count)
                         .read(reduction_source.value())
                         .write(reduction_output.value())
                         .push(element_count)
                         .submit();
        if (!event || !event.value().wait()) throw std::runtime_error("Vulkan reduction dispatch failed");
    };
    run_reduction();
    auto reduced_values = reduction_output.value().read();
    if (!reduced_values || !std::all_of(reduced_values.value().begin(), reduced_values.value().end(),
                                        [](float value) { return value == 64.0F; }))
        return false;
    results.push_back(run("gpu_reduction_4096_dispatch_wait", run_reduction, options));

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
    std::vector<std::uint32_t> scan_input(element_count, 1U);
    auto scan_source = runtime.create_buffer<std::uint32_t>(element_count);
    auto scan_output = runtime.create_buffer<std::uint32_t>(element_count);
    if (!scan_source || !scan_output || !scan_source.value().write(scan_input)) return false;
    const auto run_scan = [&] {
        auto event = runtime.dispatch(scan.value())
                         .global(element_count)
                         .read(scan_source.value())
                         .write(scan_output.value())
                         .push(element_count)
                         .submit();
        if (!event || !event.value().wait()) throw std::runtime_error("Vulkan scan dispatch failed");
    };
    run_scan();
    auto scanned_values = scan_output.value().read();
    if (!scanned_values || !std::all_of(scanned_values.value().begin(), scanned_values.value().end(),
                                        [index = std::uint32_t{0}](std::uint32_t value) mutable {
                                            return value == (index++ % 64) + 1;
                                        }))
        return false;
    results.push_back(run("gpu_tiled_scan_4096_dispatch_wait", run_scan, options));

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
    constexpr std::uint32_t matrix_dimension = 64;
    constexpr std::uint32_t matrix_elements = matrix_dimension * matrix_dimension;
    std::vector<float> matrix_values(matrix_elements, 1.0F);
    auto matrix_a = runtime.create_buffer<float>(matrix_elements);
    auto matrix_b = runtime.create_buffer<float>(matrix_elements);
    auto matrix_c = runtime.create_buffer<float>(matrix_elements);
    if (!matrix_a || !matrix_b || !matrix_c || !matrix_a.value().write(matrix_values) ||
        !matrix_b.value().write(matrix_values))
        return false;
    const auto run_matmul = [&] {
        auto event = runtime.dispatch(matmul.value())
                         .global(matrix_dimension, matrix_dimension, 1)
                         .read(matrix_a.value())
                         .read(matrix_b.value())
                         .write(matrix_c.value())
                         .push(matrix_dimension)
                         .submit();
        if (!event || !event.value().wait()) throw std::runtime_error("Vulkan tiled matmul failed");
    };
    run_matmul();
    auto matrix_values_out = matrix_c.value().read();
    if (!matrix_values_out || !std::all_of(matrix_values_out.value().begin(), matrix_values_out.value().end(),
                                           [](float value) { return value == 64.0F; }))
        return false;
    results.push_back(run("gpu_tiled_matmul_64_dispatch_wait", run_matmul, options));

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
    std::vector<float> stencil_input(element_count, 1.0F);
    auto stencil_source = runtime.create_buffer<float>(element_count);
    auto stencil_output = runtime.create_buffer<float>(element_count);
    if (!stencil_source || !stencil_output || !stencil_source.value().write(stencil_input)) return false;
    const auto run_stencil = [&] {
        auto event = runtime.dispatch(stencil.value())
                         .global(element_count)
                         .read(stencil_source.value())
                         .write(stencil_output.value())
                         .push(element_count)
                         .submit();
        if (!event || !event.value().wait()) throw std::runtime_error("Vulkan stencil dispatch failed");
    };
    run_stencil();
    auto stencil_values = stencil_output.value().read();
    if (!stencil_values || stencil_values.value().front() != 2.0F || stencil_values.value().back() != 2.0F ||
        !std::all_of(stencil_values.value().begin() + 1, stencil_values.value().end() - 1,
                     [](float value) { return value == 3.0F; }))
        return false;
    results.push_back(run("gpu_stencil_4096_dispatch_wait", run_stencil, options));

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
            results.push_back(run(
                name,
                [count] {
                    if (!cpu_vector_add_operation(count)) throw std::runtime_error("CPU vector-add failed");
                },
                options));
        }
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
    std::cout << ",\"build_type\":";
#ifdef KORA_BENCH_BUILD_TYPE
    json_string(std::cout, KORA_BENCH_BUILD_TYPE);
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
        std::cout << "],\"paths\":[\"kora_runtime\",\"kora_transfer_engine\",\"raw_vulkan\"]}";
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
