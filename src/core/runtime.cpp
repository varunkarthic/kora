#include "kora/runtime.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/buffer_manager_access.hpp"
#include "core/calibration.hpp"
#include "core/calibration_spv.hpp"
#include "core/device_caps.hpp"
#include "core/device_initialization.hpp"
#include "core/event_state.hpp"
#include "core/json.hpp"
#include "core/performance_db.hpp"
#include "core/runtime_test_access.hpp"
#include "core/scheduler_placement.hpp"
#include "kernel/autotuner.hpp"
#include "kernel/pipeline_manager.hpp"
#include "kernel/spirv.hpp"
#include "memory/eviction.hpp"
#include "memory/memory_manager.hpp"
#include "memory/memory_pool.hpp"
#include "memory/transfer_engine.hpp"
#include "profiling/timestamp.hpp"
#include "vulkan/api.hpp"
#include "vulkan/command_pool_set.hpp"
#include "vulkan/queue_context.hpp"
#include "vulkan/queue_family_transfer.hpp"

namespace kora {

namespace detail {

struct DeviceQuarantineRegistry {
    mutable std::mutex mutex;
    std::vector<DeviceId> devices;

    void add(const DeviceId& id) {
        std::lock_guard lock(mutex);
        if (std::find(devices.begin(), devices.end(), id) == devices.end()) devices.push_back(id);
    }

    std::vector<DeviceId> snapshot() const {
        std::lock_guard lock(mutex);
        return devices;
    }
};

struct BufferIdHash {
    std::size_t operator()(const BufferId& id) const noexcept {
        return static_cast<std::size_t>(id.generation ^ (static_cast<std::uint64_t>(id.slot) << 32U));
    }
};

struct RuntimeBufferState {
    vk::Api api;
    VkDevice device = VK_NULL_HANDLE;
    std::function<Status()> refresh_budget;
    detail::TransferEngine* transfers = nullptr;
    std::vector<detail::MemoryPool*> pools;
    detail::MemoryPool* host_pool = nullptr;
};

struct KernelState {
    std::string entry_point;
    KernelMetadata metadata;
    std::vector<std::uint32_t> words;
    std::uint64_t runtime_id = 0;
    std::uint64_t device_index = 0;
    std::uint64_t semantic_identity = 0;
    bool local_size_id_based = false;
    std::array<std::uint32_t, 3> local_size_spec_ids{};
    std::shared_ptr<detail::kernel::WorkgroupAutotuner> autotuner;
    std::vector<std::string> autotune_keys;
};

struct PendingPerformanceObservation {
    std::string execution_key;
    std::string submission_key;
    std::string transfer_key;
    std::string queue_completion_key;
    std::uint64_t submission_nanoseconds = 0;
    std::uint64_t transfer_nanoseconds = 0;
    std::uint64_t completion_elapsed_nanoseconds = 0;
    std::uint64_t completion_worker_delay_nanoseconds = 0;
    std::uint64_t device_completion_wait_nanoseconds = 0;
    std::uint64_t timestamp_collection_nanoseconds = 0;
    std::uint64_t completion_callback_nanoseconds = 0;
    std::uint64_t dispatch_validation_nanoseconds = 0;
    std::uint64_t binding_preflight_nanoseconds = 0;
    std::uint64_t pipeline_lookup_nanoseconds = 0;
    std::uint64_t buffer_residency_nanoseconds = 0;
    std::uint64_t submission_assembly_nanoseconds = 0;
    std::uint64_t queue_enqueue_nanoseconds = 0;
    std::uint64_t command_buffer_acquire_nanoseconds = 0;
    std::uint64_t descriptor_setup_nanoseconds = 0;
    std::uint64_t command_recording_nanoseconds = 0;
};

struct TraceSchedulerScore {
    std::uint64_t selected_device_index = 0;
    detail::ScoreBreakdown score;
};

struct DispatchContext {
    std::atomic<bool> accepting_work{true};
    std::atomic<bool> quarantined{false};
    std::shared_ptr<RuntimeBufferState> buffer_state;
    std::uint64_t runtime_id = 0;
    std::size_t device_index = 0;
    const DeviceInfo* info = nullptr;
    std::string device_uuid;
    std::uint32_t driver_version = 0;
    detail::kernel::PipelineManager* pipelines = nullptr;
    detail::TransferEngine* transfers = nullptr;
    vk::QueueContext* queue = nullptr;
    vk::QueueContext* transfer_queue = nullptr;
    vk::QueueFamilyPolicy queue_family_policy;
    const vk::Device* device = nullptr;
    // Unified memory: host-visible runtime buffers are written and read through their mapping
    // whenever no queued GPU operation can access them, instead of through staged transfers.
    bool direct_host_access = false;
    VkDeviceSize non_coherent_atom_size = 1;
    std::mutex submit_mutex;
    detail::EvictionPolicy eviction_policy;
    std::uint64_t access_tick = 0;
    std::unique_ptr<vk::CommandPoolSet> command_pool;
    std::unordered_map<BufferId, detail::TransferBuffer, BufferIdHash> buffers;
    std::unordered_map<BufferId, std::uint64_t, BufferIdHash> resident_host_versions;
    std::unordered_set<BufferId, BufferIdHash> device_newer_buffers;
    // Timeline values belong to this device's queue and are incomparable across devices.
    std::unordered_map<BufferId, std::uint64_t, BufferIdHash> device_last_use;
    std::atomic<std::uint64_t> upload_count{0};
    std::atomic<std::uint64_t> download_count{0};
    std::unique_ptr<detail::TimestampRing> timestamps;
    bool phase_profile_enabled = false;
    std::uint64_t latest_profile_submission_id = 0;
    detail::PendingPerformanceObservation latest_profile_observation;
    bool trace_enabled = false;
    bool autotuning_enabled = false;
    // Record GPU timestamps on every Nth ordinary dispatch (0 = never). Timestamp queries are not
    // free on every driver, so they are taken only as often as telemetry needs them; autotuning
    // trials and adaptive observations always record. Guarded by submit_mutex.
    std::uint32_t timestamp_sample_interval = 1;
    std::uint64_t timestamp_sample_counter = 0;
    struct PendingAutotune {
        std::shared_ptr<detail::kernel::WorkgroupAutotuner> tuner;
        detail::kernel::WorkgroupAutotuner::Trial trial;
        std::string key;
    };
    std::unordered_map<std::uint64_t, PendingAutotune> pending_autotune;
    std::unordered_map<std::uint64_t, PendingPerformanceObservation> pending_observations;
    std::unordered_map<std::uint64_t, TraceSchedulerScore> pending_trace_scores;
    std::deque<std::uint64_t> pending_trace_score_order;
    std::unordered_map<std::uint64_t, TraceSchedulerScore> completed_trace_scores;
    std::deque<std::uint64_t> completed_trace_score_order;
    std::shared_ptr<detail::PerformanceDatabase> performance_database;
    std::filesystem::path performance_database_path;
    std::unordered_map<BufferId, detail::PoolAllocation, BufferIdHash> allocations;
    std::unordered_map<BufferId, detail::MemoryPool*, BufferIdHash> allocation_pools;
    // Buffers whose last handle was released; their device copies are freed once idle.
    std::mutex released_mutex;
    std::vector<BufferId> released_buffers;

    void queue_released_buffer(BufferId id) {
        std::lock_guard lock(released_mutex);
        released_buffers.push_back(id);
    }

    // Frees device copies of released buffers that no queued or executing GPU work can access.
    // Buffers still in use stay queued and are retried on the next call.
    void reap_released_buffers() {
        std::vector<BufferId> candidates;
        {
            std::lock_guard lock(released_mutex);
            candidates.swap(released_buffers);
        }
        if (candidates.empty()) return;
        std::vector<BufferId> still_busy;
        {
            std::lock_guard lock(submit_mutex);
            for (const auto id : candidates) {
                const auto found = buffers.find(id);
                if (found != buffers.end()) {
                    if (found->second.queue_use &&
                        found->second.queue_use->pending.load(std::memory_order_acquire) != 0) {
                        still_busy.push_back(id);
                        continue;
                    }
                    if (device && found->second.buffer != VK_NULL_HANDLE)
                        device->api().destroy_buffer(device->handle(), found->second.buffer, nullptr);
                    buffers.erase(found);
                }
                const auto allocation = allocations.find(id);
                const auto pool = allocation_pools.find(id);
                if (allocation != allocations.end() && pool != allocation_pools.end())
                    (void)pool->second->deallocate(allocation->second);
                if (allocation != allocations.end()) allocations.erase(allocation);
                if (pool != allocation_pools.end()) allocation_pools.erase(pool);
                resident_host_versions.erase(id);
                device_newer_buffers.erase(id);
                device_last_use.erase(id);
                eviction_policy.forget(id);
            }
        }
        if (still_busy.empty()) return;
        std::lock_guard lock(released_mutex);
        released_buffers.insert(released_buffers.end(), still_busy.begin(), still_busy.end());
    }

    void release_buffers() noexcept {
        for (const auto& [id, buffer] : buffers) {
            (void)id;
            if (device && buffer.buffer != VK_NULL_HANDLE)
                device->api().destroy_buffer(device->handle(), buffer.buffer, nullptr);
        }
        for (const auto& [id, allocation] : allocations) {
            auto pool = allocation_pools.find(id);
            if (pool != allocation_pools.end()) (void)pool->second->deallocate(allocation);
        }
        buffers.clear();
        allocations.clear();
        allocation_pools.clear();
    }
    ~DispatchContext() { release_buffers(); }
};

struct VerificationTargetSet {
    struct Target {
        DeviceId id;
        std::shared_ptr<DispatchContext> context;
    };
    std::vector<Target> targets;
    std::shared_ptr<DeviceQuarantineRegistry> quarantine;
};

struct AdaptiveDispatchState {
    DeviceId original_id;
    std::shared_ptr<DispatchContext> original_context;
    std::shared_ptr<const KernelState> original_kernel;
    AdaptiveDispatchWorkload workload;
    std::optional<detail::ScoreBreakdown> selected_score;
    std::vector<std::shared_ptr<DispatchContext>> contexts;
};

void observe_autotune_samples(const std::shared_ptr<DispatchContext>& dispatch,
                              const std::vector<GpuTimestampSample>& samples,
                              const std::shared_ptr<PerformanceDatabase>& database,
                              const std::filesystem::path& database_path) {
    if (!dispatch || !database) return;
    bool save_required = false;
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    for (const auto& sample : samples) {
        DispatchContext::PendingAutotune pending;
        PendingPerformanceObservation observation;
        {
            std::lock_guard lock(dispatch->submit_mutex);
            const auto found = dispatch->pending_autotune.find(sample.submission_id);
            if (found != dispatch->pending_autotune.end()) {
                pending = std::move(found->second);
                dispatch->pending_autotune.erase(found);
            }
            const auto observed = dispatch->pending_observations.find(sample.submission_id);
            if (observed != dispatch->pending_observations.end()) {
                observation = std::move(observed->second);
                dispatch->pending_observations.erase(observed);
            }
            const auto trace_score = dispatch->pending_trace_scores.find(sample.submission_id);
            if (trace_score != dispatch->pending_trace_scores.end()) {
                dispatch->completed_trace_scores.insert_or_assign(sample.submission_id,
                                                                  std::move(trace_score->second));
                dispatch->completed_trace_score_order.push_back(sample.submission_id);
                dispatch->pending_trace_scores.erase(trace_score);
                constexpr std::size_t kTraceScoreHistoryLimit = 4096;
                while (dispatch->completed_trace_score_order.size() > kTraceScoreHistoryLimit) {
                    dispatch->completed_trace_scores.erase(dispatch->completed_trace_score_order.front());
                    dispatch->completed_trace_score_order.pop_front();
                }
            }
        }
        const auto duration = static_cast<double>(sample.duration_nanoseconds);
        if (pending.tuner && pending.tuner->observe(pending.trial, duration) && now_ms >= 0 &&
            !pending.trial.warmup) {
            save_required = static_cast<bool>(database->observe(pending.key, duration,
                                                                static_cast<std::uint64_t>(now_ms))) ||
                            save_required;
        }
        if (now_ms >= 0) {
            const auto observed_at = static_cast<std::uint64_t>(now_ms);
            const auto observe_domain = [&](const std::string& key, std::uint64_t nanoseconds) {
                if (key.empty() || nanoseconds == 0) return;
                const auto before = database->query(key);
                const auto before_count = before ? before.value().count : 0;
                if (!database->observe(key, static_cast<double>(nanoseconds), observed_at)) return;
                const auto after = database->query(key);
                if (after && after.value().count != before_count &&
                    (after.value().count == 1 || after.value().count == 5 || after.value().count % 8 == 0))
                    save_required = true;
            };
            observe_domain(observation.execution_key, sample.duration_nanoseconds);
            observe_domain(observation.submission_key, observation.submission_nanoseconds);
            observe_domain(observation.transfer_key, observation.transfer_nanoseconds);
            const auto queue_wait =
                observation.completion_elapsed_nanoseconds > sample.duration_nanoseconds
                    ? observation.completion_elapsed_nanoseconds - sample.duration_nanoseconds
                    : 0;
            observe_domain(observation.queue_completion_key, queue_wait);
        }
    }
    if (save_required) (void)database->save(database_path);
}

}  // namespace detail

namespace {
// Basic telemetry keeps profiling_stats() fresh while paying for timestamp queries on only one
// ordinary dispatch in 16; Profile and Trace need every dispatch.
std::uint32_t timestamp_sample_interval(TelemetryLevel level) {
    switch (level) {
        case TelemetryLevel::Off:
            return 0;
        case TelemetryLevel::Basic:
            return 16;
        case TelemetryLevel::Profile:
        case TelemetryLevel::Trace:
            return 1;
    }
    return 1;
}

std::string uuid_string(const VkPhysicalDeviceIDProperties& properties) {
    static constexpr char kHex[] = "0123456789abcdef";
    if (std::all_of(std::begin(properties.deviceUUID), std::end(properties.deviceUUID),
                    [](std::uint8_t byte) { return byte == 0; })) {
        return {};
    }
    std::string value;
    value.reserve(sizeof(properties.deviceUUID) * 2);
    for (const auto byte : properties.deviceUUID) {
        value.push_back(kHex[byte >> 4U]);
        value.push_back(kHex[byte & 0x0fU]);
    }
    return value;
}

std::uint64_t semantic_kernel_identity(const detail::KernelState& kernel) {
    std::uint64_t hash = 1469598103934665603ULL;
    auto append = [&](std::uint8_t byte) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    };
    for (const auto word : kernel.words)
        for (unsigned shift = 0; shift < 32; shift += 8) append(static_cast<std::uint8_t>(word >> shift));
    for (const unsigned char character : kernel.entry_point) append(character);
    return hash == 0 ? 1 : hash;
}

std::uint8_t workload_bucket(std::uint64_t work) {
    std::uint8_t bucket = 0;
    while (work > 1) {
        work >>= 1U;
        ++bucket;
    }
    return bucket;
}

std::optional<detail::PerformanceStats> query_execution_observation(
    const std::shared_ptr<detail::PerformanceDatabase>& database, const DeviceId& id,
    std::uint32_t driver_version, std::uint64_t kernel_identity, const AdaptiveDispatchWorkload& workload,
    std::uint64_t now_ms) {
    if (!database || id.uuid.empty()) return std::nullopt;
    const detail::PerformanceObservationKey identity{
        .device_uuid = id.uuid,
        .driver_version = driver_version,
        .runtime_schema = static_cast<std::uint32_t>(kora::kSchemaVersion),
        .kernel_identity = kernel_identity,
        .workload_class = static_cast<std::uint8_t>(workload.kernel_kind),
        .workload_bucket = workload_bucket(workload.kernel_work),
        .timing_domain = detail::PerformanceTimingDomain::DeviceExecution,
    };
    auto key = detail::performance_observation_key(identity);
    if (!key) return std::nullopt;
    return database->query_confident(key.value(), now_ms);
}

DeviceKind device_kind(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return DeviceKind::Integrated;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return DeviceKind::Discrete;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return DeviceKind::Virtual;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return DeviceKind::Cpu;
        default:
            return DeviceKind::Other;
    }
}

}  // namespace

struct Runtime::Impl {
    Impl(vk::Loader loader_value, vk::Instance instance_value)
        : loader(std::move(loader_value)), instance(std::move(instance_value)) {
        static std::atomic<std::uint64_t> next{1};
        runtime_id = next.fetch_add(1, std::memory_order_relaxed);
    }

    struct DeviceEntry {
        DeviceEntry(DeviceId device_id, DeviceInfo device_info, detail::MemoryManager manager,
                    vk::Device logical_device,
                    std::unique_ptr<detail::kernel::PipelineManager> pipeline_manager,
                    std::uint64_t runtime_identity, bool force_fallback,
                    std::vector<VkTimeDomainKHR> calibrated_time_domains,
                    ShutdownMode configured_shutdown_mode)
            : id(std::move(device_id)),
              info(std::move(device_info)),
              memory_manager(std::move(manager)),
              device(std::move(logical_device)),
              queue_family_policy{device.compute_queue_family(), device.transfer_queue_family(), true},
              pipelines(std::move(pipeline_manager)),
              runtime_id(runtime_identity),
              force_legacy_sync(force_fallback),
              calibrated_time_domains(std::move(calibrated_time_domains)),
              shutdown_mode(configured_shutdown_mode) {}

        Status initialize() {
#ifdef VK_EXT_device_fault
            if (device.api().get_device_fault_info != nullptr) {
                const auto get_fault_info = device.api().get_device_fault_info;
                const auto logical_device = device.handle();
                fault_state->set_diagnostics_query([get_fault_info, logical_device] {
                    vk::DeviceFaultSnapshot snapshot;
                    snapshot.supported = true;
                    VkDeviceFaultCountsEXT counts{};
                    counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
                    VkResult result = get_fault_info(logical_device, &counts, nullptr);
                    snapshot.query_result = result;
                    if (result != VK_SUCCESS) return snapshot;

                    constexpr std::uint32_t kMaximumFaultRecords = 256;
                    const auto address_capacity = std::min(counts.addressInfoCount, kMaximumFaultRecords);
                    const auto vendor_capacity = std::min(counts.vendorInfoCount, kMaximumFaultRecords);
                    snapshot.truncated = counts.addressInfoCount > address_capacity ||
                                         counts.vendorInfoCount > vendor_capacity;
                    std::vector<VkDeviceFaultAddressInfoEXT> addresses(address_capacity);
                    std::vector<VkDeviceFaultVendorInfoEXT> vendors(vendor_capacity);
                    VkDeviceFaultInfoEXT info{};
                    info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
                    info.pAddressInfos = addresses.data();
                    info.pVendorInfos = vendors.data();
                    counts.addressInfoCount = address_capacity;
                    counts.vendorInfoCount = vendor_capacity;
                    counts.vendorBinarySize = 0;
                    result = get_fault_info(logical_device, &counts, &info);
                    snapshot.query_result = result;
                    snapshot.truncated = snapshot.truncated || result == VK_INCOMPLETE;
                    const auto description_end =
                        std::find(std::begin(info.description), std::end(info.description), '\0');
                    snapshot.description.assign(std::begin(info.description), description_end);
                    const auto returned_addresses = std::min(counts.addressInfoCount, address_capacity);
                    snapshot.addresses.reserve(returned_addresses);
                    for (std::uint32_t index = 0; index < returned_addresses; ++index) {
                        const auto& value = addresses[index];
                        snapshot.addresses.push_back({static_cast<std::uint32_t>(value.addressType),
                                                      value.reportedAddress, value.addressPrecision});
                    }
                    const auto returned_vendors = std::min(counts.vendorInfoCount, vendor_capacity);
                    snapshot.vendor_records.reserve(returned_vendors);
                    for (std::uint32_t index = 0; index < returned_vendors; ++index) {
                        const auto& value = vendors[index];
                        const auto vendor_description_end =
                            std::find(std::begin(value.description), std::end(value.description), '\0');
                        snapshot.vendor_records.push_back(
                            {std::string(std::begin(value.description), vendor_description_end),
                             value.vendorFaultCode, value.vendorFaultData});
                    }
                    snapshot.truncated = snapshot.truncated || counts.addressInfoCount > address_capacity ||
                                         counts.vendorInfoCount > vendor_capacity;
                    return snapshot;
                });
            }
#endif
            vk::QueueContext::CoalescingOptions coalescing;
            coalescing.enabled = true;
            // Coalesce work already queued together, but don't delay an isolated dispatch in
            // anticipation of a future peer submission.
            coalescing.maximum_wait_ns = 0;
            auto queue_result =
                vk::QueueContext::create(device.api(), device.handle(), device.compute_queue(),
                                         force_legacy_sync, coalescing, fault_state);
            if (!queue_result) return Status(std::move(queue_result).error());
            queue = std::move(queue_result).value();
            transfer_queue = queue.get();
            if (queue_family_policy.compute_family != queue_family_policy.transfer_family) {
                auto transfer_result = vk::QueueContext::create(
                    device.api(), device.handle(), device.transfer_queue(), force_legacy_sync, fault_state);
                if (!transfer_result) return Status(std::move(transfer_result).error());
                dedicated_transfer_queue = std::move(transfer_result).value();
                transfer_queue = dedicated_transfer_queue.get();
            }

            const auto memory = device.buffer_memory_info();
            std::optional<std::uint32_t> host_pool_index;
            for (std::uint32_t index = 0; index < info.memory_types().size(); ++index) {
                const auto& selected_type = info.memory_types()[index];
                detail::MemoryPoolConfig config;
                config.memory_type_index = index;
                config.heap_index = selected_type.heap_index;
                config.host_visible = selected_type.host_visible;
                config.host_coherent = selected_type.host_coherent;
                config.heap_size = info.memory_heaps()[selected_type.heap_index].size_bytes;
                auto initial_budget = memory_manager.budget(config.heap_index);
                if (!initial_budget) return Status(std::move(initial_budget).error());
                config.budget = initial_budget.value().limit_bytes;
                config.block_size_hint = std::max<std::uint64_t>(
                    std::min<std::uint64_t>(16U * 1024U * 1024U, config.budget), 1U * 1024U * 1024U);
                config.reserve_budget = [this, heap = config.heap_index](std::uint64_t bytes) {
                    std::lock_guard lock(budget_mutex);
                    return memory_manager.reserve(heap, bytes);
                };
                config.release_budget = [this, heap = config.heap_index](std::uint64_t bytes) {
                    std::lock_guard lock(budget_mutex);
                    return memory_manager.release(heap, bytes);
                };
                config.allocation_failed = [this, heap = config.heap_index] {
                    std::lock_guard lock(budget_mutex);
                    return memory_manager.record_allocation_failure(heap);
                };
                config.refresh_budget = [this]() -> Result<bool> {
                    auto refreshed = refresh_budget_if_due(true);
                    if (!refreshed) return std::move(refreshed).error();
                    return true;
                };
                auto pool = detail::MemoryPool::create(device, config);
                if (!pool) return Status(std::move(pool).error());
                if (selected_type.host_visible && !host_pool_index) host_pool_index = index;
                pools.push_back(std::move(pool).value());
            }
            if (pools.empty() || !host_pool_index)
                return Error(
                    ErrorCode::UnsupportedFeature,
                    "selected Vulkan device has no memory types suitable for runtime buffers and transfers");

            detail::TransferEngine::Options transfer_options;
            transfer_options.queue_family = queue_family_policy.transfer_family;
            transfer_options.cross_queue_concurrent = queue_family_policy.uses_concurrent_sharing();
            // Keep staging rings on every topology. Unified-memory contexts bypass them per buffer
            // (see DispatchContext::direct_host_access) and fall back to them while a buffer still
            // has queued GPU work or has no host mapping.
            transfer_options.topology = MemoryTopology::Discrete;
            transfer_options.non_coherent_atom_size =
                std::max<VkDeviceSize>(memory.properties.limits.nonCoherentAtomSize, 1);
            auto transfer_result =
                detail::TransferEngine::create(device, *transfer_queue, *pools[*host_pool_index],
                                               *pools[*host_pool_index], transfer_options);
            if (!transfer_result) return Status(std::move(transfer_result).error());
            transfers = std::move(transfer_result).value();

            auto buffer_state = std::make_shared<detail::RuntimeBufferState>();
            buffer_state->api = device.api();
            buffer_state->device = device.handle();
            buffer_state->refresh_budget = [this] { return refresh_budget_if_due(); };
            buffer_state->transfers = transfers.get();
            for (auto& pool : pools) buffer_state->pools.push_back(pool.get());
            buffer_state->host_pool = pools[*host_pool_index].get();
            dispatch = std::make_shared<detail::DispatchContext>();
            dispatch->device = &device;
            dispatch->buffer_state = std::move(buffer_state);
            dispatch->runtime_id = runtime_id;
            dispatch->device_index = id.index;
            dispatch->info = &this->info;
            dispatch->pipelines = pipelines.get();
            dispatch->transfers = transfers.get();
            dispatch->queue = queue.get();
            dispatch->transfer_queue = transfer_queue;
            dispatch->queue_family_policy = queue_family_policy;
            dispatch->direct_host_access = info.memory_topology() == MemoryTopology::Uma;
            dispatch->non_coherent_atom_size = transfer_options.non_coherent_atom_size;
            const auto compute_family = queue_family_policy.compute_family;
            if (compute_family < info.queue_families().size()) {
                const auto valid_bits = info.queue_families()[compute_family].timestamp_valid_bits;
                auto timestamps = std::make_unique<detail::TimestampRing>(
                    device.api(), device.handle(), valid_bits, info.timestamp_period_nanoseconds(),
                    calibrated_time_domains);
                auto timestamp_status = timestamps->initialize();
                if (timestamp_status)
                    dispatch->timestamps = std::move(timestamps);
                else if (timestamp_status.error().code != ErrorCode::UnsupportedFeature)
                    return timestamp_status;
            }
            return {};
        }

        void configure_budget_refresh(vk::Instance& instance_value, double fallback_fraction) {
            instance = &instance_value;
            memory_budget_fraction = fallback_fraction;
        }

        Status refresh_budget_if_due(bool force = false) const {
            const auto now = std::chrono::steady_clock::now();
            std::lock_guard lock(budget_mutex);
            if (!force && now - last_budget_refresh < std::chrono::seconds(1)) return {};
            auto physical_devices = instance->physical_devices();
            if (!physical_devices) return Status(std::move(physical_devices).error());
            if (id.index >= physical_devices.value().size())
                return Error(ErrorCode::InternalInvariantViolation,
                             "physical device index changed during budget refresh");
            auto status =
                memory_manager.refresh_budget(physical_devices.value()[id.index], memory_budget_fraction);
            if (status) last_budget_refresh = now;
            return status;
        }

        void calibrate(const std::filesystem::path& directory, std::uint32_t budget_ms,
                       bool force = false) noexcept {
            try {
                const auto started = std::chrono::steady_clock::now();
                const auto hard_deadline = started + std::chrono::milliseconds(budget_ms);
                calibration_deadline = hard_deadline;
                const auto write_reserve =
                    std::min(std::chrono::milliseconds(10), std::chrono::milliseconds(budget_ms / 5));
                const auto compute_reserve =
                    std::min(std::chrono::milliseconds(150), std::chrono::milliseconds(budget_ms / 2));
                calibration_deadline = hard_deadline - write_reserve;
                const auto deadline = calibration_deadline - compute_reserve;
                const auto key = id.uuid.empty() ? std::to_string(id.index) : id.uuid;
                const auto path = directory / "calibration" / (key + ".json");
                detail::calibration::Record cached;
                if (id.uuid.empty()) return;
                if (!force && detail::calibration::read(path, id.uuid, driver_version, cached)) {
                    calibration_record = cached;
                    compute_calibration_pending = cached.tiny_dispatch_ns == 0 ||
                                                  cached.streaming_bytes_per_second == 0 ||
                                                  cached.alu_operations_per_second == 0;
                    return;
                }
                detail::calibration::Record record;
                record.device_uuid = id.uuid;
                record.driver_version = driver_version;
                std::array<std::uint64_t, 3> submit_samples{};
                std::size_t submit_sample_count = 0;
                for (auto& sample : submit_samples) {
                    if (std::chrono::steady_clock::now() >= deadline || !queue) break;
                    const auto sample_start = std::chrono::steady_clock::now();
                    auto submitted = queue->submit(vk::QueueContext::Submission{});
                    if (!submitted) break;
                    const auto remaining = deadline - std::chrono::steady_clock::now();
                    if (remaining <= std::chrono::steady_clock::duration::zero()) break;
                    const auto remaining_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(remaining);
                    auto completed = queue->wait_for_value(submitted.value(),
                                                           static_cast<std::uint64_t>(remaining_ns.count()));
                    if (!completed) break;
                    sample = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                            std::chrono::steady_clock::now() - sample_start)
                                                            .count());
                    ++submit_sample_count;
                }
                if (submit_sample_count > 0) {
                    std::sort(submit_samples.begin(), submit_samples.begin() + submit_sample_count);
                    record.submit_overhead_ns = submit_samples[submit_sample_count / 2];
                }
                // Transfer measurements use private, temporary buffers and are bounded by the
                // remaining startup budget. They never inspect or modify application buffers.
                if (std::chrono::steady_clock::now() < deadline && transfers) {
                    struct ScratchBuffer {
                        const vk::Device* device = nullptr;
                        detail::MemoryPool* pool = nullptr;
                        detail::PoolAllocation allocation;
                        detail::TransferBuffer transfer;
                        ~ScratchBuffer() {
                            if (device && transfer.buffer != VK_NULL_HANDLE)
                                device->api().destroy_buffer(device->handle(), transfer.buffer, nullptr);
                            if (pool && allocation.token != 0) (void)pool->deallocate(allocation);
                        }
                    };
                    const auto make_scratch = [&](VkDeviceSize size) -> std::shared_ptr<ScratchBuffer> {
                        auto scratch = std::make_shared<ScratchBuffer>();
                        scratch->device = &device;
                        VkBufferCreateInfo buffer_info{};
                        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                        buffer_info.size = size;
                        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                        const std::array families{queue_family_policy.compute_family,
                                                  queue_family_policy.transfer_family};
                        if (queue_family_policy.uses_concurrent_sharing()) {
                            buffer_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
                            buffer_info.queueFamilyIndexCount = static_cast<std::uint32_t>(families.size());
                            buffer_info.pQueueFamilyIndices = families.data();
                        } else {
                            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                        }
                        VkBuffer buffer = VK_NULL_HANDLE;
                        if (device.api().create_buffer(device.handle(), &buffer_info, nullptr, &buffer) !=
                            VK_SUCCESS)
                            return {};
                        scratch->transfer.buffer = buffer;
                        VkMemoryRequirements requirements{};
                        device.api().get_buffer_memory_requirements(device.handle(), buffer, &requirements);
                        auto type = detail::MemoryManager::select_memory_type(
                            info.memory_types(), info.memory_type_preferences(),
                            detail::MemoryPolicy::DevicePreferred, requirements.memoryTypeBits);
                        if (!type) return {};
                        auto pool = std::find_if(pools.begin(), pools.end(), [&](const auto& candidate) {
                            return candidate->memory_type_index() == type.value();
                        });
                        if (pool == pools.end()) return {};
                        auto allocation =
                            (*pool)->allocate({requirements.size, requirements.alignment, false, false});
                        if (!allocation) return {};
                        scratch->pool = pool->get();
                        scratch->allocation = allocation.value();
                        const auto bound = device.api().bind_buffer_memory(
                            device.handle(), buffer, allocation.value().memory, allocation.value().offset);
                        if (bound != VK_SUCCESS) return {};
                        const auto& memory_type = info.memory_types()[type.value()];
                        scratch->transfer = detail::TransferBuffer{
                            .buffer = buffer,
                            .size = size,
                            .memory = allocation.value().memory,
                            .memory_offset = allocation.value().offset,
                            .memory_allocation_size = allocation.value().dedicated
                                                          ? allocation.value().allocated_size
                                                          : (*pool)->block_size(),
                            .mapped_data = static_cast<std::byte*>(allocation.value().mapped_data),
                            .host_visible = memory_type.host_visible,
                            .host_coherent = memory_type.host_coherent,
                            .queue_use = std::make_shared<detail::QueueTimelineUse>()};
                        return scratch;
                    };

                    const auto bytes_per_second = [](std::uint64_t bytes, std::uint64_t nanoseconds) {
                        if (nanoseconds == 0) return std::uint64_t{0};
                        const auto rate = static_cast<long double>(bytes) * 1'000'000'000.0L /
                                          static_cast<long double>(nanoseconds);
                        return rate >= static_cast<long double>(UINT64_MAX)
                                   ? UINT64_MAX
                                   : static_cast<std::uint64_t>(rate);
                    };
                    const auto first = make_scratch(256U * 1024U);
                    const auto second = first ? make_scratch(256U * 1024U) : nullptr;
                    const auto measure = [&](std::uint64_t bytes, const auto& submit,
                                             std::uint64_t& best_rate,
                                             const std::shared_ptr<ScratchBuffer>& keep_first,
                                             const std::shared_ptr<ScratchBuffer>& keep_second = {},
                                             const std::shared_ptr<void>& keep_extra = {}) {
                        if (std::chrono::steady_clock::now() >= deadline) return false;
                        const auto started = std::chrono::steady_clock::now();
                        auto event = submit();
                        if (!event) return false;
                        auto retained =
                            event.value().on_complete([keep_first, keep_second, keep_extra](Status) {});
                        if (!retained) {
                            (void)event.value().wait();
                            return false;
                        }
                        const auto remaining = deadline - std::chrono::steady_clock::now();
                        if (remaining <= std::chrono::steady_clock::duration::zero()) return false;
                        auto completed = event.value().wait_for(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(remaining));
                        if (!completed) return false;
                        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count();
                        best_rate =
                            std::max(best_rate, bytes_per_second(bytes, static_cast<std::uint64_t>(elapsed)));
                        return true;
                    };
                    if (first && second) {
                        for (const std::uint64_t size : {4'096U, 65'536U, 256U * 1024U}) {
                            if (std::chrono::steady_clock::now() >= deadline) break;
                            auto input = std::make_shared<std::vector<std::byte>>(
                                static_cast<std::size_t>(size), std::byte{0x5a});
                            auto output =
                                std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(size));
                            const bool uploaded = measure(
                                size, [&] { return transfers->upload(first->transfer, 0, *input); },
                                record.h2d_bytes_per_second, first, second, input);
                            if (!uploaded) break;
                            const bool downloaded = measure(
                                size, [&] { return transfers->download(first->transfer, 0, *output); },
                                record.d2h_bytes_per_second, first, second, output);
                            if (!downloaded || *output != *input) break;
                            const bool copied = measure(
                                size,
                                [&] {
                                    return transfers->copy(first->transfer, 0, second->transfer, 0, size);
                                },
                                record.device_copy_bytes_per_second, first, second);
                            if (!copied) break;
                            std::fill(output->begin(), output->end(), std::byte{0});
                            const bool copy_readback = measure(
                                size, [&] { return transfers->download(second->transfer, 0, *output); },
                                record.d2h_bytes_per_second, first, second, output);
                            if (!copy_readback || *output != *input) break;
                        }
                    }
                }
                // Leave a compute window after queue/transfer probes and before record publication.
                if (std::chrono::steady_clock::now() >= hard_deadline) return;
                calibration_record = record;
                compute_calibration_pending = true;
                (void)detail::calibration::write_atomic(path, record);
            } catch (...) {
                // Calibration is best-effort; it must never prevent Runtime creation.
            }
        }

        void shutdown(ShutdownMode mode) noexcept {
            if (stopped) return;
            stopped = true;
            if (dispatch) dispatch->accepting_work.store(false, std::memory_order_release);
            if (queue) (void)queue->shutdown(mode);
            if (dedicated_transfer_queue) (void)dedicated_transfer_queue->shutdown(mode);
            if (dispatch) dispatch->release_buffers();
        }

        ~DeviceEntry() { shutdown(shutdown_mode); }

        DeviceId id;
        DeviceInfo info;
        mutable detail::MemoryManager memory_manager;
        vk::Device device;
        vk::QueueFamilyPolicy queue_family_policy;
        std::unique_ptr<detail::kernel::PipelineManager> pipelines;
        mutable std::mutex pools_mutex;
        mutable std::mutex budget_mutex;
        mutable std::chrono::steady_clock::time_point last_budget_refresh{};
        vk::Instance* instance = nullptr;
        double memory_budget_fraction = 0.8;
        std::vector<std::unique_ptr<detail::MemoryPool>> pools;
        std::unique_ptr<detail::TransferEngine> transfers;
        std::shared_ptr<detail::DispatchContext> dispatch;
        // Queue destruction drains callbacks before their dispatch/transfer/pool dependencies go away.
        std::unique_ptr<vk::QueueContext> queue;
        std::unique_ptr<vk::QueueContext> dedicated_transfer_queue;
        vk::QueueContext* transfer_queue = nullptr;
        std::uint64_t runtime_id = 0;
        std::shared_ptr<vk::DeviceFaultState> fault_state = std::make_shared<vk::DeviceFaultState>();
        bool force_legacy_sync = false;
        std::vector<VkTimeDomainKHR> calibrated_time_domains;
        std::uint32_t driver_version = 0;
        detail::calibration::Record calibration_record;
        std::chrono::steady_clock::time_point calibration_deadline{};
        bool compute_calibration_pending = false;
        ShutdownMode shutdown_mode = ShutdownMode::Graceful;
        bool stopped = false;
    };

    vk::Loader loader;
    vk::Instance instance;
    std::vector<std::unique_ptr<DeviceEntry>> devices;
    // Immutable device membership is shared by dispatch builders; quarantine is checked live
    // when duplicate verification is submitted. Avoid rebuilding this list on every dispatch().
    std::shared_ptr<const detail::VerificationTargetSet> verification_targets;
    std::shared_ptr<detail::DeviceQuarantineRegistry> quarantine =
        std::make_shared<detail::DeviceQuarantineRegistry>();
    std::uint64_t runtime_id = 0;
    std::filesystem::path calibration_directory;
    std::uint32_t calibration_budget_ms = 250;
    bool calibrate_on_start = true;
    ShutdownMode shutdown_mode = ShutdownMode::Graceful;
    std::shared_ptr<detail::PerformanceDatabase> performance_database =
        std::make_shared<detail::PerformanceDatabase>();
    std::filesystem::path performance_database_path;
    std::mutex cpu_mutex;
    std::unordered_map<std::string, CpuFallbackFunction> cpu_fallbacks;
    std::shared_ptr<CpuExecutor> cpu_executor;

    ~Impl() {
        // Drop cached target references before tearing down their Vulkan devices.
        verification_targets.reset();
        for (auto& device : devices)
            if (device && device->dispatch)
                device->dispatch->accepting_work.store(false, std::memory_order_release);
        for (auto& device : devices)
            if (device) device->shutdown(shutdown_mode);
        if (cpu_executor) (void)cpu_executor->shutdown(shutdown_mode);
        devices.clear();
    }
};

namespace {
Result<detail::TransferBuffer> ensure_transfer_buffer(const std::shared_ptr<detail::DispatchContext>& context,
                                                      const BufferHandle& handle, VkBufferUsageFlags usage);

// Copies a completed device result from a mapped allocation into the host shadow. The caller
// guarantees the producing work completed with a shader-write to host-read barrier and that no
// other queued work can access the buffer.
Status read_mapped_output(const std::shared_ptr<detail::DispatchContext>& context,
                          const detail::TransferBuffer& output, const BufferHandle& handle,
                          std::uint64_t expected_host_version) {
    auto invalidated =
        detail::mapped_memory_operation(context->device->api(), context->device->handle(), output, 0,
                                        output.size, context->non_coherent_atom_size, false);
    if (!invalidated) return invalidated;
    const auto bytes = std::span<const std::byte>(output.mapped_data, static_cast<std::size_t>(output.size));
    auto copied = handle.write_bytes_if_version(bytes, expected_host_version);
    if (!copied) return std::move(copied).error();
    context->download_count.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(context->submit_mutex);
    if (copied.value().written) context->resident_host_versions[handle.id()] = copied.value().host_version;
    context->device_newer_buffers.erase(handle.id());
    return ok_status();
}
}  // namespace

Result<Event> DispatchBuilder::submit() {
    const auto host_submit_started = std::chrono::steady_clock::now();
    std::uint64_t upload_wait_nanoseconds = 0;
    if (adaptive_state_) {
        if (verification_policy_)
            return Error(ErrorCode::UnsupportedFeature,
                         "adaptive dispatch cannot be combined with duplicate verification");
        if (buffers_.size() == accesses_.size()) {
            std::unordered_set<std::size_t> input_sources;
            for (std::size_t index = 0; index < buffers_.size(); ++index) {
                if (accesses_[index] == KernelAccess::Write) continue;
                for (const auto& candidate : adaptive_state_->contexts) {
                    std::lock_guard lock(candidate->submit_mutex);
                    if (candidate->device_newer_buffers.contains(buffers_[index].id()))
                        input_sources.insert(candidate->device_index);
                }
            }
            if (input_sources.size() > 1)
                return Error(
                    ErrorCode::UnsupportedFeature,
                    "adaptive dispatch inputs have newer copies on multiple devices; read them back first");
            if (!input_sources.empty() && *input_sources.begin() != context_->device_index) {
                if (*input_sources.begin() != adaptive_state_->original_id.index)
                    return Error(
                        ErrorCode::UnsupportedFeature,
                        "adaptive dispatch input is newer on another device; read it back before migration");
                context_ = adaptive_state_->original_context;
                kernel_ = adaptive_state_->original_kernel;
                selected_device_ = adaptive_state_->original_id;
            }
        }
    }
    const auto validation_started = context_ && context_->phase_profile_enabled
                                        ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
    if (verification_policy_) return submit_verified();
    if (!context_ || !context_->accepting_work.load(std::memory_order_acquire) || !kernel_ ||
        !context_->pipelines || !context_->queue || !context_->device)
        return Error(ErrorCode::InvalidArgument, "dispatch builder has an invalid runtime or kernel");
    if (context_->quarantined.load(std::memory_order_acquire))
        return Error(ErrorCode::ValidationError, "device is quarantined after a verification mismatch");
    if (kernel_->runtime_id != context_->runtime_id || kernel_->device_index != context_->device_index)
        return Error(ErrorCode::InvalidArgument, "kernel was not loaded for this runtime device");
    context_->reap_released_buffers();
    auto metadata_status = validate_kernel_metadata(kernel_->metadata);
    if (!metadata_status) return std::move(metadata_status).error();
    if (buffers_.size() != kernel_->metadata.bindings.size() || accesses_.size() != buffers_.size())
        return Error(ErrorCode::InvalidKernelBinding, "dispatch buffer count does not match kernel metadata");
    if (push_constants_.size() != kernel_->metadata.push_constant_size)
        return Error(ErrorCode::InvalidArgument, "push-constant byte size does not match kernel metadata");
    if (kernel_->metadata.push_constant_size > context_->info->limits().max_push_constants_size)
        return Error(ErrorCode::InvalidArgument, "push-constant block exceeds the selected device limit");

    const auto& limits = context_->info->limits();
    auto local_size = kernel_->metadata.local_size;
    detail::kernel::WorkgroupAutotuner::Trial autotune_trial;
    const bool autotune = context_->autotuning_enabled && context_->timestamps && kernel_->autotuner;
    if (autotune) {
        autotune_trial = kernel_->autotuner->reserve();
        if (autotune_trial.tracked)
            local_size = kernel_->autotuner->candidates()[autotune_trial.candidate_index].local_size;
        else
            local_size = kernel_->autotuner->selected_local_size();
    }
    auto cancel_autotune_trial = [&] {
        if (autotune_trial.tracked) kernel_->autotuner->cancel(autotune_trial);
        autotune_trial.tracked = false;
    };
    struct AutotuneTrialGuard {
        std::function<void()> cancel;
        bool active = true;
        ~AutotuneTrialGuard() {
            if (active) cancel();
        }
    } trial_guard{cancel_autotune_trial};
    std::array<std::uint32_t, 3> groups{};
    for (std::size_t dimension = 0; dimension < 3; ++dimension) {
        if (global_size_[dimension] == 0)
            return Error(ErrorCode::InvalidDispatch, "global dispatch dimensions must all be nonzero");
        const auto local = local_size[dimension];
        const auto group_count = 1U + ((global_size_[dimension] - 1U) / local);
        if (group_count > UINT32_MAX || group_count > limits.max_compute_work_group_count[dimension])
            return Error(ErrorCode::InvalidDispatch, "dispatch group count exceeds a selected-device limit");
        groups[dimension] = static_cast<std::uint32_t>(group_count);
        if (local > limits.max_compute_work_group_size[dimension])
            return Error(ErrorCode::InvalidDispatch,
                         "kernel local size exceeds a selected-device dimension limit");
    }
    std::uint64_t invocations = 1;
    for (const auto dimension : local_size) {
        if (dimension > UINT32_MAX / invocations)
            return Error(ErrorCode::InvalidDispatch, "kernel local-size invocation product overflows uint32");
        invocations *= dimension;
    }
    if (invocations > limits.max_compute_work_group_invocations)
        return Error(ErrorCode::InvalidDispatch, "kernel local size exceeds maxComputeWorkGroupInvocations");

    const auto validation_elapsed = context_->phase_profile_enabled
                                        ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - validation_started)
                                              .count()
                                        : 0;

    const auto binding_preflight_started = context_->phase_profile_enabled
                                               ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
    std::vector<detail::TransferBuffer> vk_buffers;
    vk_buffers.reserve(buffers_.size());
    for (std::size_t i = 0; i < buffers_.size(); ++i) {
        if (!buffers_[i])
            return Error(ErrorCode::InvalidKernelBinding, "dispatch contains an empty buffer handle");
        if (accesses_[i] != kernel_->metadata.bindings[i].access)
            return Error(ErrorCode::InvalidKernelBinding,
                         "dispatch access mode does not match kernel metadata");
        if (kernel_->metadata.bindings[i].kind != KernelBindingKind::StorageBuffer)
            return Error(ErrorCode::UnsupportedFeature,
                         "runtime dispatch currently supports storage-buffer bindings only");
    }
    {
        std::lock_guard lock(context_->submit_mutex);
        for (const auto& buffer : buffers_) {
            if (context_->access_tick == std::numeric_limits<std::uint64_t>::max())
                return Error(ErrorCode::Overflow, "runtime buffer access-history tick overflow");
            auto recorded = context_->eviction_policy.record_access(buffer.id(), ++context_->access_tick);
            if (!recorded) return std::move(recorded).error();
        }
    }
    auto specialization = kernel_->metadata.specialization_constants;
    if (autotune && autotune_trial.tracked) {
        for (std::size_t dimension = 0; dimension < local_size.size(); ++dimension) {
            const auto found = std::find_if(
                specialization.begin(), specialization.end(),
                [&](const auto& value) { return value.id == kernel_->local_size_spec_ids[dimension]; });
            if (found == specialization.end()) {
                cancel_autotune_trial();
                return Error(ErrorCode::InternalInvariantViolation,
                             "reflected LocalSizeId specialization is missing from kernel metadata");
            }
            found->default_value = static_cast<std::uint64_t>(local_size[dimension]);
        }
    }
    const auto binding_preflight_elapsed =
        context_->phase_profile_enabled ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - binding_preflight_started)
                                              .count()
                                        : 0;
    const auto pipeline_lookup_started = context_->phase_profile_enabled
                                             ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
    std::optional<detail::kernel::PipelineManager::PipelineRef> cached_pipeline;
    if (!kernel_->local_size_id_based)
        cached_pipeline = context_->pipelines->find_cached(kernel_->words, kernel_->metadata, specialization,
                                                           kernel_->entry_point);
    detail::kernel::PipelineManager::PipelineRef pipeline_ref;
    if (cached_pipeline) {
        pipeline_ref = std::move(*cached_pipeline);
    } else {
        auto pipeline = context_->pipelines->get_or_create(kernel_->words, kernel_->metadata, specialization,
                                                           kernel_->entry_point);
        if (!pipeline) {
            cancel_autotune_trial();
            return std::move(pipeline).error();
        }
        pipeline_ref = std::move(pipeline).value();
    }
    const auto pipeline_lookup_elapsed = context_->phase_profile_enabled
                                             ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() - pipeline_lookup_started)
                                                   .count()
                                             : 0;
    const auto buffer_residency_started = context_->phase_profile_enabled
                                              ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
    for (std::size_t i = 0; i < buffers_.size(); ++i) {
        auto buffer = ensure_transfer_buffer(context_, buffers_[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        if (!buffer) {
            cancel_autotune_trial();
            return std::move(buffer).error();
        }
        vk_buffers.push_back(buffer.value());
    }

    // Reuse an existing current device copy. Host writes advance host_version(), so a changed
    // host shadow automatically forces the next consumer to upload again.
    std::uint64_t transfer_wait_value = 0;
    std::vector<std::uint64_t> bound_host_versions(buffers_.size(), 0);
    for (std::size_t i = 0; i < buffers_.size(); ++i) {
        auto version = buffers_[i].host_version();
        if (!version) return std::move(version).error();
        bound_host_versions[i] = version.value();
        bool device_copy_current = false;
        {
            std::lock_guard lock(context_->submit_mutex);
            const auto found = context_->resident_host_versions.find(buffers_[i].id());
            device_copy_current =
                found != context_->resident_host_versions.end() && found->second == version.value();
        }
        if (accesses_[i] == KernelAccess::Write || device_copy_current) continue;

        // A mapped buffer with no queued or executing GPU use can take the host bytes directly;
        // vkQueueSubmit makes host writes made before it visible to the device. Anything else
        // keeps the staged upload, which is ordered behind earlier work on the queue.
        const auto& target = vk_buffers[i];
        const bool direct = context_->direct_host_access && target.host_visible &&
                            target.mapped_data != nullptr && target.queue_use &&
                            target.queue_use->pending.load(std::memory_order_acquire) == 0;
        const auto transfer_started = std::chrono::steady_clock::now();
        std::optional<Event> uploaded;
        Status copied = Error(ErrorCode::StaleHandle, "host buffer changed while preparing an upload");
        for (int attempt = 0; attempt < 3 && !copied; ++attempt) {
            version = buffers_[i].host_version();
            if (!version) return std::move(version).error();
            {
                auto shadow = buffers_[i].map(MapAccess::Read);
                if (!shadow) return std::move(shadow).error();
                const auto bytes = std::span<const std::byte>(shadow.value().data(), shadow.value().size());
                if (direct) {
                    std::memcpy(target.mapped_data, bytes.data(), bytes.size());
                } else {
                    auto staged = context_->transfers->upload(target, 0, bytes);
                    if (!staged) return std::move(staged).error();
                    uploaded = std::move(staged).value();
                }
            }
            const auto after = buffers_[i].host_version();
            if (!after) return std::move(after).error();
            if (uploaded) {
                // The staging copy already captured the bytes; wait before deciding to retry.
                auto upload_status = uploaded->wait();
                if (!upload_status) return std::move(upload_status).error();
                uploaded.reset();
            }
            if (after.value() == version.value()) {
                bound_host_versions[i] = version.value();
                copied = ok_status();
            }
        }
        if (!copied) return std::move(copied).error();
        if (direct) {
            auto flushed =
                detail::mapped_memory_operation(context_->device->api(), context_->device->handle(), target,
                                                0, target.size, context_->non_coherent_atom_size, true);
            if (!flushed) return std::move(flushed).error();
        }
        const auto transfer_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - transfer_started)
                                          .count();
        if (transfer_elapsed > 0 && static_cast<std::uint64_t>(transfer_elapsed) <=
                                        std::numeric_limits<std::uint64_t>::max() - upload_wait_nanoseconds)
            upload_wait_nanoseconds += static_cast<std::uint64_t>(transfer_elapsed);
        context_->upload_count.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard lock(context_->submit_mutex);
            context_->resident_host_versions[buffers_[i].id()] = bound_host_versions[i];
            context_->device_newer_buffers.erase(buffers_[i].id());
        }
        if (context_->transfer_queue != context_->queue && vk_buffers[i].queue_use) {
            transfer_wait_value = std::max(
                transfer_wait_value, vk_buffers[i].queue_use->transfer_value.load(std::memory_order_acquire));
        }
    }

    const auto buffer_residency_elapsed =
        context_->phase_profile_enabled ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - buffer_residency_started)
                                              .count()
                                        : 0;
    const auto submission_assembly_started = context_->phase_profile_enabled
                                                 ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point{};
    auto queue = context_->queue;
    auto ctx = context_;
    auto bound_buffers = std::make_shared<std::vector<detail::TransferBuffer>>(std::move(vk_buffers));
    auto bound_handles = buffers_;
    std::vector<BufferId> written_buffers;
    for (std::size_t index = 0; index < accesses_.size(); ++index) {
        if (accesses_[index] != KernelAccess::Read) written_buffers.push_back(buffers_[index].id());
    }
    auto push_constants = push_constants_;
    auto groups_copy = groups;
    auto promise = std::make_shared<detail::EventPromise>();
    auto record_status = std::make_shared<Status>();
    auto event = promise->event();
    detail::PendingPerformanceObservation observation;
    if (validation_elapsed > 0)
        observation.dispatch_validation_nanoseconds = static_cast<std::uint64_t>(validation_elapsed);
    if (binding_preflight_elapsed > 0)
        observation.binding_preflight_nanoseconds = static_cast<std::uint64_t>(binding_preflight_elapsed);
    if (pipeline_lookup_elapsed > 0)
        observation.pipeline_lookup_nanoseconds = static_cast<std::uint64_t>(pipeline_lookup_elapsed);
    if (buffer_residency_elapsed > 0)
        observation.buffer_residency_nanoseconds = static_cast<std::uint64_t>(buffer_residency_elapsed);
    if (adaptive_state_ && !ctx->device_uuid.empty()) {
        detail::PerformanceObservationKey observation_identity{
            .device_uuid = ctx->device_uuid,
            .driver_version = ctx->driver_version,
            .runtime_schema = static_cast<std::uint32_t>(kora::kSchemaVersion),
            .kernel_identity = kernel_->semantic_identity,
            .workload_class = static_cast<std::uint8_t>(adaptive_state_->workload.kernel_kind),
            .workload_bucket = workload_bucket(adaptive_state_->workload.kernel_work),
            .timing_domain = detail::PerformanceTimingDomain::DeviceExecution,
        };
        const auto encode_domain = [&](detail::PerformanceTimingDomain domain) {
            observation_identity.timing_domain = domain;
            auto encoded = detail::performance_observation_key(observation_identity);
            return encoded ? std::move(encoded).value() : std::string{};
        };
        observation.execution_key = encode_domain(detail::PerformanceTimingDomain::DeviceExecution);
        observation.submission_key = encode_domain(detail::PerformanceTimingDomain::HostSubmission);
        observation.transfer_key = encode_domain(detail::PerformanceTimingDomain::HostTransfer);
        observation.queue_completion_key = encode_domain(detail::PerformanceTimingDomain::QueueCompletion);
    }
    struct DispatchWork {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        std::uint64_t submission_id = 0;
        bool timestamp_recorded = false;
        detail::kernel::WorkgroupAutotuner::Trial autotune_trial;
        detail::PendingPerformanceObservation observation;
        std::optional<detail::TraceSchedulerScore> trace_score;
        std::chrono::steady_clock::time_point queued_at;
    };
    auto work = std::make_shared<DispatchWork>();
    work->autotune_trial = autotune_trial;
    work->observation = std::move(observation);
    if (ctx->trace_enabled && adaptive_state_ && adaptive_state_->selected_score)
        work->trace_score = detail::TraceSchedulerScore{.selected_device_index = ctx->device_index,
                                                        .score = *adaptive_state_->selected_score};
    auto allocated_descriptors = std::make_shared<std::vector<VkDescriptorSet>>();
    vk::QueueContext::Submission submission;
    submission.coalescible = true;
    submission.dependency_free = true;
    submission.coalescing_key =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pipeline_ref.get()));
    submission.priority = static_cast<std::uint8_t>(priority_);
    submission.resources.reserve(buffers_.size());
    for (std::size_t index = 0; index < buffers_.size(); ++index) {
        const auto id = buffers_[index].id();
        submission.resources.push_back({id.slot, id.generation, accesses_[index] != KernelAccess::Read});
    }
    if (transfer_wait_value != 0) {
        const auto semaphore = context_->transfer_queue->completion_semaphore();
        if (semaphore == VK_NULL_HANDLE)
            return Error(ErrorCode::UnsupportedFeature,
                         "Cross-family runtime buffers require a transfer queue timeline semaphore");
        submission.wait_semaphores.push_back(semaphore);
        submission.wait_stages.push_back(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        submission.wait_values.push_back(transfer_wait_value);
    }
    // Only a dispatch whose outputs are read back pays for the host-read barrier; it can stall
    // the queue on some drivers, so deferred dispatches leave outputs to an explicit readback.
    const bool host_read_barrier = context_->direct_host_access && !defer_readback_;
    for (std::size_t index = 0; index < accesses_.size(); ++index)
        if (accesses_[index] != KernelAccess::Read && (*bound_buffers)[index].queue_use)
            (*bound_buffers)[index].queue_use->host_readable.store(host_read_barrier,
                                                                   std::memory_order_release);
    submission.record_commands = [ctx, pipeline_ref, bound_buffers, push_constants, groups_copy, work,
                                  record_status, allocated_descriptors, host_read_barrier](
                                     std::uint64_t submission_id, std::vector<VkCommandBuffer>& commands) {
        std::lock_guard lock(ctx->submit_mutex);
        const auto command_acquire_started = ctx->phase_profile_enabled
                                                 ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point{};
        if (!ctx->command_pool) {
            vk::CommandPoolSet::Options options;
            options.queue_family = ctx->device->compute_queue_family();
            options.ring_size = 32;
            ctx->command_pool =
                std::make_unique<vk::CommandPoolSet>(ctx->device->api(), ctx->device->handle(), options);
        }
        auto completed = ctx->queue->completed_value();
        if (!completed) return Status(std::move(completed).error());
        auto command = ctx->command_pool->acquire(completed.value());
        if (!command) return Status(std::move(command).error());
        if (ctx->phase_profile_enabled)
            work->observation.command_buffer_acquire_nanoseconds =
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - command_acquire_started)
                                               .count());
        work->command_buffer = command.value();
        work->submission_id = submission_id;
        auto api = ctx->device->api();
        const bool timestamp_needed = work->autotune_trial.tracked ||
                                      !work->observation.execution_key.empty() ||
                                      (ctx->timestamp_sample_interval != 0 &&
                                       ctx->timestamp_sample_counter++ % ctx->timestamp_sample_interval == 0);
        if (ctx->timestamps && timestamp_needed)
            work->timestamp_recorded = ctx->timestamps->record_start(command.value(), submission_id);
        const auto descriptor_setup_started = ctx->phase_profile_enabled
                                                  ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point{};
        std::vector<VkDescriptorSet> descriptor_sets;
        for (std::uint32_t set = 0; set < pipeline_ref->descriptor_set_layouts.size(); ++set) {
            auto descriptor = ctx->pipelines->allocate_descriptor_set(pipeline_ref, set);
            if (!descriptor) {
                *record_status = Status(std::move(descriptor).error());
                (void)ctx->command_pool->discard(command.value());
                work->command_buffer = VK_NULL_HANDLE;
                return *record_status;
            }
            descriptor_sets.push_back(descriptor.value());
            allocated_descriptors->push_back(descriptor.value());
        }
        std::vector<VkDescriptorBufferInfo> infos;
        std::vector<VkWriteDescriptorSet> writes;
        infos.resize(bound_buffers->size());
        writes.reserve(bound_buffers->size());
        for (std::size_t index = 0; index < bound_buffers->size(); ++index) {
            infos[index] = {(*bound_buffers)[index].buffer, 0, (*bound_buffers)[index].size};
            const auto& binding = pipeline_ref->bindings[index];
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                              descriptor_sets.at(binding.set), binding.binding, 0, 1,
                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[index], nullptr});
        }
        api.update_descriptor_sets(ctx->device->handle(), static_cast<std::uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
        if (ctx->phase_profile_enabled)
            work->observation.descriptor_setup_nanoseconds =
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - descriptor_setup_started)
                                               .count());
        const auto command_recording_started = ctx->phase_profile_enabled
                                                   ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
        api.cmd_bind_pipeline(command.value(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_ref->handle);
        if (!descriptor_sets.empty())
            api.cmd_bind_descriptor_sets(
                command.value(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_ref->layout, 0,
                static_cast<std::uint32_t>(descriptor_sets.size()), descriptor_sets.data(), 0, nullptr);
        if (!push_constants.empty())
            api.cmd_push_constants(command.value(), pipeline_ref->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   static_cast<std::uint32_t>(push_constants.size()), push_constants.data());
        api.cmd_dispatch(command.value(), groups_copy[0], groups_copy[1], groups_copy[2]);
        if (host_read_barrier) {
            // Make shader writes available to host reads of the mapping once the queue completes.
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                                    VK_ACCESS_HOST_READ_BIT};
            api.cmd_pipeline_barrier(command.value(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        if (ctx->timestamps && work->timestamp_recorded)
            ctx->timestamps->record_end(command.value(), submission_id);
        auto ended = ctx->command_pool->end_recording(command.value());
        if (!ended) return ended;
        if (ctx->phase_profile_enabled)
            work->observation.command_recording_nanoseconds =
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - command_recording_started)
                                               .count());
        commands.push_back(command.value());
        return Status{};
    };
    submission.on_submitted =
        [ctx, work, bound_handles = std::move(bound_handles), written_buffers, tuner = kernel_->autotuner,
         key = autotune_trial.tracked ? kernel_->autotune_keys[autotune_trial.candidate_index]
                                      : std::string{}](std::uint64_t timeline, Status status) {
            std::lock_guard lock(ctx->submit_mutex);
            if (status) {
                if (work->command_buffer != VK_NULL_HANDLE)
                    (void)ctx->command_pool->submitted(work->command_buffer, timeline);
                if (ctx->timestamps && work->timestamp_recorded)
                    ctx->timestamps->submitted(work->submission_id, timeline);
                if (tuner && work->autotune_trial.tracked && work->timestamp_recorded && ctx->timestamps)
                    ctx->pending_autotune.insert_or_assign(
                        work->submission_id,
                        detail::DispatchContext::PendingAutotune{tuner, work->autotune_trial, key});
                if (work->timestamp_recorded && !work->observation.execution_key.empty())
                    ctx->pending_observations.insert_or_assign(work->submission_id, work->observation);
                if (ctx->trace_enabled && work->timestamp_recorded && work->trace_score) {
                    ctx->pending_trace_scores.insert_or_assign(work->submission_id, *work->trace_score);
                    ctx->pending_trace_score_order.push_back(work->submission_id);
                    constexpr std::size_t kTraceScoreHistoryLimit = 4096;
                    while (ctx->pending_trace_score_order.size() > kTraceScoreHistoryLimit) {
                        ctx->pending_trace_scores.erase(ctx->pending_trace_score_order.front());
                        ctx->pending_trace_score_order.pop_front();
                    }
                }
                for (const auto& buffer : bound_handles) {
                    ctx->device_last_use[buffer.id()] = timeline;
                }
            } else if (work->command_buffer != VK_NULL_HANDLE && ctx->command_pool) {
                (void)ctx->command_pool->discard(work->command_buffer);
            }
            if (!status && ctx->timestamps && work->timestamp_recorded)
                ctx->timestamps->cancel(work->submission_id);
            if ((!status || !work->timestamp_recorded) && tuner && work->autotune_trial.tracked)
                tuner->cancel(work->autotune_trial);
            if (!status) {
                for (const auto id : written_buffers) {
                    ctx->resident_host_versions.erase(id);
                    ctx->device_newer_buffers.erase(id);
                }
            }
        };
    if (ctx->phase_profile_enabled) {
        submission.on_completion_profile = [work](std::uint64_t worker_delay, std::uint64_t device_wait) {
            work->observation.completion_worker_delay_nanoseconds = worker_delay;
            work->observation.device_completion_wait_nanoseconds = device_wait;
        };
    }
    std::vector<std::shared_ptr<detail::QueueTimelineUse>> buffer_uses;
    buffer_uses.reserve(bound_buffers->size());
    for (const auto& buffer : *bound_buffers) buffer_uses.push_back(buffer.queue_use);
    auto pending_uses = std::make_shared<detail::PendingUses>(std::move(buffer_uses));
    submission.on_complete = [promise, ctx, work, pipeline_ref, allocated_descriptors, bound_buffers,
                              bound_handles = buffers_, accesses = accesses_, written_buffers,
                              bound_host_versions, pending_uses,
                              defer_readback = defer_readback_](std::uint64_t compute_value, Status status) {
        const auto completion_callback_started = ctx->phase_profile_enabled
                                                     ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point{};
        (void)pipeline_ref;
        if (!status && status.error().code == ErrorCode::DeviceLost && compute_value != 0) {
            for (std::size_t index = 0; index < accesses.size(); ++index)
                if (accesses[index] != KernelAccess::Read) (void)bound_handles[index].mark_data_lost();
        }
        if ((ctx->phase_profile_enabled || ctx->timestamps) && status) {
            const auto completion_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                std::chrono::steady_clock::now() - work->queued_at)
                                                .count();
            if (completion_elapsed > 0) {
                std::lock_guard lock(ctx->submit_mutex);
                work->observation.completion_elapsed_nanoseconds =
                    static_cast<std::uint64_t>(completion_elapsed);
                if (ctx->phase_profile_enabled) {
                    ctx->latest_profile_submission_id = work->submission_id;
                    ctx->latest_profile_observation = work->observation;
                }
                const auto pending = ctx->pending_observations.find(work->submission_id);
                if (pending != ctx->pending_observations.end())
                    pending->second.completion_elapsed_nanoseconds =
                        static_cast<std::uint64_t>(completion_elapsed);
            }
        }
        if (ctx->timestamps && status) {
            const auto timestamp_collection_started = ctx->phase_profile_enabled
                                                          ? std::chrono::steady_clock::now()
                                                          : std::chrono::steady_clock::time_point{};
            auto samples = ctx->timestamps->collect(compute_value);
            if (ctx->phase_profile_enabled) {
                const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - timestamp_collection_started)
                                         .count();
                if (elapsed > 0)
                    work->observation.timestamp_collection_nanoseconds = static_cast<std::uint64_t>(elapsed);
            }
            if (samples)
                detail::observe_autotune_samples(ctx, samples.value(), ctx->performance_database,
                                                 ctx->performance_database_path);
            else {
                // Query failures have no valid measurement to consume these entries. Remove them
                // just as we do for failed submissions so repeated failures cannot grow the maps.
                std::lock_guard lock(ctx->submit_mutex);
                ctx->pending_observations.erase(work->submission_id);
                ctx->pending_trace_scores.erase(work->submission_id);
                const auto pending = ctx->pending_autotune.find(work->submission_id);
                if (pending != ctx->pending_autotune.end()) {
                    pending->second.tuner->cancel(pending->second.trial);
                    ctx->pending_autotune.erase(pending);
                }
            }
        } else if (ctx->timestamps) {
            std::lock_guard lock(ctx->submit_mutex);
            ctx->pending_observations.erase(work->submission_id);
            ctx->pending_trace_scores.erase(work->submission_id);
            const auto pending = ctx->pending_autotune.find(work->submission_id);
            if (pending != ctx->pending_autotune.end()) {
                pending->second.tuner->cancel(pending->second.trial);
                ctx->pending_autotune.erase(pending);
            }
        }
        const auto publish_completion_callback_profile = [&] {
            if (!ctx->phase_profile_enabled) return;
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - completion_callback_started)
                                     .count();
            if (elapsed > 0)
                work->observation.completion_callback_nanoseconds = static_cast<std::uint64_t>(elapsed);
            std::lock_guard lock(ctx->submit_mutex);
            if (ctx->latest_profile_submission_id == work->submission_id)
                ctx->latest_profile_observation = work->observation;
        };
        for (const auto descriptor : *allocated_descriptors)
            (void)ctx->pipelines->free_descriptor_set(descriptor);
        if (status && !defer_readback) {
            auto readback_contents = std::make_shared<std::vector<std::vector<std::byte>>>(accesses.size());
            std::vector<Event> readbacks;
            std::vector<std::size_t> output_indices;
            for (std::size_t index = 0; index < accesses.size(); ++index) {
                if (accesses[index] == KernelAccess::Read) continue;
                const auto& output = (*bound_buffers)[index];
                // This submission still counts as a pending use, so other queued work is the only
                // thing that can make a direct copy unsafe.
                if (ctx->direct_host_access && output.host_visible && output.mapped_data != nullptr &&
                    output.queue_use && output.queue_use->pending.load(std::memory_order_acquire) == 1) {
                    auto copied =
                        read_mapped_output(ctx, output, bound_handles[index], bound_host_versions[index]);
                    if (!copied) {
                        status = std::move(copied);
                        break;
                    }
                    continue;
                }
                auto& contents = (*readback_contents)[index];
                contents.resize(static_cast<std::size_t>((*bound_buffers)[index].size));
                detail::QueueWait dependency;
                if (ctx->transfer_queue != ctx->queue) {
                    dependency.semaphore = ctx->queue->completion_semaphore();
                    dependency.value = compute_value;
                    dependency.stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                }
                auto downloaded = ctx->transfers->download((*bound_buffers)[index], 0, contents, dependency);
                if (!downloaded) {
                    status = Status(std::move(downloaded).error());
                    break;
                }
                ctx->download_count.fetch_add(1, std::memory_order_relaxed);
                readbacks.push_back(std::move(downloaded).value());
                output_indices.push_back(index);
            }
            if (status && !readbacks.empty()) {
                auto aggregate = Event::when_all(readbacks);
                if (!aggregate) {
                    publish_completion_callback_profile();
                    promise->complete(Status(std::move(aggregate).error()));
                    return;
                }
                (void)aggregate.value().on_complete([promise, readback_contents, bound_handles,
                                                     output_indices, bound_host_versions,
                                                     ctx](Status readback_status) {
                    if (readback_status) {
                        for (const auto index : output_indices) {
                            const auto copied = bound_handles[index].write_bytes_if_version(
                                (*readback_contents)[index], bound_host_versions[index]);
                            if (!copied) {
                                readback_status = Status(std::move(copied).error());
                                break;
                            }
                            if (!copied.value().written) {
                                std::lock_guard lock(ctx->submit_mutex);
                                ctx->device_newer_buffers.erase(bound_handles[index].id());
                                continue;
                            }
                            std::lock_guard lock(ctx->submit_mutex);
                            ctx->resident_host_versions[bound_handles[index].id()] =
                                copied.value().host_version;
                            ctx->device_newer_buffers.erase(bound_handles[index].id());
                        }
                    }
                    promise->complete(std::move(readback_status));
                    return ok_status();
                });
                return;
            }
        }
        if (!status) {
            std::lock_guard lock(ctx->submit_mutex);
            for (const auto id : written_buffers) {
                ctx->resident_host_versions.erase(id);
                ctx->device_newer_buffers.erase(id);
            }
        }
        publish_completion_callback_profile();
        promise->complete(std::move(status));
    };
    {
        std::lock_guard lock(context_->submit_mutex);
        for (std::size_t index = 0; index < accesses_.size(); ++index) {
            if (accesses_[index] == KernelAccess::Read) continue;
            const auto id = buffers_[index].id();
            context_->resident_host_versions[id] = bound_host_versions[index];
            context_->device_newer_buffers.insert(id);
        }
    }
    const auto submission_assembly_elapsed =
        context_->phase_profile_enabled ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - submission_assembly_started)
                                              .count()
                                        : 0;
    if (submission_assembly_elapsed > 0)
        work->observation.submission_assembly_nanoseconds =
            static_cast<std::uint64_t>(submission_assembly_elapsed);
    trial_guard.active = false;  // Submission callbacks now own cancellation/observation.
    const auto before_enqueue = std::chrono::steady_clock::now();
    const auto host_submit_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(before_enqueue - host_submit_started).count();
    work->observation.transfer_nanoseconds = upload_wait_nanoseconds;
    work->observation.submission_nanoseconds =
        host_submit_elapsed > 0 && static_cast<std::uint64_t>(host_submit_elapsed) > upload_wait_nanoseconds
            ? static_cast<std::uint64_t>(host_submit_elapsed) - upload_wait_nanoseconds
            : 0;
    work->queued_at = before_enqueue;
    const auto queue_enqueue_started = ctx->phase_profile_enabled ? std::chrono::steady_clock::now()
                                                                  : std::chrono::steady_clock::time_point{};
    auto submitted = queue->submit(std::move(submission));
    if (ctx->phase_profile_enabled) {
        const auto queue_enqueue_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - queue_enqueue_started)
                                               .count();
        if (queue_enqueue_elapsed > 0) {
            std::lock_guard lock(ctx->submit_mutex);
            work->observation.queue_enqueue_nanoseconds = static_cast<std::uint64_t>(queue_enqueue_elapsed);
            if (ctx->latest_profile_submission_id == work->submission_id)
                ctx->latest_profile_observation.queue_enqueue_nanoseconds =
                    work->observation.queue_enqueue_nanoseconds;
        }
    }
    if (!submitted) {
        cancel_autotune_trial();
        std::lock_guard lock(context_->submit_mutex);
        for (const auto id : written_buffers) {
            context_->resident_host_versions.erase(id);
            context_->device_newer_buffers.erase(id);
        }
        if (!*record_status) return std::move(*record_status).error();
        return std::move(submitted).error();
    }
    return event;
}

namespace {
Result<detail::TransferBuffer> ensure_transfer_buffer(const std::shared_ptr<detail::DispatchContext>& context,
                                                      const BufferHandle& handle, VkBufferUsageFlags usage) {
    if (!context || !handle) return Error(ErrorCode::InvalidArgument, "runtime or buffer handle is invalid");
    std::lock_guard lock(context->submit_mutex);
    const auto found = context->buffers.find(handle.id());
    if (found != context->buffers.end()) {
        if (found->second.size != handle.size_bytes())
            return Error(ErrorCode::StaleHandle, "buffer handle generation changed after Vulkan allocation");
        return found->second;
    }
    const auto size = handle.size_bytes();
    if (size == 0 || size > context->info->limits().max_storage_buffer_range)
        return Error(ErrorCode::InvalidArgument, "buffer size is zero or exceeds maxStorageBufferRange");
    if (context->buffer_state->refresh_budget) {
        auto refreshed = context->buffer_state->refresh_budget();
        if (!refreshed) return std::move(refreshed).error();
    }

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const std::array queue_families{context->queue_family_policy.compute_family,
                                    context->queue_family_policy.transfer_family};
    if (context->queue_family_policy.uses_concurrent_sharing()) {
        buffer_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        buffer_info.queueFamilyIndexCount = static_cast<std::uint32_t>(queue_families.size());
        buffer_info.pQueueFamilyIndices = queue_families.data();
    } else {
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    VkBuffer buffer = VK_NULL_HANDLE;
    auto result =
        context->device->api().create_buffer(context->device->handle(), &buffer_info, nullptr, &buffer);
    if (result != VK_SUCCESS)
        return Error(ErrorCode::AllocationFailed, "vkCreateBuffer failed for runtime buffer").with_vk(result);
    VkMemoryRequirements requirements{};
    context->device->api().get_buffer_memory_requirements(context->device->handle(), buffer, &requirements);
    auto memory_type = context->device->buffer_memory_info().memory_properties.memoryTypeCount == 0
                           ? Result<std::uint32_t>(Error(ErrorCode::UnsupportedFeature,
                                                         "physical-device memory properties are unavailable"))
                           : detail::MemoryManager::select_memory_type(
                                 context->info->memory_types(), context->info->memory_type_preferences(),
                                 detail::MemoryPolicy::Auto, requirements.memoryTypeBits);
    if (!memory_type) {
        context->device->api().destroy_buffer(context->device->handle(), buffer, nullptr);
        return std::move(memory_type).error();
    }
    auto pool = std::find_if(context->buffer_state->pools.begin(), context->buffer_state->pools.end(),
                             [&](auto* item) { return item->memory_type_index() == memory_type.value(); });
    if (pool == context->buffer_state->pools.end()) {
        context->device->api().destroy_buffer(context->device->handle(), buffer, nullptr);
        return Error(ErrorCode::UnsupportedFeature, "selected Vulkan buffer memory type has no runtime pool");
    }
    auto allocation = (*pool)->allocate({requirements.size, requirements.alignment, false, false});
    if (!allocation) {
        context->device->api().destroy_buffer(context->device->handle(), buffer, nullptr);
        return std::move(allocation).error();
    }
    result = context->device->api().bind_buffer_memory(context->device->handle(), buffer,
                                                       allocation.value().memory, allocation.value().offset);
    if (result != VK_SUCCESS) {
        (void)(*pool)->deallocate(allocation.value());
        context->device->api().destroy_buffer(context->device->handle(), buffer, nullptr);
        return Error(ErrorCode::AllocationFailed, "vkBindBufferMemory failed for runtime buffer")
            .with_vk(result);
    }
    const auto& memory_type_info = context->info->memory_types()[memory_type.value()];
    detail::TransferBuffer transfer_buffer{
        .buffer = buffer,
        .size = size,
        .memory = allocation.value().memory,
        .memory_offset = allocation.value().offset,
        .memory_allocation_size =
            allocation.value().dedicated ? allocation.value().allocated_size : (*pool)->block_size(),
        .mapped_data = static_cast<std::byte*>(allocation.value().mapped_data),
        .host_visible = memory_type_info.host_visible,
        .host_coherent = memory_type_info.host_coherent,
        .queue_use = std::make_shared<detail::QueueTimelineUse>()};
    context->buffers.emplace(handle.id(), transfer_buffer);
    context->allocations.emplace(handle.id(), allocation.value());
    context->allocation_pools.emplace(handle.id(), *pool);
    return transfer_buffer;
}
}  // namespace

Result<Event> DispatchBuilder::submit_verified() {
    if (!verification_policy_ || !verification_targets_ || !context_ || !kernel_)
        return Error(ErrorCode::InvalidArgument, "verified dispatch has no policy or runtime targets");
    if (verification_targets_->targets.size() < 2)
        return Error(ErrorCode::UnsupportedFeature,
                     "duplicate verification requires at least two initialized, non-quarantined devices");
    if (buffers_.size() != accesses_.size() || buffers_.size() != kernel_->metadata.bindings.size())
        return Error(ErrorCode::InvalidKernelBinding, "dispatch buffer count does not match kernel metadata");
    auto policy_valid = verification_policy_->validate();
    if (!policy_valid) return std::move(policy_valid).error();

    std::vector<std::size_t> output_indices;
    std::unordered_set<BufferId, detail::BufferIdHash> unique_outputs;
    std::vector<std::uint64_t> host_versions(buffers_.size());
    for (std::size_t index = 0; index < buffers_.size(); ++index) {
        if (accesses_[index] != KernelAccess::Read && unique_outputs.insert(buffers_[index].id()).second)
            output_indices.push_back(index);
    }
    if (output_indices.empty())
        return Error(ErrorCode::InvalidArgument, "verified dispatch requires at least one writable output");
    for (std::size_t index = 0; index < buffers_.size(); ++index) {
        if (!buffers_[index]) return Error(ErrorCode::InvalidKernelBinding, "dispatch has an empty buffer");
        if (accesses_[index] != KernelAccess::Read) {
            const auto element_size =
                verification_type_ == VerificationDataType::Float32 ? sizeof(float) : sizeof(double);
            if (buffers_[index].size_bytes() % element_size != 0 &&
                verification_policy_->mode != VerificationMode::Custom)
                return Error(ErrorCode::InvalidArgument,
                             "verified output size is not a multiple of the selected floating-point type");
        }
        auto version = buffers_[index].host_version();
        if (!version) return std::move(version).error();
        host_versions[index] = version.value();
    }

    std::vector<Event> dispatch_events;
    std::vector<detail::VerificationTargetSet::Target> targets;
    std::vector<std::shared_ptr<const detail::KernelState>> kernels;
    dispatch_events.reserve(verification_targets_->targets.size());
    targets.reserve(verification_targets_->targets.size());
    kernels.reserve(verification_targets_->targets.size());
    for (const auto& target : verification_targets_->targets) {
        if (!target.context || target.context->quarantined.load(std::memory_order_acquire)) continue;
        if (!target.context->accepting_work.load(std::memory_order_acquire))
            return Error(ErrorCode::Cancelled, "a verification target is shutting down");
        auto reflection = detail::spirv::validate_and_reflect(
            kernel_->words, kernel_->metadata, detail::spirv::capabilities_from(*target.context->info),
            kernel_->entry_point);
        if (!reflection) return std::move(reflection).error();
        auto target_kernel = std::make_shared<detail::KernelState>();
        target_kernel->entry_point = kernel_->entry_point;
        target_kernel->metadata = kernel_->metadata;
        target_kernel->words = kernel_->words;
        target_kernel->runtime_id = kernel_->runtime_id;
        target_kernel->device_index = target.id.index;
        targets.push_back(target);
        kernels.push_back(std::move(target_kernel));
    }
    if (targets.size() < 2)
        return Error(ErrorCode::UnsupportedFeature,
                     "fewer than two verification targets support this kernel");

    std::optional<Error> submission_error;
    for (std::size_t index = 0; index < targets.size(); ++index) {
        const auto& target = targets[index];
        const auto& target_kernel = kernels[index];
        DispatchBuilder duplicate(target.context, target_kernel, target.id);
        duplicate.global_size_ = global_size_;
        duplicate.buffers_ = buffers_;
        duplicate.accesses_ = accesses_;
        duplicate.push_constants_ = push_constants_;
        duplicate.defer_readback_ = true;
        duplicate.priority_ = priority_;
        auto submitted = duplicate.submit();
        if (!submitted) {
            submission_error = std::move(submitted).error();
            break;
        }
        dispatch_events.push_back(std::move(submitted).value());
    }
    if (dispatch_events.empty()) return std::move(*submission_error);

    auto dispatches_complete = Event::when_all(dispatch_events);
    if (!dispatches_complete) return std::move(dispatches_complete).error();
    auto promise = std::make_shared<detail::EventPromise>();
    const auto verified_event = promise->event();
    auto registered = dispatches_complete.value().on_complete(
        [promise, targets = std::move(targets), kernels = std::move(kernels), buffers = buffers_,
         output_indices = std::move(output_indices), host_versions = std::move(host_versions),
         policy = *verification_policy_, type = verification_type_,
         submission_error = std::move(submission_error),
         registry = verification_targets_->quarantine](Status dispatch_status) mutable {
            (void)kernels;
            const auto mark_outputs_lost = [&] {
                for (const auto index : output_indices) (void)buffers[index].mark_data_lost();
            };
            if (!dispatch_status) {
                mark_outputs_lost();
                promise->complete(std::move(dispatch_status));
                return;
            }
            if (submission_error) {
                mark_outputs_lost();
                promise->complete(std::move(*submission_error));
                return;
            }
            auto snapshots = std::make_shared<std::vector<std::vector<std::vector<std::byte>>>>(
                targets.size(), std::vector<std::vector<std::byte>>(output_indices.size()));
            std::vector<Event> downloads;
            downloads.reserve(targets.size() * output_indices.size());
            for (std::size_t target_index = 0; target_index < targets.size(); ++target_index) {
                const auto& context = targets[target_index].context;
                for (std::size_t output_index = 0; output_index < output_indices.size(); ++output_index) {
                    const auto buffer_index = output_indices[output_index];
                    const auto& handle = buffers[buffer_index];
                    auto device_buffer =
                        ensure_transfer_buffer(context, handle, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                    if (!device_buffer) {
                        mark_outputs_lost();
                        promise->complete(Status(std::move(device_buffer).error()));
                        return;
                    }
                    auto& bytes = (*snapshots)[target_index][output_index];
                    bytes.resize(static_cast<std::size_t>(device_buffer.value().size));
                    detail::QueueWait dependency;
                    if (context->transfer_queue != context->queue) {
                        dependency.semaphore = context->queue->completion_semaphore();
                        {
                            std::lock_guard lock(context->submit_mutex);
                            const auto last_use = context->device_last_use.find(handle.id());
                            dependency.value =
                                last_use == context->device_last_use.end() ? 0 : last_use->second;
                        }
                        dependency.stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                    }
                    auto downloaded =
                        context->transfers->download(device_buffer.value(), 0, bytes, dependency);
                    if (!downloaded) {
                        mark_outputs_lost();
                        promise->complete(Status(std::move(downloaded).error()));
                        return;
                    }
                    context->download_count.fetch_add(1, std::memory_order_relaxed);
                    downloads.push_back(std::move(downloaded).value());
                }
            }
            auto all_downloads = Event::when_all(downloads);
            if (!all_downloads) {
                mark_outputs_lost();
                promise->complete(Status(std::move(all_downloads).error()));
                return;
            }
            (void)all_downloads.value().on_complete(
                [promise, targets = std::move(targets), buffers = std::move(buffers), snapshots,
                 output_indices = std::move(output_indices), host_versions = std::move(host_versions),
                 policy = std::move(policy), type,
                 registry = std::move(registry)](Status download_status) mutable {
                    const auto mark_outputs_lost = [&] {
                        for (const auto index : output_indices) (void)buffers[index].mark_data_lost();
                    };
                    if (!download_status) {
                        mark_outputs_lost();
                        promise->complete(std::move(download_status));
                        return;
                    }
                    for (std::size_t output = 0; output < output_indices.size(); ++output) {
                        const auto& reference = (*snapshots)[0][output];
                        for (std::size_t target = 1; target < targets.size(); ++target) {
                            auto equal = policy.compare_bytes(reference, (*snapshots)[target][output], type);
                            if (!equal) {
                                mark_outputs_lost();
                                promise->complete(Status(std::move(equal).error()));
                                return;
                            }
                            if (equal.value()) continue;
                            mark_outputs_lost();
                            for (const auto& participant : targets) {
                                participant.context->quarantined.store(true, std::memory_order_release);
                                if (registry) registry->add(participant.id);
                            }
                            promise->complete(
                                Error(ErrorCode::ValidationError,
                                      "duplicate dispatch outputs differ; all participants quarantined"));
                            return;
                        }
                    }
                    for (std::size_t output = 0; output < output_indices.size(); ++output) {
                        const auto index = output_indices[output];
                        auto copied = buffers[index].write_bytes_if_version((*snapshots)[0][output],
                                                                            host_versions[index]);
                        if (!copied) {
                            mark_outputs_lost();
                            promise->complete(Status(std::move(copied).error()));
                            return;
                        }
                        if (!copied.value().written) {
                            promise->complete(Error(ErrorCode::StaleHandle,
                                                    "host buffer changed during verified dispatch"));
                            return;
                        }
                        for (const auto& target : targets) {
                            std::lock_guard lock(target.context->submit_mutex);
                            target.context->resident_host_versions[buffers[index].id()] =
                                copied.value().host_version;
                            target.context->device_newer_buffers.erase(buffers[index].id());
                        }
                    }
                    promise->complete(ok_status());
                });
        });
    if (!registered) promise->complete(Status(std::move(registered).error()));
    return verified_event;
}

Runtime::Runtime() = default;
namespace {
// Lets each device context free its copy of a buffer once the buffer's last handle is released.
template <class DeviceEntries>
void register_buffer_release(BufferManager& buffers, const DeviceEntries& devices) {
    for (const auto& device : devices) {
        if (!device || !device->dispatch) continue;
        std::weak_ptr<detail::DispatchContext> context = device->dispatch;
        detail::BufferManagerAccess::add_release_listener(buffers, context, [context](BufferId id) {
            if (auto locked = context.lock()) locked->queue_released_buffer(id);
        });
    }
}
}  // namespace

Runtime::Runtime(std::unique_ptr<Impl> impl, RuntimeOptions options)
    : impl_(std::move(impl)), options_(std::move(options)) {
    if (impl_) register_buffer_release(buffers_, impl_->devices);
}
Runtime::~Runtime() noexcept = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

Result<std::span<const std::byte>> CpuTaskContext::read(std::size_t index) const {
    if (index >= buffers_.size() || index >= accesses_.size() || index >= mappings_.size())
        return Error(ErrorCode::InvalidArgument, "CPU task buffer index is out of range");
    if (accesses_[index] == KernelAccess::Write)
        return Error(ErrorCode::InvalidKernelBinding,
                     "CPU task requested read access to a write-only buffer");
    return std::span<const std::byte>(mappings_[index].data(), mappings_[index].size());
}

Result<std::span<std::byte>> CpuTaskContext::write(std::size_t index) {
    if (index >= buffers_.size() || index >= accesses_.size() || index >= mappings_.size())
        return Error(ErrorCode::InvalidArgument, "CPU task buffer index is out of range");
    if (accesses_[index] == KernelAccess::Read)
        return Error(ErrorCode::InvalidKernelBinding,
                     "CPU task requested write access to a read-only buffer");
    return mappings_[index].bytes();
}

Result<std::span<std::byte>> CpuTaskContext::read_write(std::size_t index) {
    if (index >= buffers_.size() || index >= accesses_.size() || index >= mappings_.size())
        return Error(ErrorCode::InvalidArgument, "CPU task buffer index is out of range");
    if (accesses_[index] != KernelAccess::ReadWrite)
        return Error(ErrorCode::InvalidKernelBinding,
                     "CPU task requested read_write on a differently declared buffer");
    return mappings_[index].bytes();
}

const KernelMetadata& Kernel::metadata() const noexcept {
    static const KernelMetadata empty_metadata;
    return state_ ? state_->metadata : empty_metadata;
}

const std::string& Kernel::entry_point() const noexcept {
    static const std::string empty_entry_point;
    return state_ ? state_->entry_point : empty_entry_point;
}

Result<BufferHandle> Runtime::create_buffer(std::uint64_t size_bytes, BufferUsage usage) {
    if (!impl_ || impl_->devices.empty())
        return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    return buffers_.create(size_bytes, usage);
}

DispatchBuilder::DispatchBuilder(std::shared_ptr<detail::DispatchContext> context,
                                 std::shared_ptr<const detail::KernelState> kernel,
                                 std::optional<DeviceId> selected_device)
    : context_(std::move(context)),
      kernel_(std::move(kernel)),
      selected_device_(std::move(selected_device)) {}

DispatchBuilder& DispatchBuilder::global(std::uint64_t x) {
    return global(x, 1, 1);
}

DispatchBuilder& DispatchBuilder::global(std::uint64_t x, std::uint64_t y, std::uint64_t z) {
    global_size_ = {x, y, z};
    return *this;
}

DispatchBuilder& DispatchBuilder::read(const BufferHandle& buffer) {
    buffers_.push_back(buffer);
    accesses_.push_back(KernelAccess::Read);
    return *this;
}

DispatchBuilder& DispatchBuilder::write(const BufferHandle& buffer) {
    buffers_.push_back(buffer);
    accesses_.push_back(KernelAccess::Write);
    return *this;
}

DispatchBuilder& DispatchBuilder::read_write(const BufferHandle& buffer) {
    buffers_.push_back(buffer);
    accesses_.push_back(KernelAccess::ReadWrite);
    return *this;
}

DispatchBuilder& DispatchBuilder::defer_readback() {
    defer_readback_ = true;
    return *this;
}

DispatchBuilder& DispatchBuilder::priority(DispatchPriority priority) {
    priority_ = priority;
    return *this;
}

DispatchBuilder& DispatchBuilder::verify_with(VerificationPolicy policy, VerificationDataType type) {
    verification_policy_ = std::move(policy);
    verification_type_ = type;
    return *this;
}

DispatchBuilder& DispatchBuilder::push(std::span<const std::byte> bytes) {
    push_constants_.assign(bytes.begin(), bytes.end());
    return *this;
}

Result<Kernel> Runtime::load_kernel(const std::filesystem::path& spirv_path, std::string entry_point,
                                    KernelMetadata metadata) {
    if (!impl_ || impl_->devices.empty())
        return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    return load_kernel(spirv_path, std::move(entry_point), std::move(metadata), impl_->devices.front()->id);
}

Result<Kernel> Runtime::load_kernel(const std::filesystem::path& spirv_path, std::string entry_point,
                                    KernelMetadata metadata, const DeviceId& device_id) {
    if (!impl_ || impl_->devices.empty())
        return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    const auto selected_device = std::find_if(impl_->devices.begin(), impl_->devices.end(),
                                              [&](const auto& device) { return device->id == device_id; });
    if (selected_device == impl_->devices.end())
        return Error(ErrorCode::InvalidArgument, "device is not owned by runtime");
    auto metadata_status = validate_kernel_metadata(metadata);
    if (!metadata_status) return std::move(metadata_status).error();
    std::ifstream input(spirv_path, std::ios::binary | std::ios::ate);
    if (!input) return Error(ErrorCode::InvalidKernel, "cannot open SPIR-V module: " + spirv_path.string());
    const auto end = input.tellg();
    if (end <= 0 || static_cast<std::uint64_t>(end) > 16U * 1024U * 1024U)
        return Error(ErrorCode::InvalidKernel, "SPIR-V module is empty or exceeds 16 MiB");
    if (static_cast<std::uint64_t>(end) % sizeof(std::uint32_t) != 0)
        return Error(ErrorCode::InvalidKernel, "SPIR-V module byte count is not divisible by four");
    std::vector<std::uint32_t> words(static_cast<std::size_t>(end) / sizeof(std::uint32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(end));
    if (!input) return Error(ErrorCode::IoError, "failed to read complete SPIR-V module");
    auto structure = detail::spirv::validate_module_structure(words);
    if (!structure) return std::move(structure).error();
    auto& device = **selected_device;
    auto reflection = detail::spirv::validate_and_reflect(
        words, metadata, detail::spirv::capabilities_from(device.info), entry_point);
    if (!reflection) return std::move(reflection).error();
    auto state = std::make_shared<detail::KernelState>();
    state->entry_point = std::move(entry_point);
    state->metadata = std::move(metadata);
    state->words = std::move(words);
    state->semantic_identity = semantic_kernel_identity(*state);
    state->runtime_id = impl_->runtime_id;
    state->device_index = device.id.index;
    if (reflection.value().local_size_id_based) {
        state->local_size_id_based = true;
        state->local_size_spec_ids = reflection.value().local_size_spec_ids;
        auto candidates = detail::kernel::generate_workgroup_candidates(
            reflection.value().local_size, true, device.info.limits(), device.info.subgroup());
        if (!candidates) return std::move(candidates).error();
        const bool x_dimension_is_independent =
            reflection.value().local_size_spec_ids[0] != reflection.value().local_size_spec_ids[1] &&
            reflection.value().local_size_spec_ids[0] != reflection.value().local_size_spec_ids[2];
        if (candidates.value().size() > 1 && x_dimension_is_independent) {
            std::size_t base_index = 0;
            for (std::size_t index = 0; index < candidates.value().size(); ++index) {
                if (candidates.value()[index].local_size == reflection.value().local_size) base_index = index;
            }
            state->autotuner = std::make_shared<detail::kernel::WorkgroupAutotuner>(
                std::move(candidates).value(), base_index);
            const auto device_key = device.id.uuid.empty()
                                        ? device.info.name() + "/" + std::to_string(device.id.index)
                                        : device.id.uuid;
            std::uint64_t module_hash = 1469598103934665603ULL;
            for (const auto word : state->words) {
                module_hash ^= word;
                module_hash *= 1099511628211ULL;
            }
            const auto tuning_kernel_id = state->metadata.kernel_id + "/" + std::to_string(module_hash);
            for (const auto& candidate : state->autotuner->candidates()) {
                auto key = detail::kernel::workgroup_performance_key(
                    device_key, device.driver_version, tuning_kernel_id, state->entry_point, candidate);
                auto saved = impl_->performance_database->query(key);
                if (saved) {
                    auto restored = state->autotuner->restore(state->autotune_keys.size(), saved.value());
                    if (!restored) return std::move(restored).error();
                }
                state->autotune_keys.push_back(std::move(key));
            }
        }
    }
    return Kernel(std::move(state));
}

DispatchBuilder Runtime::dispatch(const Kernel& kernel) const {
    if (!impl_ || !kernel.state_ || kernel.state_->runtime_id != impl_->runtime_id) return {};
    const auto selected_device =
        std::find_if(impl_->devices.begin(), impl_->devices.end(),
                     [&](const auto& device) { return device->id.index == kernel.state_->device_index; });
    DispatchBuilder builder(
        selected_device != impl_->devices.end() ? (*selected_device)->dispatch : nullptr, kernel.state_,
        selected_device != impl_->devices.end() ? std::optional<DeviceId>((*selected_device)->id)
                                                : std::nullopt);
    if (selected_device != impl_->devices.end()) builder.verification_targets_ = impl_->verification_targets;
    return builder;
}

Result<DispatchBuilder> Runtime::dispatch_adaptive(const Kernel& kernel,
                                                   const AdaptiveDispatchWorkload& workload) const {
    if (!impl_ || !kernel.state_ || kernel.state_->runtime_id != impl_->runtime_id)
        return Error(ErrorCode::InvalidArgument, "adaptive dispatch kernel does not belong to this runtime");
    if (workload.kernel_work == 0 || static_cast<std::uint8_t>(workload.kernel_kind) >
                                         static_cast<std::uint8_t>(AdaptiveKernelWorkKind::AluOperations))
        return Error(ErrorCode::InvalidArgument, "adaptive dispatch workload is invalid");
    const auto current = std::find_if(impl_->devices.begin(), impl_->devices.end(), [&](const auto& device) {
        return device->id.index == kernel.state_->device_index;
    });
    if (current == impl_->devices.end())
        return Error(ErrorCode::InvalidArgument,
                     "adaptive dispatch kernel device is not owned by this runtime");

    std::vector<detail::RuntimePlacementCandidate> candidates;
    candidates.reserve(impl_->devices.size());
    const auto wall_now = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
    const auto observation_now = wall_now < 0 ? 0 : static_cast<std::uint64_t>(wall_now);
    for (const auto& device : impl_->devices) {
        if (device->dispatch->quarantined.load(std::memory_order_acquire) ||
            !device->dispatch->accepting_work.load(std::memory_order_acquire))
            continue;
        auto reflection = detail::spirv::validate_and_reflect(kernel.state_->words, kernel.state_->metadata,
                                                              detail::spirv::capabilities_from(device->info),
                                                              kernel.state_->entry_point);
        if (!reflection || (kernel.state_->autotuner && device.get() != current->get())) continue;
        auto stats = memory_stats(device->id);
        if (!stats) {
            if (device.get() == current->get()) return dispatch(kernel);
            continue;
        }
        std::uint64_t available_device_memory = 0;
        for (const auto& heap : stats.value().heaps) {
            if (heap.heap_index >= device->info.memory_heaps().size() ||
                !device->info.memory_heaps()[heap.heap_index].device_local)
                continue;
            available_device_memory = std::max(available_device_memory, heap.available_bytes);
        }
        candidates.push_back(detail::RuntimePlacementCandidate{
            .id = device->id,
            .driver_version = device->driver_version,
            .calibration_record = device->calibration_record,
            .online_execution =
                query_execution_observation(impl_->performance_database, device->id, device->driver_version,
                                            kernel.state_->semantic_identity, workload, observation_now),
            .available_device_memory_bytes = available_device_memory,
            .supported = true,
            .healthy = true,
            .allowed = true,
        });
    }
    auto selected = detail::select_adaptive_device(candidates, (*current)->id, workload);
    if (!selected) return std::move(selected).error();
    const auto target = std::find_if(impl_->devices.begin(), impl_->devices.end(), [&](const auto& device) {
        return device->id == selected.value().device;
    });
    if (target == impl_->devices.end())
        return Error(ErrorCode::InternalInvariantViolation, "adaptive scheduler selected an unknown device");
    std::shared_ptr<const detail::KernelState> selected_kernel = kernel.state_;
    if (target->get() != current->get()) {
        auto migrated_kernel = std::make_shared<detail::KernelState>(*kernel.state_);
        migrated_kernel->device_index = (*target)->id.index;
        migrated_kernel->autotuner.reset();
        migrated_kernel->autotune_keys.clear();
        selected_kernel = std::move(migrated_kernel);
    }
    DispatchBuilder builder((*target)->dispatch, std::move(selected_kernel), (*target)->id);
    builder.adaptive_state_ = std::make_shared<detail::AdaptiveDispatchState>();
    builder.adaptive_state_->original_id = (*current)->id;
    builder.adaptive_state_->original_context = (*current)->dispatch;
    builder.adaptive_state_->original_kernel = kernel.state_;
    builder.adaptive_state_->workload = workload;
    builder.adaptive_state_->selected_score = selected.value().score;
    builder.adaptive_state_->contexts.reserve(impl_->devices.size());
    for (const auto& device : impl_->devices) builder.adaptive_state_->contexts.push_back(device->dispatch);
    return builder;
}

Result<Runtime> Runtime::create(const RuntimeOptions& options) {
    const auto resolved = resolve(options);
    if (resolved.debug_safety) log::set_category_level(log::Category::Sync, log::Level::Trace);
    auto loader_result = vk::Loader::load();
    if (!loader_result) return std::move(loader_result).error();

    auto instance_options = vk::Loader::InstanceOptions{};
    instance_options.enable_validation = resolved.enable_validation || resolved.debug_safety;
    auto instance_result = loader_result.value().create_instance(instance_options);
    if (!instance_result) return std::move(instance_result).error();

    auto physical_devices = instance_result.value().physical_devices();
    if (!physical_devices) return std::move(physical_devices).error();

    std::vector<DeviceCandidate> candidates;
    candidates.reserve(physical_devices.value().size());
    for (std::size_t index = 0; index < physical_devices.value().size(); ++index) {
        const auto& physical = physical_devices.value()[index];
        const auto info = detail::DeviceCaps::from_vulkan(physical).to_device_info();
        const auto compute_queues = std::accumulate(
            physical.queue_family_properties.begin(), physical.queue_family_properties.end(),
            std::uint32_t{0}, [](std::uint32_t total, const VkQueueFamilyProperties& family) {
                return (family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 ? total + family.queueCount : total;
            });
        std::uint64_t local_memory = 0;
        for (const auto& heap : info.memory_heaps()) {
            if (heap.device_local) local_memory += heap.size_bytes;
        }
        candidates.push_back(DeviceCandidate{
            .id = DeviceId{static_cast<std::uint32_t>(index), uuid_string(physical.id_properties)},
            .kind = device_kind(physical.properties.deviceType),
            .tier = info.tier(),
            .device_local_memory_bytes = local_memory,
            .compute_queue_count = compute_queues,
            .satisfies_requirements = true,
        });
    }

    auto selected = select_devices(resolved, candidates);
    if (!selected) return std::move(selected).error();

    auto impl = std::make_unique<Impl>(std::move(loader_result).value(), std::move(instance_result).value());
    impl->calibration_directory = resolved.cache_dir;
    impl->calibration_budget_ms = resolved.calibration_budget_ms;
    impl->shutdown_mode = resolved.shutdown_mode;
    impl->calibrate_on_start = resolved.enable_startup_calibration;
    impl->performance_database_path = std::filesystem::path(resolved.cache_dir) / "performance-db.json";
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    if (now_ms >= 0) {
        (void)impl->performance_database->load(impl->performance_database_path,
                                               static_cast<std::uint64_t>(now_ms));
    }
    const bool isolate_device_failures = resolved.device_policy == DevicePolicy::AllEligible;
    const auto initialize_device = [&](const DeviceId& id) -> Status {
        if (id.index >= physical_devices.value().size()) {
            return Error(ErrorCode::InternalInvariantViolation,
                         "device selector returned an index outside the Vulkan enumeration");
        }
        const auto& physical = physical_devices.value()[id.index];
        vk::DeviceOptions device_options;
        device_options.enable_timeline_semaphore = !resolved.force_legacy_sync;
        device_options.enable_synchronization2 = !resolved.force_legacy_sync;
        const bool core_12 = physical.properties.apiVersion >= VK_API_VERSION_1_2;
        const bool timeline_available =
            core_12 ? physical.vulkan12_features.timelineSemaphore == VK_TRUE
                    : physical.timeline_semaphore_features.timelineSemaphore == VK_TRUE;
        device_options.enable_dedicated_transfer_queue = !resolved.force_legacy_sync && timeline_available;
        auto logical_device = impl->instance.create_device(physical, device_options);
        if (!logical_device) return std::move(logical_device).error();
        auto info = detail::DeviceCaps::from_vulkan(physical).to_device_info();
        auto memory_manager = detail::MemoryManager::create(physical, info, resolved.memory_budget_fraction);
        if (!memory_manager) return std::move(memory_manager).error();
        detail::kernel::PipelineCacheIdentity pipeline_identity;
        pipeline_identity.vendor_id = physical.properties.vendorID;
        pipeline_identity.device_id = physical.properties.deviceID;
        pipeline_identity.driver_version = physical.properties.driverVersion;
        std::copy(std::begin(physical.id_properties.deviceUUID), std::end(physical.id_properties.deviceUUID),
                  pipeline_identity.device_uuid.begin());
        std::copy(std::begin(physical.id_properties.driverUUID), std::end(physical.id_properties.driverUUID),
                  pipeline_identity.driver_uuid.begin());
        const auto cache_key = id.uuid.empty() ? std::to_string(pipeline_identity.vendor_id) + "-" +
                                                     std::to_string(pipeline_identity.device_id) + "-" +
                                                     std::to_string(pipeline_identity.driver_version)
                                               : id.uuid;
        auto pipeline_manager = detail::kernel::PipelineManager::create(
            logical_device.value().api(), logical_device.value().handle(),
            detail::spirv::capabilities_from(info), pipeline_identity,
            std::filesystem::path(resolved.cache_dir) / "pipelines" / (cache_key + ".bin"));
        if (!pipeline_manager) return std::move(pipeline_manager).error();
        auto entry = std::make_unique<Impl::DeviceEntry>(
            id, std::move(info), std::move(memory_manager).value(), std::move(logical_device).value(),
            std::move(pipeline_manager).value(), impl->runtime_id, resolved.force_legacy_sync,
            physical.calibrated_time_domains, resolved.shutdown_mode);
        entry->configure_budget_refresh(impl->instance, resolved.memory_budget_fraction);
        entry->driver_version = physical.properties.driverVersion;
        auto initialized = entry->initialize();
        if (!initialized) return std::move(initialized).error();
        entry->dispatch->trace_enabled = resolved.telemetry == TelemetryLevel::Trace;
        entry->dispatch->phase_profile_enabled =
            resolved.telemetry == TelemetryLevel::Profile || resolved.telemetry == TelemetryLevel::Trace;
        entry->dispatch->timestamp_sample_interval = timestamp_sample_interval(resolved.telemetry);
        if (entry->dispatch->timestamps)
            entry->dispatch->timestamps->enable_trace_history(entry->dispatch->trace_enabled);
        entry->dispatch->device_uuid = entry->id.uuid;
        entry->dispatch->driver_version = entry->driver_version;
        if (impl->calibrate_on_start)
            entry->calibrate(impl->calibration_directory, impl->calibration_budget_ms);
        entry->dispatch->autotuning_enabled = !resolved.disable_autotune;
        entry->dispatch->performance_database = impl->performance_database;
        entry->dispatch->performance_database_path = impl->performance_database_path;
        impl->devices.push_back(std::move(entry));
        return {};
    };
    auto device_initialization = detail::initialize_selected_devices(
        selected.value(), isolate_device_failures, initialize_device,
        [&](const DeviceId& id) { impl->quarantine->add(id); },
        [](const DeviceId& id, const Error& error) {
            KORA_LOG(Recovery, Warn, "Quarantined device {} during AllEligible initialization: {}", id.index,
                     error.describe());
        });
    if (!device_initialization) return std::move(device_initialization).error();
    auto verification_targets = std::make_shared<detail::VerificationTargetSet>();
    verification_targets->quarantine = impl->quarantine;
    verification_targets->targets.reserve(impl->devices.size());
    for (const auto& device : impl->devices)
        verification_targets->targets.push_back({device->id, device->dispatch});
    impl->verification_targets = std::move(verification_targets);
    RuntimeOptions effective_options;
    effective_options.enable_validation = resolved.enable_validation;
    effective_options.debug_safety = resolved.debug_safety;
    effective_options.device_policy = resolved.device_policy;
    effective_options.device_index = resolved.device_index;
    effective_options.device_uuid = resolved.device_uuid;
    effective_options.memory_budget_fraction = resolved.memory_budget_fraction;
    effective_options.telemetry = resolved.telemetry;
    effective_options.log_level = resolved.log_level;
    effective_options.cache_dir = resolved.cache_dir;
    effective_options.deterministic = resolved.deterministic;
    effective_options.shutdown_mode = resolved.shutdown_mode;
    effective_options.disable_autotune = resolved.disable_autotune;
    effective_options.disable_multi_queue = resolved.disable_multi_queue;
    effective_options.disable_cpu_backend = resolved.disable_cpu_backend;
    effective_options.force_legacy_sync = resolved.force_legacy_sync;
    effective_options.trace = options.trace;
    effective_options.enable_startup_calibration = resolved.enable_startup_calibration;
    effective_options.calibration_budget_ms = resolved.calibration_budget_ms;
    Runtime runtime(std::move(impl), std::move(effective_options));
    if (resolved.enable_startup_calibration) runtime.calibrate_compute_probes();
    return runtime;
}

Status Runtime::recover() {
    if (!impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    const bool device_lost = std::any_of(impl_->devices.begin(), impl_->devices.end(), [](const auto& entry) {
        return entry->fault_state && !entry->fault_state->healthy();
    });
    if (!device_lost)
        return Error(ErrorCode::InvalidArgument, "runtime recovery requires an observed device loss");

    auto created = Runtime::create(options_);
    if (!created) return Status(std::move(created).error());
    auto replacement = std::move(created).value();
    auto new_quarantines = replacement.impl_->quarantine->snapshot();
    for (const auto& id : new_quarantines) impl_->quarantine->add(id);
    replacement.impl_->quarantine = impl_->quarantine;
    auto replacement_targets =
        std::make_shared<detail::VerificationTargetSet>(*replacement.impl_->verification_targets);
    replacement_targets->quarantine = impl_->quarantine;
    replacement.impl_->verification_targets = std::move(replacement_targets);
    for (const auto& id : replacement.impl_->quarantine->snapshot()) {
        const auto quarantined =
            std::find_if(replacement.impl_->devices.begin(), replacement.impl_->devices.end(),
                         [&](const auto& entry) { return entry->id == id; });
        if (quarantined != replacement.impl_->devices.end())
            (*quarantined)->dispatch->quarantined.store(true, std::memory_order_release);
    }
    {
        std::scoped_lock lock(impl_->cpu_mutex, replacement.impl_->cpu_mutex);
        replacement.impl_->cpu_fallbacks = impl_->cpu_fallbacks;
    }
    replacement.buffers_ = std::move(buffers_);
    impl_.swap(replacement.impl_);
    buffers_ = std::move(replacement.buffers_);
    register_buffer_release(buffers_, impl_->devices);
    options_ = std::move(replacement.options_);
    // `replacement` now owns the failed implementation; its destructor drains old queue callbacks
    // before this method returns. Host-backed BufferHandles remain backed by the retained state.
    return {};
}

Status detail::RuntimeTestAccess::mark_device_lost(Runtime& runtime, const DeviceId& id) {
    if (!runtime.impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    const auto found = std::find_if(runtime.impl_->devices.begin(), runtime.impl_->devices.end(),
                                    [&](const auto& entry) { return entry->id == id; });
    if (found == runtime.impl_->devices.end())
        return Error(ErrorCode::InvalidArgument, "device is not owned by runtime");
    (*found)->fault_state->mark_lost(VK_ERROR_DEVICE_LOST, false);
    return {};
}

Status detail::RuntimeTestAccess::set_calibration_record(Runtime& runtime, const DeviceId& id,
                                                         detail::calibration::Record record) {
    if (!runtime.impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    const auto found = std::find_if(runtime.impl_->devices.begin(), runtime.impl_->devices.end(),
                                    [&](const auto& entry) { return entry->id == id; });
    if (found == runtime.impl_->devices.end())
        return Error(ErrorCode::InvalidArgument, "device is not owned by runtime");
    record.device_uuid = (*found)->id.uuid;
    record.driver_version = (*found)->driver_version;
    record.kora_schema = static_cast<std::uint32_t>(kora::kSchemaVersion);
    (*found)->calibration_record = std::move(record);
    return {};
}

Result<std::string> detail::RuntimeTestAccess::performance_observation_key(
    const Runtime& runtime, const DeviceId& id, const Kernel& kernel,
    const AdaptiveDispatchWorkload& workload, detail::PerformanceTimingDomain domain) {
    if (!runtime.impl_ || !kernel.state_ || kernel.state_->runtime_id != runtime.impl_->runtime_id)
        return Error(ErrorCode::InvalidArgument, "kernel does not belong to the initialized runtime");
    const auto found = std::find_if(runtime.impl_->devices.begin(), runtime.impl_->devices.end(),
                                    [&](const auto& device) { return device->id == id; });
    if (found == runtime.impl_->devices.end())
        return Error(ErrorCode::InvalidArgument, "device is not owned by runtime");
    return detail::performance_observation_key(detail::PerformanceObservationKey{
        .device_uuid = id.uuid,
        .driver_version = (*found)->driver_version,
        .runtime_schema = static_cast<std::uint32_t>(kora::kSchemaVersion),
        .kernel_identity = kernel.state_->semantic_identity,
        .workload_class = static_cast<std::uint8_t>(workload.kernel_kind),
        .workload_bucket = workload_bucket(workload.kernel_work),
        .timing_domain = domain,
    });
}

Result<detail::PerformanceStats> detail::RuntimeTestAccess::performance_stats(const Runtime& runtime,
                                                                              const std::string& key) {
    if (!runtime.impl_ || !runtime.impl_->performance_database)
        return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    return runtime.impl_->performance_database->query(key);
}

std::vector<DeviceId> Runtime::devices() const {
    std::vector<DeviceId> ids;
    if (!impl_) return ids;
    ids.reserve(impl_->devices.size());
    for (const auto& device : impl_->devices) ids.push_back(device->id);
    return ids;
}

std::vector<DeviceId> Runtime::quarantined_devices() const {
    if (!impl_) return {};
    return impl_->quarantine->snapshot();
}

Result<DeviceInfo> Runtime::device_info(const DeviceId& id) const {
    if (!impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    const auto found = std::find_if(impl_->devices.begin(), impl_->devices.end(),
                                    [&](const auto& device) { return device->id == id; });
    if (found == impl_->devices.end())
        return Error(ErrorCode::InvalidArgument, "device is not owned by runtime");
    return (*found)->info;
}

Result<MemoryStats> Runtime::memory_stats(const DeviceId& id) const {
    if (!impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    const auto found = std::find_if(impl_->devices.begin(), impl_->devices.end(),
                                    [&](const auto& device) { return device->id == id; });
    if (found == impl_->devices.end())
        return Error(ErrorCode::InvalidArgument, "device is not owned by runtime");
    const auto& entry = **found;
    if (entry.dispatch) entry.dispatch->reap_released_buffers();
    auto refreshed = entry.refresh_budget_if_due(true);
    if (!refreshed) return std::move(refreshed).error();
    std::lock_guard lock(entry.pools_mutex);
    std::vector<detail::MemoryPoolStats> pool_stats;
    pool_stats.reserve(entry.pools.size());
    for (const auto& pool : entry.pools) pool_stats.push_back(pool->stats());
    auto aggregate = detail::MemoryManager::aggregate_stats(pool_stats);
    if (!aggregate) return std::move(aggregate).error();
    std::lock_guard budget_lock(entry.budget_mutex);
    for (std::uint32_t index = 0; index < entry.info.memory_heaps().size(); ++index) {
        auto budget = entry.memory_manager.budget(index);
        if (!budget) return std::move(budget).error();
        const auto& value = budget.value();
        aggregate.value().heaps.push_back(MemoryHeapStats{index, value.limit_bytes, value.reserved_bytes,
                                                          value.available_bytes, value.externally_used_bytes,
                                                          value.queried});
    }
    return std::move(aggregate).value();
}

Result<RuntimeTransferStats> Runtime::transfer_stats() const {
    if (!impl_ || impl_->devices.empty())
        return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    RuntimeTransferStats stats;
    for (const auto& device : impl_->devices) {
        const auto& context = device->dispatch;
        const auto uploads = context->upload_count.load(std::memory_order_relaxed);
        const auto downloads = context->download_count.load(std::memory_order_relaxed);
        if (uploads > std::numeric_limits<std::uint64_t>::max() - stats.uploads ||
            downloads > std::numeric_limits<std::uint64_t>::max() - stats.downloads)
            return Error(ErrorCode::Overflow, "runtime transfer counter aggregation overflowed");
        stats.uploads += uploads;
        stats.downloads += downloads;
    }
    return stats;
}

Result<RuntimeProfilingStats> Runtime::profiling_stats(const DeviceId& id) const {
    if (!impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    const auto found = std::find_if(impl_->devices.begin(), impl_->devices.end(),
                                    [&](const auto& device) { return device->id == id; });
    if (found == impl_->devices.end())
        return Error(ErrorCode::InvalidArgument, "device is not owned by runtime");
    const auto& dispatch = (*found)->dispatch;
    if (!dispatch || (!dispatch->timestamps && !dispatch->phase_profile_enabled))
        return Error(ErrorCode::UnsupportedFeature, "GPU timestamp profiling is unavailable for this device");
    RuntimeProfilingStats stats;
    if (dispatch->timestamps) {
        const auto completed = dispatch->queue->completed_value();
        if (!completed) return std::move(completed).error();
        // Diagnostics may wait briefly for results a driver publishes after the queue signal.
        auto collected = dispatch->timestamps->collect(completed.value(), std::chrono::milliseconds(2));
        if (!collected) return std::move(collected).error();
        detail::observe_autotune_samples(dispatch, collected.value(), impl_->performance_database,
                                         impl_->performance_database_path);
        const auto latest = dispatch->timestamps->latest_sample();
        stats.samples_collected = dispatch->timestamps->samples_collected();
        stats.dropped_samples = dispatch->timestamps->dropped_samples();
        stats.last_submission_id = latest.submission_id;
        stats.last_duration_nanoseconds = latest.duration_nanoseconds;
        stats.last_host_timestamp_nanoseconds = latest.host_timestamp_nanoseconds;
        stats.last_calibration_deviation_nanoseconds = latest.calibration_deviation_nanoseconds;
    }
    {
        std::lock_guard lock(dispatch->submit_mutex);
        stats.last_profile_submission_id = dispatch->latest_profile_submission_id;
        if (dispatch->latest_profile_submission_id != 0) {
            const auto& observation = dispatch->latest_profile_observation;
            stats.last_dispatch_validation_nanoseconds = observation.dispatch_validation_nanoseconds;
            stats.last_binding_preflight_nanoseconds = observation.binding_preflight_nanoseconds;
            stats.last_pipeline_lookup_nanoseconds = observation.pipeline_lookup_nanoseconds;
            stats.last_buffer_residency_nanoseconds = observation.buffer_residency_nanoseconds;
            stats.last_submission_assembly_nanoseconds = observation.submission_assembly_nanoseconds;
            stats.last_host_submission_nanoseconds = observation.submission_nanoseconds;
            stats.last_queue_enqueue_nanoseconds = observation.queue_enqueue_nanoseconds;
            stats.last_command_buffer_acquire_nanoseconds = observation.command_buffer_acquire_nanoseconds;
            stats.last_descriptor_setup_nanoseconds = observation.descriptor_setup_nanoseconds;
            stats.last_command_recording_nanoseconds = observation.command_recording_nanoseconds;
            stats.last_queue_completion_nanoseconds = observation.completion_elapsed_nanoseconds;
            stats.last_completion_worker_delay_nanoseconds = observation.completion_worker_delay_nanoseconds;
            stats.last_device_completion_wait_nanoseconds = observation.device_completion_wait_nanoseconds;
            stats.last_timestamp_collection_nanoseconds = observation.timestamp_collection_nanoseconds;
            stats.last_completion_callback_nanoseconds = observation.completion_callback_nanoseconds;
        }
    }
    return stats;
}

Status Runtime::calibrate() {
    if (!impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    for (auto& device : impl_->devices)
        if (device) device->calibrate(impl_->calibration_directory, impl_->calibration_budget_ms, true);
    calibrate_compute_probes();
    return ok_status();
}

void Runtime::calibrate_compute_probes() noexcept {
    if (!impl_) return;
    for (auto& device_ptr : impl_->devices) {
        if (!device_ptr || !device_ptr->compute_calibration_pending) continue;
        auto& device = *device_ptr;
        try {
            if (std::chrono::steady_clock::now() >= device.calibration_deadline) continue;
            KernelMetadata metadata;
            metadata.kernel_id = "kora.internal.calibration-probe.v1";
            metadata.bindings = {
                {0, 0, "input_values", KernelBindingKind::StorageBuffer, KernelAccess::Read},
                {0, 1, "output_values", KernelBindingKind::StorageBuffer, KernelAccess::Write},
            };
            metadata.push_constant_size = 16;
            metadata.push_constants = {{"count", "u32", 0, 4},
                                       {"iterations", "u32", 4, 4},
                                       {"mode", "u32", 8, 4},
                                       {"seed", "u32", 12, 4}};
            metadata.local_size = {64, 1, 1};
            auto metadata_status = validate_kernel_metadata(metadata);
            if (!metadata_status) continue;
            const auto words = std::span<const std::uint32_t>(detail::calibration::kProbeShaderSpv);
            auto reflection = detail::spirv::validate_and_reflect(
                words, metadata, detail::spirv::capabilities_from(device.info), "main");
            if (!reflection) continue;

            auto state = std::make_shared<detail::KernelState>();
            state->entry_point = "main";
            state->metadata = metadata;
            state->words.assign(words.begin(), words.end());
            state->runtime_id = impl_->runtime_id;
            state->device_index = device.id.index;
            const Kernel probe(std::move(state));

            constexpr std::uint32_t count = 16U * 1024U;
            auto input_result = buffers_.create<std::uint32_t>(count, BufferUsage::Storage);
            auto output_result = buffers_.create<std::uint32_t>(count, BufferUsage::Storage);
            auto tiny_input_result = buffers_.create<std::uint32_t>(1, BufferUsage::Storage);
            auto tiny_output_result = buffers_.create<std::uint32_t>(1, BufferUsage::Storage);
            if (!input_result || !output_result || !tiny_input_result || !tiny_output_result) continue;
            auto input = std::move(input_result).value();
            auto output = std::move(output_result).value();
            auto tiny_input = std::move(tiny_input_result).value();
            auto tiny_output = std::move(tiny_output_result).value();
            std::vector<std::uint32_t> input_values(count);
            for (std::uint32_t i = 0; i < count; ++i) input_values[i] = i * 747796405U + 2891336453U;
            if (!input.write(input_values) ||
                !tiny_input.write(std::span<const std::uint32_t>(input_values.data(), 1)))
                continue;

            struct Parameters {
                std::uint32_t count;
                std::uint32_t iterations;
                std::uint32_t mode;
                std::uint32_t seed;
            };
            const auto run = [&](std::uint32_t elements, std::uint32_t iterations, std::uint32_t mode,
                                 std::uint32_t seed) -> std::optional<std::uint64_t> {
                if (std::chrono::steady_clock::now() >= device.calibration_deadline) return {};
                const auto started = std::chrono::steady_clock::now();
                const auto& source = elements == 1 ? tiny_input : input;
                const auto& destination = elements == 1 ? tiny_output : output;
                auto event = dispatch(probe)
                                 .global(elements)
                                 .read(source.handle())
                                 .write(destination.handle())
                                 .push(Parameters{elements, iterations, mode, seed})
                                 .submit();
                if (!event) return {};
                auto remaining = device.calibration_deadline - std::chrono::steady_clock::now();
                if (remaining <= std::chrono::steady_clock::duration::zero() ||
                    !event.value().wait_for(std::chrono::duration_cast<std::chrono::nanoseconds>(remaining)))
                    return {};
                const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - started)
                                         .count();
                if (elapsed <= 0) return {};
                return static_cast<std::uint64_t>(elapsed);
            };
            const auto verify = [&](std::uint32_t elements, std::uint32_t iterations, std::uint32_t mode,
                                    std::uint32_t seed) {
                const auto& source = elements == 1 ? tiny_output : output;
                auto values = source.read();
                if (!values || values.value().size() < elements) return false;
                for (std::uint32_t i = 0; i < elements; ++i) {
                    auto expected = input_values[i];
                    if (mode == 0) expected ^= seed;
                    if (mode == 2) {
                        for (std::uint32_t n = 0; n < iterations; ++n) {
                            expected = expected * 1664525U + 1013904223U;
                            expected ^= expected >> 13U;
                        }
                    }
                    if (values.value()[i] != expected) return false;
                }
                return true;
            };

            constexpr std::uint32_t seed = 0x9e3779b9U;
            (void)run(1, 0, 0, seed);  // Warm pipeline and the smallest dispatch path.
            std::array<std::uint64_t, 3> tiny_samples{};
            std::size_t tiny_count = 0;
            for (auto& sample : tiny_samples) {
                auto elapsed = run(1, 0, 0, seed);
                if (!elapsed || !verify(1, 0, 0, seed)) break;
                sample = *elapsed;
                ++tiny_count;
            }
            if (tiny_count != 0) {
                std::sort(tiny_samples.begin(), tiny_samples.begin() + tiny_count);
                device.calibration_record.tiny_dispatch_ns = tiny_samples[tiny_count / 2];
            }

            (void)run(count, 0, 1, seed);
            if (auto elapsed = run(count, 0, 1, seed); elapsed && verify(count, 0, 1, seed)) {
                const auto bytes = static_cast<std::uint64_t>(count) * sizeof(std::uint32_t) * 2U;
                device.calibration_record.streaming_bytes_per_second = static_cast<std::uint64_t>(
                    static_cast<long double>(bytes) * 1'000'000'000.0L / static_cast<long double>(*elapsed));
            }

            constexpr std::uint32_t iterations = 256;
            (void)run(count, iterations, 2, seed);
            if (auto elapsed = run(count, iterations, 2, seed);
                elapsed && verify(count, iterations, 2, seed)) {
                const auto operations = static_cast<std::uint64_t>(count) * iterations * 4U;
                device.calibration_record.alu_operations_per_second =
                    static_cast<std::uint64_t>(static_cast<long double>(operations) * 1'000'000'000.0L /
                                               static_cast<long double>(*elapsed));
            }

            if (std::chrono::steady_clock::now() < device.calibration_deadline) {
                const auto key = device.id.uuid.empty() ? std::to_string(device.id.index) : device.id.uuid;
                const auto path = impl_->calibration_directory / "calibration" / (key + ".json");
                if (detail::calibration::write_atomic(path, device.calibration_record))
                    device.compute_calibration_pending =
                        device.calibration_record.tiny_dispatch_ns == 0 ||
                        device.calibration_record.streaming_bytes_per_second == 0 ||
                        device.calibration_record.alu_operations_per_second == 0;
            }
        } catch (...) {
            // Compute calibration is diagnostic and cannot make Runtime creation fail.
        }
    }
}

Result<std::string> Runtime::export_trace_json() const {
    if (!impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    if (options_.telemetry.value_or(TelemetryLevel::Basic) != TelemetryLevel::Trace)
        return Error(ErrorCode::InvalidArgument, "trace export requires TelemetryLevel::Trace");
    for (const auto& entry : impl_->devices) {
        const auto& dispatch = entry->dispatch;
        if (!dispatch || !dispatch->timestamps) continue;
        const auto completed = dispatch->queue->completed_value();
        if (!completed) return std::move(completed).error();
        auto samples = dispatch->timestamps->collect(completed.value());
        if (!samples) return std::move(samples).error();
        detail::observe_autotune_samples(dispatch, samples.value(), impl_->performance_database,
                                         impl_->performance_database_path);
    }
    std::ostringstream json;
    json << "{\"traceEvents\":[";
    bool first = true;
    for (const auto& entry : impl_->devices) {
        const auto& dispatch = entry->dispatch;
        if (!dispatch || !dispatch->timestamps) continue;
        const auto history = dispatch->timestamps->history();
        std::unordered_map<std::uint64_t, detail::TraceSchedulerScore> scheduler_scores;
        {
            std::lock_guard lock(dispatch->submit_mutex);
            scheduler_scores = dispatch->completed_trace_scores;
        }
        for (const auto& sample : history) {
            if (!first) json << ',';
            first = false;
            const auto start_us = sample.host_timestamp_nanoseconds / 1000;
            const auto duration_us = sample.duration_nanoseconds / 1000;
            json << "{\"name\":\"gpu_dispatch\",\"cat\":\"kora.gpu\",\"ph\":\"X\","
                    "\"ts\":"
                 << start_us << ",\"dur\":" << duration_us << ",\"pid\":1,\"tid\":" << entry->id.index
                 << ",\"args\":{\"submission_id\":" << sample.submission_id
                 << ",\"queue_value\":" << sample.queue_value
                 << ",\"host_clock_calibrated\":" << (sample.host_timestamp_nanoseconds ? "true" : "false")
                 << ",\"calibration_deviation_ns\":" << sample.calibration_deviation_nanoseconds;
            const auto scheduler_score = scheduler_scores.find(sample.submission_id);
            if (scheduler_score != scheduler_scores.end()) {
                const auto& selected = scheduler_score->second;
                const auto& score = selected.score;
                json << ",\"scheduler_score\":{\"selected_device_index\":" << selected.selected_device_index
                     << ",\"ready_delay_ns\":" << score.ready_delay_ns
                     << ",\"queue_delay_ns\":" << score.queue_delay_ns
                     << ",\"input_transfer_ns\":" << score.input_transfer_ns
                     << ",\"execute_ns\":" << score.execute_ns
                     << ",\"required_output_ns\":" << score.required_output_ns
                     << ",\"synchronization_ns\":" << score.synchronization_ns
                     << ",\"memory_penalty_ns\":" << score.memory_penalty_ns
                     << ",\"uncertainty_penalty_ns\":" << score.uncertainty_penalty_ns
                     << ",\"locality_bonus_ns\":" << score.locality_bonus_ns
                     << ",\"downstream_movement_penalty_ns\":" << score.downstream_movement_penalty_ns
                     << ",\"predicted_finish_ns\":" << score.predicted_finish_ns << '}';
            }
            json << "}}";
        }
    }
    json << "],\"displayTimeUnit\":\"us\",\"metadata\":{\"format\":\"Chrome Trace Event\","
            "\"time_origin\":\"calibrated_monotonic_host_timestamp_when_available\"}}";
    const auto document = json.str();
    auto validated = detail::json::parse(document, {.maximum_bytes = 128U * 1024U * 1024U,
                                                    .maximum_depth = 16,
                                                    .maximum_string_bytes = 1024,
                                                    .maximum_values = 1'000'000});
    if (!validated)
        return Error(ErrorCode::InternalInvariantViolation, "trace exporter generated invalid JSON");
    return document;
}

Result<DeviceFaultReport> Runtime::device_fault_report(const DeviceId& id) const {
    if (!impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    const auto found = std::find_if(impl_->devices.begin(), impl_->devices.end(),
                                    [&](const auto& device) { return device->id == id; });
    if (found == impl_->devices.end())
        return Error(ErrorCode::InvalidArgument, "device is not owned by runtime");
    const auto& entry = **found;
    const auto captured = entry.fault_state->diagnostics();
    DeviceFaultReport report;
    report.supported = captured.supported;
    report.queried = captured.queried;
    report.truncated = captured.truncated;
    report.loss_result = static_cast<std::int32_t>(entry.fault_state->last_result());
    report.query_result = captured.query_result;
    report.description = captured.description;
    report.addresses.reserve(captured.addresses.size());
    for (const auto& address : captured.addresses)
        report.addresses.push_back({address.type, address.reported_address, address.precision});
    report.vendor_records.reserve(captured.vendor_records.size());
    for (const auto& vendor : captured.vendor_records)
        report.vendor_records.push_back({vendor.description, vendor.code, vendor.data});
    return report;
}

Result<Event> Runtime::readback(const BufferHandle& buffer) const {
    if (!impl_ || impl_->devices.empty() || !buffer)
        return Error(ErrorCode::InvalidArgument, "runtime or buffer handle is invalid");
    return readback(buffer, impl_->devices.front()->id);
}

Result<Event> Runtime::readback(const BufferHandle& buffer, const DeviceId& device_id) const {
    if (!impl_ || impl_->devices.empty() || !buffer)
        return Error(ErrorCode::InvalidArgument, "runtime or buffer handle is invalid");
    const auto selected_device = std::find_if(impl_->devices.begin(), impl_->devices.end(),
                                              [&](const auto& device) { return device->id == device_id; });
    if (selected_device == impl_->devices.end())
        return Error(ErrorCode::InvalidArgument, "device is not owned by runtime");
    const auto context = (*selected_device)->dispatch;
    context->reap_released_buffers();
    const auto host_version = buffer.host_version();
    if (!host_version) return std::move(host_version).error();
    {
        std::lock_guard lock(context->submit_mutex);
        const auto resident = context->resident_host_versions.find(buffer.id());
        if (!context->device_newer_buffers.contains(buffer.id()) ||
            resident == context->resident_host_versions.end() || resident->second != host_version.value()) {
            context->device_newer_buffers.erase(buffer.id());
            return Event::when_all(std::span<const Event>{});
        }
    }
    auto device_buffer = ensure_transfer_buffer(context, buffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (!device_buffer) return std::move(device_buffer).error();
    const auto& mapped = device_buffer.value();
    if (context->direct_host_access && mapped.host_visible && mapped.mapped_data != nullptr &&
        mapped.queue_use && mapped.queue_use->pending.load(std::memory_order_acquire) == 0 &&
        mapped.queue_use->host_readable.load(std::memory_order_acquire)) {
        // The producing dispatch completed with a host-read barrier, so copy without a transfer.
        auto copied = read_mapped_output(context, mapped, buffer, host_version.value());
        if (!copied) return std::move(copied).error();
        return Event::when_all(std::span<const Event>{});
    }
    auto info = buffer.info();
    if (!info) return std::move(info).error();
    auto contents =
        std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(info.value().size_bytes));
    detail::QueueWait dependency;
    if (context->transfer_queue != context->queue) {
        dependency.semaphore = context->queue->completion_semaphore();
        {
            std::lock_guard lock(context->submit_mutex);
            const auto last_use = context->device_last_use.find(buffer.id());
            dependency.value = last_use == context->device_last_use.end() ? 0 : last_use->second;
        }
        dependency.stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    auto downloaded = context->transfers->download(device_buffer.value(), 0, *contents, dependency);
    if (!downloaded) return std::move(downloaded).error();
    context->download_count.fetch_add(1, std::memory_order_relaxed);
    auto promise = std::make_shared<detail::EventPromise>();
    const auto event = promise->event();
    const auto completion = downloaded.value().on_complete(
        [context, buffer, contents, promise, expected_host_version = host_version.value()](Status status) {
            Result<ConditionalBufferWrite> copied = ConditionalBufferWrite{};
            if (status) copied = buffer.write_bytes_if_version(*contents, expected_host_version);
            if (!copied) {
                if (status) status = Status(std::move(copied).error());
            } else if (status && copied.value().written) {
                std::lock_guard lock(context->submit_mutex);
                context->resident_host_versions[buffer.id()] = copied.value().host_version;
                context->device_newer_buffers.erase(buffer.id());
            } else if (status) {
                std::lock_guard lock(context->submit_mutex);
                context->device_newer_buffers.erase(buffer.id());
            }
            promise->complete(std::move(status));
            return ok_status();
        });
    if (!completion) promise->complete(Status(std::move(completion).error()));
    return event;
}

Status Runtime::register_cpu_fallback(std::string key, CpuFallbackFunction function) {
    if (!impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    if (key.empty() || !function)
        return Error(ErrorCode::InvalidArgument, "CPU fallback key and function must be non-empty");
    std::lock_guard lock(impl_->cpu_mutex);
    impl_->cpu_fallbacks.insert_or_assign(std::move(key), std::move(function));
    return {};
}

Result<Event> Runtime::run_cpu_fallback(std::string_view key, std::span<const BufferHandle> buffers,
                                        std::span<const KernelAccess> accesses,
                                        std::span<const std::byte> push_constants) {
    if (!impl_ || impl_->devices.empty())
        return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    if (key.empty() || buffers.size() != accesses.size())
        return Error(ErrorCode::InvalidArgument,
                     "CPU fallback key or buffer access declarations are invalid");

    CpuFallbackFunction function;
    std::shared_ptr<CpuExecutor> executor;
    {
        std::lock_guard lock(impl_->cpu_mutex);
        const auto found = impl_->cpu_fallbacks.find(std::string(key));
        if (found == impl_->cpu_fallbacks.end())
            return Error(ErrorCode::InvalidKernel, "no CPU fallback is registered for semantic key");
        function = found->second;
        if (!impl_->cpu_executor) {
            auto created_executor = CpuExecutor::create();
            if (!created_executor) return std::move(created_executor).error();
            impl_->cpu_executor = std::shared_ptr<CpuExecutor>(std::move(created_executor).value());
        }
        executor = impl_->cpu_executor;
    }

    auto task = std::make_shared<CpuTaskContext>();
    task->buffers_.assign(buffers.begin(), buffers.end());
    task->accesses_.assign(accesses.begin(), accesses.end());
    task->push_constants_ = push_constants;
    std::vector<Event> prerequisites;
    prerequisites.reserve(buffers.size());
    std::unordered_set<BufferId, detail::BufferIdHash> unique_buffers;
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        if (!buffers[index])
            return Error(ErrorCode::InvalidKernelBinding, "CPU task contains an empty buffer");
        if (accesses[index] != KernelAccess::Read && accesses[index] != KernelAccess::Write &&
            accesses[index] != KernelAccess::ReadWrite)
            return Error(ErrorCode::InvalidKernelBinding, "CPU task contains an invalid buffer access mode");
        if (!unique_buffers.insert(buffers[index].id()).second)
            return Error(ErrorCode::InvalidKernelBinding,
                         "CPU task cannot declare the same buffer more than once");
        if (accesses[index] == KernelAccess::Write) continue;
        auto visible = readback(buffers[index]);
        if (!visible) return std::move(visible).error();
        prerequisites.push_back(std::move(visible).value());
    }
    auto ready = Event::when_all(prerequisites);
    if (!ready) return std::move(ready).error();

    auto promise = std::make_shared<detail::EventPromise>();
    const auto result_event = promise->event();
    auto continued = ready.value().on_complete([task, function = std::move(function), executor,
                                                promise](Status status) mutable {
        if (!status) {
            promise->complete(std::move(status));
            return ok_status();
        }
        for (std::size_t index = 0; index < task->buffers_.size(); ++index) {
            const auto access = task->accesses_[index];
            const auto map_access = access == KernelAccess::Read    ? MapAccess::Read
                                    : access == KernelAccess::Write ? MapAccess::Write
                                                                    : MapAccess::ReadWrite;
            auto mapping = task->buffers_[index].map(map_access);
            if (!mapping) {
                promise->complete(Status(std::move(mapping).error()));
                return ok_status();
            }
            task->mappings_.push_back(std::move(mapping).value());
        }
        auto submitted = executor->submit([task, function = std::move(function)]() mutable {
            Status status;
            try {
                status = function(*task);
            } catch (...) {
                status =
                    Error(ErrorCode::InternalInvariantViolation, "CPU fallback callback threw an exception");
            }
            task->mappings_.clear();  // Publish host versions before completing the Event.
            return status;
        });
        if (!submitted) {
            promise->complete(Status(std::move(submitted).error()));
            return ok_status();
        }
        (void)submitted.value().on_complete([promise](Status task_status) {
            promise->complete(std::move(task_status));
            return ok_status();
        });
        return ok_status();
    });
    if (!continued) promise->complete(Status(std::move(continued).error()));
    return result_event;
}

}  // namespace kora
