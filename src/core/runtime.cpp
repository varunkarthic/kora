#include "kora/runtime.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/calibration.hpp"
#include "core/device_caps.hpp"
#include "core/event_state.hpp"
#include "core/performance_db.hpp"
#include "kernel/pipeline_manager.hpp"
#include "memory/memory_manager.hpp"
#include "memory/memory_pool.hpp"
#include "memory/transfer_engine.hpp"
#include "vulkan/api.hpp"
#include "vulkan/command_pool_set.hpp"
#include "vulkan/queue_context.hpp"
#include "vulkan/queue_family_transfer.hpp"

namespace kora {

namespace detail {

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
};

struct DispatchContext {
    std::atomic<bool> accepting_work{true};
    std::shared_ptr<RuntimeBufferState> buffer_state;
    std::uint64_t runtime_id = 0;
    std::size_t device_index = 0;
    const DeviceInfo* info = nullptr;
    detail::kernel::PipelineManager* pipelines = nullptr;
    detail::TransferEngine* transfers = nullptr;
    vk::QueueContext* queue = nullptr;
    vk::QueueContext* transfer_queue = nullptr;
    vk::QueueFamilyPolicy queue_family_policy;
    const vk::Device* device = nullptr;
    std::mutex submit_mutex;
    std::unique_ptr<vk::CommandPoolSet> command_pool;
    std::unordered_map<BufferId, detail::TransferBuffer, BufferIdHash> buffers;
    std::unordered_map<BufferId, std::uint64_t, BufferIdHash> resident_host_versions;
    std::unordered_set<BufferId, BufferIdHash> device_newer_buffers;
    std::atomic<std::uint64_t> upload_count{0};
    std::atomic<std::uint64_t> download_count{0};
    std::unordered_map<BufferId, detail::PoolAllocation, BufferIdHash> allocations;
    std::unordered_map<BufferId, detail::MemoryPool*, BufferIdHash> allocation_pools;
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

}  // namespace detail

namespace {

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
                    std::uint64_t runtime_identity, bool force_fallback)
            : id(std::move(device_id)),
              info(std::move(device_info)),
              memory_manager(std::move(manager)),
              device(std::move(logical_device)),
              queue_family_policy{device.compute_queue_family(), device.transfer_queue_family(), true},
              pipelines(std::move(pipeline_manager)),
              runtime_id(runtime_identity),
              force_legacy_sync(force_fallback) {}

        Status initialize() {
            vk::QueueContext::CoalescingOptions coalescing;
            coalescing.enabled = true;
            auto queue_result = vk::QueueContext::create(
                device.api(), device.handle(), device.compute_queue(), force_legacy_sync, coalescing);
            if (!queue_result) return Status(std::move(queue_result).error());
            queue = std::move(queue_result).value();
            transfer_queue = queue.get();
            if (queue_family_policy.compute_family != queue_family_policy.transfer_family) {
                auto transfer_result = vk::QueueContext::create(device.api(), device.handle(),
                                                                device.transfer_queue(), force_legacy_sync);
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
            // The runtime may select a device-local, non-host-visible pool even on UMA devices.
            // Use staging for this API until buffer placement and measured direct-map eligibility
            // are selected per allocation.
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

        void calibrate(const std::filesystem::path& directory, std::uint32_t budget_ms) noexcept {
            try {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(budget_ms);
                const auto key = id.uuid.empty() ? std::to_string(id.index) : id.uuid;
                const auto path = directory / "calibration" / (key + ".json");
                detail::calibration::Record cached;
                if (id.uuid.empty()) return;
                if (detail::calibration::read(path, id.uuid, driver_version, cached)) {
                    calibration_record = cached;
                    return;
                }
                detail::calibration::Record record;
                record.device_uuid = id.uuid;
                record.driver_version = driver_version;
                constexpr auto iterations = 1000;
                const auto start = std::chrono::steady_clock::now();
                for (std::uint32_t index = 0; index < iterations; ++index) {
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                }
                const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - start)
                                         .count();
                record.submit_overhead_ns = static_cast<std::uint64_t>(elapsed) / iterations;
                // No transfer or arithmetic probe is enabled until a safe scratch-buffer API
                // exists; zero explicitly means "not measured", not a zero-cost transfer.
                if (std::chrono::steady_clock::now() >= deadline) return;
                auto status = detail::calibration::write_atomic(path, record);
                if (status && std::chrono::steady_clock::now() <= deadline) calibration_record = record;
            } catch (...) {
                // Calibration is best-effort; it must never prevent Runtime creation.
            }
        }

        ~DeviceEntry() {
            if (dispatch) dispatch->accepting_work.store(false, std::memory_order_release);
            if (queue) (void)queue->shutdown(ShutdownMode::Graceful);
            if (dedicated_transfer_queue) (void)dedicated_transfer_queue->shutdown(ShutdownMode::Graceful);
            if (dispatch) dispatch->release_buffers();
        }

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
        bool force_legacy_sync = false;
        std::uint32_t driver_version = 0;
        detail::calibration::Record calibration_record;
    };

    vk::Loader loader;
    vk::Instance instance;
    std::vector<std::unique_ptr<DeviceEntry>> devices;
    std::uint64_t runtime_id = 0;
    std::filesystem::path calibration_directory;
    std::uint32_t calibration_budget_ms = 250;
    bool calibrate_on_start = true;
    detail::PerformanceDatabase performance_database;
    std::mutex cpu_mutex;
    std::unordered_map<std::string, CpuFallbackFunction> cpu_fallbacks;
    std::shared_ptr<CpuExecutor> cpu_executor;

    ~Impl() {
        if (cpu_executor) (void)cpu_executor->shutdown();
    }
};

namespace {
Result<detail::TransferBuffer> ensure_transfer_buffer(const std::shared_ptr<detail::DispatchContext>& context,
                                                      const BufferHandle& handle, VkBufferUsageFlags usage);
}

Result<Event> DispatchBuilder::submit() {
    if (!context_ || !context_->accepting_work.load(std::memory_order_acquire) || !kernel_ ||
        !context_->pipelines || !context_->queue || !context_->device)
        return Error(ErrorCode::InvalidArgument, "dispatch builder has an invalid runtime or kernel");
    if (kernel_->runtime_id != context_->runtime_id || kernel_->device_index != context_->device_index)
        return Error(ErrorCode::InvalidArgument, "kernel was not loaded for this runtime device");
    auto metadata_status = validate_kernel_metadata(kernel_->metadata);
    if (!metadata_status) return std::move(metadata_status).error();
    if (buffers_.size() != kernel_->metadata.bindings.size() || accesses_.size() != buffers_.size())
        return Error(ErrorCode::InvalidKernelBinding, "dispatch buffer count does not match kernel metadata");
    if (push_constants_.size() != kernel_->metadata.push_constant_size)
        return Error(ErrorCode::InvalidArgument, "push-constant byte size does not match kernel metadata");
    if (kernel_->metadata.push_constant_size > context_->info->limits().max_push_constants_size)
        return Error(ErrorCode::InvalidArgument, "push-constant block exceeds the selected device limit");

    const auto& limits = context_->info->limits();
    std::array<std::uint32_t, 3> groups{};
    for (std::size_t dimension = 0; dimension < 3; ++dimension) {
        if (global_size_[dimension] == 0)
            return Error(ErrorCode::InvalidDispatch, "global dispatch dimensions must all be nonzero");
        const auto local = kernel_->metadata.local_size[dimension];
        const auto group_count = 1U + ((global_size_[dimension] - 1U) / local);
        if (group_count > UINT32_MAX || group_count > limits.max_compute_work_group_count[dimension])
            return Error(ErrorCode::InvalidDispatch, "dispatch group count exceeds a selected-device limit");
        groups[dimension] = static_cast<std::uint32_t>(group_count);
        if (local > limits.max_compute_work_group_size[dimension])
            return Error(ErrorCode::InvalidDispatch,
                         "kernel local size exceeds a selected-device dimension limit");
    }
    std::uint64_t invocations = 1;
    for (const auto dimension : kernel_->metadata.local_size) {
        if (dimension > UINT32_MAX / invocations)
            return Error(ErrorCode::InvalidDispatch, "kernel local-size invocation product overflows uint32");
        invocations *= dimension;
    }
    if (invocations > limits.max_compute_work_group_invocations)
        return Error(ErrorCode::InvalidDispatch, "kernel local size exceeds maxComputeWorkGroupInvocations");

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
    auto specialization = kernel_->metadata.specialization_constants;
    auto pipeline = context_->pipelines->get_or_create(kernel_->words, kernel_->metadata, specialization,
                                                       kernel_->entry_point);
    if (!pipeline) return std::move(pipeline).error();
    for (std::size_t i = 0; i < buffers_.size(); ++i) {
        auto buffer = ensure_transfer_buffer(context_, buffers_[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        if (!buffer) return std::move(buffer).error();
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

        Result<std::vector<std::byte>> contents =
            Error(ErrorCode::InternalInvariantViolation, "host snapshot was not captured");
        for (int attempt = 0; attempt < 3; ++attempt) {
            version = buffers_[i].host_version();
            if (!version) return std::move(version).error();
            contents = buffers_[i].read_bytes();
            if (!contents) return std::move(contents).error();
            const auto after = buffers_[i].host_version();
            if (!after) return std::move(after).error();
            if (after.value() == version.value()) {
                bound_host_versions[i] = version.value();
                break;
            }
            contents = Error(ErrorCode::StaleHandle, "host buffer changed while preparing an upload");
        }
        if (!contents) return std::move(contents).error();
        auto uploaded = context_->transfers->upload(vk_buffers[i], 0, contents.value());
        if (!uploaded) return std::move(uploaded).error();
        auto upload_status = uploaded.value().wait();
        if (!upload_status) return std::move(upload_status).error();
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

    auto queue = context_->queue;
    auto ctx = context_;
    auto pipeline_ref = pipeline.value();
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
    auto work = std::make_shared<std::pair<VkCommandBuffer, std::uint64_t>>(VK_NULL_HANDLE, 0);
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
    submission.record_commands = [ctx, pipeline_ref, bound_buffers, push_constants, groups_copy, work,
                                  record_status, allocated_descriptors](
                                     std::uint64_t, std::vector<VkCommandBuffer>& commands) {
        std::lock_guard lock(ctx->submit_mutex);
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
        work->first = command.value();
        auto api = ctx->device->api();
        api.cmd_bind_pipeline(command.value(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_ref->handle);
        std::vector<VkDescriptorSet> descriptor_sets;
        for (std::uint32_t set = 0; set < pipeline_ref->descriptor_set_layouts.size(); ++set) {
            auto descriptor = ctx->pipelines->allocate_descriptor_set(pipeline_ref, set);
            if (!descriptor) {
                *record_status = Status(std::move(descriptor).error());
                (void)ctx->command_pool->discard(command.value());
                work->first = VK_NULL_HANDLE;
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
        if (!descriptor_sets.empty())
            api.cmd_bind_descriptor_sets(
                command.value(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_ref->layout, 0,
                static_cast<std::uint32_t>(descriptor_sets.size()), descriptor_sets.data(), 0, nullptr);
        if (!push_constants.empty())
            api.cmd_push_constants(command.value(), pipeline_ref->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   static_cast<std::uint32_t>(push_constants.size()), push_constants.data());
        api.cmd_dispatch(command.value(), groups_copy[0], groups_copy[1], groups_copy[2]);
        auto ended = ctx->command_pool->end_recording(command.value());
        if (!ended) return ended;
        commands.push_back(command.value());
        return Status{};
    };
    submission.on_submitted = [ctx, work, bound_handles = std::move(bound_handles), written_buffers](
                                  std::uint64_t timeline, Status status) {
        std::lock_guard lock(ctx->submit_mutex);
        if (status) {
            work->second = timeline;
            if (work->first != VK_NULL_HANDLE) (void)ctx->command_pool->submitted(work->first, timeline);
            for (const auto& buffer : bound_handles) (void)buffer.mark_used(timeline);
        } else if (work->first != VK_NULL_HANDLE && ctx->command_pool) {
            (void)ctx->command_pool->discard(work->first);
        }
        if (!status) {
            for (const auto id : written_buffers) {
                ctx->resident_host_versions.erase(id);
                ctx->device_newer_buffers.erase(id);
            }
        }
    };
    submission.on_complete = [promise, ctx, pipeline_ref, allocated_descriptors, bound_buffers,
                              bound_handles = buffers_, accesses = accesses_, written_buffers,
                              bound_host_versions,
                              defer_readback = defer_readback_](std::uint64_t compute_value, Status status) {
        (void)pipeline_ref;
        for (const auto descriptor : *allocated_descriptors)
            (void)ctx->pipelines->free_descriptor_set(descriptor);
        if (status && !defer_readback) {
            auto readback_contents = std::make_shared<std::vector<std::vector<std::byte>>>(accesses.size());
            std::vector<Event> readbacks;
            std::vector<std::size_t> output_indices;
            for (std::size_t index = 0; index < accesses.size(); ++index) {
                if (accesses[index] == KernelAccess::Read) continue;
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
    auto submitted = queue->submit(std::move(submission));
    if (!submitted) {
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
                                 detail::MemoryPolicy::DevicePreferred, requirements.memoryTypeBits);
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

Runtime::Runtime() = default;
Runtime::Runtime(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Runtime::~Runtime() = default;
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
                                 std::shared_ptr<const detail::KernelState> kernel)
    : context_(std::move(context)), kernel_(std::move(kernel)) {}

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

DispatchBuilder& DispatchBuilder::push(std::span<const std::byte> bytes) {
    push_constants_.assign(bytes.begin(), bytes.end());
    return *this;
}

Result<Kernel> Runtime::load_kernel(const std::filesystem::path& spirv_path, std::string entry_point,
                                    KernelMetadata metadata) {
    if (!impl_ || impl_->devices.empty())
        return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
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
    auto state = std::make_shared<detail::KernelState>();
    state->entry_point = std::move(entry_point);
    state->metadata = std::move(metadata);
    state->words = std::move(words);
    state->runtime_id = impl_->runtime_id;
    state->device_index = impl_->devices.front()->id.index;
    return Kernel(std::move(state));
}

DispatchBuilder Runtime::dispatch(const Kernel& kernel) const {
    return DispatchBuilder(impl_ && !impl_->devices.empty() ? impl_->devices.front()->dispatch : nullptr,
                           kernel.state_);
}

Result<Runtime> Runtime::create(const RuntimeOptions& options) {
    const auto resolved = resolve(options);
    auto loader_result = vk::Loader::load();
    if (!loader_result) return std::move(loader_result).error();

    auto instance_options = vk::Loader::InstanceOptions{};
    instance_options.enable_validation = resolved.enable_validation;
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
    impl->calibrate_on_start = resolved.enable_startup_calibration;
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    if (now_ms >= 0) {
        (void)impl->performance_database.load(
            std::filesystem::path(resolved.cache_dir) / "performance-db.json",
            static_cast<std::uint64_t>(now_ms));
    }
    for (const auto& id : selected.value()) {
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
            std::move(pipeline_manager).value(), impl->runtime_id, resolved.force_legacy_sync);
        entry->configure_budget_refresh(impl->instance, resolved.memory_budget_fraction);
        entry->driver_version = physical.properties.driverVersion;
        auto initialized = entry->initialize();
        if (!initialized) return std::move(initialized).error();
        if (impl->calibrate_on_start)
            entry->calibrate(impl->calibration_directory, impl->calibration_budget_ms);
        impl->devices.push_back(std::move(entry));
    }
    return Runtime(std::move(impl));
}

std::vector<DeviceId> Runtime::devices() const {
    std::vector<DeviceId> ids;
    if (!impl_) return ids;
    ids.reserve(impl_->devices.size());
    for (const auto& device : impl_->devices) ids.push_back(device->id);
    return ids;
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
    const auto& context = impl_->devices.front()->dispatch;
    return RuntimeTransferStats{context->upload_count.load(std::memory_order_relaxed),
                                context->download_count.load(std::memory_order_relaxed)};
}

Result<Event> Runtime::readback(const BufferHandle& buffer) const {
    if (!impl_ || impl_->devices.empty() || !buffer)
        return Error(ErrorCode::InvalidArgument, "runtime or buffer handle is invalid");
    const auto context = impl_->devices.front()->dispatch;
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
    auto info = buffer.info();
    if (!info) return std::move(info).error();
    auto contents =
        std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(info.value().size_bytes));
    detail::QueueWait dependency;
    if (context->transfer_queue != context->queue) {
        dependency.semaphore = context->queue->completion_semaphore();
        dependency.value = info.value().last_use_timeline_value;
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
