#include "memory/transfer_engine.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

#include "core/event_state.hpp"
#include "kora/detail/checked.hpp"
#include "vulkan/command_pool_set.hpp"
#include "vulkan/hazard_tracker.hpp"

namespace kora::detail {
namespace {

Error vulkan_error(const char* message, VkResult result) {
    return Error(ErrorCode::TransferFailed, message).with_vk(result);
}

Status validate_buffer_range(const TransferBuffer& buffer, VkDeviceSize offset, VkDeviceSize size,
                             const char* description) {
    if (buffer.buffer == VK_NULL_HANDLE)
        return Error(ErrorCode::InvalidArgument, "transfer requires a valid Vulkan buffer");
    auto end = checked::range_end(offset, size, buffer.size, description);
    if (!end) return std::move(end).error();
    return ok_status();
}

}  // namespace

Status mapped_memory_operation(const vk::Api& api, VkDevice device, const TransferBuffer& buffer,
                               VkDeviceSize offset, VkDeviceSize size, VkDeviceSize atom_size, bool flush) {
    if (buffer.host_coherent || size == 0) return ok_status();
    if (!buffer.host_visible || buffer.memory == VK_NULL_HANDLE || buffer.memory_allocation_size == 0 ||
        !checked::is_pow2(atom_size)) {
        return Error(ErrorCode::InvalidArgument, "non-coherent transfer memory metadata is incomplete");
    }
    auto absolute_begin = checked::add(buffer.memory_offset, offset);
    auto relative_end = checked::add(offset, size);
    if (!absolute_begin || !relative_end || *relative_end > buffer.size)
        return Error(ErrorCode::Overflow, "mapped transfer range overflowed");
    auto absolute_end = checked::add(buffer.memory_offset, *relative_end);
    if (!absolute_end || *absolute_end > buffer.memory_allocation_size)
        return Error(ErrorCode::InvalidArgument, "mapped transfer range exceeds its memory allocation");
    const auto start = checked::align_down(*absolute_begin, atom_size);
    const auto rounded_end = checked::align_up(*absolute_end, atom_size);
    if (!start || !rounded_end || *start >= buffer.memory_allocation_size)
        return Error(ErrorCode::Overflow, "mapped transfer cache range alignment failed");

    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = buffer.memory;
    range.offset = *start;
    if (*rounded_end > buffer.memory_allocation_size) {
        range.size = VK_WHOLE_SIZE;
    } else {
        range.size = *rounded_end - *start;
    }
    const auto result = flush ? api.flush_mapped_memory_ranges(device, 1, &range)
                              : api.invalidate_mapped_memory_ranges(device, 1, &range);
    if (result != VK_SUCCESS)
        return vulkan_error(
            flush ? "Flushing mapped upload bytes failed" : "Invalidating mapped readback bytes failed",
            result);
    return ok_status();
}

namespace {

void record_hazards(const vk::Api& api, VkCommandBuffer command_buffer,
                    const std::vector<vk::BarrierPlan>& plans) {
    for (const auto& plan : plans) {
        if (api.synchronization2_enabled() && api.cmd_pipeline_barrier2) {
            VkMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            barrier.srcStageMask = plan.source_stage;
            barrier.srcAccessMask = plan.source_access;
            barrier.dstStageMask = plan.destination_stage;
            barrier.dstAccessMask = plan.destination_access;
            VkDependencyInfo dependency{};
            dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &barrier;
            api.cmd_pipeline_barrier2(command_buffer, &dependency);
            continue;
        }
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = plan.source_access;
        barrier.dstAccessMask = plan.destination_access;
        api.cmd_pipeline_barrier(command_buffer, plan.source_stage, plan.destination_stage, 0, 1, &barrier, 0,
                                 nullptr, 0, nullptr);
    }
}

}  // namespace

struct TransferEngine::Impl {
    struct Work {
        std::optional<StagingSlice> staging;
        StagingRing* ring = nullptr;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        std::atomic<bool> submission_processed{false};
    };

    struct RingState {
        std::mutex mutex;
        std::uint64_t last_enqueued = 0;
    };

    const vk::Device* device = nullptr;
    vk::QueueContext* queue = nullptr;
    Options options;
    std::unique_ptr<StagingRing> upload_ring;
    std::unique_ptr<StagingRing> readback_ring;
    RingState upload_state;
    RingState readback_state;
    std::unique_ptr<vk::CommandPoolSet> command_pool;
};

TransferEngine::TransferEngine(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
TransferEngine::~TransferEngine() = default;

Result<std::unique_ptr<TransferEngine>> TransferEngine::create(const vk::Device& device,
                                                               vk::QueueContext& queue,
                                                               MemoryPool& upload_pool,
                                                               MemoryPool& readback_pool, Options options) {
    if (device.handle() == VK_NULL_HANDLE || options.command_buffer_count == 0 ||
        options.staging_initial_capacity == 0 ||
        options.staging_maximum_capacity < options.staging_initial_capacity ||
        !checked::is_pow2(options.non_coherent_atom_size)) {
        return Error(ErrorCode::InvalidArgument, "TransferEngine device, ring size, or atom size is invalid");
    }
    const auto& api = device.api();
    if (!api.cmd_pipeline_barrier || !api.cmd_copy_buffer || !api.cmd_fill_buffer ||
        !api.reset_command_buffer) {
        return Error(ErrorCode::UnsupportedFeature, "Vulkan transfer command dispatch is unavailable");
    }
    auto impl = std::make_shared<Impl>();
    impl->device = &device;
    impl->queue = &queue;
    impl->options = options;
    if (StagingRing::should_use_staging(options.topology, options.measured_staging_benefit)) {
        StagingRing::Options upload_options;
        upload_options.direction = StagingDirection::Upload;
        upload_options.topology = options.topology;
        upload_options.measured_staging_benefit = options.measured_staging_benefit;
        upload_options.non_coherent_atom_size = options.non_coherent_atom_size;
        upload_options.initial_capacity = options.staging_initial_capacity;
        upload_options.maximum_capacity = options.staging_maximum_capacity;
        auto upload = StagingRing::create(device, upload_pool, upload_options);
        if (!upload) return std::move(upload).error();
        impl->upload_ring = std::move(upload).value();

        StagingRing::Options readback_options = upload_options;
        readback_options.direction = StagingDirection::Readback;
        auto readback = StagingRing::create(device, readback_pool, readback_options);
        if (!readback) return std::move(readback).error();
        impl->readback_ring = std::move(readback).value();
    }
    return std::unique_ptr<TransferEngine>(new TransferEngine(std::move(impl)));
}

Result<Event> TransferEngine::enqueue(std::function<Status(VkCommandBuffer)> record,
                                      std::function<Status()> complete, std::optional<StagingSlice> staging,
                                      StagingRing* ring, QueueWait dependency,
                                      std::function<void(std::uint64_t)> on_submitted,
                                      std::vector<std::shared_ptr<QueueTimelineUse>> uses) {
    auto work = std::make_shared<Impl::Work>();
    work->staging = staging;
    work->ring = ring;
    auto promise = std::make_shared<EventPromise>();
    auto event = promise->event();
    auto impl = impl_;
    Impl::RingState* ring_state = ring == impl->upload_ring.get()     ? &impl->upload_state
                                  : ring == impl->readback_ring.get() ? &impl->readback_state
                                                                      : nullptr;
    if (work->staging && ring_state) {
        std::unique_lock lock(ring_state->mutex);
        if (ring_state->last_enqueued != 0) {
            auto waited = impl->queue->wait_for_value(ring_state->last_enqueued);
            if (!waited) return std::move(waited).error();
            auto completed = impl->queue->completed_value();
            if (!completed) return std::move(completed).error();
            lock.unlock();
            auto reclaimed = ring_state == &impl->upload_state
                                 ? impl->upload_ring->reclaim_completed(completed.value())
                                 : impl->readback_ring->reclaim_completed(completed.value());
            if (!reclaimed) return std::move(reclaimed).error();
        } else {
            lock.unlock();
        }
    }

    vk::QueueContext::Submission submission;
    if (dependency.semaphore != VK_NULL_HANDLE) {
        submission.wait_semaphores.push_back(dependency.semaphore);
        submission.wait_stages.push_back(dependency.stage);
        submission.wait_values.push_back(dependency.value);
    }
    submission.record_commands = [impl, work, record = std::move(record)](
                                     uint64_t, std::vector<VkCommandBuffer>& commands) mutable -> Status {
        if (!impl->command_pool) {
            vk::CommandPoolSet::Options pool_options;
            pool_options.queue_family = impl->options.queue_family;
            pool_options.ring_size = impl->options.command_buffer_count;
            impl->command_pool = std::make_unique<vk::CommandPoolSet>(impl->device->api(),
                                                                      impl->device->handle(), pool_options);
            if (impl->command_pool->capacity() == 0)
                return Error(ErrorCode::NoVulkanImplementation,
                             "Transfer command-buffer ring could not be created");
        }

        auto completed = impl->queue->completed_value();
        if (!completed) return std::move(completed).error();
        auto command = impl->command_pool->acquire(completed.value());
        while (!command && command.error().code == ErrorCode::OutOfMemoryBudget) {
            const auto oldest = impl->command_pool->oldest_pending_value();
            if (oldest == 0) break;
            auto waited = impl->queue->wait_for_value(oldest);
            if (!waited) return std::move(waited).error();
            completed = impl->queue->completed_value();
            if (!completed) return std::move(completed).error();
            command = impl->command_pool->acquire(completed.value());
        }
        if (!command) return std::move(command).error();
        work->command_buffer = command.value();
        auto recorded = record(command.value());
        if (!recorded) return std::move(recorded).error();
        auto ended = impl->command_pool->end_recording(command.value());
        if (!ended) return std::move(ended).error();
        commands.push_back(command.value());
        return ok_status();
    };
    submission.on_submitted = [impl, work, ring_state, on_submitted = std::move(on_submitted)](
                                  uint64_t timeline_value, Status status) {
        if (status) {
            if (work->command_buffer != VK_NULL_HANDLE) {
                auto tracked = impl->command_pool->submitted(work->command_buffer, timeline_value);
                if (!tracked) {
                    // Keep the command and any staging bytes pinned on an invariant failure.
                    work->submission_processed.store(true, std::memory_order_release);
                    return;
                }
            }
            if (work->staging && work->ring) (void)work->ring->retire(*work->staging, timeline_value);
        } else {
            if (work->command_buffer != VK_NULL_HANDLE && impl->command_pool)
                (void)impl->command_pool->discard(work->command_buffer);
            if (work->staging && work->ring) (void)work->ring->cancel(*work->staging);
        }
        if (status && on_submitted) on_submitted(timeline_value);
        if (work->staging && ring_state && status) {
            std::lock_guard lock(ring_state->mutex);
            ring_state->last_enqueued = timeline_value;
        }
        work->submission_processed.store(true, std::memory_order_release);
    };
    submission.on_complete = [work, promise, complete = std::move(complete),
                              pending = std::make_shared<PendingUses>(std::move(uses))](
                                 uint64_t, Status status) mutable {
        if (!work->submission_processed.exchange(true, std::memory_order_acq_rel)) {
            if (work->staging && work->ring) (void)work->ring->cancel(*work->staging);
        }
        if (status && complete) {
            try {
                status = complete();
            } catch (...) {
                status = Error(ErrorCode::InternalInvariantViolation,
                               "Transfer completion callback threw an exception");
            }
        }
        promise->complete(std::move(status));
    };

    auto submitted = impl_->queue->submit(std::move(submission));
    if (!submitted) {
        if (staging && ring) (void)ring->cancel(*staging);
        return std::move(submitted).error();
    }
    if (work->staging && ring_state) {
        std::lock_guard lock(ring_state->mutex);
        ring_state->last_enqueued = submitted.value();
    }
    return event;
}

Result<Event> TransferEngine::upload(TransferBuffer destination, VkDeviceSize destination_offset,
                                     std::span<const std::byte> source) {
    if (destination.buffer == VK_NULL_HANDLE ||
        !checked::in_range(destination_offset, source.size(), destination.size))
        return Error(ErrorCode::InvalidArgument, "upload destination range is invalid");
    if (source.empty()) {
        auto promise = std::make_shared<EventPromise>();
        auto event = promise->event();
        promise->complete(ok_status());
        return event;
    }

    const bool direct = !impl_->upload_ring && destination.host_visible && destination.mapped_data != nullptr;
    std::optional<StagingSlice> staging;
    StagingRing* ring = nullptr;
    std::shared_ptr<std::vector<std::byte>> owned_source;
    if (direct) {
        std::memcpy(destination.mapped_data + destination_offset, source.data(), source.size());
        auto flushed = mapped_memory_operation(impl_->device->api(), impl_->device->handle(), destination,
                                               destination_offset, source.size(),
                                               impl_->options.non_coherent_atom_size, true);
        if (!flushed) return std::move(flushed).error();
    } else {
        if (!impl_->upload_ring)
            return Error(
                ErrorCode::UnsupportedFeature,
                "transfer policy selected direct access but this buffer has no host-visible mapping");
        auto completed = impl_->queue->completed_value();
        if (!completed) return std::move(completed).error();
        auto reclaimed = impl_->upload_ring->reclaim_completed(completed.value());
        if (!reclaimed) return std::move(reclaimed).error();
        auto reserved = impl_->upload_ring->reserve(source.size(), 1, completed.value(),
                                                    [this](uint64_t value) -> Result<uint64_t> {
                                                        auto waited = impl_->queue->wait_for_value(value);
                                                        if (!waited) return std::move(waited).error();
                                                        return impl_->queue->completed_value();
                                                    });
        if (!reserved) return std::move(reserved).error();
        staging = reserved.value();
        ring = impl_->upload_ring.get();
        std::memcpy(staging->mapped_data, source.data(), source.size());
        auto flushed = ring->flush(*staging);
        if (!flushed) {
            (void)ring->cancel(*staging);
            return std::move(flushed).error();
        }
    }

    auto api = impl_->device->api();
    const bool cross_queue = impl_->options.cross_queue_concurrent;
    auto record = [api, destination, destination_offset, size = static_cast<VkDeviceSize>(source.size()),
                   direct, staging, cross_queue](VkCommandBuffer command) {
        if (direct) {
            vk::HazardTracker hazards;
            (void)hazards.access(vk::AccessKind::HostWrite);
            if (cross_queue) {
                record_hazards(api, command, hazards.access(vk::AccessKind::TransferRead));
            } else {
                record_hazards(api, command, hazards.access(vk::AccessKind::ComputeRead));
                record_hazards(api, command, hazards.access(vk::AccessKind::ComputeWrite));
            }
        } else {
            vk::HazardTracker staging_hazards;
            (void)staging_hazards.access(vk::AccessKind::HostWrite);
            record_hazards(api, command, staging_hazards.access(vk::AccessKind::TransferRead));
            vk::HazardTracker destination_hazards;
            if (cross_queue) {
                (void)destination_hazards.access(vk::AccessKind::TransferWrite);
            } else {
                (void)destination_hazards.access(vk::AccessKind::ComputeRead);
                (void)destination_hazards.access(vk::AccessKind::ComputeWrite);
            }
            record_hazards(api, command, destination_hazards.access(vk::AccessKind::TransferWrite));
            VkBufferCopy region{};
            region.srcOffset = staging->offset;
            region.dstOffset = destination_offset;
            region.size = size;
            api.cmd_copy_buffer(command, staging->buffer, destination.buffer, 1, &region);
            if (!cross_queue) {
                record_hazards(api, command, destination_hazards.access(vk::AccessKind::ComputeRead));
                record_hazards(api, command, destination_hazards.access(vk::AccessKind::ComputeWrite));
            }
        }
        return ok_status();
    };
    return enqueue(
        std::move(record), {}, staging, ring, {},
        destination.queue_use
            ? std::function<void(std::uint64_t)>([use = destination.queue_use](std::uint64_t value) {
                  use->transfer_value.store(value, std::memory_order_release);
              })
            : std::function<void(std::uint64_t)>{},
        {destination.queue_use});
}

Result<Event> TransferEngine::download(TransferBuffer source, VkDeviceSize source_offset,
                                       std::span<std::byte> destination, QueueWait dependency) {
    if (source.buffer == VK_NULL_HANDLE || !checked::in_range(source_offset, destination.size(), source.size))
        return Error(ErrorCode::InvalidArgument, "download source range is invalid");
    if (destination.empty()) {
        auto promise = std::make_shared<EventPromise>();
        auto event = promise->event();
        promise->complete(ok_status());
        return event;
    }

    const bool direct = !impl_->readback_ring && source.host_visible && source.mapped_data != nullptr;
    std::optional<StagingSlice> staging;
    StagingRing* ring = nullptr;
    if (!direct) {
        if (!impl_->readback_ring)
            return Error(
                ErrorCode::UnsupportedFeature,
                "transfer policy selected direct access but this source has no host-visible mapping");
        auto completed = impl_->queue->completed_value();
        if (!completed) return std::move(completed).error();
        auto reclaimed = impl_->readback_ring->reclaim_completed(completed.value());
        if (!reclaimed) return std::move(reclaimed).error();
        auto reserved = impl_->readback_ring->reserve(destination.size(), 1, completed.value(),
                                                      [this](uint64_t value) -> Result<uint64_t> {
                                                          auto waited = impl_->queue->wait_for_value(value);
                                                          if (!waited) return std::move(waited).error();
                                                          return impl_->queue->completed_value();
                                                      });
        if (!reserved) return std::move(reserved).error();
        staging = reserved.value();
        ring = impl_->readback_ring.get();
    }

    auto api = impl_->device->api();
    auto output = destination;
    const auto size = static_cast<VkDeviceSize>(destination.size());
    const bool cross_queue = impl_->options.cross_queue_concurrent;
    auto record = [api, source, source_offset, size, direct, staging, cross_queue](VkCommandBuffer command) {
        if (direct) {
            if (!cross_queue) {
                vk::HazardTracker hazards;
                (void)hazards.access(vk::AccessKind::ComputeWrite);
                record_hazards(api, command, hazards.access(vk::AccessKind::HostRead));
            }
        } else {
            if (!cross_queue) {
                vk::HazardTracker source_hazards;
                (void)source_hazards.access(vk::AccessKind::ComputeWrite);
                record_hazards(api, command, source_hazards.access(vk::AccessKind::TransferRead));
            }
            vk::HazardTracker staging_hazards;
            (void)staging_hazards.access(vk::AccessKind::HostRead);
            record_hazards(api, command, staging_hazards.access(vk::AccessKind::TransferWrite));
            VkBufferCopy region{};
            region.srcOffset = source_offset;
            region.dstOffset = staging->offset;
            region.size = size;
            api.cmd_copy_buffer(command, source.buffer, staging->buffer, 1, &region);
            record_hazards(api, command, staging_hazards.access(vk::AccessKind::HostRead));
        }
        return ok_status();
    };
    auto complete = [api, device = impl_->device->handle(), source, source_offset, size, direct, staging,
                     ring, output, atom_size = impl_->options.non_coherent_atom_size]() mutable -> Status {
        const std::byte* input = nullptr;
        if (direct) {
            auto invalidated =
                mapped_memory_operation(api, device, source, source_offset, size, atom_size, false);
            if (!invalidated) return std::move(invalidated).error();
            input = source.mapped_data + source_offset;
        } else {
            auto invalidated = ring->invalidate(*staging);
            if (!invalidated) return std::move(invalidated).error();
            input = staging->mapped_data;
        }
        std::memcpy(output.data(), input, output.size());
        return ok_status();
    };
    return enqueue(std::move(record), std::move(complete), staging, ring, dependency, {}, {source.queue_use});
}

Result<Event> TransferEngine::copy(TransferBuffer source, VkDeviceSize source_offset,
                                   TransferBuffer destination, VkDeviceSize destination_offset,
                                   VkDeviceSize size) {
    KORA_TRY_STATUS(validate_buffer_range(source, source_offset, size, "device-copy source"));
    KORA_TRY_STATUS(validate_buffer_range(destination, destination_offset, size, "device-copy destination"));
    if (size == 0) {
        auto promise = std::make_shared<EventPromise>();
        auto event = promise->event();
        promise->complete(ok_status());
        return event;
    }
    if (source.buffer == destination.buffer) {
        const auto source_end = source_offset + size;
        const auto destination_end = destination_offset + size;
        if (source_offset < destination_end && destination_offset < source_end)
            return Error(ErrorCode::InvalidArgument, "overlapping same-buffer copy ranges are not supported");
    }
    auto api = impl_->device->api();
    auto record = [api, source, source_offset, destination, destination_offset,
                   size](VkCommandBuffer command) {
        vk::HazardTracker source_hazards;
        (void)source_hazards.access(vk::AccessKind::ComputeWrite);
        record_hazards(api, command, source_hazards.access(vk::AccessKind::TransferRead));
        vk::HazardTracker destination_hazards;
        (void)destination_hazards.access(vk::AccessKind::ComputeWrite);
        record_hazards(api, command, destination_hazards.access(vk::AccessKind::TransferWrite));
        VkBufferCopy region{};
        region.srcOffset = source_offset;
        region.dstOffset = destination_offset;
        region.size = size;
        api.cmd_copy_buffer(command, source.buffer, destination.buffer, 1, &region);
        vk::HazardTracker source_result;
        (void)source_result.access(vk::AccessKind::TransferRead);
        record_hazards(api, command, source_result.access(vk::AccessKind::ComputeRead));
        record_hazards(api, command, source_result.access(vk::AccessKind::ComputeWrite));
        vk::HazardTracker destination_result;
        (void)destination_result.access(vk::AccessKind::TransferWrite);
        record_hazards(api, command, destination_result.access(vk::AccessKind::ComputeRead));
        record_hazards(api, command, destination_result.access(vk::AccessKind::ComputeWrite));
        return ok_status();
    };
    return enqueue(std::move(record), {}, std::nullopt, nullptr, {}, {},
                   {source.queue_use, destination.queue_use});
}

Result<Event> TransferEngine::fill(TransferBuffer destination, VkDeviceSize offset, VkDeviceSize size,
                                   std::uint32_t value) {
    KORA_TRY_STATUS(validate_buffer_range(destination, offset, size, "buffer fill"));
    if (size == 0 || offset % 4 != 0 || size % 4 != 0)
        return Error(ErrorCode::InvalidArgument,
                     "vkCmdFillBuffer requires a nonzero size and 4-byte alignment");
    auto api = impl_->device->api();
    auto record = [api, destination, offset, size, value](VkCommandBuffer command) {
        vk::HazardTracker hazards;
        (void)hazards.access(vk::AccessKind::ComputeWrite);
        record_hazards(api, command, hazards.access(vk::AccessKind::TransferWrite));
        api.cmd_fill_buffer(command, destination.buffer, offset, size, value);
        record_hazards(api, command, hazards.access(vk::AccessKind::ComputeRead));
        record_hazards(api, command, hazards.access(vk::AccessKind::ComputeWrite));
        return ok_status();
    };
    return enqueue(std::move(record), {}, std::nullopt, nullptr, {}, {}, {destination.queue_use});
}

}  // namespace kora::detail
