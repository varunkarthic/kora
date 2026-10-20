#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "kora/event.hpp"
#include "memory/memory_pool.hpp"
#include "memory/staging_ring.hpp"
#include "vulkan/queue_context.hpp"

namespace kora::detail {

/// Per-buffer queue bookkeeping shared by every TransferBuffer copy of one runtime buffer.
struct QueueTimelineUse {
    /// Last transfer-queue timeline value that accessed a cross-queue runtime buffer.
    std::atomic<std::uint64_t> transfer_value{0};
    /// GPU operations that are queued or executing and may access the buffer. Host code may
    /// write or read the mapped allocation directly only while this is zero.
    std::atomic<std::uint32_t> pending{0};
};

/// Holds QueueTimelineUse::pending raised for each listed buffer until destroyed. Capture it in a
/// submission's completion callback so it is released after completion, failure, or cancellation.
class PendingUses final {
public:
    explicit PendingUses(std::vector<std::shared_ptr<QueueTimelineUse>> uses) noexcept
        : uses_(std::move(uses)) {
        for (const auto& use : uses_)
            if (use) use->pending.fetch_add(1, std::memory_order_acq_rel);
    }
    ~PendingUses() {
        for (const auto& use : uses_)
            if (use) use->pending.fetch_sub(1, std::memory_order_acq_rel);
    }
    PendingUses(const PendingUses&) = delete;
    PendingUses& operator=(const PendingUses&) = delete;

private:
    std::vector<std::shared_ptr<QueueTimelineUse>> uses_;
};

/// Vulkan buffer plus optional mapped allocation metadata for the UMA direct path.
struct TransferBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize memory_offset = 0;
    VkDeviceSize memory_allocation_size = 0;
    std::byte* mapped_data = nullptr;  // Points at byte zero of this buffer.
    bool host_visible = false;
    bool host_coherent = false;
    std::shared_ptr<QueueTimelineUse> queue_use;
};

/// Flushes host writes to, or invalidates the host view of, a mapped buffer range. Coherent memory
/// needs neither, so this returns immediately for it.
Status mapped_memory_operation(const vk::Api& api, VkDevice device, const TransferBuffer& buffer,
                               VkDeviceSize offset, VkDeviceSize size, VkDeviceSize atom_size, bool flush);

struct QueueWait {
    VkSemaphore semaphore = VK_NULL_HANDLE;
    std::uint64_t value = 0;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
};

/// Internal asynchronous upload/download/copy/fill tasks sharing QueueContext submission ordering.
class TransferEngine final {
public:
    struct Options {
        std::uint32_t queue_family = 0;
        std::size_t command_buffer_count = 32;
        VkDeviceSize staging_initial_capacity = 4U * 1024U * 1024U;
        VkDeviceSize staging_maximum_capacity = 128U * 1024U * 1024U;
        MemoryTopology topology = MemoryTopology::Unknown;
        bool measured_staging_benefit = false;
        VkDeviceSize non_coherent_atom_size = 1;
        bool cross_queue_concurrent = false;
    };

    static Result<std::unique_ptr<TransferEngine>> create(const vk::Device& device, vk::QueueContext& queue,
                                                          MemoryPool& upload_pool, MemoryPool& readback_pool,
                                                          Options options);
    ~TransferEngine();
    TransferEngine(const TransferEngine&) = delete;
    TransferEngine& operator=(const TransferEngine&) = delete;

    Result<Event> upload(TransferBuffer destination, VkDeviceSize destination_offset,
                         std::span<const std::byte> source);
    Result<Event> download(TransferBuffer source, VkDeviceSize source_offset,
                           std::span<std::byte> destination, QueueWait dependency = {});
    Result<Event> copy(TransferBuffer source, VkDeviceSize source_offset, TransferBuffer destination,
                       VkDeviceSize destination_offset, VkDeviceSize size);
    Result<Event> fill(TransferBuffer destination, VkDeviceSize offset, VkDeviceSize size,
                       std::uint32_t value);

private:
    struct Impl;
    explicit TransferEngine(std::shared_ptr<Impl> impl);
    Result<Event> enqueue(std::function<Status(VkCommandBuffer)> record, std::function<Status()> complete,
                          std::optional<StagingSlice> staging, StagingRing* ring, QueueWait dependency = {},
                          std::function<void(std::uint64_t)> on_submitted = {},
                          std::vector<std::shared_ptr<QueueTimelineUse>> uses = {});
    std::shared_ptr<Impl> impl_;
};

}  // namespace kora::detail
