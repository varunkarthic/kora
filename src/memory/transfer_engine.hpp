#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>

#include "kora/event.hpp"
#include "memory/memory_pool.hpp"
#include "memory/staging_ring.hpp"
#include "vulkan/queue_context.hpp"

namespace kora::detail {

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
                           std::span<std::byte> destination);
    Result<Event> copy(TransferBuffer source, VkDeviceSize source_offset, TransferBuffer destination,
                       VkDeviceSize destination_offset, VkDeviceSize size);
    Result<Event> fill(TransferBuffer destination, VkDeviceSize offset, VkDeviceSize size,
                       std::uint32_t value);

private:
    struct Impl;
    explicit TransferEngine(std::shared_ptr<Impl> impl);
    Result<Event> enqueue(std::function<Status(VkCommandBuffer)> record, std::function<Status()> complete,
                          std::optional<StagingSlice> staging, StagingRing* ring);
    std::shared_ptr<Impl> impl_;
};

}  // namespace kora::detail
