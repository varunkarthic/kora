#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "kora/device_info.hpp"
#include "memory/memory_pool.hpp"

namespace kora::detail {

enum class StagingDirection : std::uint8_t { Upload, Readback };

/// A byte range in a persistently mapped staging buffer. The range stays reserved until cancel() or
/// until its release timeline value has completed and a later allocation reclaims it.
struct StagingSlice {
    std::uint64_t reservation = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    std::uint64_t reserved_size = 0;
    VkDeviceSize memory_offset = 0;
    std::byte* mapped_data = nullptr;
};

/// Bounded collection of persistently mapped Vulkan staging buffers with timeline-safe ring ranges.
class StagingRing final {
public:
    struct Options {
        StagingDirection direction = StagingDirection::Upload;
        VkDeviceSize initial_capacity = 4U * 1024U * 1024U;
        VkDeviceSize maximum_capacity = 64U * 1024U * 1024U;
        VkBufferUsageFlags additional_usage = 0;
        MemoryTopology topology = MemoryTopology::Unknown;
        bool measured_staging_benefit = false;
        VkDeviceSize non_coherent_atom_size = 1;
    };

    /// Waits for a release value and returns an updated completed-timeline snapshot.
    using WaitForTimeline = std::function<Result<std::uint64_t>(std::uint64_t)>;

    static Result<std::unique_ptr<StagingRing>> create(const vk::Device& device, MemoryPool& host_pool,
                                                       Options options);
    ~StagingRing();
    StagingRing(const StagingRing&) = delete;
    StagingRing& operator=(const StagingRing&) = delete;

    /// Reserves a non-overlapping range. Reclaims completed ranges, grows within maximum_capacity, or
    /// calls wait_for on the oldest submitted range before retrying. Unsubmitted reservations are never
    /// waited on and can therefore produce AllocationFailed when they fill the ring.
    Result<StagingSlice> reserve(VkDeviceSize size, VkDeviceSize alignment, std::uint64_t completed_timeline,
                                 const WaitForTimeline& wait_for);
    Status retire(const StagingSlice& slice, std::uint64_t release_timeline);
    Status cancel(const StagingSlice& slice);
    Status flush(const StagingSlice& slice) const;
    Status invalidate(const StagingSlice& slice) const;
    /// Marks a readback slice's bytes as copied to the host, so it can be reclaimed once its
    /// timeline completes. Until then a readback slice is never reclaimed or reused.
    Status consumed(const StagingSlice& slice);
    /// Reclaims submitted reservations whose release timeline is at or below this value.
    Status reclaim_completed(std::uint64_t completed_timeline);

    static bool should_use_staging(MemoryTopology topology, bool measured_staging_benefit = false) noexcept;
    VkDeviceSize capacity() const noexcept;
    StagingDirection direction() const noexcept { return options_.direction; }

private:
    struct Segment;
    struct Block;
    StagingRing(const vk::Device& device, MemoryPool& host_pool, Options options);
    Result<void> add_block(VkDeviceSize capacity);
    void reclaim(std::uint64_t completed_timeline);

    const vk::Device* device_ = nullptr;
    MemoryPool* host_pool_ = nullptr;
    Options options_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Block>> blocks_;
    std::uint64_t next_reservation_ = 1;
    VkDeviceSize capacity_ = 0;
};

}  // namespace kora::detail
