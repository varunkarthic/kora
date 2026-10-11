#pragma once

#include <cstdint>
#include <optional>

#include "vulkan/hazard_tracker.hpp"

namespace kora::vk {

/// Cached immutable choice of queue-family ownership strategy for a logical device.
struct QueueFamilyPolicy {
    std::uint32_t compute_family = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t transfer_family = VK_QUEUE_FAMILY_IGNORED;
    bool concurrent_sharing = true;

    bool requires_ownership_transfer() const noexcept {
        return compute_family != VK_QUEUE_FAMILY_IGNORED && transfer_family != VK_QUEUE_FAMILY_IGNORED &&
               compute_family != transfer_family && !concurrent_sharing;
    }

    bool uses_concurrent_sharing() const noexcept {
        return compute_family != VK_QUEUE_FAMILY_IGNORED && transfer_family != VK_QUEUE_FAMILY_IGNORED &&
               compute_family != transfer_family && concurrent_sharing;
    }
};

/// Queue-family release/acquire barriers for an exclusive buffer handoff.
struct OwnershipBarrierPair {
    VkBufferMemoryBarrier2 release{};
    VkBufferMemoryBarrier2 acquire{};
    VkBufferMemoryBarrier release_legacy{};
    VkBufferMemoryBarrier acquire_legacy{};
    VkPipelineStageFlags release_legacy_source_stage = 0;
    VkPipelineStageFlags acquire_legacy_destination_stage = 0;
    std::uint32_t source_family = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t destination_family = VK_QUEUE_FAMILY_IGNORED;
    bool requires_semaphore_dependency = false;
};

/// Produces no barriers for same-family use and a semaphore-ordered pair otherwise.
std::optional<OwnershipBarrierPair> make_ownership_barrier_pair(VkBuffer buffer, VkDeviceSize offset,
                                                                VkDeviceSize size, AccessScope source,
                                                                AccessScope destination,
                                                                std::uint32_t source_family,
                                                                std::uint32_t destination_family) noexcept;

}  // namespace kora::vk
