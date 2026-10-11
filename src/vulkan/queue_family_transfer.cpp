#include "vulkan/queue_family_transfer.hpp"

namespace kora::vk {

std::optional<OwnershipBarrierPair> make_ownership_barrier_pair(VkBuffer buffer, VkDeviceSize offset,
                                                                VkDeviceSize size, AccessScope source,
                                                                AccessScope destination,
                                                                std::uint32_t source_family,
                                                                std::uint32_t destination_family) noexcept {
    if (buffer == VK_NULL_HANDLE || size == 0 || source_family == destination_family ||
        source_family == VK_QUEUE_FAMILY_IGNORED || destination_family == VK_QUEUE_FAMILY_IGNORED) {
        return std::nullopt;
    }

    OwnershipBarrierPair pair;
    pair.source_family = source_family;
    pair.destination_family = destination_family;
    pair.requires_semaphore_dependency = true;

    pair.release.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    pair.release.srcStageMask = source.stage;
    pair.release.srcAccessMask = source.access;
    pair.release.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    pair.release.dstAccessMask = VK_ACCESS_2_NONE;
    pair.release.srcQueueFamilyIndex = source_family;
    pair.release.dstQueueFamilyIndex = destination_family;
    pair.release.buffer = buffer;
    pair.release.offset = offset;
    pair.release.size = size;

    pair.release_legacy.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    pair.release_legacy.srcAccessMask = source.access;
    pair.release_legacy.dstAccessMask = 0;
    pair.release_legacy.srcQueueFamilyIndex = source_family;
    pair.release_legacy.dstQueueFamilyIndex = destination_family;
    pair.release_legacy.buffer = buffer;
    pair.release_legacy.offset = offset;
    pair.release_legacy.size = size;
    pair.release_legacy_source_stage = source.stage;

    pair.acquire.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    pair.acquire.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    pair.acquire.srcAccessMask = VK_ACCESS_2_NONE;
    pair.acquire.dstStageMask = destination.stage;
    pair.acquire.dstAccessMask = destination.access;
    pair.acquire.srcQueueFamilyIndex = source_family;
    pair.acquire.dstQueueFamilyIndex = destination_family;
    pair.acquire.buffer = buffer;
    pair.acquire.offset = offset;
    pair.acquire.size = size;

    pair.acquire_legacy.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    pair.acquire_legacy.srcAccessMask = 0;
    pair.acquire_legacy.dstAccessMask = destination.access;
    pair.acquire_legacy.srcQueueFamilyIndex = source_family;
    pair.acquire_legacy.dstQueueFamilyIndex = destination_family;
    pair.acquire_legacy.buffer = buffer;
    pair.acquire_legacy.offset = offset;
    pair.acquire_legacy.size = size;
    pair.acquire_legacy_destination_stage = destination.stage;
    return pair;
}

}  // namespace kora::vk
