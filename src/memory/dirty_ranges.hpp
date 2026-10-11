#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "kora/buffer.hpp"
#include "vulkan/api.hpp"

namespace kora::detail {

/// Sorted, non-empty, non-overlapping dirty byte intervals for one allocation.
class DirtyRangeSet final {
public:
    Status add(DirtyRange range);
    void clear() noexcept { ranges_.clear(); }
    const std::vector<DirtyRange>& ranges() const noexcept { return ranges_; }

    /// Expand each interval to atom boundaries, clamping only the allocation's final endpoint.
    Result<std::vector<DirtyRange>> aligned(std::uint64_t allocation_size,
                                            std::uint64_t non_coherent_atom_size) const;

private:
    std::vector<DirtyRange> ranges_;
};

/// Flush or invalidate mapped Vulkan memory. Coherent allocations are no-ops.
Status flush_mapped_ranges(const vk::Api& api, VkDevice device, VkDeviceMemory memory,
                           std::uint64_t allocation_size, std::uint64_t non_coherent_atom_size,
                           std::span<const DirtyRange> ranges, bool host_coherent);
Status invalidate_mapped_ranges(const vk::Api& api, VkDevice device, VkDeviceMemory memory,
                                std::uint64_t allocation_size, std::uint64_t non_coherent_atom_size,
                                std::span<const DirtyRange> ranges, bool host_coherent);

}  // namespace kora::detail
