#include "memory/dirty_ranges.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

#include "kora/detail/checked.hpp"

namespace kora::detail {
namespace {

Result<std::vector<VkMappedMemoryRange>> vk_ranges(VkDevice device, VkDeviceMemory memory,
                                                   std::uint64_t allocation_size, std::uint64_t atom_size,
                                                   std::span<const DirtyRange> ranges, bool host_coherent) {
    if (host_coherent || ranges.empty()) return std::vector<VkMappedMemoryRange>{};
    if (device == VK_NULL_HANDLE || memory == VK_NULL_HANDLE) {
        return Error(ErrorCode::InvalidArgument,
                     "mapped-memory operation requires live device and memory handles");
    }
    if (allocation_size > std::numeric_limits<VkDeviceSize>::max()) {
        return Error(ErrorCode::Overflow, "mapped allocation size exceeds VkDeviceSize");
    }
    DirtyRangeSet dirty;
    for (const auto range : ranges) {
        auto status = dirty.add(range);
        if (!status) return status.error();
    }
    auto aligned = dirty.aligned(allocation_size, atom_size);
    if (!aligned) return aligned.error();
    std::vector<VkMappedMemoryRange> vk_ranges;
    try {
        vk_ranges.reserve(aligned.value().size());
        for (const auto range : aligned.value()) {
            VkMappedMemoryRange mapped{};
            mapped.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            mapped.memory = memory;
            mapped.offset = static_cast<VkDeviceSize>(range.offset);
            const auto end = range.offset + range.size;
            mapped.size = end == allocation_size ? VK_WHOLE_SIZE : static_cast<VkDeviceSize>(range.size);
            vk_ranges.push_back(mapped);
        }
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to prepare Vulkan mapped-memory ranges");
    } catch (const std::length_error&) {
        return Error(ErrorCode::Overflow, "Vulkan mapped-memory range count exceeds container capacity");
    }
    return vk_ranges;
}

Status mapped_memory_operation(const vk::Api& api, VkDevice device, VkDeviceMemory memory,
                               std::uint64_t allocation_size, std::uint64_t atom_size,
                               std::span<const DirtyRange> ranges, bool host_coherent, bool flush) {
    auto prepared = vk_ranges(device, memory, allocation_size, atom_size, ranges, host_coherent);
    if (!prepared) return prepared.error();
    auto& mapped = prepared.value();
    if (mapped.empty()) return ok_status();
    if (mapped.size() > std::numeric_limits<std::uint32_t>::max()) {
        return Error(ErrorCode::Overflow, "mapped-memory range count exceeds Vulkan limit");
    }
    if ((flush && !api.flush_mapped_memory_ranges) || (!flush && !api.invalidate_mapped_memory_ranges)) {
        return Error(ErrorCode::UnsupportedFeature,
                     flush ? "Vulkan dispatch does not provide vkFlushMappedMemoryRanges"
                           : "Vulkan dispatch does not provide vkInvalidateMappedMemoryRanges");
    }
    const auto result = flush ? api.flush_mapped_memory_ranges(
                                    device, static_cast<std::uint32_t>(mapped.size()), mapped.data())
                              : api.invalidate_mapped_memory_ranges(
                                    device, static_cast<std::uint32_t>(mapped.size()), mapped.data());
    if (result != VK_SUCCESS) {
        return Error(ErrorCode::TransferFailed,
                     std::string(flush ? "vkFlushMappedMemoryRanges" : "vkInvalidateMappedMemoryRanges") +
                         " failed")
            .with_vk(result);
    }
    return ok_status();
}

}  // namespace

Status DirtyRangeSet::add(DirtyRange range) {
    if (range.size == 0) return ok_status();
    if (!checked::add(range.offset, range.size)) {
        return Error(ErrorCode::Overflow, "dirty range endpoint overflows uint64_t");
    }
    const auto start = range.offset;
    const auto end = range.offset + range.size;
    try {
        auto first = std::lower_bound(ranges_.begin(), ranges_.end(), start,
                                      [](const DirtyRange& current, std::uint64_t value) {
                                          return current.offset + current.size < value;
                                      });
        auto merged_start = start;
        auto merged_end = end;
        auto last = first;
        while (last != ranges_.end() && last->offset <= merged_end) {
            merged_start = std::min(merged_start, last->offset);
            merged_end = std::max(merged_end, last->offset + last->size);
            ++last;
        }
        const auto merged_size = merged_end - merged_start;
        first = ranges_.erase(first, last);
        ranges_.insert(first, DirtyRange{merged_start, merged_size});
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to store dirty-range metadata");
    } catch (const std::length_error&) {
        return Error(ErrorCode::Overflow, "dirty-range metadata exceeds container capacity");
    }
    return ok_status();
}

Result<std::vector<DirtyRange>> DirtyRangeSet::aligned(std::uint64_t allocation_size,
                                                       std::uint64_t atom_size) const {
    if (atom_size == 0 || !checked::is_pow2(atom_size)) {
        return Error(ErrorCode::InvalidArgument, "non-coherent atom size must be a non-zero power of two");
    }
    std::vector<DirtyRange> result;
    try {
        result.reserve(ranges_.size());
        for (const auto range : ranges_) {
            if (!checked::in_range(range.offset, range.size, allocation_size)) {
                return Error(ErrorCode::InvalidArgument, "dirty range exceeds mapped allocation");
            }
            const auto end = range.offset + range.size;
            const auto aligned_start = *checked::align_down(range.offset, atom_size);
            auto aligned_end = checked::align_up(end, atom_size);
            if (!aligned_end) {
                if (end <= allocation_size)
                    aligned_end = allocation_size;
                else
                    return Error(ErrorCode::Overflow, "aligned dirty range endpoint overflows uint64_t");
            }
            const auto clamped_end = std::min(*aligned_end, allocation_size);
            if (!result.empty() && aligned_start <= result.back().offset + result.back().size) {
                const auto previous_end = result.back().offset + result.back().size;
                result.back().size = std::max(previous_end, clamped_end) - result.back().offset;
            } else {
                result.push_back({aligned_start, clamped_end - aligned_start});
            }
        }
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to allocate aligned dirty-range metadata");
    } catch (const std::length_error&) {
        return Error(ErrorCode::Overflow, "aligned dirty-range metadata exceeds container capacity");
    }
    return result;
}

Status flush_mapped_ranges(const vk::Api& api, VkDevice device, VkDeviceMemory memory,
                           std::uint64_t allocation_size, std::uint64_t atom_size,
                           std::span<const DirtyRange> ranges, bool host_coherent) {
    return mapped_memory_operation(api, device, memory, allocation_size, atom_size, ranges, host_coherent,
                                   true);
}

Status invalidate_mapped_ranges(const vk::Api& api, VkDevice device, VkDeviceMemory memory,
                                std::uint64_t allocation_size, std::uint64_t atom_size,
                                std::span<const DirtyRange> ranges, bool host_coherent) {
    return mapped_memory_operation(api, device, memory, allocation_size, atom_size, ranges, host_coherent,
                                   false);
}

}  // namespace kora::detail
