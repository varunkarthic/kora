#pragma once
/// @file memory_stats.hpp
/// Stable, Vulkan-header-free memory telemetry returned by Runtime.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kora {

struct MemoryHeapStats {
    std::uint32_t heap_index = 0;
    std::uint64_t budget_bytes = 0;
    std::uint64_t reserved_bytes = 0;
    std::uint64_t available_bytes = 0;
    std::uint64_t externally_used_bytes = 0;
    bool budget_queried = false;
};

/// Point-in-time sum of the tracked device's memory pools.
struct MemoryStats {
    std::uint64_t reserved_bytes = 0;
    std::uint64_t allocated_bytes = 0;
    std::uint64_t requested_bytes = 0;
    std::uint64_t free_bytes = 0;
    std::uint64_t internal_fragmentation_bytes = 0;
    std::uint64_t external_fragmentation_bytes = 0;
    std::uint64_t largest_free_block = 0;
    std::uint64_t high_water_mark = 0;
    std::size_t pools = 0;
    std::size_t blocks = 0;
    std::size_t dedicated_allocations = 0;
    std::size_t live_allocations = 0;
    std::uint64_t device_allocation_calls = 0;
    std::vector<MemoryHeapStats> heaps;
};

}  // namespace kora
