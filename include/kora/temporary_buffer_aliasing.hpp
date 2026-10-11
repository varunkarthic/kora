#pragma once
/// @file temporary_buffer_aliasing.hpp
/// Conservative temporary-buffer lifetime planning for finalized task DAGs.

#include <cstdint>
#include <span>
#include <vector>

#include "kora/task_graph.hpp"

namespace kora {

/// Requirements supplied by the resource-creation layer for one graph temporary.
struct TemporaryBufferRequirements {
    BufferId buffer;
    std::uint64_t size_bytes = 0;
    std::uint64_t alignment = 1;
    /// Exact compatibility class (for example, a normalized Vulkan buffer-usage combination).
    std::uint32_t usage_class = 0;
    /// Vulkan-style memory type bit mask; aliases must have a non-empty intersection.
    std::uint32_t memory_type_bits = 0;
    /// Persistent/user-visible resources get a private allocation and are never aliased.
    bool persistent = false;
};

struct TemporaryBufferPlacement {
    BufferId buffer;
    std::uint32_t allocation_id = 0;
    std::uint64_t offset = 0;
};

struct TemporaryAllocationPlan {
    std::uint32_t id = 0;
    std::uint64_t size_bytes = 0;
    std::uint64_t alignment = 1;
    std::uint32_t usage_class = 0;
    std::uint32_t memory_type_bits = 0;
    bool persistent_exclusive = false;
};

struct TemporaryBufferAliasingPlan {
    std::vector<TemporaryBufferPlacement> placements;
    std::vector<TemporaryAllocationPlan> allocations;
    std::uint64_t unaliased_bytes = 0;
    std::uint64_t planned_bytes = 0;

    std::uint64_t planned_savings_bytes() const noexcept {
        return unaliased_bytes >= planned_bytes ? unaliased_bytes - planned_bytes : 0;
    }
};

/// Colors graph-temporary lifetimes into compatible allocation slots. Reuse is allowed only when
/// every use of one buffer happens-before every use of the other in the DAG.
Result<TemporaryBufferAliasingPlan> plan_temporary_buffer_aliasing(
    const TaskGraphPlan& graph, std::span<const TemporaryBufferRequirements> buffers);

}  // namespace kora
