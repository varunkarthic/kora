#pragma once

#include <cstdint>
#include <vector>

#include "vulkan/api.hpp"

namespace kora::vk {

/// A queue-independent description of an access to a buffer.
enum class AccessKind : std::uint8_t {
    HostRead,
    HostWrite,
    TransferRead,
    TransferWrite,
    ComputeRead,
    ComputeWrite
};

/// The Vulkan synchronization scope corresponding to an access.
struct AccessScope {
    VkPipelineStageFlags stage = 0;
    VkAccessFlags access = 0;
    bool writes = false;
    friend bool operator==(const AccessScope&, const AccessScope&) = default;
};

/// One minimal legacy barrier plan. The same logical scopes can be lowered to synchronization2.
struct BarrierPlan {
    VkPipelineStageFlags source_stage = 0;
    VkAccessFlags source_access = 0;
    VkPipelineStageFlags destination_stage = 0;
    VkAccessFlags destination_access = 0;

    friend bool operator==(const BarrierPlan&, const BarrierPlan&) = default;
};

/// Returns whether two accesses to the same buffer require an ordering dependency.
bool accesses_conflict(AccessScope previous, AccessScope next) noexcept;

/// Tracks the last writer and readers since that writer for one logical buffer.
class HazardTracker final {
public:
    /// Adds an access and returns the barriers required to order it after conflicting accesses.
    std::vector<BarrierPlan> access(AccessKind kind);
    /// For tests/debugging, returns the stored latest writer and reader scopes.
    const std::vector<AccessScope>& active_readers() const noexcept { return readers_; }
    bool has_writer() const noexcept { return has_writer_; }
    AccessScope last_writer() const noexcept { return writer_; }

private:
    AccessScope writer_{};
    bool has_writer_ = false;
    std::vector<AccessScope> readers_;
};

/// Maps a high-level access class to its precise Vulkan stage/access scope.
AccessScope access_scope(AccessKind kind) noexcept;

}  // namespace kora::vk
