#include "vulkan/hazard_tracker.hpp"

namespace kora::vk {

AccessScope access_scope(AccessKind kind) noexcept {
    switch (kind) {
        case AccessKind::HostRead:
            return {VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT, false};
        case AccessKind::HostWrite:
            return {VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT, true};
        case AccessKind::TransferRead:
            return {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT, false};
        case AccessKind::TransferWrite:
            return {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, true};
        case AccessKind::ComputeRead:
            return {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, false};
        case AccessKind::ComputeWrite:
            return {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, true};
    }
    return {};
}

bool accesses_conflict(AccessScope previous, AccessScope next) noexcept {
    return previous.writes || next.writes;
}

std::vector<BarrierPlan> HazardTracker::access(AccessKind kind) {
    const auto next = access_scope(kind);
    std::vector<BarrierPlan> barriers;
    VkPipelineStageFlags source_stage = 0;
    VkAccessFlags source_access = 0;
    if (has_writer_ && accesses_conflict(writer_, next)) {
        source_stage |= writer_.stage;
        source_access |= writer_.access;
    }
    if (next.writes) {
        for (const auto& reader : readers_) {
            if (!accesses_conflict(reader, next)) continue;
            source_stage |= reader.stage;
            source_access |= reader.access;
        }
    }
    if (source_stage != 0) barriers.push_back({source_stage, source_access, next.stage, next.access});

    if (next.writes) {
        writer_ = next;
        has_writer_ = true;
        readers_.clear();
    } else {
        readers_.push_back(next);
    }
    return barriers;
}

}  // namespace kora::vk
