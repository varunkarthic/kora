#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "core/calibration.hpp"
#include "core/performance_db.hpp"
#include "kora/runtime.hpp"

namespace kora::detail {

struct RuntimePlacementCandidate {
    DeviceId id;
    std::uint32_t driver_version = 0;
    calibration::Record calibration_record;
    std::optional<PerformanceStats> online_execution;
    std::uint64_t available_device_memory_bytes = 0;
    bool supported = true;
    bool healthy = true;
    bool allowed = true;
};

Result<DeviceId> select_adaptive_device(const std::vector<RuntimePlacementCandidate>& candidates,
                                        const DeviceId& current, const AdaptiveDispatchWorkload& workload);

}  // namespace kora::detail
