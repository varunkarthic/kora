#pragma once

#include "core/calibration.hpp"
#include "core/performance_db.hpp"
#include "kora/runtime.hpp"

namespace kora::detail {

/// Internal test seam for exercising recovery without relying on physical device failure.
struct RuntimeTestAccess {
    static Status mark_device_lost(Runtime& runtime, const DeviceId& id);
    static Status set_calibration_record(Runtime& runtime, const DeviceId& id, calibration::Record record);
    static Result<std::string> performance_observation_key(const Runtime& runtime, const DeviceId& id,
                                                           const Kernel& kernel,
                                                           const AdaptiveDispatchWorkload& workload,
                                                           PerformanceTimingDomain domain);
    static Result<PerformanceStats> performance_stats(const Runtime& runtime, const std::string& key);
};

}  // namespace kora::detail
