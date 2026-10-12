#pragma once

#include "core/calibration.hpp"
#include "kora/runtime.hpp"

namespace kora::detail {

/// Internal test seam for exercising recovery without relying on physical device failure.
struct RuntimeTestAccess {
    static Status mark_device_lost(Runtime& runtime, const DeviceId& id);
    static Status set_calibration_record(Runtime& runtime, const DeviceId& id, calibration::Record record);
};

}  // namespace kora::detail
