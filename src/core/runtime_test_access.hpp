#pragma once

#include "kora/runtime.hpp"

namespace kora::detail {

/// Internal test seam for exercising recovery without relying on physical device failure.
struct RuntimeTestAccess {
    static Status mark_device_lost(Runtime& runtime, const DeviceId& id);
};

}  // namespace kora::detail
