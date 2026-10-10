#pragma once

#include "kora/device_info.hpp"
#include "vulkan/api.hpp"

namespace kora::detail {

/// Converts raw Vulkan queries into KORA's immutable, vendor-neutral device snapshot.
class DeviceCaps final {
public:
    static DeviceCaps from_vulkan(const vk::PhysicalDeviceInfo& physical_device);

    CapabilityTier tier() const noexcept { return data_.tier; }
    DeviceInfo to_device_info() const { return DeviceInfo(data_); }

private:
    explicit DeviceCaps(DeviceInfo::Data data) : data_(std::move(data)) {}

    DeviceInfo::Data data_;
};

}  // namespace kora::detail
