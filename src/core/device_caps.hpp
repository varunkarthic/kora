#pragma once

#include <vector>

#include "kora/device_info.hpp"
#include "vulkan/api.hpp"

namespace kora::detail {

enum class MemoryPolicy : std::uint8_t {
    Auto,
    DevicePreferred,
    HostVisible,
    ReadMostly,
    WriteMostly,
    Streaming,
    Readback,
};

struct MemoryClassification {
    MemoryTopology topology = MemoryTopology::Unknown;
    bool needs_benchmark = false;
    DeviceInfo::MemoryTypePreferences type_preferences;
};

/// Converts raw Vulkan queries into KORA's immutable, vendor-neutral device snapshot.
class DeviceCaps final {
public:
    static DeviceCaps from_vulkan(const vk::PhysicalDeviceInfo& physical_device);
    static std::vector<DeviceInfo::QueueFamily> classify_queue_families(
        const std::vector<VkQueueFamilyProperties>& queue_families);
    static MemoryClassification classify_memory_properties(
        const VkPhysicalDeviceMemoryProperties& memory_properties);

    CapabilityTier tier() const noexcept { return data_.tier; }
    DeviceInfo to_device_info() const { return DeviceInfo(data_); }

private:
    explicit DeviceCaps(DeviceInfo::Data data) : data_(std::move(data)) {}

    DeviceInfo::Data data_;
};

}  // namespace kora::detail
