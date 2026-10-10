#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace kora {

namespace detail {
class DeviceCaps;
}

enum class CapabilityTier : std::uint8_t { A, B, C, D };

/// Immutable, vendor-neutral snapshot of a Vulkan physical device's capabilities.
class DeviceInfo final {
public:
    struct ApiVersion {
        std::uint32_t major = 0;
        std::uint32_t minor = 0;
        std::uint32_t patch = 0;
    };

    struct Limits {
        std::array<std::uint32_t, 3> max_compute_work_group_count{};
        std::array<std::uint32_t, 3> max_compute_work_group_size{};
        std::uint32_t max_compute_work_group_invocations = 0;
        std::uint32_t max_compute_shared_memory_size = 0;
        std::uint64_t max_storage_buffer_range = 0;
        std::uint32_t max_push_constants_size = 0;
    };

    struct Features {
        bool robust_buffer_access = false;
        bool shader_int16 = false;
        bool shader_int64 = false;
        bool shader_float64 = false;
        bool timeline_semaphore = false;
        bool buffer_device_address = false;
        bool descriptor_indexing = false;
        bool runtime_descriptor_array = false;
        bool synchronization2 = false;
        bool portability_subset = false;
    };

    struct Subgroup {
        std::uint32_t size = 0;
        bool supported = false;
        bool basic = false;
        bool vote = false;
        bool arithmetic = false;
        bool ballot = false;
        bool shuffle = false;
        bool shuffle_relative = false;
        bool clustered = false;
        bool quad = false;
    };

    struct QueueFamily {
        std::uint32_t queue_count = 0;
        std::uint32_t timestamp_valid_bits = 0;
        std::array<std::uint32_t, 3> min_image_transfer_granularity{};
        bool graphics = false;
        bool compute = false;
        bool transfer = false;
        bool sparse_binding = false;
    };

    struct MemoryHeap {
        std::uint64_t size_bytes = 0;
        bool device_local = false;
        bool multi_instance = false;
    };

    struct MemoryType {
        std::uint32_t heap_index = 0;
        bool device_local = false;
        bool host_visible = false;
        bool host_coherent = false;
        bool host_cached = false;
        bool lazily_allocated = false;
        bool protected_memory = false;
    };

    DeviceInfo(const DeviceInfo&) = default;
    DeviceInfo(DeviceInfo&&) noexcept = default;
    DeviceInfo& operator=(const DeviceInfo&) = delete;
    DeviceInfo& operator=(DeviceInfo&&) = delete;

    const std::string& name() const noexcept { return data_.name; }
    std::uint32_t vendor_id() const noexcept { return data_.vendor_id; }
    std::uint32_t device_id() const noexcept { return data_.device_id; }
    ApiVersion api_version() const noexcept { return data_.api_version; }
    const Limits& limits() const noexcept { return data_.limits; }
    const Features& features() const noexcept { return data_.features; }
    const Subgroup& subgroup() const noexcept { return data_.subgroup; }
    float timestamp_period_nanoseconds() const noexcept { return data_.timestamp_period_nanoseconds; }
    const std::vector<QueueFamily>& queue_families() const noexcept { return data_.queue_families; }
    const std::vector<MemoryHeap>& memory_heaps() const noexcept { return data_.memory_heaps; }
    const std::vector<MemoryType>& memory_types() const noexcept { return data_.memory_types; }
    CapabilityTier tier() const noexcept { return data_.tier; }

private:
    struct Data {
        std::string name;
        std::uint32_t vendor_id = 0;
        std::uint32_t device_id = 0;
        ApiVersion api_version;
        Limits limits;
        Features features;
        Subgroup subgroup;
        float timestamp_period_nanoseconds = 0.0F;
        std::vector<QueueFamily> queue_families;
        std::vector<MemoryHeap> memory_heaps;
        std::vector<MemoryType> memory_types;
        CapabilityTier tier = CapabilityTier::A;
    };

    explicit DeviceInfo(Data data) : data_(std::move(data)) {}

    Data data_;
    friend class detail::DeviceCaps;
};

}  // namespace kora
