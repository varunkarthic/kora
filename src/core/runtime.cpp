#include "kora/runtime.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/device_caps.hpp"
#include "memory/memory_manager.hpp"
#include "memory/memory_pool.hpp"
#include "vulkan/api.hpp"

namespace kora {
namespace {

std::string uuid_string(const VkPhysicalDeviceIDProperties& properties) {
    static constexpr char kHex[] = "0123456789abcdef";
    if (std::all_of(std::begin(properties.deviceUUID), std::end(properties.deviceUUID),
                    [](std::uint8_t byte) { return byte == 0; })) {
        return {};
    }
    std::string value;
    value.reserve(sizeof(properties.deviceUUID) * 2);
    for (const auto byte : properties.deviceUUID) {
        value.push_back(kHex[byte >> 4U]);
        value.push_back(kHex[byte & 0x0fU]);
    }
    return value;
}

DeviceKind device_kind(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return DeviceKind::Integrated;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return DeviceKind::Discrete;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return DeviceKind::Virtual;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return DeviceKind::Cpu;
        default:
            return DeviceKind::Other;
    }
}

}  // namespace

struct Runtime::Impl {
    Impl(vk::Loader loader_value, vk::Instance instance_value)
        : loader(std::move(loader_value)), instance(std::move(instance_value)) {}

    struct DeviceEntry {
        DeviceEntry(DeviceId device_id, DeviceInfo device_info, detail::MemoryManager manager,
                    vk::Device logical_device)
            : id(std::move(device_id)),
              info(std::move(device_info)),
              memory_manager(std::move(manager)),
              device(std::move(logical_device)) {}

        DeviceId id;
        DeviceInfo info;
        detail::MemoryManager memory_manager;
        vk::Device device;
        mutable std::mutex pools_mutex;
        // The buffer/transfer layers register their pools here as those allocation paths are connected.
        std::vector<std::unique_ptr<detail::MemoryPool>> pools;
    };

    vk::Loader loader;
    vk::Instance instance;
    std::vector<std::unique_ptr<DeviceEntry>> devices;
};

Runtime::Runtime() = default;
Runtime::Runtime(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

Result<Runtime> Runtime::create(const RuntimeOptions& options) {
    const auto resolved = resolve(options);
    auto loader_result = vk::Loader::load();
    if (!loader_result) return std::move(loader_result).error();

    auto instance_options = vk::Loader::InstanceOptions{};
    instance_options.enable_validation = resolved.enable_validation;
    auto instance_result = loader_result.value().create_instance(instance_options);
    if (!instance_result) return std::move(instance_result).error();

    auto physical_devices = instance_result.value().physical_devices();
    if (!physical_devices) return std::move(physical_devices).error();

    std::vector<DeviceCandidate> candidates;
    candidates.reserve(physical_devices.value().size());
    for (std::size_t index = 0; index < physical_devices.value().size(); ++index) {
        const auto& physical = physical_devices.value()[index];
        const auto info = detail::DeviceCaps::from_vulkan(physical).to_device_info();
        const auto compute_queues = std::accumulate(
            physical.queue_family_properties.begin(), physical.queue_family_properties.end(),
            std::uint32_t{0}, [](std::uint32_t total, const VkQueueFamilyProperties& family) {
                return (family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 ? total + family.queueCount : total;
            });
        std::uint64_t local_memory = 0;
        for (const auto& heap : info.memory_heaps()) {
            if (heap.device_local) local_memory += heap.size_bytes;
        }
        candidates.push_back(DeviceCandidate{
            .id = DeviceId{static_cast<std::uint32_t>(index), uuid_string(physical.id_properties)},
            .kind = device_kind(physical.properties.deviceType),
            .tier = info.tier(),
            .device_local_memory_bytes = local_memory,
            .compute_queue_count = compute_queues,
            .satisfies_requirements = true,
        });
    }

    auto selected = select_devices(resolved, candidates);
    if (!selected) return std::move(selected).error();

    auto impl = std::make_unique<Impl>(std::move(loader_result).value(), std::move(instance_result).value());
    for (const auto& id : selected.value()) {
        if (id.index >= physical_devices.value().size()) {
            return Error(ErrorCode::InternalInvariantViolation,
                         "device selector returned an index outside the Vulkan enumeration");
        }
        const auto& physical = physical_devices.value()[id.index];
        auto logical_device = impl->instance.create_device(physical);
        if (!logical_device) return std::move(logical_device).error();
        auto info = detail::DeviceCaps::from_vulkan(physical).to_device_info();
        auto memory_manager = detail::MemoryManager::create(physical, info, resolved.memory_budget_fraction);
        if (!memory_manager) return std::move(memory_manager).error();
        impl->devices.push_back(std::make_unique<Impl::DeviceEntry>(
            id, std::move(info), std::move(memory_manager).value(), std::move(logical_device).value()));
    }
    return Runtime(std::move(impl));
}

std::vector<DeviceId> Runtime::devices() const {
    std::vector<DeviceId> ids;
    if (!impl_) return ids;
    ids.reserve(impl_->devices.size());
    for (const auto& device : impl_->devices) ids.push_back(device->id);
    return ids;
}

Result<DeviceInfo> Runtime::device_info(const DeviceId& id) const {
    if (!impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    const auto found = std::find_if(impl_->devices.begin(), impl_->devices.end(),
                                    [&](const auto& device) { return device->id == id; });
    if (found == impl_->devices.end())
        return Error(ErrorCode::InvalidArgument, "device is not owned by runtime");
    return (*found)->info;
}

Result<MemoryStats> Runtime::memory_stats(const DeviceId& id) const {
    if (!impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
    const auto found = std::find_if(impl_->devices.begin(), impl_->devices.end(),
                                    [&](const auto& device) { return device->id == id; });
    if (found == impl_->devices.end())
        return Error(ErrorCode::InvalidArgument, "device is not owned by runtime");
    const auto& entry = **found;
    std::lock_guard lock(entry.pools_mutex);
    std::vector<detail::MemoryPoolStats> pool_stats;
    pool_stats.reserve(entry.pools.size());
    for (const auto& pool : entry.pools) pool_stats.push_back(pool->stats());
    auto aggregate = detail::MemoryManager::aggregate_stats(pool_stats);
    if (!aggregate) return std::move(aggregate).error();
    for (std::uint32_t index = 0; index < entry.info.memory_heaps().size(); ++index) {
        auto budget = entry.memory_manager.budget(index);
        if (!budget) return std::move(budget).error();
        const auto& value = budget.value();
        aggregate.value().heaps.push_back(MemoryHeapStats{index, value.limit_bytes, value.reserved_bytes,
                                                          value.available_bytes, value.externally_used_bytes,
                                                          value.queried});
    }
    return std::move(aggregate).value();
}

}  // namespace kora
