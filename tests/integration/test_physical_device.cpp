#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "core/device_caps.hpp"
#include "kora/log.hpp"
#include "memory/memory_pool.hpp"
#include "support/vulkan_test.hpp"
#include "vulkan/api.hpp"

namespace {

class ValidationErrors {
public:
    ValidationErrors() {
        kora::log::set_level(kora::log::Level::Trace);
        kora::log::set_sink(
            [this](kora::log::Level level, kora::log::Category category, std::string_view message) {
                if (category != kora::log::Category::Device || level != kora::log::Level::Error) return;
                std::lock_guard lock(mutex_);
                messages_.emplace_back(message);
            });
    }

    ~ValidationErrors() {
        kora::log::set_sink(nullptr);
        kora::log::set_level(kora::log::Level::Warn);
    }

    std::vector<std::string> messages() const {
        std::lock_guard lock(mutex_);
        return messages_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<std::string> messages_;
};

}  // namespace

TEST(VulkanPhysicalDevice, EnumeratesAndQueriesEveryDevice) {
    ValidationErrors errors;
    auto loader = kora::vk::Loader::load();
    if (!loader) KORA_REQUIRE_VULKAN(loader);

    kora::vk::Loader::InstanceOptions options;
    options.enable_validation = true;
    auto instance = loader.value().create_instance(options);
    if (!instance && instance.error().code == kora::ErrorCode::NoVulkanImplementation)
        KORA_REQUIRE_VULKAN(false);
    ASSERT_TRUE(instance) << instance.error().describe();

    auto devices = instance.value().physical_devices();
    ASSERT_TRUE(devices) << devices.error().describe();
    ASSERT_FALSE(devices.value().empty());

    for (std::size_t i = 0; i < devices.value().size(); ++i) {
        const auto& device = devices.value()[i];
        SCOPED_TRACE(i);
        EXPECT_NE(device.handle, VK_NULL_HANDLE);
        EXPECT_NE(device.properties.apiVersion, 0u);
        EXPECT_NE(device.properties.deviceName[0], '\0');
        EXPECT_LE(device.api_version, instance.value().api_version());
        EXPECT_GE(device.memory_properties.memoryTypeCount, 1u);
        EXPECT_GE(device.memory_properties.memoryHeapCount, 1u);
        EXPECT_FALSE(device.queue_family_properties.empty());
        EXPECT_EQ(device.subgroup_properties.pNext, nullptr);
        EXPECT_EQ(device.vulkan11_features.pNext, nullptr);
        EXPECT_EQ(device.vulkan12_features.pNext, nullptr);
        EXPECT_EQ(device.vulkan13_features.pNext, nullptr);
        EXPECT_EQ(device.vulkan11_properties.pNext, nullptr);
        EXPECT_EQ(device.vulkan12_properties.pNext, nullptr);
        EXPECT_EQ(device.vulkan13_properties.pNext, nullptr);
        EXPECT_EQ(device.timeline_semaphore_features.pNext, nullptr);
        EXPECT_EQ(device.buffer_device_address_features.pNext, nullptr);
        EXPECT_EQ(device.buffer_device_address_features_ext.pNext, nullptr);
        EXPECT_EQ(device.descriptor_indexing_features.pNext, nullptr);
        EXPECT_EQ(device.synchronization2_features.pNext, nullptr);
        EXPECT_EQ(device.subgroup_size_control_features.pNext, nullptr);
        EXPECT_EQ(device.subgroup_size_control_properties.pNext, nullptr);
        EXPECT_EQ(device.memory_priority_features.pNext, nullptr);
#ifdef VK_KHR_device_fault
        EXPECT_EQ(device.device_fault_features.pNext, nullptr);
#endif
#ifdef VK_EXT_device_fault
        EXPECT_EQ(device.device_fault_features_ext.pNext, nullptr);
#endif
        EXPECT_EQ(device.portability_subset_features.pNext, nullptr);
        EXPECT_EQ(device.portability_subset_properties.pNext, nullptr);
        EXPECT_EQ(device.robustness2_features.pNext, nullptr);
        EXPECT_EQ(device.robustness2_properties.pNext, nullptr);
        EXPECT_EQ(device.memory_budget_properties.pNext, nullptr);

        if (device.api_version >= VK_API_VERSION_1_1) {
            EXPECT_EQ(device.vulkan11_features.sType, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
            EXPECT_EQ(device.vulkan11_properties.sType,
                      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES);
        }
        if (device.api_version >= VK_API_VERSION_1_2) {
            EXPECT_EQ(device.vulkan12_features.sType, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
            EXPECT_EQ(device.vulkan12_properties.sType,
                      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES);
        }
        if (device.api_version >= VK_API_VERSION_1_3) {
            EXPECT_EQ(device.vulkan13_features.sType, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
            EXPECT_EQ(device.vulkan13_properties.sType,
                      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES);
        }

        const auto public_info = kora::detail::DeviceCaps::from_vulkan(device).to_device_info();
        EXPECT_EQ(public_info.name(), device.properties.deviceName);
        EXPECT_EQ(public_info.queue_families().size(), device.queue_family_properties.size());
        EXPECT_EQ(public_info.memory_heaps().size(), device.memory_properties.memoryHeapCount);
        EXPECT_EQ(public_info.memory_types().size(), device.memory_properties.memoryTypeCount);

        std::cout << "Physical device " << i << ": " << device.properties.deviceName << " (Vulkan "
                  << VK_VERSION_MAJOR(device.properties.apiVersion) << '.'
                  << VK_VERSION_MINOR(device.properties.apiVersion) << ", " << device.extensions.size()
                  << " device extensions)\n";
    }

    for (const auto& message : errors.messages()) {
        ADD_FAILURE() << message;
    }
}

TEST(VulkanPhysicalDevice, CreatesLogicalDeviceForEveryComputeCapableDevice) {
    ValidationErrors errors;
    auto loader = kora::vk::Loader::load();
    if (!loader) KORA_REQUIRE_VULKAN(loader);

    kora::vk::Loader::InstanceOptions options;
    options.enable_validation = true;
    auto instance = loader.value().create_instance(options);
    if (!instance && instance.error().code == kora::ErrorCode::NoVulkanImplementation)
        KORA_REQUIRE_VULKAN(false);
    ASSERT_TRUE(instance) << instance.error().describe();

    auto devices = instance.value().physical_devices();
    ASSERT_TRUE(devices) << devices.error().describe();
    std::size_t eligible = 0;
    for (const auto& physical_device : devices.value()) {
        const bool has_compute = std::any_of(
            physical_device.queue_family_properties.begin(), physical_device.queue_family_properties.end(),
            [](const VkQueueFamilyProperties& family) {
                return family.queueCount != 0 && (family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
            });
        if (!has_compute) continue;
        ++eligible;
        SCOPED_TRACE(physical_device.properties.deviceName);
        auto device = instance.value().create_device(physical_device);
        ASSERT_TRUE(device) << device.error().describe();
        EXPECT_NE(device.value().handle(), VK_NULL_HANDLE);
        EXPECT_NE(device.value().compute_queue(), VK_NULL_HANDLE);
        EXPECT_NE(device.value().transfer_queue(), VK_NULL_HANDLE);
        EXPECT_TRUE(device.value().api().allocate_memory);
        EXPECT_TRUE(device.value().api().queue_submit);

        for (const auto& extension : device.value().enabled_extensions()) {
            EXPECT_NE(std::find_if(physical_device.extensions.begin(), physical_device.extensions.end(),
                                   [&](const VkExtensionProperties& available) {
                                       return extension == available.extensionName;
                                   }),
                      physical_device.extensions.end());
        }
        const bool portability_advertised = std::any_of(
            physical_device.extensions.begin(), physical_device.extensions.end(),
            [](const VkExtensionProperties& extension) {
                return std::string_view(extension.extensionName) == VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
            });
        EXPECT_EQ(
            std::find(device.value().enabled_extensions().begin(), device.value().enabled_extensions().end(),
                      VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) != device.value().enabled_extensions().end(),
            portability_advertised);
    }
    if (eligible == 0) KORA_REQUIRE_VULKAN(false);

    for (const auto& message : errors.messages()) ADD_FAILURE() << message;
}

TEST(VulkanMemoryPool, TenThousandSmallAllocationsUseBoundedDeviceMemoryBlocks) {
    auto loader = kora::vk::Loader::load();
    if (!loader) KORA_REQUIRE_VULKAN(loader);
    auto instance = loader.value().create_instance();
    if (!instance && instance.error().code == kora::ErrorCode::NoVulkanImplementation)
        KORA_REQUIRE_VULKAN(false);
    ASSERT_TRUE(instance) << instance.error().describe();
    auto physical_devices = instance.value().physical_devices();
    ASSERT_TRUE(physical_devices) << physical_devices.error().describe();

    for (const auto& physical : physical_devices.value()) {
        const auto compute =
            std::find_if(physical.queue_family_properties.begin(), physical.queue_family_properties.end(),
                         [](const auto& queue) { return (queue.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0; });
        if (compute == physical.queue_family_properties.end()) continue;
        auto device = instance.value().create_device(physical);
        ASSERT_TRUE(device) << device.error().describe();

        std::uint32_t type_index = 0;
        for (std::uint32_t i = 0; i < physical.memory_properties.memoryTypeCount; ++i) {
            if ((physical.memory_properties.memoryTypes[i].propertyFlags &
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
                type_index = i;
                break;
            }
        }
        const auto flags = physical.memory_properties.memoryTypes[type_index].propertyFlags;
        const auto heap_index = physical.memory_properties.memoryTypes[type_index].heapIndex;
        kora::detail::MemoryPoolConfig config;
        config.memory_type_index = type_index;
        config.host_visible = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        config.host_coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        config.heap_size = physical.memory_properties.memoryHeaps[heap_index].size;
        config.budget = physical.memory_budget_properties.heapBudget[heap_index];
        auto pool = kora::detail::MemoryPool::create(device.value(), config);
        ASSERT_TRUE(pool) << pool.error().describe();

        std::vector<kora::detail::PoolAllocation> allocations;
        allocations.reserve(10'000);
        for (int i = 0; i < 10'000; ++i) {
            auto allocation = pool.value()->allocate({.size = 256, .alignment = 256});
            ASSERT_TRUE(allocation) << allocation.error().describe();
            EXPECT_EQ(allocation.value().offset % 256, 0u);
            EXPECT_EQ(allocation.value().mapped_data != nullptr, config.host_visible);
            allocations.push_back(allocation.value());
        }
        const auto stats = pool.value()->stats();
        EXPECT_EQ(stats.live_allocations, 10'000u);
        EXPECT_LE(stats.device_allocation_calls, 2u);
        EXPECT_LT(stats.device_allocation_calls, allocations.size());
        for (const auto& allocation : allocations) ASSERT_TRUE(pool.value()->deallocate(allocation));
        EXPECT_EQ(pool.value()->stats().live_allocations, 0u);

        auto dedicated = pool.value()->allocate({.size = 4096, .alignment = 256, .dedicated_required = true});
        ASSERT_TRUE(dedicated) << dedicated.error().describe();
        EXPECT_TRUE(dedicated.value().dedicated);
        EXPECT_EQ(pool.value()->stats().dedicated_allocations, 1u);
        EXPECT_TRUE(pool.value()->deallocate(dedicated.value()));
        break;
    }
}
