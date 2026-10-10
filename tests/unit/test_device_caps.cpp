#include <gtest/gtest.h>

#include <cstring>
#include <type_traits>
#include <utility>

#include "core/device_caps.hpp"

static_assert(!std::is_default_constructible_v<kora::DeviceInfo>);
static_assert(!std::is_copy_assignable_v<kora::DeviceInfo>);
static_assert(!std::is_move_assignable_v<kora::DeviceInfo>);
static_assert(std::is_same_v<decltype(std::declval<const kora::DeviceInfo&>().limits()),
                             const kora::DeviceInfo::Limits&>);
static_assert(std::is_same_v<decltype(std::declval<const kora::DeviceInfo&>().features()),
                             const kora::DeviceInfo::Features&>);

namespace {

kora::vk::PhysicalDeviceInfo make_device() {
    kora::vk::PhysicalDeviceInfo device;
    std::memcpy(device.properties.deviceName, "Synthetic Vulkan device", sizeof("Synthetic Vulkan device"));
    device.properties.vendorID = 0x1234;
    device.properties.deviceID = 0x5678;
    device.properties.apiVersion = VK_API_VERSION_1_3;
    device.properties.limits.maxComputeWorkGroupCount[0] = 65535;
    device.properties.limits.maxComputeWorkGroupCount[1] = 65535;
    device.properties.limits.maxComputeWorkGroupCount[2] = 65535;
    device.properties.limits.maxComputeWorkGroupSize[0] = 1024;
    device.properties.limits.maxComputeWorkGroupSize[1] = 1024;
    device.properties.limits.maxComputeWorkGroupSize[2] = 64;
    device.properties.limits.maxComputeWorkGroupInvocations = 1024;
    device.properties.limits.maxComputeSharedMemorySize = 32768;
    device.properties.limits.maxStorageBufferRange = 1U << 30;
    device.properties.limits.maxPushConstantsSize = 256;
    device.properties.limits.timestampPeriod = 2.5F;
    device.features.robustBufferAccess = VK_TRUE;
    device.features.shaderInt16 = VK_TRUE;
    device.features.shaderInt64 = VK_TRUE;
    device.features.shaderFloat64 = VK_TRUE;

    device.subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    device.subgroup_properties.subgroupSize = 32;
    device.subgroup_properties.supportedOperations =
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT;

    device.vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    device.vulkan13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

    device.memory_properties.memoryHeapCount = 1;
    device.memory_properties.memoryHeaps[0].size = 8U * 1024U * 1024U * 1024U;
    device.memory_properties.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    device.memory_properties.memoryTypeCount = 1;
    device.memory_properties.memoryTypes[0].heapIndex = 0;
    device.memory_properties.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkQueueFamilyProperties queue_family{};
    queue_family.queueCount = 2;
    queue_family.queueFlags = VK_QUEUE_COMPUTE_BIT;
    queue_family.timestampValidBits = 64;
    queue_family.minImageTransferGranularity = {1, 2, 3};
    device.queue_family_properties.push_back(queue_family);
    return device;
}

void enable_tier_b(kora::vk::PhysicalDeviceInfo& device) {
    device.vulkan12_features.timelineSemaphore = VK_TRUE;
}

}  // namespace

TEST(DeviceCaps, StartsAtBaselineAndDoesNotRequireHigherTierFeatures) {
    auto raw = make_device();
    const auto caps = kora::detail::DeviceCaps::from_vulkan(raw);

    EXPECT_EQ(caps.tier(), kora::CapabilityTier::A);
    const auto info = caps.to_device_info();
    EXPECT_EQ(info.tier(), kora::CapabilityTier::A);
    EXPECT_FALSE(info.features().timeline_semaphore);
    EXPECT_FALSE(info.features().buffer_device_address);
}

TEST(DeviceCaps, AssignsAdditiveCapabilityTiers) {
    auto raw = make_device();
    enable_tier_b(raw);

    EXPECT_EQ(kora::detail::DeviceCaps::from_vulkan(raw).tier(), kora::CapabilityTier::B);

    raw.vulkan12_features.bufferDeviceAddress = VK_TRUE;
    EXPECT_EQ(kora::detail::DeviceCaps::from_vulkan(raw).tier(), kora::CapabilityTier::C);

    raw.vulkan12_features.runtimeDescriptorArray = VK_TRUE;
    raw.vulkan13_features.synchronization2 = VK_TRUE;
    EXPECT_EQ(kora::detail::DeviceCaps::from_vulkan(raw).tier(), kora::CapabilityTier::D);
}

TEST(DeviceCaps, MapsLimitsFeaturesQueuesAndMemoryWithoutVulkanTypes) {
    const auto raw = make_device();
    const auto info = kora::detail::DeviceCaps::from_vulkan(raw).to_device_info();

    EXPECT_EQ(info.name(), "Synthetic Vulkan device");
    EXPECT_EQ(info.vendor_id(), 0x1234U);
    EXPECT_EQ(info.device_id(), 0x5678U);
    EXPECT_EQ(info.api_version().major, 1U);
    EXPECT_EQ(info.api_version().minor, 3U);
    EXPECT_EQ(info.limits().max_compute_work_group_count[0], 65535U);
    EXPECT_EQ(info.limits().max_compute_work_group_size[2], 64U);
    EXPECT_EQ(info.limits().max_compute_work_group_invocations, 1024U);
    EXPECT_EQ(info.limits().max_compute_shared_memory_size, 32768U);
    EXPECT_EQ(info.limits().max_storage_buffer_range, 1U << 30);
    EXPECT_EQ(info.limits().max_push_constants_size, 256U);
    EXPECT_FLOAT_EQ(info.timestamp_period_nanoseconds(), 2.5F);
    EXPECT_TRUE(info.features().robust_buffer_access);
    EXPECT_TRUE(info.features().shader_int16);
    EXPECT_TRUE(info.features().shader_int64);
    EXPECT_TRUE(info.features().shader_float64);

    ASSERT_TRUE(info.subgroup().supported);
    EXPECT_EQ(info.subgroup().size, 32U);
    EXPECT_TRUE(info.subgroup().basic);
    EXPECT_TRUE(info.subgroup().arithmetic);
    EXPECT_TRUE(info.subgroup().ballot);

    ASSERT_EQ(info.queue_families().size(), 1U);
    EXPECT_EQ(info.queue_families()[0].queue_count, 2U);
    EXPECT_TRUE(info.queue_families()[0].compute);
    EXPECT_TRUE(info.queue_families()[0].transfer);
    EXPECT_EQ(info.queue_families()[0].timestamp_valid_bits, 64U);
    EXPECT_EQ(info.queue_families()[0].min_image_transfer_granularity[2], 3U);

    ASSERT_EQ(info.memory_heaps().size(), 1U);
    EXPECT_EQ(info.memory_heaps()[0].size_bytes, 8U * 1024U * 1024U * 1024U);
    EXPECT_TRUE(info.memory_heaps()[0].device_local);
    ASSERT_EQ(info.memory_types().size(), 1U);
    EXPECT_TRUE(info.memory_types()[0].device_local);
    EXPECT_TRUE(info.memory_types()[0].host_visible);
    EXPECT_TRUE(info.memory_types()[0].host_coherent);
}

TEST(DeviceCaps, ReadsPromotedFeaturesFromExtensionQueryStructures) {
    auto raw = make_device();
    raw.vulkan12_features.sType = static_cast<VkStructureType>(0);
    raw.vulkan13_features.sType = static_cast<VkStructureType>(0);
    raw.timeline_semaphore_features.timelineSemaphore = VK_TRUE;
    raw.buffer_device_address_features.bufferDeviceAddress = VK_TRUE;
    raw.descriptor_indexing_features.descriptorBindingPartiallyBound = VK_TRUE;
    raw.descriptor_indexing_features.runtimeDescriptorArray = VK_TRUE;
    raw.synchronization2_features.synchronization2 = VK_TRUE;

    const auto info = kora::detail::DeviceCaps::from_vulkan(raw).to_device_info();

    EXPECT_TRUE(info.features().timeline_semaphore);
    EXPECT_TRUE(info.features().buffer_device_address);
    EXPECT_TRUE(info.features().descriptor_indexing);
    EXPECT_TRUE(info.features().runtime_descriptor_array);
    EXPECT_TRUE(info.features().synchronization2);
    EXPECT_EQ(info.tier(), kora::CapabilityTier::D);
}

TEST(DeviceCaps, ReportsIndividualDescriptorIndexingSupport) {
    auto raw = make_device();
    raw.vulkan12_features.sType = static_cast<VkStructureType>(0);
    raw.descriptor_indexing_features.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;

    const auto info = kora::detail::DeviceCaps::from_vulkan(raw).to_device_info();

    EXPECT_TRUE(info.features().descriptor_indexing);
    EXPECT_FALSE(info.features().runtime_descriptor_array);
    EXPECT_EQ(info.tier(), kora::CapabilityTier::A);
}
