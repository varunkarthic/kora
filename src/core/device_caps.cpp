#include "device_caps.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace kora::detail {
namespace {

template <class Flags, class Flag>
bool has_flag(Flags flags, Flag flag) {
    return (flags & static_cast<Flags>(flag)) != 0;
}

bool has_extension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto& extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

bool has_descriptor_indexing_feature(const VkPhysicalDeviceDescriptorIndexingFeatures& features) {
    const std::array<VkBool32, 20> values{
        features.shaderInputAttachmentArrayDynamicIndexing,
        features.shaderUniformTexelBufferArrayDynamicIndexing,
        features.shaderStorageTexelBufferArrayDynamicIndexing,
        features.shaderUniformBufferArrayNonUniformIndexing,
        features.shaderSampledImageArrayNonUniformIndexing,
        features.shaderStorageBufferArrayNonUniformIndexing,
        features.shaderStorageImageArrayNonUniformIndexing,
        features.shaderInputAttachmentArrayNonUniformIndexing,
        features.shaderUniformTexelBufferArrayNonUniformIndexing,
        features.shaderStorageTexelBufferArrayNonUniformIndexing,
        features.descriptorBindingUniformBufferUpdateAfterBind,
        features.descriptorBindingSampledImageUpdateAfterBind,
        features.descriptorBindingStorageImageUpdateAfterBind,
        features.descriptorBindingStorageBufferUpdateAfterBind,
        features.descriptorBindingUniformTexelBufferUpdateAfterBind,
        features.descriptorBindingStorageTexelBufferUpdateAfterBind,
        features.descriptorBindingUpdateUnusedWhilePending,
        features.descriptorBindingPartiallyBound,
        features.descriptorBindingVariableDescriptorCount,
        features.runtimeDescriptorArray,
    };
    return std::any_of(values.begin(), values.end(), [](VkBool32 value) { return value == VK_TRUE; });
}

CapabilityTier classify_tier(const DeviceInfo::Features& features, const DeviceInfo::Subgroup& subgroup) {
    if (!features.timeline_semaphore || !subgroup.supported || !subgroup.basic) {
        return CapabilityTier::A;
    }
    if (!features.buffer_device_address) return CapabilityTier::B;
    if (!features.runtime_descriptor_array || !features.synchronization2) return CapabilityTier::C;
    return CapabilityTier::D;
}

}  // namespace

DeviceCaps DeviceCaps::from_vulkan(const vk::PhysicalDeviceInfo& physical_device) {
    DeviceInfo::Data data;
    data.name = physical_device.properties.deviceName;
    data.vendor_id = physical_device.properties.vendorID;
    data.device_id = physical_device.properties.deviceID;

    const auto api_version = physical_device.properties.apiVersion;
    data.api_version = {VK_VERSION_MAJOR(api_version), VK_VERSION_MINOR(api_version),
                        VK_VERSION_PATCH(api_version)};

    const auto& properties = physical_device.properties;
    const auto& limits = properties.limits;
    std::copy_n(limits.maxComputeWorkGroupCount, data.limits.max_compute_work_group_count.size(),
                data.limits.max_compute_work_group_count.begin());
    std::copy_n(limits.maxComputeWorkGroupSize, data.limits.max_compute_work_group_size.size(),
                data.limits.max_compute_work_group_size.begin());
    data.limits.max_compute_work_group_invocations = limits.maxComputeWorkGroupInvocations;
    data.limits.max_compute_shared_memory_size = limits.maxComputeSharedMemorySize;
    data.limits.max_storage_buffer_range = limits.maxStorageBufferRange;
    data.limits.max_push_constants_size = limits.maxPushConstantsSize;
    data.timestamp_period_nanoseconds = limits.timestampPeriod;

    const auto& features = physical_device.features;
    data.features.robust_buffer_access = features.robustBufferAccess == VK_TRUE;
    data.features.shader_int16 = features.shaderInt16 == VK_TRUE;
    data.features.shader_int64 = features.shaderInt64 == VK_TRUE;
    data.features.shader_float64 = features.shaderFloat64 == VK_TRUE;

    const bool core12_features =
        physical_device.vulkan12_features.sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    data.features.timeline_semaphore =
        (core12_features && physical_device.vulkan12_features.timelineSemaphore == VK_TRUE) ||
        physical_device.timeline_semaphore_features.timelineSemaphore == VK_TRUE;
    data.features.buffer_device_address =
        (core12_features && physical_device.vulkan12_features.bufferDeviceAddress == VK_TRUE) ||
        physical_device.buffer_device_address_features.bufferDeviceAddress == VK_TRUE ||
        physical_device.buffer_device_address_features_ext.bufferDeviceAddress == VK_TRUE;
    data.features.descriptor_indexing =
        (core12_features && physical_device.vulkan12_features.descriptorIndexing == VK_TRUE) ||
        has_descriptor_indexing_feature(physical_device.descriptor_indexing_features);
    data.features.runtime_descriptor_array =
        (core12_features && physical_device.vulkan12_features.runtimeDescriptorArray == VK_TRUE) ||
        physical_device.descriptor_indexing_features.runtimeDescriptorArray == VK_TRUE;

    const bool core13_features =
        physical_device.vulkan13_features.sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    data.features.synchronization2 =
        (core13_features && physical_device.vulkan13_features.synchronization2 == VK_TRUE) ||
        physical_device.synchronization2_features.synchronization2 == VK_TRUE;

    data.features.portability_subset =
        has_extension(physical_device.extensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);

    const auto& subgroup = physical_device.subgroup_properties;
    data.subgroup.supported =
        subgroup.sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES && subgroup.subgroupSize != 0;
    data.subgroup.size = data.subgroup.supported ? subgroup.subgroupSize : 0;
    data.subgroup.basic =
        data.subgroup.supported && has_flag(subgroup.supportedOperations, VK_SUBGROUP_FEATURE_BASIC_BIT);
    data.subgroup.vote =
        data.subgroup.supported && has_flag(subgroup.supportedOperations, VK_SUBGROUP_FEATURE_VOTE_BIT);
    data.subgroup.arithmetic =
        data.subgroup.supported && has_flag(subgroup.supportedOperations, VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
    data.subgroup.ballot =
        data.subgroup.supported && has_flag(subgroup.supportedOperations, VK_SUBGROUP_FEATURE_BALLOT_BIT);
    data.subgroup.shuffle =
        data.subgroup.supported && has_flag(subgroup.supportedOperations, VK_SUBGROUP_FEATURE_SHUFFLE_BIT);
    data.subgroup.shuffle_relative =
        data.subgroup.supported &&
        has_flag(subgroup.supportedOperations, VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT);
    data.subgroup.clustered =
        data.subgroup.supported && has_flag(subgroup.supportedOperations, VK_SUBGROUP_FEATURE_CLUSTERED_BIT);
    data.subgroup.quad =
        data.subgroup.supported && has_flag(subgroup.supportedOperations, VK_SUBGROUP_FEATURE_QUAD_BIT);

    data.queue_families.reserve(physical_device.queue_family_properties.size());
    for (const auto& family : physical_device.queue_family_properties) {
        DeviceInfo::QueueFamily result;
        result.queue_count = family.queueCount;
        result.timestamp_valid_bits = family.timestampValidBits;
        result.min_image_transfer_granularity = {
            family.minImageTransferGranularity.width,
            family.minImageTransferGranularity.height,
            family.minImageTransferGranularity.depth,
        };
        result.graphics = has_flag(family.queueFlags, VK_QUEUE_GRAPHICS_BIT);
        result.compute = has_flag(family.queueFlags, VK_QUEUE_COMPUTE_BIT);
        result.transfer =
            has_flag(family.queueFlags, VK_QUEUE_TRANSFER_BIT) || result.graphics || result.compute;
        result.sparse_binding = has_flag(family.queueFlags, VK_QUEUE_SPARSE_BINDING_BIT);
        data.queue_families.push_back(result);
    }

    const auto& memory = physical_device.memory_properties;
    data.memory_heaps.reserve(memory.memoryHeapCount);
    for (std::uint32_t index = 0; index < memory.memoryHeapCount; ++index) {
        const auto& heap = memory.memoryHeaps[index];
        data.memory_heaps.push_back({
            heap.size,
            has_flag(heap.flags, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT),
            has_flag(heap.flags, VK_MEMORY_HEAP_MULTI_INSTANCE_BIT),
        });
    }

    data.memory_types.reserve(memory.memoryTypeCount);
    for (std::uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
        const auto& type = memory.memoryTypes[index];
        data.memory_types.push_back({
            type.heapIndex,
            has_flag(type.propertyFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            has_flag(type.propertyFlags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
            has_flag(type.propertyFlags, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
            has_flag(type.propertyFlags, VK_MEMORY_PROPERTY_HOST_CACHED_BIT),
            has_flag(type.propertyFlags, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT),
            has_flag(type.propertyFlags, VK_MEMORY_PROPERTY_PROTECTED_BIT),
        });
    }

    data.tier = classify_tier(data.features, data.subgroup);
    return DeviceCaps(std::move(data));
}

}  // namespace kora::detail
