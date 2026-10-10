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

bool type_is_device_local(const VkPhysicalDeviceMemoryProperties& memory, const VkMemoryType& type) {
    if (has_flag(type.propertyFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return true;
    const auto heap_count = std::min<std::uint32_t>(memory.memoryHeapCount, VK_MAX_MEMORY_HEAPS);
    return type.heapIndex < heap_count &&
           has_flag(memory.memoryHeaps[type.heapIndex].flags, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
}

int preference_score(MemoryPolicy policy, MemoryTopology topology, bool device_local, bool host_visible,
                     bool host_coherent, bool host_cached) {
    const auto score = [](bool first, bool second, bool third) {
        return static_cast<int>(first) * 4 + static_cast<int>(second) * 2 + static_cast<int>(third);
    };
    switch (policy) {
        case MemoryPolicy::Auto:
            switch (topology) {
                case MemoryTopology::Uma:
                    return score(device_local && host_visible, host_visible && host_coherent, device_local);
                case MemoryTopology::DiscreteWithBar:
                case MemoryTopology::Discrete:
                    return score(device_local && !host_visible, device_local && host_visible,
                                 host_visible && host_cached);
                case MemoryTopology::Unknown:
                    return score(device_local && !host_visible, device_local && host_visible, host_visible);
            }
            break;
        case MemoryPolicy::DevicePreferred:
            return score(device_local && !host_visible, device_local && host_visible, device_local);
        case MemoryPolicy::HostVisible:
            return score(host_visible, host_coherent, host_cached);
        case MemoryPolicy::ReadMostly:
            return score(device_local, host_cached, host_visible);
        case MemoryPolicy::WriteMostly:
            return score(device_local, host_coherent, host_visible);
        case MemoryPolicy::Streaming:
            return score(host_visible, host_coherent, device_local);
        case MemoryPolicy::Readback:
            return score(host_visible, host_cached, host_coherent);
    }
    return 0;
}

std::vector<std::uint32_t> order_memory_types(const VkPhysicalDeviceMemoryProperties& memory,
                                              MemoryPolicy policy, MemoryTopology topology) {
    const auto type_count = std::min<std::uint32_t>(memory.memoryTypeCount, VK_MAX_MEMORY_TYPES);
    std::vector<std::uint32_t> indices(type_count);
    for (std::uint32_t index = 0; index < type_count; ++index) indices[index] = index;
    std::stable_sort(indices.begin(), indices.end(), [&](std::uint32_t left, std::uint32_t right) {
        const auto score = [&](std::uint32_t index) {
            const auto& type = memory.memoryTypes[index];
            return preference_score(policy, topology, type_is_device_local(memory, type),
                                    has_flag(type.propertyFlags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
                                    has_flag(type.propertyFlags, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
                                    has_flag(type.propertyFlags, VK_MEMORY_PROPERTY_HOST_CACHED_BIT));
        };
        return score(left) > score(right);
    });
    return indices;
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

    data.queue_families = classify_queue_families(physical_device.queue_family_properties);

    const auto& memory = physical_device.memory_properties;
    const auto memory_heap_count = std::min<std::uint32_t>(memory.memoryHeapCount, VK_MAX_MEMORY_HEAPS);
    data.memory_heaps.reserve(memory_heap_count);
    for (std::uint32_t index = 0; index < memory_heap_count; ++index) {
        const auto& heap = memory.memoryHeaps[index];
        data.memory_heaps.push_back({
            heap.size,
            has_flag(heap.flags, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT),
            has_flag(heap.flags, VK_MEMORY_HEAP_MULTI_INSTANCE_BIT),
        });
    }

    const auto memory_type_count = std::min<std::uint32_t>(memory.memoryTypeCount, VK_MAX_MEMORY_TYPES);
    data.memory_types.reserve(memory_type_count);
    for (std::uint32_t index = 0; index < memory_type_count; ++index) {
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

    const auto memory_classification = classify_memory_properties(memory);
    data.memory_topology = memory_classification.topology;
    data.memory_topology_needs_benchmark = memory_classification.needs_benchmark;
    data.memory_type_preferences = memory_classification.type_preferences;

    data.tier = classify_tier(data.features, data.subgroup);
    return DeviceCaps(std::move(data));
}

std::vector<DeviceInfo::QueueFamily> DeviceCaps::classify_queue_families(
    const std::vector<VkQueueFamilyProperties>& queue_families) {
    std::vector<DeviceInfo::QueueFamily> classified;
    classified.reserve(queue_families.size());
    for (const auto& family : queue_families) {
        DeviceInfo::QueueFamily family_info;
        family_info.queue_count = family.queueCount;
        family_info.timestamp_valid_bits = family.timestampValidBits;
        family_info.min_image_transfer_granularity = {
            family.minImageTransferGranularity.width,
            family.minImageTransferGranularity.height,
            family.minImageTransferGranularity.depth,
        };
        family_info.graphics = has_flag(family.queueFlags, VK_QUEUE_GRAPHICS_BIT);
        family_info.compute = has_flag(family.queueFlags, VK_QUEUE_COMPUTE_BIT);
        family_info.transfer =
            has_flag(family.queueFlags, VK_QUEUE_TRANSFER_BIT) || family_info.graphics || family_info.compute;
        family_info.sparse_binding = has_flag(family.queueFlags, VK_QUEUE_SPARSE_BINDING_BIT);
        family_info.dedicated_compute = family_info.compute && !family_info.graphics;
        family_info.dedicated_transfer =
            family_info.transfer && !family_info.graphics && !family_info.compute;
        classified.push_back(family_info);
    }
    return classified;
}

MemoryClassification DeviceCaps::classify_memory_properties(
    const VkPhysicalDeviceMemoryProperties& memory_properties) {
    MemoryClassification classification;
    bool has_local_host_visible = false;
    bool has_local_host_invisible = false;
    const auto type_count = std::min<std::uint32_t>(memory_properties.memoryTypeCount, VK_MAX_MEMORY_TYPES);
    for (std::uint32_t index = 0; index < type_count; ++index) {
        const auto& type = memory_properties.memoryTypes[index];
        const bool local = type_is_device_local(memory_properties, type);
        const bool visible = has_flag(type.propertyFlags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        has_local_host_visible |= local && visible;
        has_local_host_invisible |= local && !visible;
    }

    if (has_local_host_visible && has_local_host_invisible) {
        classification.topology = MemoryTopology::DiscreteWithBar;
        classification.needs_benchmark = true;
    } else if (has_local_host_visible) {
        classification.topology = MemoryTopology::Uma;
    } else if (has_local_host_invisible) {
        classification.topology = MemoryTopology::Discrete;
    }

    classification.type_preferences.automatic =
        order_memory_types(memory_properties, MemoryPolicy::Auto, classification.topology);
    classification.type_preferences.device_preferred =
        order_memory_types(memory_properties, MemoryPolicy::DevicePreferred, classification.topology);
    classification.type_preferences.host_visible =
        order_memory_types(memory_properties, MemoryPolicy::HostVisible, classification.topology);
    classification.type_preferences.read_mostly =
        order_memory_types(memory_properties, MemoryPolicy::ReadMostly, classification.topology);
    classification.type_preferences.write_mostly =
        order_memory_types(memory_properties, MemoryPolicy::WriteMostly, classification.topology);
    classification.type_preferences.streaming =
        order_memory_types(memory_properties, MemoryPolicy::Streaming, classification.topology);
    classification.type_preferences.readback =
        order_memory_types(memory_properties, MemoryPolicy::Readback, classification.topology);
    return classification;
}

}  // namespace kora::detail
