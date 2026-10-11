#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kora/log.hpp"
#include "vulkan/api.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace kora::vk {
namespace {

struct LoaderState {
#if defined(_WIN32)
    HMODULE library = nullptr;
#else
    void* library = nullptr;
#endif
    PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
    uint32_t supported_api_version = VK_API_VERSION_1_0;

    ~LoaderState() {
#if defined(_WIN32)
        if (library != nullptr) FreeLibrary(library);
#else
        if (library != nullptr) dlclose(library);
#endif
    }
};

struct DeviceState {
    std::shared_ptr<LoaderState> loader;
    PFN_vkCreateShaderModule create_shader_module = nullptr;
    PFN_vkDestroyShaderModule destroy_shader_module = nullptr;
    PFN_vkCreateDescriptorSetLayout create_descriptor_set_layout = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroy_descriptor_set_layout = nullptr;
    PFN_vkCreatePipelineLayout create_pipeline_layout = nullptr;
    PFN_vkDestroyPipelineLayout destroy_pipeline_layout = nullptr;
    PFN_vkCreateDescriptorPool create_descriptor_pool = nullptr;
    PFN_vkDestroyDescriptorPool destroy_descriptor_pool = nullptr;
    PFN_vkAllocateDescriptorSets allocate_descriptor_sets = nullptr;
    PFN_vkFreeDescriptorSets free_descriptor_sets = nullptr;
    PFN_vkUpdateDescriptorSets update_descriptor_sets = nullptr;
    PFN_vkCreateComputePipelines create_compute_pipelines = nullptr;
    PFN_vkDestroyPipeline destroy_pipeline = nullptr;
    PFN_vkCreatePipelineCache create_pipeline_cache = nullptr;
    PFN_vkDestroyPipelineCache destroy_pipeline_cache = nullptr;
    PFN_vkGetPipelineCacheData get_pipeline_cache_data = nullptr;
    PFN_vkCreateBuffer create_buffer = nullptr;
    PFN_vkDestroyBuffer destroy_buffer = nullptr;
    PFN_vkGetBufferMemoryRequirements get_buffer_memory_requirements = nullptr;
    PFN_vkBindBufferMemory bind_buffer_memory = nullptr;
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier = nullptr;
    PFN_vkCmdCopyBuffer cmd_copy_buffer = nullptr;
    PFN_vkCmdFillBuffer cmd_fill_buffer = nullptr;
    PFN_vkResetCommandBuffer reset_command_buffer = nullptr;
    PFN_vkAllocateMemory allocate_memory = nullptr;
    PFN_vkFreeMemory free_memory = nullptr;
    PFN_vkMapMemory map_memory = nullptr;
    PFN_vkUnmapMemory unmap_memory = nullptr;
    PFN_vkFlushMappedMemoryRanges flush_mapped_memory_ranges = nullptr;
    PFN_vkInvalidateMappedMemoryRanges invalidate_mapped_memory_ranges = nullptr;
    PFN_vkQueueSubmit queue_submit = nullptr;
    PFN_vkCreateCommandPool create_command_pool = nullptr;
    PFN_vkDestroyCommandPool destroy_command_pool = nullptr;
    PFN_vkResetCommandPool reset_command_pool = nullptr;
    PFN_vkAllocateCommandBuffers allocate_command_buffers = nullptr;
    PFN_vkFreeCommandBuffers free_command_buffers = nullptr;
    PFN_vkBeginCommandBuffer begin_command_buffer = nullptr;
    PFN_vkEndCommandBuffer end_command_buffer = nullptr;
    PFN_vkCreateSemaphore create_semaphore = nullptr;
    PFN_vkDestroySemaphore destroy_semaphore = nullptr;
    PFN_vkGetSemaphoreCounterValue get_semaphore_counter_value = nullptr;
    PFN_vkWaitSemaphores wait_semaphores = nullptr;
};

VkResult VKAPI_PTR create_buffer(void* context, VkDevice device, const VkBufferCreateInfo* info,
                                 const VkAllocationCallbacks* allocator, VkBuffer* buffer) noexcept {
    const auto function = static_cast<DeviceState*>(context)->create_buffer;
    return function == nullptr ? VK_ERROR_INITIALIZATION_FAILED : function(device, info, allocator, buffer);
}
void VKAPI_PTR destroy_buffer(void* context, VkDevice device, VkBuffer buffer,
                              const VkAllocationCallbacks* allocator) noexcept {
    const auto function = static_cast<DeviceState*>(context)->destroy_buffer;
    if (function != nullptr) function(device, buffer, allocator);
}
void VKAPI_PTR get_buffer_memory_requirements(void* context, VkDevice device, VkBuffer buffer,
                                              VkMemoryRequirements* requirements) noexcept {
    const auto function = static_cast<DeviceState*>(context)->get_buffer_memory_requirements;
    if (function != nullptr) function(device, buffer, requirements);
}
VkResult VKAPI_PTR bind_buffer_memory(void* context, VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                                      VkDeviceSize offset) noexcept {
    const auto function = static_cast<DeviceState*>(context)->bind_buffer_memory;
    return function == nullptr ? VK_ERROR_INITIALIZATION_FAILED : function(device, buffer, memory, offset);
}
void VKAPI_PTR cmd_pipeline_barrier(void* context, VkCommandBuffer command_buffer,
                                    VkPipelineStageFlags source_stage, VkPipelineStageFlags destination_stage,
                                    VkDependencyFlags dependency_flags, uint32_t memory_barrier_count,
                                    const VkMemoryBarrier* memory_barriers, uint32_t buffer_barrier_count,
                                    const VkBufferMemoryBarrier* buffer_barriers,
                                    uint32_t image_barrier_count,
                                    const VkImageMemoryBarrier* image_barriers) noexcept {
    const auto function = static_cast<DeviceState*>(context)->cmd_pipeline_barrier;
    if (function != nullptr)
        function(command_buffer, source_stage, destination_stage, dependency_flags, memory_barrier_count,
                 memory_barriers, buffer_barrier_count, buffer_barriers, image_barrier_count, image_barriers);
}
void VKAPI_PTR cmd_copy_buffer(void* context, VkCommandBuffer command_buffer, VkBuffer source,
                               VkBuffer destination, uint32_t region_count,
                               const VkBufferCopy* regions) noexcept {
    const auto function = static_cast<DeviceState*>(context)->cmd_copy_buffer;
    if (function != nullptr) function(command_buffer, source, destination, region_count, regions);
}
void VKAPI_PTR cmd_fill_buffer(void* context, VkCommandBuffer command_buffer, VkBuffer destination,
                               VkDeviceSize offset, VkDeviceSize size, uint32_t data) noexcept {
    const auto function = static_cast<DeviceState*>(context)->cmd_fill_buffer;
    if (function != nullptr) function(command_buffer, destination, offset, size, data);
}
VkResult VKAPI_PTR reset_command_buffer(void* context, VkCommandBuffer command_buffer,
                                        VkCommandBufferResetFlags flags) noexcept {
    const auto function = static_cast<DeviceState*>(context)->reset_command_buffer;
    return function == nullptr ? VK_ERROR_INITIALIZATION_FAILED : function(command_buffer, flags);
}

VkResult VKAPI_PTR allocate_memory(void* context, VkDevice device, const VkMemoryAllocateInfo* info,
                                   const VkAllocationCallbacks* allocator, VkDeviceMemory* memory) noexcept {
    const auto& state = *static_cast<DeviceState*>(context);
    return state.allocate_memory(device, info, allocator, memory);
}

void VKAPI_PTR free_memory(void* context, VkDevice device, VkDeviceMemory memory,
                           const VkAllocationCallbacks* allocator) noexcept {
    const auto& state = *static_cast<DeviceState*>(context);
    state.free_memory(device, memory, allocator);
}

VkResult VKAPI_PTR map_memory(void* context, VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                              VkDeviceSize size, VkMemoryMapFlags flags, void** data) noexcept {
    const auto& state = *static_cast<DeviceState*>(context);
    return state.map_memory(device, memory, offset, size, flags, data);
}

void VKAPI_PTR unmap_memory(void* context, VkDevice device, VkDeviceMemory memory) noexcept {
    const auto& state = *static_cast<DeviceState*>(context);
    state.unmap_memory(device, memory);
}

VkResult VKAPI_PTR flush_mapped_memory_ranges(void* context, VkDevice device, uint32_t count,
                                              const VkMappedMemoryRange* ranges) noexcept {
    const auto& state = *static_cast<DeviceState*>(context);
    return state.flush_mapped_memory_ranges(device, count, ranges);
}

VkResult VKAPI_PTR invalidate_mapped_memory_ranges(void* context, VkDevice device, uint32_t count,
                                                   const VkMappedMemoryRange* ranges) noexcept {
    const auto& state = *static_cast<DeviceState*>(context);
    return state.invalidate_mapped_memory_ranges(device, count, ranges);
}

VkResult VKAPI_PTR queue_submit(void* context, VkQueue queue, uint32_t count, const VkSubmitInfo* submits,
                                VkFence fence) noexcept {
    const auto& state = *static_cast<DeviceState*>(context);
    return state.queue_submit(queue, count, submits, fence);
}

VkResult VKAPI_PTR create_command_pool(void* context, VkDevice device, const VkCommandPoolCreateInfo* info,
                                       const VkAllocationCallbacks* allocator, VkCommandPool* pool) noexcept {
    const auto function = static_cast<DeviceState*>(context)->create_command_pool;
    return function == nullptr ? VK_ERROR_INITIALIZATION_FAILED : function(device, info, allocator, pool);
}
void VKAPI_PTR destroy_command_pool(void* context, VkDevice device, VkCommandPool pool,
                                    const VkAllocationCallbacks* allocator) noexcept {
    const auto function = static_cast<DeviceState*>(context)->destroy_command_pool;
    if (function != nullptr) function(device, pool, allocator);
}
VkResult VKAPI_PTR reset_command_pool(void* context, VkDevice device, VkCommandPool pool,
                                      VkCommandPoolResetFlags flags) noexcept {
    const auto function = static_cast<DeviceState*>(context)->reset_command_pool;
    return function == nullptr ? VK_ERROR_INITIALIZATION_FAILED : function(device, pool, flags);
}
VkResult VKAPI_PTR allocate_command_buffers(void* context, VkDevice device,
                                            const VkCommandBufferAllocateInfo* info,
                                            VkCommandBuffer* buffers) noexcept {
    const auto function = static_cast<DeviceState*>(context)->allocate_command_buffers;
    return function == nullptr ? VK_ERROR_INITIALIZATION_FAILED : function(device, info, buffers);
}
void VKAPI_PTR free_command_buffers(void* context, VkDevice device, VkCommandPool pool, uint32_t count,
                                    const VkCommandBuffer* buffers) noexcept {
    const auto function = static_cast<DeviceState*>(context)->free_command_buffers;
    if (function != nullptr) function(device, pool, count, buffers);
}
VkResult VKAPI_PTR begin_command_buffer(void* context, VkCommandBuffer buffer,
                                        const VkCommandBufferBeginInfo* info) noexcept {
    const auto function = static_cast<DeviceState*>(context)->begin_command_buffer;
    return function == nullptr ? VK_ERROR_INITIALIZATION_FAILED : function(buffer, info);
}
VkResult VKAPI_PTR end_command_buffer(void* context, VkCommandBuffer buffer) noexcept {
    const auto function = static_cast<DeviceState*>(context)->end_command_buffer;
    return function == nullptr ? VK_ERROR_INITIALIZATION_FAILED : function(buffer);
}
VkResult VKAPI_PTR create_semaphore(void* context, VkDevice device, const VkSemaphoreCreateInfo* info,
                                    const VkAllocationCallbacks* allocator, VkSemaphore* semaphore) noexcept {
    const auto function = static_cast<DeviceState*>(context)->create_semaphore;
    return function == nullptr ? VK_ERROR_INITIALIZATION_FAILED
                               : function(device, info, allocator, semaphore);
}
void VKAPI_PTR destroy_semaphore(void* context, VkDevice device, VkSemaphore semaphore,
                                 const VkAllocationCallbacks* allocator) noexcept {
    const auto function = static_cast<DeviceState*>(context)->destroy_semaphore;
    if (function != nullptr) function(device, semaphore, allocator);
}
VkResult VKAPI_PTR get_semaphore_counter_value(void* context, VkDevice device, VkSemaphore semaphore,
                                               uint64_t* value) noexcept {
    const auto function = static_cast<DeviceState*>(context)->get_semaphore_counter_value;
    return function == nullptr ? VK_ERROR_INITIALIZATION_FAILED : function(device, semaphore, value);
}
VkResult VKAPI_PTR wait_semaphores(void* context, VkDevice device, const VkSemaphoreWaitInfo* info,
                                   uint64_t timeout) noexcept {
    const auto function = static_cast<DeviceState*>(context)->wait_semaphores;
    return function == nullptr ? VK_ERROR_INITIALIZATION_FAILED : function(device, info, timeout);
}
const char* library_name() {
#if defined(_WIN32)
    return "vulkan-1.dll";
#elif defined(__APPLE__)
    return "libvulkan.1.dylib";
#else
    return "libvulkan.so.1";
#endif
}

#if defined(_WIN32)
HMODULE open_vulkan_library() {
    return LoadLibraryA(library_name());
}
#else
void* open_vulkan_library() {
    constexpr int flags = RTLD_NOW | RTLD_LOCAL;
    if (void* library = dlopen(library_name(), flags)) return library;
#if defined(__APPLE__)
    if (const char* sdk = std::getenv("VULKAN_SDK"); sdk != nullptr && sdk[0] != '\0') {
        const std::array<std::string, 2> sdk_paths = {
            std::string(sdk) + "/lib/libvulkan.1.dylib",
            std::string(sdk) + "/macOS/lib/libvulkan.1.dylib",
        };
        for (const auto& path : sdk_paths) {
            if (void* library = dlopen(path.c_str(), flags)) return library;
        }
    }
    constexpr std::array<const char*, 2> package_paths = {
        "/opt/homebrew/lib/libvulkan.1.dylib",
        "/usr/local/lib/libvulkan.1.dylib",
    };
    for (const auto* path : package_paths) {
        if (void* library = dlopen(path, flags)) return library;
    }
#endif
    return nullptr;
}
#endif

#if defined(_WIN32)
template <class T>
T load_symbol(HMODULE library, const char* name) {
    return reinterpret_cast<T>(GetProcAddress(library, name));
}
#else
template <class T>
T load_symbol(void* library, const char* name) {
    return reinterpret_cast<T>(dlsym(library, name));
}
#endif

template <class T>
T load_device_function(PFN_vkGetDeviceProcAddr get_proc, VkDevice device, const char* name) {
    return get_proc == nullptr ? nullptr : reinterpret_cast<T>(get_proc(device, name));
}

template <class T>
T load_instance_function(PFN_vkGetInstanceProcAddr get_proc, VkInstance instance, const char* name) {
    return get_proc == nullptr ? nullptr : reinterpret_cast<T>(get_proc(instance, name));
}

bool has_extension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    for (const auto& extension : extensions) {
        if (std::string_view(extension.extensionName) == name) return true;
    }
    return false;
}

bool has_layer(const std::vector<VkLayerProperties>& layers, const char* name) {
    for (const auto& layer : layers) {
        if (std::string_view(layer.layerName) == name) return true;
    }
    return false;
}

template <class T, class Enumerate>
Result<std::vector<T>> enumerate(Enumerate function, const char* description) {
    while (true) {
        uint32_t count = 0;
        VkResult result = function(&count, nullptr);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::NoVulkanImplementation, std::string("Failed to enumerate ") + description)
                .with_vk(result);
        }
        std::vector<T> values(count);
        uint32_t written = count;
        result = function(&written, values.data());
        if (result == VK_SUCCESS) {
            values.resize(written);
            return values;
        }
        if (result != VK_INCOMPLETE) {
            return Error(ErrorCode::NoVulkanImplementation, std::string("Failed to enumerate ") + description)
                .with_vk(result);
        }
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    log::Level level = log::Level::Debug;
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        level = log::Level::Error;
    } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
        level = log::Level::Warn;
    } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) != 0) {
        level = log::Level::Info;
    }
    log::write(level, log::Category::Device,
               data != nullptr && data->pMessage != nullptr ? data->pMessage : "Vulkan validation message");
    return VK_FALSE;
}

template <class T>
void append_query_structure(void*& chain, T& structure, VkStructureType type) {
    structure.sType = type;
    structure.pNext = chain;
    chain = &structure;
}

Result<PhysicalDeviceInfo> query_physical_device(PFN_vkGetInstanceProcAddr get_proc, VkInstance instance,
                                                 uint32_t instance_api_version, bool properties2_enabled,
                                                 VkPhysicalDevice physical_device) {
    PhysicalDeviceInfo info;
    info.handle = physical_device;

    const auto enumerate_extensions = load_instance_function<PFN_vkEnumerateDeviceExtensionProperties>(
        get_proc, instance, "vkEnumerateDeviceExtensionProperties");
    if (enumerate_extensions == nullptr) {
        return Error(ErrorCode::NoVulkanImplementation,
                     "Vulkan instance does not provide vkEnumerateDeviceExtensionProperties");
    }
    auto extensions = enumerate<VkExtensionProperties>(
        [enumerate_extensions, physical_device](uint32_t* count, VkExtensionProperties* values) {
            return enumerate_extensions(physical_device, nullptr, count, values);
        },
        "device extensions");
    if (!extensions) return std::move(extensions).error();
    info.extensions = std::move(extensions).value();

    PFN_vkGetPhysicalDeviceProperties2 get_properties2 = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 get_features2 = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties2 get_memory_properties2 = nullptr;
    if (properties2_enabled) {
        get_properties2 = load_instance_function<PFN_vkGetPhysicalDeviceProperties2>(
            get_proc, instance, "vkGetPhysicalDeviceProperties2");
        if (get_properties2 == nullptr) {
            get_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                get_proc(instance, "vkGetPhysicalDeviceProperties2KHR"));
        }
        get_features2 = load_instance_function<PFN_vkGetPhysicalDeviceFeatures2>(
            get_proc, instance, "vkGetPhysicalDeviceFeatures2");
        if (get_features2 == nullptr) {
            get_features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                get_proc(instance, "vkGetPhysicalDeviceFeatures2KHR"));
        }
        get_memory_properties2 = load_instance_function<PFN_vkGetPhysicalDeviceMemoryProperties2>(
            get_proc, instance, "vkGetPhysicalDeviceMemoryProperties2");
        if (get_memory_properties2 == nullptr) {
            get_memory_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
                get_proc(instance, "vkGetPhysicalDeviceMemoryProperties2KHR"));
        }
    }
    if (properties2_enabled &&
        (get_properties2 == nullptr || get_features2 == nullptr || get_memory_properties2 == nullptr)) {
        return Error(ErrorCode::NoVulkanImplementation,
                     "Vulkan instance is missing a required properties2 entry point");
    }

    const auto get_properties = load_instance_function<PFN_vkGetPhysicalDeviceProperties>(
        get_proc, instance, "vkGetPhysicalDeviceProperties");
    const auto get_features = load_instance_function<PFN_vkGetPhysicalDeviceFeatures>(
        get_proc, instance, "vkGetPhysicalDeviceFeatures");
    const auto get_memory_properties = load_instance_function<PFN_vkGetPhysicalDeviceMemoryProperties>(
        get_proc, instance, "vkGetPhysicalDeviceMemoryProperties");
    if (get_properties2 != nullptr) {
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        get_properties2(physical_device, &properties);
        info.properties = properties.properties;
    } else if (get_properties != nullptr) {
        get_properties(physical_device, &info.properties);
    } else {
        return Error(ErrorCode::NoVulkanImplementation,
                     "Vulkan instance does not provide physical-device properties");
    }
    info.api_version = std::min(instance_api_version, info.properties.apiVersion);

    if (!properties2_enabled && get_features == nullptr) {
        return Error(ErrorCode::NoVulkanImplementation,
                     "Vulkan instance does not provide physical-device features");
    }
    if (!properties2_enabled && get_memory_properties == nullptr) {
        return Error(ErrorCode::NoVulkanImplementation,
                     "Vulkan instance does not provide physical-device memory properties");
    }

    const bool core11 = info.api_version >= VK_API_VERSION_1_1;
    const bool core12 = info.api_version >= VK_API_VERSION_1_2;
    const bool core13 = info.api_version >= VK_API_VERSION_1_3;
    const bool has_timeline = has_extension(info.extensions, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    const bool has_bda_khr = has_extension(info.extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    const bool has_bda_ext = has_extension(info.extensions, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    const bool has_descriptor_indexing =
        has_extension(info.extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    const bool has_sync2 = has_extension(info.extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    const bool has_subgroup_size_control =
        has_extension(info.extensions, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    const bool has_memory_budget = has_extension(info.extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    const bool has_memory_priority = has_extension(info.extensions, VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
#ifdef VK_KHR_device_fault
    const bool has_device_fault_khr = has_extension(info.extensions, VK_KHR_DEVICE_FAULT_EXTENSION_NAME);
#else
    const bool has_device_fault_khr = false;
#endif
#ifdef VK_EXT_device_fault
    const bool has_device_fault_ext = has_extension(info.extensions, VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
#else
    const bool has_device_fault_ext = false;
#endif
    const bool has_portability_subset =
        has_extension(info.extensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    bool has_robustness2 = has_extension(info.extensions, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
#ifdef VK_KHR_robustness2
    has_robustness2 = has_robustness2 || has_extension(info.extensions, VK_KHR_ROBUSTNESS_2_EXTENSION_NAME);
#endif

    if (get_features2 != nullptr) {
        void* chain = nullptr;
        if (core11) {
            append_query_structure(chain, info.vulkan11_features,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
        }
        if (core12) {
            append_query_structure(chain, info.vulkan12_features,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
        } else {
            if (has_timeline) {
                append_query_structure(chain, info.timeline_semaphore_features,
                                       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES);
            }
            if (has_bda_khr) {
                append_query_structure(chain, info.buffer_device_address_features,
                                       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES);
            } else if (has_bda_ext) {
                append_query_structure(chain, info.buffer_device_address_features_ext,
                                       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT);
            }
            if (has_descriptor_indexing) {
                append_query_structure(chain, info.descriptor_indexing_features,
                                       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES);
            }
        }
        if (core13) {
            append_query_structure(chain, info.vulkan13_features,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
        } else {
            if (has_sync2) {
                append_query_structure(chain, info.synchronization2_features,
                                       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES);
            }
            if (has_subgroup_size_control) {
                append_query_structure(chain, info.subgroup_size_control_features,
                                       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES);
            }
        }
        if (has_memory_priority) {
            append_query_structure(chain, info.memory_priority_features,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT);
        }
#ifdef VK_KHR_device_fault
        if (has_device_fault_khr) {
            append_query_structure(chain, info.device_fault_features,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_KHR);
        }
#endif
#ifdef VK_EXT_device_fault
        if (
#ifdef VK_KHR_device_fault
            !has_device_fault_khr &&
#endif
            has_device_fault_ext) {
            append_query_structure(chain, info.device_fault_features_ext,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT);
        }
#endif
        if (has_portability_subset) {
            append_query_structure(chain, info.portability_subset_features,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR);
        }
        if (has_robustness2) {
            append_query_structure(chain, info.robustness2_features,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT);
        }

        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = chain;
        get_features2(physical_device, &features);
        info.features = features.features;
    } else {
        get_features(physical_device, &info.features);
    }

    if (get_properties2 != nullptr) {
        void* chain = nullptr;
        if (core11) {
            append_query_structure(chain, info.id_properties,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES);
        }
        if (core11) {
            append_query_structure(chain, info.subgroup_properties,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES);
        }
        if (core11) {
            append_query_structure(chain, info.vulkan11_properties,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES);
        }
        if (core12) {
            append_query_structure(chain, info.vulkan12_properties,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES);
        }
        if (core13) {
            append_query_structure(chain, info.vulkan13_properties,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES);
        } else if (has_subgroup_size_control) {
            append_query_structure(chain, info.subgroup_size_control_properties,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES);
        }
#ifdef VK_KHR_device_fault
        if (has_device_fault_khr) {
            append_query_structure(chain, info.device_fault_properties,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_PROPERTIES_KHR);
        }
#endif
        if (has_portability_subset) {
            append_query_structure(chain, info.portability_subset_properties,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_PROPERTIES_KHR);
        }
        if (has_robustness2) {
            append_query_structure(chain, info.robustness2_properties,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT);
        }

        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = chain;
        get_properties2(physical_device, &properties);
        info.properties = properties.properties;
        info.id_properties.pNext = nullptr;
    } else {
        get_properties(physical_device, &info.properties);
    }

    if (get_memory_properties2 != nullptr) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT memory_budget{};
        void* chain = nullptr;
        if (has_memory_budget) {
            append_query_structure(chain, memory_budget,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT);
        }
        VkPhysicalDeviceMemoryProperties2 memory{};
        memory.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        memory.pNext = chain;
        get_memory_properties2(physical_device, &memory);
        info.memory_properties = memory.memoryProperties;
        info.memory_budget_properties = memory_budget;
        info.memory_budget_properties.pNext = nullptr;
    } else {
        get_memory_properties(physical_device, &info.memory_properties);
    }

    const auto get_queue_family_properties =
        load_instance_function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            get_proc, instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    if (get_queue_family_properties == nullptr) {
        return Error(ErrorCode::NoVulkanImplementation,
                     "Vulkan instance does not provide vkGetPhysicalDeviceQueueFamilyProperties");
    }
    uint32_t queue_family_count = 0;
    get_queue_family_properties(physical_device, &queue_family_count, nullptr);
    info.queue_family_properties.resize(queue_family_count);
    if (queue_family_count != 0) {
        get_queue_family_properties(physical_device, &queue_family_count,
                                    info.queue_family_properties.data());
        info.queue_family_properties.resize(queue_family_count);
    }

    if (has_extension(info.extensions, VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME) ||
        has_extension(info.extensions, "VK_EXT_calibrated_timestamps")) {
        auto get_time_domains = load_instance_function<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
            get_proc, instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR");
        if (get_time_domains == nullptr) {
            get_time_domains = reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
                get_proc(instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));
        }
        if (get_time_domains == nullptr) {
            return Error(ErrorCode::UnsupportedFeature,
                         "Calibrated timestamp extension is advertised without its query entry point");
        }
        auto domains = enumerate<VkTimeDomainKHR>(
            [get_time_domains, physical_device](uint32_t* count, VkTimeDomainKHR* values) {
                return get_time_domains(physical_device, count, values);
            },
            "calibrated timestamp domains");
        if (!domains) return std::move(domains).error();
        info.calibrated_time_domains = std::move(domains).value();
    }

    info.vulkan11_features.pNext = nullptr;
    info.vulkan11_properties.pNext = nullptr;
    info.vulkan12_features.pNext = nullptr;
    info.vulkan12_properties.pNext = nullptr;
    info.vulkan13_features.pNext = nullptr;
    info.vulkan13_properties.pNext = nullptr;
    info.subgroup_properties.pNext = nullptr;
    info.timeline_semaphore_features.pNext = nullptr;
    info.buffer_device_address_features.pNext = nullptr;
    info.buffer_device_address_features_ext.pNext = nullptr;
    info.descriptor_indexing_features.pNext = nullptr;
    info.synchronization2_features.pNext = nullptr;
    info.subgroup_size_control_features.pNext = nullptr;
    info.subgroup_size_control_properties.pNext = nullptr;
    info.memory_priority_features.pNext = nullptr;
#ifdef VK_KHR_device_fault
    info.device_fault_features.pNext = nullptr;
    info.device_fault_properties.pNext = nullptr;
#endif
#ifdef VK_EXT_device_fault
    info.device_fault_features_ext.pNext = nullptr;
#endif
    info.portability_subset_features.pNext = nullptr;
    info.portability_subset_properties.pNext = nullptr;
    info.robustness2_features.pNext = nullptr;
    info.robustness2_properties.pNext = nullptr;
    return info;
}

}  // namespace

struct Instance::State {
    std::shared_ptr<LoaderState> loader;
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    PFN_vkDestroyInstance destroy_instance = nullptr;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug_messenger = nullptr;
    uint32_t api_version = VK_API_VERSION_1_0;
    bool validation_enabled = false;
    bool properties2_enabled = false;

    ~State() {
        if (instance != VK_NULL_HANDLE && debug_messenger != VK_NULL_HANDLE &&
            destroy_debug_messenger != nullptr) {
            destroy_debug_messenger(instance, debug_messenger, nullptr);
        }
        if (instance != VK_NULL_HANDLE && destroy_instance != nullptr) destroy_instance(instance, nullptr);
    }
};

struct Device::State {
    std::shared_ptr<const void> instance_lifetime;
    std::shared_ptr<LoaderState> loader;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkQueue transfer_queue = VK_NULL_HANDLE;
    uint32_t compute_queue_family = std::numeric_limits<uint32_t>::max();
    uint32_t transfer_queue_family = std::numeric_limits<uint32_t>::max();
    EnabledFeatures enabled_features;
    std::vector<std::string> enabled_extensions;
    PFN_vkDestroyDevice destroy_device = nullptr;

    ~State() {
        if (device != VK_NULL_HANDLE && destroy_device != nullptr) destroy_device(device, nullptr);
    }
};

Result<Loader> Loader::load() {
    auto state = std::make_shared<LoaderState>();
    state->library = open_vulkan_library();
    if (state->library == nullptr) {
        return Error(ErrorCode::NoVulkanImplementation,
                     std::string("Unable to load Vulkan loader: ") + library_name());
    }

    const auto get_instance_proc =
        load_symbol<PFN_vkGetInstanceProcAddr>(state->library, "vkGetInstanceProcAddr");
    if (get_instance_proc == nullptr) {
        return Error(ErrorCode::NoVulkanImplementation,
                     "Vulkan loader does not export vkGetInstanceProcAddr");
    }

    state->get_instance_proc_addr = get_instance_proc;
    const auto enumerate_version = load_instance_function<PFN_vkEnumerateInstanceVersion>(
        get_instance_proc, VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    if (enumerate_version != nullptr) {
        const VkResult result = enumerate_version(&state->supported_api_version);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::NoVulkanImplementation, "Failed to query Vulkan loader version")
                .with_vk(result);
        }
    }
    return Loader(std::move(state));
}

Result<Instance> Loader::create_instance() const {
    return create_instance(InstanceOptions{});
}

Result<Instance> Loader::create_instance(const InstanceOptions& options) const {
    if (!state_) return Error(ErrorCode::NoVulkanImplementation, "Vulkan loader is not initialized");
    auto loader = std::static_pointer_cast<LoaderState>(state_);
    const uint32_t requested =
        options.requested_api_version == 0 ? VK_API_VERSION_1_0 : options.requested_api_version;
    const uint32_t api_version =
        std::min({requested, loader->supported_api_version, static_cast<uint32_t>(VK_API_VERSION_1_3)});

    auto enumerate_extensions = load_instance_function<PFN_vkEnumerateInstanceExtensionProperties>(
        loader->get_instance_proc_addr, VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties");
    auto enumerate_layers = load_instance_function<PFN_vkEnumerateInstanceLayerProperties>(
        loader->get_instance_proc_addr, VK_NULL_HANDLE, "vkEnumerateInstanceLayerProperties");
    auto create_instance = load_instance_function<PFN_vkCreateInstance>(loader->get_instance_proc_addr,
                                                                        VK_NULL_HANDLE, "vkCreateInstance");
    if (enumerate_extensions == nullptr || enumerate_layers == nullptr || create_instance == nullptr) {
        return Error(ErrorCode::NoVulkanImplementation,
                     "Vulkan loader is missing an instance-level entry point");
    }
    auto extensions_result = enumerate<VkExtensionProperties>(
        [enumerate_extensions](uint32_t* count, VkExtensionProperties* values) {
            return enumerate_extensions(nullptr, count, values);
        },
        "instance extensions");
    if (!extensions_result) return std::move(extensions_result).error();
    auto layers_result = enumerate<VkLayerProperties>(
        [enumerate_layers](uint32_t* count, VkLayerProperties* values) {
            return enumerate_layers(count, values);
        },
        "instance layers");
    if (!layers_result) return std::move(layers_result).error();
    const auto& available_extensions = extensions_result.value();
    const auto& available_layers = layers_result.value();

    std::vector<const char*> enabled_extensions;
    VkInstanceCreateFlags flags = 0;
    if (has_extension(available_extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        enabled_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    const bool properties2_extension_enabled =
        api_version == VK_API_VERSION_1_0 &&
        has_extension(available_extensions, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    if (properties2_extension_enabled) {
        enabled_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }

    const char* validation_layer = "VK_LAYER_KHRONOS_validation";
    const bool enable_validation = options.enable_validation && has_layer(available_layers, validation_layer);
    std::vector<const char*> enabled_layers;
    if (enable_validation)
        enabled_layers.push_back(validation_layer);
    else if (options.enable_validation) {
        KORA_LOG(Device, Warn, "Vulkan validation requested but VK_LAYER_KHRONOS_validation is unavailable");
    }

    const bool enable_debug_utils =
        enable_validation && has_extension(available_extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (enable_debug_utils) enabled_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    VkDebugUtilsMessengerCreateInfoEXT debug_info{};
    debug_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_info.pfnUserCallback = debug_message;

    VkApplicationInfo application_info{};
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pApplicationName =
        options.application_name.empty() ? "KORA" : options.application_name.c_str();
    application_info.applicationVersion = options.application_version;
    application_info.pEngineName = "KORA";
    application_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    application_info.apiVersion = api_version;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pNext = enable_debug_utils ? &debug_info : nullptr;
    instance_info.flags = flags;
    instance_info.pApplicationInfo = &application_info;
    instance_info.enabledLayerCount = static_cast<uint32_t>(enabled_layers.size());
    instance_info.ppEnabledLayerNames = enabled_layers.data();
    instance_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    instance_info.ppEnabledExtensionNames = enabled_extensions.data();

    auto instance_state = std::make_shared<Instance::State>();
    instance_state->loader = loader;
    instance_state->api_version = api_version;
    instance_state->validation_enabled = enable_validation;
    instance_state->properties2_enabled = api_version >= VK_API_VERSION_1_1 || properties2_extension_enabled;
    VkResult result = create_instance(&instance_info, nullptr, &instance_state->instance);
    if (result != VK_SUCCESS) {
        return Error(ErrorCode::NoVulkanImplementation, "Failed to create Vulkan instance").with_vk(result);
    }

    instance_state->destroy_instance = load_instance_function<PFN_vkDestroyInstance>(
        loader->get_instance_proc_addr, instance_state->instance, "vkDestroyInstance");
    auto enumerate_physical_devices = load_instance_function<PFN_vkEnumeratePhysicalDevices>(
        loader->get_instance_proc_addr, instance_state->instance, "vkEnumeratePhysicalDevices");
    if (instance_state->destroy_instance == nullptr || enumerate_physical_devices == nullptr) {
        return Error(ErrorCode::NoVulkanImplementation,
                     "Vulkan instance is missing a required instance entry point");
    }

    auto physical_devices = enumerate<VkPhysicalDevice>(
        [enumerate_physical_devices, instance = instance_state->instance](uint32_t* count,
                                                                          VkPhysicalDevice* devices) {
            return enumerate_physical_devices(instance, count, devices);
        },
        "physical devices");
    if (!physical_devices) return std::move(physical_devices).error();
    if (physical_devices.value().empty()) {
        return Error(ErrorCode::NoVulkanImplementation, "Vulkan loader found no physical devices");
    }

    if (enable_debug_utils) {
        const auto create_debug_messenger = load_instance_function<PFN_vkCreateDebugUtilsMessengerEXT>(
            loader->get_instance_proc_addr, instance_state->instance, "vkCreateDebugUtilsMessengerEXT");
        instance_state->destroy_debug_messenger = load_instance_function<PFN_vkDestroyDebugUtilsMessengerEXT>(
            loader->get_instance_proc_addr, instance_state->instance, "vkDestroyDebugUtilsMessengerEXT");
        if (create_debug_messenger != nullptr && instance_state->destroy_debug_messenger != nullptr) {
            result = create_debug_messenger(instance_state->instance, &debug_info, nullptr,
                                            &instance_state->debug_messenger);
            if (result != VK_SUCCESS) {
                return Error(ErrorCode::NoVulkanImplementation, "Failed to create Vulkan debug messenger")
                    .with_vk(result);
            }
        }
    }
    return Instance(std::move(instance_state));
}

Instance::~Instance() = default;
Instance::Instance(Instance&&) noexcept = default;
Instance& Instance::operator=(Instance&&) noexcept = default;

VkInstance Instance::handle() const noexcept {
    return state_ ? state_->instance : VK_NULL_HANDLE;
}

uint32_t Instance::api_version() const noexcept {
    return state_ ? state_->api_version : VK_API_VERSION_1_0;
}

bool Instance::validation_enabled() const noexcept {
    return state_ && state_->validation_enabled;
}

uint32_t Loader::supported_api_version() const noexcept {
    if (!state_) return VK_API_VERSION_1_0;
    return std::static_pointer_cast<LoaderState>(state_)->supported_api_version;
}

Result<std::vector<PhysicalDeviceInfo>> Instance::physical_devices() const {
    if (!state_ || state_->instance == VK_NULL_HANDLE) {
        return Error(ErrorCode::NoVulkanImplementation, "Vulkan instance is not initialized");
    }
    const auto get_proc = state_->loader->get_instance_proc_addr;
    const auto enumerate_physical_devices = load_instance_function<PFN_vkEnumeratePhysicalDevices>(
        get_proc, state_->instance, "vkEnumeratePhysicalDevices");
    if (enumerate_physical_devices == nullptr) {
        return Error(ErrorCode::NoVulkanImplementation,
                     "Vulkan instance does not provide vkEnumeratePhysicalDevices");
    }
    auto handles = enumerate<VkPhysicalDevice>(
        [enumerate_physical_devices, instance = state_->instance](uint32_t* count,
                                                                  VkPhysicalDevice* devices) {
            return enumerate_physical_devices(instance, count, devices);
        },
        "physical devices");
    if (!handles) return std::move(handles).error();

    std::vector<PhysicalDeviceInfo> devices;
    devices.reserve(handles.value().size());
    for (const auto handle : handles.value()) {
        auto info = query_physical_device(get_proc, state_->instance, state_->api_version,
                                          state_->properties2_enabled, handle);
        if (!info) return std::move(info).error();
        devices.push_back(std::move(info).value());
    }
    return devices;
}

Result<Device> Instance::create_device(const PhysicalDeviceInfo& physical_device,
                                       const DeviceOptions& options) const {
    if (!state_ || state_->instance == VK_NULL_HANDLE || physical_device.handle == VK_NULL_HANDLE) {
        return Error(ErrorCode::InvalidArgument, "An initialized instance and physical device are required");
    }

    uint32_t compute_family = std::numeric_limits<uint32_t>::max();
    uint32_t dedicated_transfer_family = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < physical_device.queue_family_properties.size(); ++i) {
        const auto& family = physical_device.queue_family_properties[i];
        if (family.queueCount == 0) continue;
        if (compute_family == std::numeric_limits<uint32_t>::max() &&
            (family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            compute_family = i;
        }
        if (dedicated_transfer_family == std::numeric_limits<uint32_t>::max() &&
            (family.queueFlags & VK_QUEUE_TRANSFER_BIT) != 0 &&
            (family.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0) {
            dedicated_transfer_family = i;
        }
    }
    if (compute_family == std::numeric_limits<uint32_t>::max()) {
        return Error(ErrorCode::NoComputeQueue, "Physical device exposes no compute-capable queue")
            .with_device(physical_device.properties.deviceName);
    }
    const uint32_t transfer_family = options.enable_dedicated_transfer_queue &&
                                             dedicated_transfer_family != std::numeric_limits<uint32_t>::max()
                                         ? dedicated_transfer_family
                                         : compute_family;

    const float priority = 1.0F;
    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    VkDeviceQueueCreateInfo compute_queue_info{};
    compute_queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    compute_queue_info.queueFamilyIndex = compute_family;
    compute_queue_info.queueCount = 1;
    compute_queue_info.pQueuePriorities = &priority;
    queue_infos.push_back(compute_queue_info);
    if (transfer_family != compute_family) {
        VkDeviceQueueCreateInfo transfer_queue_info = compute_queue_info;
        transfer_queue_info.queueFamilyIndex = transfer_family;
        queue_infos.push_back(transfer_queue_info);
    }

    const bool core12 = physical_device.api_version >= VK_API_VERSION_1_2;
    const bool core13 = physical_device.api_version >= VK_API_VERSION_1_3;
    const bool timeline = core12 ? physical_device.vulkan12_features.timelineSemaphore == VK_TRUE
                                 : physical_device.timeline_semaphore_features.timelineSemaphore == VK_TRUE;
    const bool bda_khr = physical_device.buffer_device_address_features.bufferDeviceAddress == VK_TRUE;
    const bool bda_ext = physical_device.buffer_device_address_features_ext.bufferDeviceAddress == VK_TRUE;
    const bool bda =
        core12 ? physical_device.vulkan12_features.bufferDeviceAddress == VK_TRUE : bda_khr || bda_ext;
    const bool descriptor_indexing =
        core12 ? physical_device.vulkan12_features.descriptorIndexing == VK_TRUE
               : physical_device.descriptor_indexing_features.runtimeDescriptorArray == VK_TRUE;
    const bool sync2 = core13 ? physical_device.vulkan13_features.synchronization2 == VK_TRUE
                              : physical_device.synchronization2_features.synchronization2 == VK_TRUE;

    std::vector<const char*> extensions;
    auto enable_extension = [&](const char* name) {
        if (has_extension(physical_device.extensions, name)) extensions.push_back(name);
    };
    if (!core12 && timeline) enable_extension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    if (!core12 && bda) {
        enable_extension(bda_khr ? VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME
                                 : VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    }
    if (!core12 && descriptor_indexing) enable_extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    if (!core13 && sync2) enable_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    enable_extension(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);

    void* feature_chain = nullptr;
    VkPhysicalDeviceVulkan12Features features12{};
    VkPhysicalDeviceVulkan13Features features13{};
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda_features{};
    VkPhysicalDeviceBufferDeviceAddressFeaturesEXT bda_features_ext{};
    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_features{};
    VkPhysicalDeviceSynchronization2Features sync2_features{};
    if (core12) {
        features12.timelineSemaphore = timeline;
        features12.bufferDeviceAddress = bda;
        features12.descriptorIndexing = descriptor_indexing;
        features12.runtimeDescriptorArray =
            descriptor_indexing && physical_device.vulkan12_features.runtimeDescriptorArray;
        features12.descriptorBindingPartiallyBound =
            descriptor_indexing && physical_device.vulkan12_features.descriptorBindingPartiallyBound;
        features12.descriptorBindingVariableDescriptorCount =
            descriptor_indexing && physical_device.vulkan12_features.descriptorBindingVariableDescriptorCount;
        append_query_structure(feature_chain, features12,
                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    } else {
        if (timeline) {
            timeline_features.timelineSemaphore = VK_TRUE;
            append_query_structure(feature_chain, timeline_features,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES);
        }
        if (bda_khr) {
            bda_features.bufferDeviceAddress = VK_TRUE;
            append_query_structure(feature_chain, bda_features,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES);
        } else if (bda_ext) {
            bda_features_ext.bufferDeviceAddress = VK_TRUE;
            append_query_structure(feature_chain, bda_features_ext,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT);
        }
        if (descriptor_indexing) {
            descriptor_features.runtimeDescriptorArray =
                physical_device.descriptor_indexing_features.runtimeDescriptorArray;
            descriptor_features.descriptorBindingPartiallyBound =
                physical_device.descriptor_indexing_features.descriptorBindingPartiallyBound;
            descriptor_features.descriptorBindingVariableDescriptorCount =
                physical_device.descriptor_indexing_features.descriptorBindingVariableDescriptorCount;
            append_query_structure(feature_chain, descriptor_features,
                                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES);
        }
    }
    if (core13) {
        features13.synchronization2 = sync2;
        append_query_structure(feature_chain, features13,
                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
    } else if (sync2) {
        sync2_features.synchronization2 = VK_TRUE;
        append_query_structure(feature_chain, sync2_features,
                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES);
    }

    VkPhysicalDeviceFeatures base_features{};
    base_features.robustBufferAccess = physical_device.features.robustBufferAccess;
    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = feature_chain;
    features.features = base_features;

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    if (physical_device.api_version >= VK_API_VERSION_1_1 || state_->properties2_enabled) {
        create_info.pNext = &features;
    } else {
        create_info.pEnabledFeatures = &base_features;
    }
    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_infos.size());
    create_info.pQueueCreateInfos = queue_infos.data();
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    const auto get_proc = state_->loader->get_instance_proc_addr;
    const auto create_device =
        load_instance_function<PFN_vkCreateDevice>(get_proc, state_->instance, "vkCreateDevice");
    if (create_device == nullptr) {
        return Error(ErrorCode::UnsupportedFeature, "Vulkan instance does not provide vkCreateDevice");
    }
    const auto get_device_proc =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(get_proc(state_->instance, "vkGetDeviceProcAddr"));
    if (get_device_proc == nullptr) {
        return Error(ErrorCode::UnsupportedFeature, "Vulkan instance does not provide vkGetDeviceProcAddr");
    }
    auto device_state = std::make_shared<Device::State>();
    device_state->instance_lifetime = state_;
    device_state->loader = state_->loader;
    VkResult result = create_device(physical_device.handle, &create_info, nullptr, &device_state->device);
    if (result != VK_SUCCESS) {
        return Error(ErrorCode::NoEligibleDevice, "Failed to create Vulkan logical device")
            .with_vk(result)
            .with_device(physical_device.properties.deviceName);
    }

    device_state->destroy_device =
        load_device_function<PFN_vkDestroyDevice>(get_device_proc, device_state->device, "vkDestroyDevice");
    const auto get_device_queue =
        load_device_function<PFN_vkGetDeviceQueue>(get_device_proc, device_state->device, "vkGetDeviceQueue");
    if (device_state->destroy_device == nullptr || get_device_queue == nullptr) {
        return Error(ErrorCode::UnsupportedFeature,
                     "Vulkan device is missing required lifecycle or queue entry points");
    }
    get_device_queue(device_state->device, compute_family, 0, &device_state->compute_queue);
    get_device_queue(device_state->device, transfer_family, 0, &device_state->transfer_queue);
    device_state->compute_queue_family = compute_family;
    device_state->transfer_queue_family = transfer_family;
    device_state->enabled_features = {timeline, bda, descriptor_indexing, sync2};
    for (const char* extension : extensions) device_state->enabled_extensions.emplace_back(extension);

    auto api_state = std::make_shared<DeviceState>();
    api_state->loader = state_->loader;
    api_state->create_shader_module = load_device_function<PFN_vkCreateShaderModule>(
        get_device_proc, device_state->device, "vkCreateShaderModule");
    api_state->destroy_shader_module = load_device_function<PFN_vkDestroyShaderModule>(
        get_device_proc, device_state->device, "vkDestroyShaderModule");
    api_state->create_descriptor_set_layout = load_device_function<PFN_vkCreateDescriptorSetLayout>(
        get_device_proc, device_state->device, "vkCreateDescriptorSetLayout");
    api_state->destroy_descriptor_set_layout = load_device_function<PFN_vkDestroyDescriptorSetLayout>(
        get_device_proc, device_state->device, "vkDestroyDescriptorSetLayout");
    api_state->create_pipeline_layout = load_device_function<PFN_vkCreatePipelineLayout>(
        get_device_proc, device_state->device, "vkCreatePipelineLayout");
    api_state->destroy_pipeline_layout = load_device_function<PFN_vkDestroyPipelineLayout>(
        get_device_proc, device_state->device, "vkDestroyPipelineLayout");
    api_state->create_descriptor_pool = load_device_function<PFN_vkCreateDescriptorPool>(
        get_device_proc, device_state->device, "vkCreateDescriptorPool");
    api_state->destroy_descriptor_pool = load_device_function<PFN_vkDestroyDescriptorPool>(
        get_device_proc, device_state->device, "vkDestroyDescriptorPool");
    api_state->allocate_descriptor_sets = load_device_function<PFN_vkAllocateDescriptorSets>(
        get_device_proc, device_state->device, "vkAllocateDescriptorSets");
    api_state->free_descriptor_sets = load_device_function<PFN_vkFreeDescriptorSets>(
        get_device_proc, device_state->device, "vkFreeDescriptorSets");
    api_state->update_descriptor_sets = load_device_function<PFN_vkUpdateDescriptorSets>(
        get_device_proc, device_state->device, "vkUpdateDescriptorSets");
    api_state->create_compute_pipelines = load_device_function<PFN_vkCreateComputePipelines>(
        get_device_proc, device_state->device, "vkCreateComputePipelines");
    api_state->destroy_pipeline = load_device_function<PFN_vkDestroyPipeline>(
        get_device_proc, device_state->device, "vkDestroyPipeline");
    api_state->create_pipeline_cache = load_device_function<PFN_vkCreatePipelineCache>(
        get_device_proc, device_state->device, "vkCreatePipelineCache");
    api_state->destroy_pipeline_cache = load_device_function<PFN_vkDestroyPipelineCache>(
        get_device_proc, device_state->device, "vkDestroyPipelineCache");
    api_state->get_pipeline_cache_data = load_device_function<PFN_vkGetPipelineCacheData>(
        get_device_proc, device_state->device, "vkGetPipelineCacheData");
    api_state->create_buffer =
        load_device_function<PFN_vkCreateBuffer>(get_device_proc, device_state->device, "vkCreateBuffer");
    api_state->destroy_buffer =
        load_device_function<PFN_vkDestroyBuffer>(get_device_proc, device_state->device, "vkDestroyBuffer");
    api_state->get_buffer_memory_requirements = load_device_function<PFN_vkGetBufferMemoryRequirements>(
        get_device_proc, device_state->device, "vkGetBufferMemoryRequirements");
    api_state->bind_buffer_memory = load_device_function<PFN_vkBindBufferMemory>(
        get_device_proc, device_state->device, "vkBindBufferMemory");
    api_state->cmd_pipeline_barrier = load_device_function<PFN_vkCmdPipelineBarrier>(
        get_device_proc, device_state->device, "vkCmdPipelineBarrier");
    api_state->cmd_copy_buffer =
        load_device_function<PFN_vkCmdCopyBuffer>(get_device_proc, device_state->device, "vkCmdCopyBuffer");
    api_state->cmd_fill_buffer =
        load_device_function<PFN_vkCmdFillBuffer>(get_device_proc, device_state->device, "vkCmdFillBuffer");
    api_state->reset_command_buffer = load_device_function<PFN_vkResetCommandBuffer>(
        get_device_proc, device_state->device, "vkResetCommandBuffer");
    api_state->allocate_memory =
        load_device_function<PFN_vkAllocateMemory>(get_device_proc, device_state->device, "vkAllocateMemory");
    api_state->free_memory =
        load_device_function<PFN_vkFreeMemory>(get_device_proc, device_state->device, "vkFreeMemory");
    api_state->map_memory =
        load_device_function<PFN_vkMapMemory>(get_device_proc, device_state->device, "vkMapMemory");
    api_state->unmap_memory =
        load_device_function<PFN_vkUnmapMemory>(get_device_proc, device_state->device, "vkUnmapMemory");
    api_state->flush_mapped_memory_ranges = load_device_function<PFN_vkFlushMappedMemoryRanges>(
        get_device_proc, device_state->device, "vkFlushMappedMemoryRanges");
    api_state->invalidate_mapped_memory_ranges = load_device_function<PFN_vkInvalidateMappedMemoryRanges>(
        get_device_proc, device_state->device, "vkInvalidateMappedMemoryRanges");
    api_state->queue_submit =
        load_device_function<PFN_vkQueueSubmit>(get_device_proc, device_state->device, "vkQueueSubmit");
    api_state->create_command_pool = load_device_function<PFN_vkCreateCommandPool>(
        get_device_proc, device_state->device, "vkCreateCommandPool");
    api_state->destroy_command_pool = load_device_function<PFN_vkDestroyCommandPool>(
        get_device_proc, device_state->device, "vkDestroyCommandPool");
    api_state->reset_command_pool = load_device_function<PFN_vkResetCommandPool>(
        get_device_proc, device_state->device, "vkResetCommandPool");
    api_state->allocate_command_buffers = load_device_function<PFN_vkAllocateCommandBuffers>(
        get_device_proc, device_state->device, "vkAllocateCommandBuffers");
    api_state->free_command_buffers = load_device_function<PFN_vkFreeCommandBuffers>(
        get_device_proc, device_state->device, "vkFreeCommandBuffers");
    api_state->begin_command_buffer = load_device_function<PFN_vkBeginCommandBuffer>(
        get_device_proc, device_state->device, "vkBeginCommandBuffer");
    api_state->end_command_buffer = load_device_function<PFN_vkEndCommandBuffer>(
        get_device_proc, device_state->device, "vkEndCommandBuffer");
    api_state->create_semaphore = load_device_function<PFN_vkCreateSemaphore>(
        get_device_proc, device_state->device, "vkCreateSemaphore");
    api_state->destroy_semaphore = load_device_function<PFN_vkDestroySemaphore>(
        get_device_proc, device_state->device, "vkDestroySemaphore");
    api_state->get_semaphore_counter_value = load_device_function<PFN_vkGetSemaphoreCounterValue>(
        get_device_proc, device_state->device, "vkGetSemaphoreCounterValue");
    if (api_state->get_semaphore_counter_value == nullptr)
        api_state->get_semaphore_counter_value = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
            get_device_proc(device_state->device, "vkGetSemaphoreCounterValueKHR"));
    api_state->wait_semaphores =
        load_device_function<PFN_vkWaitSemaphores>(get_device_proc, device_state->device, "vkWaitSemaphores");
    if (api_state->wait_semaphores == nullptr)
        api_state->wait_semaphores = reinterpret_cast<PFN_vkWaitSemaphores>(
            get_device_proc(device_state->device, "vkWaitSemaphoresKHR"));
    if (api_state->allocate_memory == nullptr || api_state->free_memory == nullptr ||
        api_state->map_memory == nullptr || api_state->unmap_memory == nullptr ||
        api_state->flush_mapped_memory_ranges == nullptr ||
        api_state->invalidate_mapped_memory_ranges == nullptr || api_state->queue_submit == nullptr) {
        return Error(ErrorCode::UnsupportedFeature, "Vulkan device is missing required backend entry points");
    }
    Api api;
    api.create_shader_module = api_state->create_shader_module;
    api.destroy_shader_module = api_state->destroy_shader_module;
    api.create_descriptor_set_layout = api_state->create_descriptor_set_layout;
    api.destroy_descriptor_set_layout = api_state->destroy_descriptor_set_layout;
    api.create_pipeline_layout = api_state->create_pipeline_layout;
    api.destroy_pipeline_layout = api_state->destroy_pipeline_layout;
    api.create_descriptor_pool = api_state->create_descriptor_pool;
    api.destroy_descriptor_pool = api_state->destroy_descriptor_pool;
    api.allocate_descriptor_sets = api_state->allocate_descriptor_sets;
    api.free_descriptor_sets = api_state->free_descriptor_sets;
    api.update_descriptor_sets = api_state->update_descriptor_sets;
    api.create_compute_pipelines = api_state->create_compute_pipelines;
    api.destroy_pipeline = api_state->destroy_pipeline;
    api.create_pipeline_cache = api_state->create_pipeline_cache;
    api.destroy_pipeline_cache = api_state->destroy_pipeline_cache;
    api.get_pipeline_cache_data = api_state->get_pipeline_cache_data;
    api.create_buffer = {api_state.get(), create_buffer};
    api.destroy_buffer = {api_state.get(), destroy_buffer};
    api.get_buffer_memory_requirements = {api_state.get(), get_buffer_memory_requirements};
    api.bind_buffer_memory = {api_state.get(), bind_buffer_memory};
    api.cmd_pipeline_barrier = {api_state.get(), cmd_pipeline_barrier};
    api.cmd_copy_buffer = {api_state.get(), cmd_copy_buffer};
    api.cmd_fill_buffer = {api_state.get(), cmd_fill_buffer};
    api.reset_command_buffer = {api_state.get(), reset_command_buffer};
    api.allocate_memory = {api_state.get(), allocate_memory};
    api.free_memory = {api_state.get(), free_memory};
    api.map_memory = {api_state.get(), map_memory};
    api.unmap_memory = {api_state.get(), unmap_memory};
    api.flush_mapped_memory_ranges = {api_state.get(), flush_mapped_memory_ranges};
    api.invalidate_mapped_memory_ranges = {api_state.get(), invalidate_mapped_memory_ranges};
    api.queue_submit = {api_state.get(), queue_submit};
    api.create_command_pool = {api_state.get(), create_command_pool};
    api.destroy_command_pool = {api_state.get(), destroy_command_pool};
    api.reset_command_pool = {api_state.get(), reset_command_pool};
    api.allocate_command_buffers = {api_state.get(), allocate_command_buffers};
    api.free_command_buffers = {api_state.get(), free_command_buffers};
    api.begin_command_buffer = {api_state.get(), begin_command_buffer};
    api.end_command_buffer = {api_state.get(), end_command_buffer};
    api.create_semaphore = {api_state.get(), create_semaphore};
    api.destroy_semaphore = {api_state.get(), destroy_semaphore};
    api.get_semaphore_counter_value = {api_state.get(), get_semaphore_counter_value};
    api.wait_semaphores = {api_state.get(), wait_semaphores};
    api.lifetime_ = std::move(api_state);
    return Device(std::move(device_state), std::move(api));
}

Device::~Device() = default;
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;

VkDevice Device::handle() const noexcept {
    return state_ ? state_->device : VK_NULL_HANDLE;
}
VkQueue Device::compute_queue() const noexcept {
    return state_ ? state_->compute_queue : VK_NULL_HANDLE;
}
VkQueue Device::transfer_queue() const noexcept {
    return state_ ? state_->transfer_queue : VK_NULL_HANDLE;
}
uint32_t Device::compute_queue_family() const noexcept {
    return state_ ? state_->compute_queue_family : std::numeric_limits<uint32_t>::max();
}
uint32_t Device::transfer_queue_family() const noexcept {
    return state_ ? state_->transfer_queue_family : std::numeric_limits<uint32_t>::max();
}
const Device::EnabledFeatures& Device::enabled_features() const noexcept {
    static const EnabledFeatures empty;
    return state_ ? state_->enabled_features : empty;
}
const std::vector<std::string>& Device::enabled_extensions() const noexcept {
    static const std::vector<std::string> empty;
    return state_ ? state_->enabled_extensions : empty;
}
const Api& Device::api() const noexcept {
    return api_;
}

Result<Api> Loader::device_api(VkInstance instance, VkDevice device) const {
    if (!state_ || instance == VK_NULL_HANDLE || device == VK_NULL_HANDLE) {
        return Error(ErrorCode::InvalidArgument,
                     "A Vulkan instance and device are required to load device calls");
    }
    auto loader = std::static_pointer_cast<LoaderState>(state_);
    const auto get_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        loader->get_instance_proc_addr(instance, "vkGetDeviceProcAddr"));
    if (get_device_proc == nullptr) {
        return Error(ErrorCode::UnsupportedFeature, "Vulkan instance does not provide vkGetDeviceProcAddr");
    }
    auto device_state = std::make_shared<DeviceState>();
    device_state->loader = loader;
    device_state->create_shader_module =
        load_device_function<PFN_vkCreateShaderModule>(get_device_proc, device, "vkCreateShaderModule");
    device_state->destroy_shader_module =
        load_device_function<PFN_vkDestroyShaderModule>(get_device_proc, device, "vkDestroyShaderModule");
    device_state->create_descriptor_set_layout = load_device_function<PFN_vkCreateDescriptorSetLayout>(
        get_device_proc, device, "vkCreateDescriptorSetLayout");
    device_state->destroy_descriptor_set_layout = load_device_function<PFN_vkDestroyDescriptorSetLayout>(
        get_device_proc, device, "vkDestroyDescriptorSetLayout");
    device_state->create_pipeline_layout =
        load_device_function<PFN_vkCreatePipelineLayout>(get_device_proc, device, "vkCreatePipelineLayout");
    device_state->destroy_pipeline_layout =
        load_device_function<PFN_vkDestroyPipelineLayout>(get_device_proc, device, "vkDestroyPipelineLayout");
    device_state->create_descriptor_pool =
        load_device_function<PFN_vkCreateDescriptorPool>(get_device_proc, device, "vkCreateDescriptorPool");
    device_state->destroy_descriptor_pool =
        load_device_function<PFN_vkDestroyDescriptorPool>(get_device_proc, device, "vkDestroyDescriptorPool");
    device_state->allocate_descriptor_sets = load_device_function<PFN_vkAllocateDescriptorSets>(
        get_device_proc, device, "vkAllocateDescriptorSets");
    device_state->free_descriptor_sets =
        load_device_function<PFN_vkFreeDescriptorSets>(get_device_proc, device, "vkFreeDescriptorSets");
    device_state->update_descriptor_sets =
        load_device_function<PFN_vkUpdateDescriptorSets>(get_device_proc, device, "vkUpdateDescriptorSets");
    device_state->create_compute_pipelines = load_device_function<PFN_vkCreateComputePipelines>(
        get_device_proc, device, "vkCreateComputePipelines");
    device_state->destroy_pipeline =
        load_device_function<PFN_vkDestroyPipeline>(get_device_proc, device, "vkDestroyPipeline");
    device_state->create_pipeline_cache =
        load_device_function<PFN_vkCreatePipelineCache>(get_device_proc, device, "vkCreatePipelineCache");
    device_state->destroy_pipeline_cache =
        load_device_function<PFN_vkDestroyPipelineCache>(get_device_proc, device, "vkDestroyPipelineCache");
    device_state->get_pipeline_cache_data =
        load_device_function<PFN_vkGetPipelineCacheData>(get_device_proc, device, "vkGetPipelineCacheData");
    device_state->create_buffer =
        load_device_function<PFN_vkCreateBuffer>(get_device_proc, device, "vkCreateBuffer");
    device_state->destroy_buffer =
        load_device_function<PFN_vkDestroyBuffer>(get_device_proc, device, "vkDestroyBuffer");
    device_state->get_buffer_memory_requirements = load_device_function<PFN_vkGetBufferMemoryRequirements>(
        get_device_proc, device, "vkGetBufferMemoryRequirements");
    device_state->bind_buffer_memory =
        load_device_function<PFN_vkBindBufferMemory>(get_device_proc, device, "vkBindBufferMemory");
    device_state->cmd_pipeline_barrier =
        load_device_function<PFN_vkCmdPipelineBarrier>(get_device_proc, device, "vkCmdPipelineBarrier");
    device_state->cmd_copy_buffer =
        load_device_function<PFN_vkCmdCopyBuffer>(get_device_proc, device, "vkCmdCopyBuffer");
    device_state->cmd_fill_buffer =
        load_device_function<PFN_vkCmdFillBuffer>(get_device_proc, device, "vkCmdFillBuffer");
    device_state->reset_command_buffer =
        load_device_function<PFN_vkResetCommandBuffer>(get_device_proc, device, "vkResetCommandBuffer");
    device_state->allocate_memory =
        load_device_function<PFN_vkAllocateMemory>(get_device_proc, device, "vkAllocateMemory");
    device_state->free_memory =
        load_device_function<PFN_vkFreeMemory>(get_device_proc, device, "vkFreeMemory");
    device_state->map_memory = load_device_function<PFN_vkMapMemory>(get_device_proc, device, "vkMapMemory");
    device_state->unmap_memory =
        load_device_function<PFN_vkUnmapMemory>(get_device_proc, device, "vkUnmapMemory");
    device_state->flush_mapped_memory_ranges = load_device_function<PFN_vkFlushMappedMemoryRanges>(
        get_device_proc, device, "vkFlushMappedMemoryRanges");
    device_state->invalidate_mapped_memory_ranges = load_device_function<PFN_vkInvalidateMappedMemoryRanges>(
        get_device_proc, device, "vkInvalidateMappedMemoryRanges");
    device_state->queue_submit =
        load_device_function<PFN_vkQueueSubmit>(get_device_proc, device, "vkQueueSubmit");
    device_state->create_command_pool =
        load_device_function<PFN_vkCreateCommandPool>(get_device_proc, device, "vkCreateCommandPool");
    device_state->destroy_command_pool =
        load_device_function<PFN_vkDestroyCommandPool>(get_device_proc, device, "vkDestroyCommandPool");
    device_state->reset_command_pool =
        load_device_function<PFN_vkResetCommandPool>(get_device_proc, device, "vkResetCommandPool");
    device_state->allocate_command_buffers = load_device_function<PFN_vkAllocateCommandBuffers>(
        get_device_proc, device, "vkAllocateCommandBuffers");
    device_state->free_command_buffers =
        load_device_function<PFN_vkFreeCommandBuffers>(get_device_proc, device, "vkFreeCommandBuffers");
    device_state->begin_command_buffer =
        load_device_function<PFN_vkBeginCommandBuffer>(get_device_proc, device, "vkBeginCommandBuffer");
    device_state->end_command_buffer =
        load_device_function<PFN_vkEndCommandBuffer>(get_device_proc, device, "vkEndCommandBuffer");
    device_state->create_semaphore =
        load_device_function<PFN_vkCreateSemaphore>(get_device_proc, device, "vkCreateSemaphore");
    device_state->destroy_semaphore =
        load_device_function<PFN_vkDestroySemaphore>(get_device_proc, device, "vkDestroySemaphore");
    device_state->get_semaphore_counter_value = load_device_function<PFN_vkGetSemaphoreCounterValue>(
        get_device_proc, device, "vkGetSemaphoreCounterValue");
    if (device_state->get_semaphore_counter_value == nullptr)
        device_state->get_semaphore_counter_value = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
            get_device_proc(device, "vkGetSemaphoreCounterValueKHR"));
    device_state->wait_semaphores =
        load_device_function<PFN_vkWaitSemaphores>(get_device_proc, device, "vkWaitSemaphores");
    if (device_state->wait_semaphores == nullptr)
        device_state->wait_semaphores =
            reinterpret_cast<PFN_vkWaitSemaphores>(get_device_proc(device, "vkWaitSemaphoresKHR"));
    if (device_state->allocate_memory == nullptr || device_state->free_memory == nullptr ||
        device_state->map_memory == nullptr || device_state->unmap_memory == nullptr ||
        device_state->flush_mapped_memory_ranges == nullptr ||
        device_state->invalidate_mapped_memory_ranges == nullptr)
        return Error(ErrorCode::UnsupportedFeature, "Vulkan device does not provide memory entry points");
    if (device_state->queue_submit == nullptr)
        return Error(ErrorCode::UnsupportedFeature, "Vulkan device does not provide vkQueueSubmit");

    Api api;
    api.create_shader_module = device_state->create_shader_module;
    api.destroy_shader_module = device_state->destroy_shader_module;
    api.create_descriptor_set_layout = device_state->create_descriptor_set_layout;
    api.destroy_descriptor_set_layout = device_state->destroy_descriptor_set_layout;
    api.create_pipeline_layout = device_state->create_pipeline_layout;
    api.destroy_pipeline_layout = device_state->destroy_pipeline_layout;
    api.create_descriptor_pool = device_state->create_descriptor_pool;
    api.destroy_descriptor_pool = device_state->destroy_descriptor_pool;
    api.allocate_descriptor_sets = device_state->allocate_descriptor_sets;
    api.free_descriptor_sets = device_state->free_descriptor_sets;
    api.update_descriptor_sets = device_state->update_descriptor_sets;
    api.create_compute_pipelines = device_state->create_compute_pipelines;
    api.destroy_pipeline = device_state->destroy_pipeline;
    api.create_pipeline_cache = device_state->create_pipeline_cache;
    api.destroy_pipeline_cache = device_state->destroy_pipeline_cache;
    api.get_pipeline_cache_data = device_state->get_pipeline_cache_data;
    api.create_buffer = {device_state.get(), create_buffer};
    api.destroy_buffer = {device_state.get(), destroy_buffer};
    api.get_buffer_memory_requirements = {device_state.get(), get_buffer_memory_requirements};
    api.bind_buffer_memory = {device_state.get(), bind_buffer_memory};
    api.cmd_pipeline_barrier = {device_state.get(), cmd_pipeline_barrier};
    api.cmd_copy_buffer = {device_state.get(), cmd_copy_buffer};
    api.cmd_fill_buffer = {device_state.get(), cmd_fill_buffer};
    api.reset_command_buffer = {device_state.get(), reset_command_buffer};
    api.allocate_memory = {device_state.get(), allocate_memory};
    api.free_memory = {device_state.get(), free_memory};
    api.map_memory = {device_state.get(), map_memory};
    api.unmap_memory = {device_state.get(), unmap_memory};
    api.flush_mapped_memory_ranges = {device_state.get(), flush_mapped_memory_ranges};
    api.invalidate_mapped_memory_ranges = {device_state.get(), invalidate_mapped_memory_ranges};
    api.queue_submit = {device_state.get(), queue_submit};
    api.create_command_pool = {device_state.get(), create_command_pool};
    api.destroy_command_pool = {device_state.get(), destroy_command_pool};
    api.reset_command_pool = {device_state.get(), reset_command_pool};
    api.allocate_command_buffers = {device_state.get(), allocate_command_buffers};
    api.free_command_buffers = {device_state.get(), free_command_buffers};
    api.begin_command_buffer = {device_state.get(), begin_command_buffer};
    api.end_command_buffer = {device_state.get(), end_command_buffer};
    api.create_semaphore = {device_state.get(), create_semaphore};
    api.destroy_semaphore = {device_state.get(), destroy_semaphore};
    api.get_semaphore_counter_value = {device_state.get(), get_semaphore_counter_value};
    api.wait_semaphores = {device_state.get(), wait_semaphores};
    api.lifetime_ = std::move(device_state);
    return api;
}

struct FaultInjector::State {
    struct Rule {
        std::atomic<uint64_t> calls{0};
        std::atomic<uint64_t> fail_at{0};
        std::atomic<int32_t> result{VK_SUCCESS};
    };

    explicit State(Api api) : inner(std::move(api)) {}

    Api inner;
    std::array<Rule, 2> rules;

    static VkResult result(State& state, Function function) noexcept {
        auto& rule = state.rules[static_cast<size_t>(function)];
        const uint64_t call = rule.calls.fetch_add(1, std::memory_order_relaxed) + 1;
        const uint64_t fail_at = rule.fail_at.load(std::memory_order_relaxed);
        if (fail_at != 0 && call == fail_at)
            return static_cast<VkResult>(rule.result.load(std::memory_order_relaxed));
        return VK_SUCCESS;
    }

    static VkResult VKAPI_PTR allocate_memory(void* context, VkDevice device,
                                              const VkMemoryAllocateInfo* info,
                                              const VkAllocationCallbacks* allocator,
                                              VkDeviceMemory* memory) noexcept {
        auto& state = *static_cast<State*>(context);
        const VkResult failure = result(state, Function::AllocateMemory);
        if (failure != VK_SUCCESS) return failure;
        return state.inner.allocate_memory(device, info, allocator, memory);
    }

    static VkResult VKAPI_PTR queue_submit(void* context, VkQueue queue, uint32_t count,
                                           const VkSubmitInfo* submits, VkFence fence) noexcept {
        auto& state = *static_cast<State*>(context);
        const VkResult failure = result(state, Function::QueueSubmit);
        if (failure != VK_SUCCESS) return failure;
        return state.inner.queue_submit(queue, count, submits, fence);
    }
};

FaultInjector::FaultInjector(Api inner) : state_(std::make_shared<State>(std::move(inner))) {}
FaultInjector::~FaultInjector() = default;
FaultInjector::FaultInjector(FaultInjector&&) noexcept = default;
FaultInjector& FaultInjector::operator=(FaultInjector&&) noexcept = default;

Status FaultInjector::fail_on_nth(Function function, uint64_t call, VkResult result) {
    if (!state_) return Error(ErrorCode::InvalidArgument, "Fault injector has no dispatch state");
    if (call == 0 || static_cast<size_t>(function) >= state_->rules.size() || result == VK_SUCCESS) {
        return Error(ErrorCode::InvalidArgument,
                     "Fault injection requires a supported function, positive call, and failure result");
    }
    auto& rule = state_->rules[static_cast<size_t>(function)];
    rule.calls.store(0, std::memory_order_relaxed);
    rule.result.store(static_cast<int32_t>(result), std::memory_order_relaxed);
    rule.fail_at.store(call, std::memory_order_relaxed);
    return ok_status();
}

Api FaultInjector::api() const {
    Api result;
    if (!state_) return result;
    result.allocate_memory = {state_.get(), State::allocate_memory};
    result.free_memory = state_->inner.free_memory;
    result.map_memory = state_->inner.map_memory;
    result.unmap_memory = state_->inner.unmap_memory;
    result.flush_mapped_memory_ranges = state_->inner.flush_mapped_memory_ranges;
    result.invalidate_mapped_memory_ranges = state_->inner.invalidate_mapped_memory_ranges;
    result.queue_submit = {state_.get(), State::queue_submit};
    result.create_command_pool = state_->inner.create_command_pool;
    result.destroy_command_pool = state_->inner.destroy_command_pool;
    result.reset_command_pool = state_->inner.reset_command_pool;
    result.allocate_command_buffers = state_->inner.allocate_command_buffers;
    result.free_command_buffers = state_->inner.free_command_buffers;
    result.begin_command_buffer = state_->inner.begin_command_buffer;
    result.end_command_buffer = state_->inner.end_command_buffer;
    result.create_semaphore = state_->inner.create_semaphore;
    result.destroy_semaphore = state_->inner.destroy_semaphore;
    result.get_semaphore_counter_value = state_->inner.get_semaphore_counter_value;
    result.wait_semaphores = state_->inner.wait_semaphores;
    result.lifetime_ = state_;
    return result;
}

}  // namespace kora::vk
