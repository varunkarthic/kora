#pragma once

#ifndef VK_ENABLE_BETA_EXTENSIONS
#define VK_ENABLE_BETA_EXTENSIONS
#endif
#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kora/errors.hpp"

namespace kora::vk {

template <class>
class DispatchFunction;

template <class R, class... Args>
class DispatchFunction<R(Args...)> {
public:
    using Callback = R(VKAPI_PTR*)(void*, Args...) noexcept;

    DispatchFunction() = default;
    DispatchFunction(void* context, Callback callback) : context_(context), callback_(callback) {}

    explicit operator bool() const noexcept { return callback_ != nullptr; }

    R operator()(Args... args) const noexcept {
        if constexpr (std::is_void_v<R>) {
            if (callback_ != nullptr) callback_(context_, args...);
        } else {
            if (callback_ == nullptr) return static_cast<R>(VK_ERROR_INITIALIZATION_FAILED);
            return callback_(context_, args...);
        }
    }

private:
    void* context_ = nullptr;
    Callback callback_ = nullptr;
};

/// Device-level Vulkan entry points used by the initial backend and fault-injection seam.
struct Api {
    bool synchronization2_enabled() const noexcept { return synchronization2_enabled_; }
    // Pipeline and descriptor entry points are raw Vulkan PFNs. The owning Api still retains the
    // DeviceState through lifetime_, and each consumer checks these optional feature-layer calls.
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
    PFN_vkCmdBindPipeline cmd_bind_pipeline = nullptr;
    PFN_vkCmdBindDescriptorSets cmd_bind_descriptor_sets = nullptr;
    PFN_vkCmdPushConstants cmd_push_constants = nullptr;
    PFN_vkCmdDispatch cmd_dispatch = nullptr;

    DispatchFunction<VkResult(VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer*)>
        create_buffer;
    DispatchFunction<void(VkDevice, VkBuffer, const VkAllocationCallbacks*)> destroy_buffer;
    DispatchFunction<void(VkDevice, VkBuffer, VkMemoryRequirements*)> get_buffer_memory_requirements;
    DispatchFunction<VkResult(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize)> bind_buffer_memory;
    DispatchFunction<void(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                          uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t,
                          const VkImageMemoryBarrier*)>
        cmd_pipeline_barrier;
    DispatchFunction<void(VkCommandBuffer, const VkDependencyInfo*)> cmd_pipeline_barrier2;
    DispatchFunction<void(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy*)>
        cmd_copy_buffer;
    DispatchFunction<void(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, uint32_t)> cmd_fill_buffer;
    DispatchFunction<VkResult(VkCommandBuffer, VkCommandBufferResetFlags)> reset_command_buffer;
    DispatchFunction<VkResult(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*,
                              VkDeviceMemory*)>
        allocate_memory;
    DispatchFunction<void(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*)> free_memory;
    DispatchFunction<VkResult(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void**)>
        map_memory;
    DispatchFunction<void(VkDevice, VkDeviceMemory)> unmap_memory;
    DispatchFunction<VkResult(VkDevice, uint32_t, const VkMappedMemoryRange*)> flush_mapped_memory_ranges;
    DispatchFunction<VkResult(VkDevice, uint32_t, const VkMappedMemoryRange*)>
        invalidate_mapped_memory_ranges;
    DispatchFunction<VkResult(VkQueue, uint32_t, const VkSubmitInfo*, VkFence)> queue_submit;
    DispatchFunction<VkResult(VkQueue, uint32_t, const VkSubmitInfo2*, VkFence)> queue_submit2;
    DispatchFunction<VkResult(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*,
                              VkCommandPool*)>
        create_command_pool;
    DispatchFunction<void(VkDevice, VkCommandPool, const VkAllocationCallbacks*)> destroy_command_pool;
    DispatchFunction<VkResult(VkDevice, VkCommandPool, VkCommandPoolResetFlags)> reset_command_pool;
    DispatchFunction<VkResult(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*)>
        allocate_command_buffers;
    DispatchFunction<void(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*)> free_command_buffers;
    DispatchFunction<VkResult(VkCommandBuffer, const VkCommandBufferBeginInfo*)> begin_command_buffer;
    DispatchFunction<VkResult(VkCommandBuffer)> end_command_buffer;
    DispatchFunction<VkResult(VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*,
                              VkSemaphore*)>
        create_semaphore;
    DispatchFunction<void(VkDevice, VkSemaphore, const VkAllocationCallbacks*)> destroy_semaphore;
    DispatchFunction<VkResult(VkDevice, VkSemaphore, uint64_t*)> get_semaphore_counter_value;
    DispatchFunction<VkResult(VkDevice, const VkSemaphoreWaitInfo*, uint64_t)> wait_semaphores;
    DispatchFunction<VkResult(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence*)>
        create_fence;
    DispatchFunction<void(VkDevice, VkFence, const VkAllocationCallbacks*)> destroy_fence;
    DispatchFunction<VkResult(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t)> wait_for_fences;
    DispatchFunction<VkResult(VkDevice, uint32_t, const VkFence*)> reset_fences;

private:
    friend class Loader;
    friend class FaultInjector;
    friend class Instance;
    std::shared_ptr<const void> lifetime_;
    bool synchronization2_enabled_ = false;
};

class Loader;
struct PhysicalDeviceInfo;

/// Raw queried capabilities for a physical device. The handle remains owned by its Instance.
struct PhysicalDeviceInfo {
    VkPhysicalDevice handle = VK_NULL_HANDLE;
    uint32_t api_version = VK_API_VERSION_1_0;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceIDProperties id_properties{};
    VkPhysicalDeviceFeatures features{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    VkPhysicalDeviceSubgroupProperties subgroup_properties{};
    std::vector<VkQueueFamilyProperties> queue_family_properties;
    std::vector<VkExtensionProperties> extensions;
    std::vector<VkTimeDomainKHR> calibrated_time_domains;

    VkPhysicalDeviceVulkan11Features vulkan11_features{};
    VkPhysicalDeviceVulkan11Properties vulkan11_properties{};
    VkPhysicalDeviceVulkan12Features vulkan12_features{};
    VkPhysicalDeviceVulkan12Properties vulkan12_properties{};
    VkPhysicalDeviceVulkan13Features vulkan13_features{};
    VkPhysicalDeviceVulkan13Properties vulkan13_properties{};

    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_features{};
    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features{};
    VkPhysicalDeviceBufferDeviceAddressFeaturesEXT buffer_device_address_features_ext{};
    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing_features{};
    VkPhysicalDeviceSynchronization2Features synchronization2_features{};
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_control_features{};
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_size_control_properties{};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT memory_budget_properties{};
    VkPhysicalDeviceMemoryPriorityFeaturesEXT memory_priority_features{};
#ifdef VK_KHR_device_fault
    VkPhysicalDeviceFaultFeaturesKHR device_fault_features{};
    VkPhysicalDeviceFaultPropertiesKHR device_fault_properties{};
#endif
#ifdef VK_EXT_device_fault
    VkPhysicalDeviceFaultFeaturesEXT device_fault_features_ext{};
#endif
    VkPhysicalDevicePortabilitySubsetFeaturesKHR portability_subset_features{};
    VkPhysicalDevicePortabilitySubsetPropertiesKHR portability_subset_properties{};
    VkPhysicalDeviceRobustness2FeaturesEXT robustness2_features{};
    VkPhysicalDeviceRobustness2PropertiesEXT robustness2_properties{};
};

struct DeviceOptions {
    bool enable_dedicated_transfer_queue = true;
    bool enable_timeline_semaphore = true;
    bool enable_synchronization2 = true;
};

struct BufferMemoryInfo {
    VkMemoryRequirements requirements{};
    std::uint32_t memory_type_count = 0;
    VkPhysicalDeviceMemoryProperties memory_properties{};
    VkPhysicalDeviceProperties properties{};
};

/// Owned logical device, selected queues, and device-level dispatch table.
class Device {
public:
    struct EnabledFeatures {
        bool timeline_semaphore = false;
        bool buffer_device_address = false;
        bool descriptor_indexing = false;
        bool synchronization2 = false;
    };

    Device() = default;
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;

    VkDevice handle() const noexcept;
    VkQueue compute_queue() const noexcept;
    VkQueue transfer_queue() const noexcept;
    uint32_t compute_queue_family() const noexcept;
    uint32_t transfer_queue_family() const noexcept;
    const EnabledFeatures& enabled_features() const noexcept;
    const std::vector<std::string>& enabled_extensions() const noexcept;
    const Api& api() const noexcept;
    const BufferMemoryInfo& buffer_memory_info() const noexcept;

private:
    friend class Instance;
    struct State;
    explicit Device(std::shared_ptr<State> state, Api api) : state_(std::move(state)), api_(std::move(api)) {}
    std::shared_ptr<State> state_;
    Api api_;
};

class Instance {
public:
    Instance() = default;
    ~Instance();
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) noexcept;
    Instance& operator=(Instance&&) noexcept;

    VkInstance handle() const noexcept;
    uint32_t api_version() const noexcept;
    bool validation_enabled() const noexcept;
    Result<std::vector<PhysicalDeviceInfo>> physical_devices() const;
    Result<Device> create_device(const PhysicalDeviceInfo& physical_device,
                                 const DeviceOptions& options = {}) const;

private:
    friend class Loader;
    struct State;
    explicit Instance(std::shared_ptr<State> state) : state_(std::move(state)) {}
    std::shared_ptr<State> state_;
};

/// Owns a dynamically loaded Vulkan loader and creates instances and device dispatch tables.
class Loader {
public:
    static Result<Loader> load();

    /// Highest Vulkan API version reported by the loader (Vulkan 1.0 when unavailable).
    uint32_t supported_api_version() const noexcept;

    struct InstanceOptions {
        uint32_t requested_api_version = VK_API_VERSION_1_3;
        std::string application_name = "KORA";
        uint32_t application_version = VK_MAKE_VERSION(0, 1, 0);
        bool enable_validation = false;
    };

    Result<Instance> create_instance(const InstanceOptions& options) const;
    Result<Instance> create_instance() const;
    Result<Api> device_api(VkInstance instance, VkDevice device) const;

private:
    explicit Loader(std::shared_ptr<void> state) : state_(std::move(state)) {}
    std::shared_ptr<void> state_;
};

enum class Function : uint8_t { AllocateMemory, QueueSubmit };

/// Decorates an Api with deterministic, per-function Nth-call failures.
class FaultInjector {
public:
    explicit FaultInjector(Api inner);
    ~FaultInjector();

    FaultInjector(const FaultInjector&) = delete;
    FaultInjector& operator=(const FaultInjector&) = delete;
    FaultInjector(FaultInjector&&) noexcept;
    FaultInjector& operator=(FaultInjector&&) noexcept;

    Status fail_on_nth(Function function, uint64_t call, VkResult result);
    Api api() const;

private:
    struct State;
    std::shared_ptr<State> state_;
};

}  // namespace kora::vk
