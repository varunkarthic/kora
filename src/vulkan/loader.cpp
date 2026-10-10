#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
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
    PFN_vkAllocateMemory allocate_memory = nullptr;
    PFN_vkQueueSubmit queue_submit = nullptr;
};

VkResult VKAPI_PTR allocate_memory(void* context, VkDevice device, const VkMemoryAllocateInfo* info,
                                   const VkAllocationCallbacks* allocator, VkDeviceMemory* memory) noexcept {
    const auto& state = *static_cast<DeviceState*>(context);
    return state.allocate_memory(device, info, allocator, memory);
}

VkResult VKAPI_PTR queue_submit(void* context, VkQueue queue, uint32_t count, const VkSubmitInfo* submits,
                                VkFence fence) noexcept {
    const auto& state = *static_cast<DeviceState*>(context);
    return state.queue_submit(queue, count, submits, fence);
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

}  // namespace

struct Instance::State {
    std::shared_ptr<LoaderState> loader;
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    PFN_vkDestroyInstance destroy_instance = nullptr;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug_messenger = nullptr;
    uint32_t api_version = VK_API_VERSION_1_0;
    bool validation_enabled = false;

    ~State() {
        if (instance != VK_NULL_HANDLE && debug_messenger != VK_NULL_HANDLE &&
            destroy_debug_messenger != nullptr) {
            destroy_debug_messenger(instance, debug_messenger, nullptr);
        }
        if (instance != VK_NULL_HANDLE && destroy_instance != nullptr) destroy_instance(instance, nullptr);
    }
};

Result<Loader> Loader::load() {
    auto state = std::make_shared<LoaderState>();
#if defined(_WIN32)
    state->library = LoadLibraryA(library_name());
#else
    state->library = dlopen(library_name(), RTLD_NOW | RTLD_LOCAL);
#endif
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
    return Loader(std::move(state));
}

Result<Instance> Loader::create_instance() const {
    return create_instance(InstanceOptions{});
}

Result<Instance> Loader::create_instance(const InstanceOptions& options) const {
    if (!state_) return Error(ErrorCode::NoVulkanImplementation, "Vulkan loader is not initialized");
    auto loader = std::static_pointer_cast<LoaderState>(state_);
    auto enumerate_version = load_instance_function<PFN_vkEnumerateInstanceVersion>(
        loader->get_instance_proc_addr, VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    uint32_t supported_version = VK_API_VERSION_1_0;
    if (enumerate_version != nullptr) {
        const VkResult result = enumerate_version(&supported_version);
        if (result != VK_SUCCESS) {
            return Error(ErrorCode::NoVulkanImplementation, "Failed to query Vulkan loader version")
                .with_vk(result);
        }
    }
    const uint32_t requested =
        options.requested_api_version == 0 ? VK_API_VERSION_1_0 : options.requested_api_version;
    const uint32_t api_version =
        std::min({requested, supported_version, static_cast<uint32_t>(VK_API_VERSION_1_3)});

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
    if (api_version == VK_API_VERSION_1_0 &&
        has_extension(available_extensions, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
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
    device_state->allocate_memory =
        load_device_function<PFN_vkAllocateMemory>(get_device_proc, device, "vkAllocateMemory");
    device_state->queue_submit =
        load_device_function<PFN_vkQueueSubmit>(get_device_proc, device, "vkQueueSubmit");
    if (device_state->allocate_memory == nullptr)
        return Error(ErrorCode::UnsupportedFeature, "Vulkan device does not provide vkAllocateMemory");
    if (device_state->queue_submit == nullptr)
        return Error(ErrorCode::UnsupportedFeature, "Vulkan device does not provide vkQueueSubmit");

    Api api;
    api.allocate_memory = {device_state.get(), allocate_memory};
    api.queue_submit = {device_state.get(), queue_submit};
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
    result.queue_submit = {state_.get(), State::queue_submit};
    result.lifetime_ = state_;
    return result;
}

}  // namespace kora::vk
