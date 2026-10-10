#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "vulkan/api.hpp"

#if defined(_WIN32)
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

}  // namespace

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
