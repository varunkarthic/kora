#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

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
        if (callback_ == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
        return callback_(context_, args...);
    }

private:
    void* context_ = nullptr;
    Callback callback_ = nullptr;
};

/// Device-level Vulkan entry points used by the initial backend and fault-injection seam.
struct Api {
    DispatchFunction<VkResult(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*,
                              VkDeviceMemory*)>
        allocate_memory;
    DispatchFunction<VkResult(VkQueue, uint32_t, const VkSubmitInfo*, VkFence)> queue_submit;

private:
    friend class Loader;
    friend class FaultInjector;
    std::shared_ptr<const void> lifetime_;
};

class Loader;

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
