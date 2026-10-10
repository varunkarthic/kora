#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
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

/// Owns a dynamically loaded Vulkan loader and creates device-scoped dispatch tables.
class Loader {
public:
    static Result<Loader> load();

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
