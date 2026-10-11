#pragma once

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>
#include <mutex>

namespace kora::vk {

/// Shared by all queues belonging to a logical device.
class DeviceFaultState final {
public:
    bool healthy() const noexcept { return !lost_.load(std::memory_order_acquire); }
    void mark_lost(VkResult result) noexcept {
        std::lock_guard lock(fault_mutex_);
        if (lost_.load(std::memory_order_relaxed)) return;
        last_result_.store(static_cast<std::int32_t>(result), std::memory_order_relaxed);
        lost_.store(true, std::memory_order_release);
    }
    VkResult last_result() const noexcept {
        return static_cast<VkResult>(last_result_.load(std::memory_order_relaxed));
    }
    std::mutex& submission_mutex() noexcept { return submission_mutex_; }

private:
    std::mutex submission_mutex_;
    std::mutex fault_mutex_;
    std::atomic<bool> lost_{false};
    std::atomic<std::int32_t> last_result_{VK_SUCCESS};
};

}  // namespace kora::vk
