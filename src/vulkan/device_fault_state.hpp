#pragma once

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace kora::vk {

struct DeviceFaultAddressRecord {
    std::uint32_t type = 0;
    std::uint64_t reported_address = 0;
    std::uint64_t precision = 0;
};

struct DeviceFaultVendorRecord {
    std::string description;
    std::uint64_t code = 0;
    std::uint64_t data = 0;
};

struct DeviceFaultSnapshot {
    bool supported = false;
    bool queried = false;
    bool truncated = false;
    std::int32_t query_result = VK_SUCCESS;
    std::string description;
    std::vector<DeviceFaultAddressRecord> addresses;
    std::vector<DeviceFaultVendorRecord> vendor_records;
};

/// Shared by all queues belonging to a logical device.
class DeviceFaultState final {
public:
    using DiagnosticsQuery = std::function<DeviceFaultSnapshot()>;

    bool healthy() const noexcept { return !lost_.load(std::memory_order_acquire); }
    void set_diagnostics_query(DiagnosticsQuery query) {
        std::lock_guard lock(fault_mutex_);
        diagnostics_query_ = std::move(query);
        diagnostics_.supported = static_cast<bool>(diagnostics_query_);
    }
    void mark_lost(VkResult result, bool capture_diagnostics = true) noexcept {
        DiagnosticsQuery query;
        {
            std::lock_guard lock(fault_mutex_);
            if (lost_.load(std::memory_order_relaxed)) return;
            last_result_.store(static_cast<std::int32_t>(result), std::memory_order_relaxed);
            lost_.store(true, std::memory_order_release);
            if (capture_diagnostics) query = diagnostics_query_;
        }
        if (!query) return;
        try {
            auto snapshot = query();
            std::lock_guard lock(fault_mutex_);
            diagnostics_ = std::move(snapshot);
            diagnostics_.supported = true;
            diagnostics_.queried = true;
        } catch (...) {
            std::lock_guard lock(fault_mutex_);
            diagnostics_.supported = true;
            diagnostics_.queried = true;
            diagnostics_.query_result = VK_ERROR_UNKNOWN;
        }
    }
    VkResult last_result() const noexcept {
        return static_cast<VkResult>(last_result_.load(std::memory_order_relaxed));
    }
    DeviceFaultSnapshot diagnostics() const {
        std::lock_guard lock(fault_mutex_);
        return diagnostics_;
    }
    std::mutex& submission_mutex() noexcept { return submission_mutex_; }

private:
    std::mutex submission_mutex_;
    mutable std::mutex fault_mutex_;
    std::atomic<bool> lost_{false};
    std::atomic<std::int32_t> last_result_{VK_SUCCESS};
    DiagnosticsQuery diagnostics_query_;
    DeviceFaultSnapshot diagnostics_;
};

}  // namespace kora::vk
