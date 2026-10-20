#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

#include "vulkan/api.hpp"

namespace kora::detail {

struct GpuTimestampSample {
    std::uint64_t submission_id = 0;
    std::uint64_t queue_value = 0;
    std::uint64_t start_ticks = 0;
    std::uint64_t end_ticks = 0;
    std::uint64_t duration_nanoseconds = 0;
    std::uint64_t host_timestamp_nanoseconds = 0;
    std::uint64_t calibration_deviation_nanoseconds = 0;
};

Result<std::uint64_t> timestamp_delta(std::uint64_t start, std::uint64_t end, std::uint32_t valid_bits);
Result<std::uint64_t> timestamp_nanoseconds(std::uint64_t ticks, float period_nanoseconds);

/// Per-queue, bounded GPU timestamp ring. Poll only after the owning queue value has completed.
class TimestampRing final {
public:
    static constexpr std::size_t kCapacity = 128;

    TimestampRing(vk::Api api, VkDevice device, std::uint32_t valid_bits, float period_nanoseconds,
                  std::vector<VkTimeDomainKHR> calibrated_domains = {});
    ~TimestampRing();
    TimestampRing(const TimestampRing&) = delete;
    TimestampRing& operator=(const TimestampRing&) = delete;

    Status initialize();
    bool enabled() const noexcept { return query_pool_ != VK_NULL_HANDLE; }
    bool record_start(VkCommandBuffer command_buffer, std::uint64_t submission_id);
    void record_end(VkCommandBuffer command_buffer, std::uint64_t submission_id) const;
    void cancel(std::uint64_t submission_id) noexcept;
    void submitted(std::uint64_t submission_id, std::uint64_t queue_value);
    /// Reads only completed slots, without WAIT_BIT. Samples still pending remain in the ring.
    /// Some drivers publish results shortly after the queue signal; `ready_wait` bounds how long
    /// to poll for a completed slot. Hot completion paths pass zero and never block.
    Result<std::vector<GpuTimestampSample>> collect(std::uint64_t completed_queue_value,
                                                    std::chrono::nanoseconds ready_wait = {});
    /// Enable bounded historical retention for opt-in Trace export.
    void enable_trace_history(bool enabled) noexcept;
    std::uint64_t dropped_samples() const noexcept;
    std::uint64_t samples_collected() const noexcept;
    GpuTimestampSample latest_sample() const noexcept;
    /// Bounded history of completed samples for trace export.
    std::vector<GpuTimestampSample> history() const;

private:
    struct Slot {
        std::uint64_t submission_id = 0;
        std::uint64_t queue_value = 0;
        std::uint32_t query_index = 0;
        std::uint64_t host_timestamp = 0;
        std::uint64_t host_deviation = 0;
        bool recording = false;
        bool submitted = false;
    };

    bool capture_host_timestamp(std::uint64_t& timestamp, std::uint64_t& deviation) const;

    vk::Api api_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    std::uint32_t valid_bits_ = 0;
    std::vector<VkTimeDomainKHR> calibrated_domains_;
    float period_nanoseconds_ = 0.0F;
    mutable std::mutex mutex_;
    std::deque<Slot> slots_;
    std::uint64_t dropped_samples_ = 0;
    std::uint64_t samples_collected_ = 0;
    GpuTimestampSample latest_sample_{};
    bool trace_history_enabled_ = false;
    std::deque<GpuTimestampSample> history_;
};

}  // namespace kora::detail
