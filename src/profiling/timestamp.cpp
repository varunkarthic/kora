#include "profiling/timestamp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace kora::detail {

Result<std::uint64_t> timestamp_delta(std::uint64_t start, std::uint64_t end, std::uint32_t valid_bits) {
    if (valid_bits == 0 || valid_bits > 64)
        return Error(ErrorCode::InvalidArgument, "timestamp valid-bit count must be in [1, 64]");
    if (valid_bits == 64) return end - start;
    const auto mask = (std::uint64_t{1} << valid_bits) - 1;
    return (end - start) & mask;
}

Result<std::uint64_t> timestamp_nanoseconds(std::uint64_t ticks, float period_nanoseconds) {
    if (!std::isfinite(period_nanoseconds) || period_nanoseconds <= 0.0F)
        return Error(ErrorCode::InvalidArgument, "timestamp period must be finite and positive");
    const auto duration = static_cast<long double>(ticks) * period_nanoseconds;
    if (duration > static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
        return Error(ErrorCode::Overflow, "GPU timestamp duration exceeds nanosecond representation");
    return static_cast<std::uint64_t>(std::llround(duration));
}

TimestampRing::TimestampRing(vk::Api api, VkDevice device, std::uint32_t valid_bits, float period_nanoseconds,
                             std::vector<VkTimeDomainKHR> calibrated_domains)
    : api_(std::move(api)),
      device_(device),
      valid_bits_(valid_bits),
      calibrated_domains_(std::move(calibrated_domains)),
      period_nanoseconds_(period_nanoseconds) {}

TimestampRing::~TimestampRing() {
    if (query_pool_ != VK_NULL_HANDLE && api_.destroy_query_pool)
        api_.destroy_query_pool(device_, query_pool_, nullptr);
}

Status TimestampRing::initialize() {
    if (device_ == VK_NULL_HANDLE || valid_bits_ == 0 || valid_bits_ > 64 ||
        !std::isfinite(period_nanoseconds_) || period_nanoseconds_ <= 0.0F || !api_.create_query_pool ||
        !api_.destroy_query_pool || !api_.get_query_pool_results || !api_.cmd_write_timestamp ||
        !api_.cmd_reset_query_pool) {
        return Error(ErrorCode::UnsupportedFeature, "GPU timestamp profiling is unavailable on this queue");
    }
    VkQueryPoolCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    create_info.queryCount = static_cast<std::uint32_t>(kCapacity * 2);
    const auto result = api_.create_query_pool(device_, &create_info, nullptr, &query_pool_);
    if (result != VK_SUCCESS)
        return Error(ErrorCode::UnsupportedFeature, "Creating GPU timestamp query pool failed")
            .with_vk(result);
    return ok_status();
}

bool TimestampRing::capture_host_timestamp(std::uint64_t& timestamp, std::uint64_t& deviation) const {
    if (!api_.get_calibrated_timestamps) return false;
    const auto monotonic =
        std::find(calibrated_domains_.begin(), calibrated_domains_.end(), VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR);
    if (std::find(calibrated_domains_.begin(), calibrated_domains_.end(), VK_TIME_DOMAIN_DEVICE_KHR) ==
            calibrated_domains_.end() ||
        monotonic == calibrated_domains_.end())
        return false;
    const VkCalibratedTimestampInfoKHR infos[] = {
        {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, nullptr, VK_TIME_DOMAIN_DEVICE_KHR},
        {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, nullptr, *monotonic}};
    std::uint64_t values[2]{};
    std::uint64_t max_deviation = 0;
    const auto result = api_.get_calibrated_timestamps(device_, 2, infos, values, &max_deviation);
    if (result != VK_SUCCESS) return false;
    timestamp = values[1];
    deviation = max_deviation;
    return true;
}

bool TimestampRing::record_start(VkCommandBuffer command_buffer, std::uint64_t submission_id) {
    std::lock_guard lock(mutex_);
    if (!enabled() || command_buffer == VK_NULL_HANDLE || slots_.size() == kCapacity) {
        ++dropped_samples_;
        return false;
    }
    Slot slot;
    slot.submission_id = submission_id;
    slot.query_index = static_cast<std::uint32_t>((submission_id % kCapacity) * 2);
    slot.recording = true;
    static_cast<void>(capture_host_timestamp(slot.host_timestamp, slot.host_deviation));
    api_.cmd_reset_query_pool(command_buffer, query_pool_, slot.query_index, 2);
    api_.cmd_write_timestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_,
                             slot.query_index);
    slots_.push_back(slot);
    return true;
}

void TimestampRing::record_end(VkCommandBuffer command_buffer, std::uint64_t submission_id) const {
    std::lock_guard lock(mutex_);
    for (auto iter = slots_.rbegin(); iter != slots_.rend(); ++iter) {
        if (iter->submission_id == submission_id && iter->recording) {
            api_.cmd_write_timestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_,
                                     iter->query_index + 1);
            return;
        }
    }
}

void TimestampRing::cancel(std::uint64_t submission_id) noexcept {
    std::lock_guard lock(mutex_);
    for (auto iter = slots_.begin(); iter != slots_.end(); ++iter) {
        if (iter->submission_id == submission_id && !iter->submitted) {
            slots_.erase(iter);
            return;
        }
    }
}

void TimestampRing::submitted(std::uint64_t submission_id, std::uint64_t queue_value) {
    std::lock_guard lock(mutex_);
    for (auto& slot : slots_) {
        if (slot.submission_id == submission_id && slot.recording) {
            slot.queue_value = queue_value;
            slot.recording = false;
            slot.submitted = true;
            return;
        }
    }
}

Result<std::vector<GpuTimestampSample>> TimestampRing::collect(std::uint64_t completed_queue_value) {
    std::vector<GpuTimestampSample> samples;
    std::lock_guard lock(mutex_);
    while (!slots_.empty()) {
        auto& slot = slots_.front();
        if (!slot.submitted || slot.queue_value > completed_queue_value) break;
        std::uint64_t values[2]{};
        const auto result =
            api_.get_query_pool_results(device_, query_pool_, slot.query_index, 2, sizeof(values), values,
                                        sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
        if (result == VK_NOT_READY) break;
        if (result != VK_SUCCESS)
            return Error(ErrorCode::TransferFailed, "Reading completed GPU timestamps failed")
                .with_vk(result);
        auto delta = timestamp_delta(values[0], values[1], valid_bits_);
        if (!delta) return std::move(delta).error();
        auto duration = timestamp_nanoseconds(delta.value(), period_nanoseconds_);
        if (!duration) return std::move(duration).error();
        latest_sample_ =
            GpuTimestampSample{slot.submission_id, slot.queue_value,    values[0],          values[1],
                               duration.value(),   slot.host_timestamp, slot.host_deviation};
        ++samples_collected_;
        if (trace_history_enabled_) {
            history_.push_back(latest_sample_);
            if (history_.size() > 4096) history_.pop_front();
        }
        samples.push_back(latest_sample_);
        slots_.pop_front();
    }
    return samples;
}

std::uint64_t TimestampRing::dropped_samples() const noexcept {
    std::lock_guard lock(mutex_);
    return dropped_samples_;
}

void TimestampRing::enable_trace_history(bool enabled) noexcept {
    std::lock_guard lock(mutex_);
    trace_history_enabled_ = enabled;
    if (!enabled) history_.clear();
}

std::uint64_t TimestampRing::samples_collected() const noexcept {
    std::lock_guard lock(mutex_);
    return samples_collected_;
}

GpuTimestampSample TimestampRing::latest_sample() const noexcept {
    std::lock_guard lock(mutex_);
    return latest_sample_;
}

std::vector<GpuTimestampSample> TimestampRing::history() const {
    std::lock_guard lock(mutex_);
    return {history_.begin(), history_.end()};
}

}  // namespace kora::detail
