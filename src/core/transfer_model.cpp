#include "core/transfer_model.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace kora::detail {

TransferSizeBucket transfer_size_bucket(std::uint64_t bytes) noexcept {
    const auto index = static_cast<std::uint8_t>(bytes == 0 ? 0 : std::bit_width(bytes) - 1);
    const auto lower = index == 0 ? 0ULL : (std::uint64_t{1} << index);
    const auto upper =
        index == 63 ? std::numeric_limits<std::uint64_t>::max() : (std::uint64_t{1} << (index + 1)) - 1;
    return {index, lower, upper};
}

TransferModelSet::TransferModelSet() : TransferModelSet(Options{0.2, 1.0}) {}

TransferModelSet::TransferModelSet(Options options) : options_(options) {
    if (!(options_.ewma_alpha > 0 && options_.ewma_alpha <= 1) ||
        !std::isfinite(options_.uma_direct_penalty) || options_.uma_direct_penalty < 1.0)
        options_ = Options{0.2, 1.0};
}

std::size_t TransferModelSet::path_index(TransferPath path) noexcept {
    const auto index = static_cast<std::size_t>(path);
    return index < kPathCount ? index : kPathCount;
}

Status TransferModelSet::observe(TransferPath path, std::uint64_t bytes, double duration_ns) {
    const auto index = path_index(path);
    if (index >= kPathCount || bytes == 0 || !std::isfinite(duration_ns) || duration_ns < 0)
        return Error(ErrorCode::InvalidArgument,
                     "transfer sample needs a valid path, nonzero size, and finite duration");
    const auto bucket = transfer_size_bucket(bytes).index;
    std::lock_guard lock(mutex_);
    auto& state = paths_[index];
    auto& stats = state.buckets[bucket];
    if (stats.observations == 0) {
        stats.ewma_bytes = static_cast<double>(bytes);
        stats.ewma_duration_ns = duration_ns;
    } else {
        const auto alpha = options_.ewma_alpha;
        stats.ewma_bytes += alpha * (static_cast<double>(bytes) - stats.ewma_bytes);
        stats.ewma_duration_ns += alpha * (duration_ns - stats.ewma_duration_ns);
    }
    ++stats.observations;
    state.initialized = true;
    ++state.model.observations;
    refit(state);
    return {};
}

void TransferModelSet::refit(PathState& state) {
    double mean_bytes = 0;
    double mean_time = 0;
    std::size_t count = 0;
    for (const auto& bucket : state.buckets) {
        if (bucket.observations == 0) continue;
        ++count;
        mean_bytes += bucket.ewma_bytes;
        mean_time += bucket.ewma_duration_ns;
    }
    if (count == 0) return;
    mean_bytes /= static_cast<double>(count);
    mean_time /= static_cast<double>(count);
    double covariance = 0;
    double variance = 0;
    for (const auto& bucket : state.buckets) {
        if (bucket.observations == 0) continue;
        const auto x = bucket.ewma_bytes - mean_bytes;
        covariance += x * (bucket.ewma_duration_ns - mean_time);
        variance += x * x;
    }
    auto slope = variance > 0 ? covariance / variance : mean_time / std::max(mean_bytes, 1.0);
    if (!std::isfinite(slope) || slope < 0) slope = 0;
    auto latency = mean_time - slope * mean_bytes;
    if (!std::isfinite(latency) || latency < 0) latency = 0;
    state.model.latency_ns = latency;
    state.model.nanoseconds_per_byte = slope;
}

Result<double> TransferModelSet::estimate_ns(TransferPath path, std::uint64_t bytes) const {
    const auto index = path_index(path);
    if (index >= kPathCount) return Error(ErrorCode::InvalidArgument, "transfer path is invalid");
    std::lock_guard lock(mutex_);
    const auto& state = paths_[index];
    if (!state.initialized) return Error(ErrorCode::InvalidArgument, "transfer path has no measured model");
    const auto size_cost = state.model.nanoseconds_per_byte * static_cast<double>(bytes);
    const auto penalty = path == TransferPath::UmaDirect ? options_.uma_direct_penalty : 1.0;
    const auto estimate = state.model.latency_ns + penalty * size_cost;
    if (!std::isfinite(estimate)) return Error(ErrorCode::Overflow, "transfer estimate overflowed");
    return estimate;
}

Result<TransferCostModel> TransferModelSet::model(TransferPath path) const {
    const auto index = path_index(path);
    if (index >= kPathCount) return Error(ErrorCode::InvalidArgument, "transfer path is invalid");
    std::lock_guard lock(mutex_);
    if (!paths_[index].initialized)
        return Error(ErrorCode::InvalidArgument, "transfer path has no measured model");
    return paths_[index].model;
}

Result<TransferBucketStats> TransferModelSet::bucket_stats(TransferPath path, std::uint8_t bucket) const {
    const auto index = path_index(path);
    if (index >= kPathCount || bucket >= kTransferSizeBucketCount)
        return Error(ErrorCode::InvalidArgument, "transfer path or size bucket is invalid");
    std::lock_guard lock(mutex_);
    return paths_[index].buckets[bucket];
}

Status TransferModelSet::set_uma_direct_penalty(double penalty) {
    if (!std::isfinite(penalty) || penalty < 1.0)
        return Error(ErrorCode::InvalidArgument, "UMA direct penalty must be finite and at least one");
    std::lock_guard lock(mutex_);
    options_.uma_direct_penalty = penalty;
    return {};
}

double TransferModelSet::uma_direct_penalty() const {
    std::lock_guard lock(mutex_);
    return options_.uma_direct_penalty;
}

}  // namespace kora::detail
