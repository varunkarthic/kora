#include "kora/task_partition.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

namespace kora {
namespace {

bool valid_rate(double value) {
    return std::isfinite(value) && value > 0;
}

struct Candidate {
    const PartitionDevice* device;
    double rate;
    std::uint64_t max_units;
};

}  // namespace

DynamicPartitioner::DynamicPartitioner(double observation_alpha) : alpha_(observation_alpha) {}

std::size_t DynamicPartitioner::DeviceKeyHash::operator()(const DeviceId& device) const noexcept {
    const auto first = std::hash<std::uint32_t>{}(device.index);
    const auto second = std::hash<std::string>{}(device.uuid);
    return first ^ (second + 0x9e3779b9U + (first << 6U) + (first >> 2U));
}

Status DynamicPartitioner::observe(const DeviceId& device, std::uint64_t work_units, double duration_ns) {
    if (work_units == 0 || !std::isfinite(duration_ns) || duration_ns <= 0 || !std::isfinite(alpha_) ||
        alpha_ <= 0 || alpha_ > 1)
        return Error(ErrorCode::InvalidArgument, "partition observation must have positive finite values");
    const auto sample = static_cast<double>(work_units) / duration_ns;
    if (!valid_rate(sample)) return Error(ErrorCode::InvalidArgument, "partition throughput is invalid");
    auto [it, inserted] = throughput_.try_emplace(device, sample);
    if (!inserted) it->second = alpha_ * sample + (1.0 - alpha_) * it->second;
    return ok_status();
}

Result<PartitionPlan> DynamicPartitioner::partition(const PartitionRequest& request,
                                                    const std::vector<PartitionDevice>& devices) const {
    const double numeric[] = {request.estimated_unsplit_duration_ns, request.partition_overhead_ns,
                              request.minimum_gain_fraction};
    for (const auto value : numeric)
        if (!std::isfinite(value) || value < 0)
            return Error(ErrorCode::InvalidArgument, "partition estimates must be finite and nonnegative");
    if (request.total_work_units == 0 || request.bytes_per_work_unit == 0 || request.granularity == 0 ||
        request.minimum_chunk_units == 0 || request.minimum_gain_fraction >= 1 || devices.empty())
        return Error(ErrorCode::InvalidArgument, "partition request or device set is empty or invalid");

    const auto chunks = request.total_work_units / request.granularity;
    const auto tail = request.total_work_units % request.granularity;
    if (chunks == 0 || request.total_work_units < request.minimum_chunk_units)
        return Error(ErrorCode::InvalidArgument, "work extent is smaller than its partition granularity");
    std::vector<Candidate> eligible;
    eligible.reserve(devices.size());
    for (const auto& device : devices) {
        const auto observed = throughput_.find(device.id);
        const auto rate = observed == throughput_.end() ? device.initial_throughput : observed->second;
        if (!valid_rate(rate) || device.available_memory_bytes < request.bytes_per_work_unit) continue;
        const auto max_units = device.available_memory_bytes / request.bytes_per_work_unit;
        if (max_units >= request.minimum_chunk_units) eligible.push_back({&device, rate, max_units});
    }
    if (eligible.empty())
        return Error(ErrorCode::OutOfMemoryBudget, "no device can fit the minimum task chunk");

    const auto choose_best = [&]() {
        return std::min_element(
            eligible.begin(), eligible.end(), [&](const Candidate& left, const Candidate& right) {
                return static_cast<double>(std::min(left.max_units, request.total_work_units)) / left.rate <
                       static_cast<double>(std::min(right.max_units, request.total_work_units)) / right.rate;
            });
    };
    auto best = choose_best();
    const auto best_capacity = std::min(best->max_units, request.total_work_units);
    const auto unsplit_time = request.estimated_unsplit_duration_ns > 0
                                  ? request.estimated_unsplit_duration_ns
                                  : static_cast<double>(request.total_work_units) / best->rate;
    PartitionPlan fallback;
    if (best_capacity == request.total_work_units) {
        fallback = {
            {PartitionSlice{best->device->id, 0, request.total_work_units, 1.0}}, unsplit_time, false};
    }
    const auto fallback_or_error = [&]() -> Result<PartitionPlan> {
        if (fallback.slices.empty())
            return Error(ErrorCode::OutOfMemoryBudget, "no single device can fit the unsplit task");
        return fallback;
    };
    if (eligible.size() < 2 || chunks < 2 || request.total_work_units < 2 * request.minimum_chunk_units)
        return fallback_or_error();

    // Split only on full granularity units. The final slice absorbs the caller's tail.
    const auto min_chunks = std::max<std::uint64_t>(
        1, (request.minimum_chunk_units + request.granularity - 1) / request.granularity);
    if (chunks < min_chunks * 2) return fallback_or_error();
    const auto total_rate = std::accumulate(eligible.begin(), eligible.end(), 0.0,
                                            [](double sum, const Candidate& c) { return sum + c.rate; });
    std::vector<std::uint64_t> allocations(eligible.size(), 0);
    std::vector<double> weights;
    weights.reserve(eligible.size());
    std::uint64_t assigned = 0;
    for (const auto& candidate : eligible) {
        weights.push_back(candidate.rate / total_rate);
        allocations[weights.size() - 1] = min_chunks;
        assigned += min_chunks;
    }
    if (assigned > chunks) return fallback_or_error();
    auto remaining = chunks - assigned;
    while (remaining > 0) {
        std::size_t selected = eligible.size();
        double largest_deficit = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < eligible.size(); ++i) {
            const auto max_chunks = eligible[i].max_units / request.granularity;
            if (allocations[i] >= max_chunks) continue;
            const auto deficit =
                weights[i] * static_cast<double>(chunks) - static_cast<double>(allocations[i]);
            if (deficit > largest_deficit) {
                largest_deficit = deficit;
                selected = i;
            }
        }
        if (selected == eligible.size()) return fallback_or_error();
        ++allocations[selected];
        --remaining;
    }

    PartitionPlan plan;
    plan.split = true;
    std::uint64_t offset = 0;
    double slowest = 0;
    for (std::size_t i = 0; i < eligible.size(); ++i) {
        auto units = allocations[i] * request.granularity;
        if (i + 1 == eligible.size()) units += tail;
        plan.slices.push_back({eligible[i].device->id, offset, units, weights[i]});
        offset += units;
        slowest = std::max(slowest, static_cast<double>(units) / eligible[i].rate);
    }
    plan.estimated_makespan_ns = slowest + request.partition_overhead_ns;
    if (offset != request.total_work_units) return fallback_or_error();
    if (plan.estimated_makespan_ns >= unsplit_time * (1.0 - request.minimum_gain_fraction) &&
        !fallback.slices.empty())
        return fallback;
    return plan;
}

}  // namespace kora
