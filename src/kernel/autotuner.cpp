#include "kernel/autotuner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

namespace kora::detail::kernel {
namespace {

bool valid_local_size(const std::array<std::uint32_t, 3>& local_size, const DeviceInfo::Limits& limits) {
    std::uint64_t invocations = 1;
    for (std::size_t dimension = 0; dimension < local_size.size(); ++dimension) {
        const auto size = local_size[dimension];
        if (size == 0 || size > limits.max_compute_work_group_size[dimension] ||
            size > limits.max_compute_work_group_invocations / invocations)
            return false;
        invocations *= size;
    }
    return invocations <= limits.max_compute_work_group_invocations;
}

double median(std::vector<double> values) {
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() % 2 != 0) return *middle;
    const auto lower = *std::max_element(values.begin(), middle);
    return (lower + *middle) / 2.0;
}

}  // namespace

Result<std::vector<WorkgroupCandidate>> generate_workgroup_candidates(
    const std::array<std::uint32_t, 3>& base_local_size, bool local_size_id_based,
    const DeviceInfo::Limits& limits, const DeviceInfo::Subgroup& subgroup, std::size_t maximum_candidates) {
    if (maximum_candidates == 0)
        return Error(ErrorCode::InvalidArgument, "workgroup candidate limit must be nonzero");
    if (!valid_local_size(base_local_size, limits))
        return Error(ErrorCode::InvalidArgument, "base local size exceeds device workgroup limits");
    std::vector<WorkgroupCandidate> candidates{{base_local_size}};
    if (!local_size_id_based) return candidates;

    const auto yz = static_cast<std::uint64_t>(base_local_size[1]) * base_local_size[2];
    if (yz == 0 || yz > limits.max_compute_work_group_invocations) return candidates;
    const auto max_x = std::min(limits.max_compute_work_group_size[0],
                                static_cast<std::uint32_t>(limits.max_compute_work_group_invocations / yz));
    std::vector<std::uint32_t> widths;
    if (subgroup.supported && subgroup.size != 0) {
        for (std::uint32_t multiplier = 1; multiplier <= 16; multiplier *= 2) {
            const auto width = static_cast<std::uint64_t>(subgroup.size) * multiplier;
            if (width <= max_x) widths.push_back(static_cast<std::uint32_t>(width));
        }
    } else {
        for (std::uint32_t width = 32; width != 0 && width <= max_x; width *= 2) widths.push_back(width);
    }
    for (const auto width : widths) {
        auto local_size = base_local_size;
        local_size[0] = width;
        if (!valid_local_size(local_size, limits)) continue;
        if (std::none_of(candidates.begin(), candidates.end(),
                         [&](const auto& candidate) { return candidate.local_size == local_size; }))
            candidates.push_back({local_size});
    }
    if (candidates.size() > maximum_candidates) candidates.resize(maximum_candidates);
    return candidates;
}

std::string workgroup_performance_key(std::string_view device_key, std::uint32_t driver_version,
                                      std::string_view kernel_id, std::string_view entry_point,
                                      const WorkgroupCandidate& candidate) {
    return "workgroup/" + std::string(device_key) + "/" + std::to_string(driver_version) + "/" +
           std::string(kernel_id) + "/" + std::string(entry_point) + "/" +
           std::to_string(candidate.local_size[0]) + "x" + std::to_string(candidate.local_size[1]) + "x" +
           std::to_string(candidate.local_size[2]);
}

WorkgroupAutotuner::WorkgroupAutotuner(std::vector<WorkgroupCandidate> candidates, std::size_t base_candidate,
                                       std::uint32_t warmup_count, std::uint32_t measurement_count,
                                       double maximum_dispersion)
    : candidates_(std::move(candidates)),
      states_(candidates_.size()),
      base_candidate_(base_candidate < candidates_.size() ? base_candidate : 0),
      warmup_count_(warmup_count),
      measurement_count_(measurement_count),
      maximum_dispersion_(maximum_dispersion) {
    if (candidates_.empty()) candidates_.push_back({});
    if (states_.empty()) states_.resize(candidates_.size());
    if (warmup_count_ == 0 || measurement_count_ == 0 || !std::isfinite(maximum_dispersion_) ||
        maximum_dispersion_ < 0.0) {
        warmup_count_ = 2;
        measurement_count_ = 5;
        maximum_dispersion_ = 0.5;
    }
}

WorkgroupAutotuner::Trial WorkgroupAutotuner::reserve() {
    std::lock_guard lock(mutex_);
    if (selected_) return {*selected_, false, false};
    for (std::size_t offset = 0; offset < states_.size(); ++offset) {
        const auto index = (cursor_ + offset) % states_.size();
        auto& state = states_[index];
        if (state.evaluated) continue;
        const bool warmup = state.warmups + state.pending_warmups < warmup_count_;
        if (warmup) {
            ++state.pending_warmups;
        } else {
            if (state.measurements.size() + state.pending_measurements >= measurement_count_) continue;
            ++state.pending_measurements;
        }
        cursor_ = (index + 1) % states_.size();
        return {index, warmup, true};
    }
    return {base_candidate_, false, false};
}

void WorkgroupAutotuner::cancel(const Trial& trial) noexcept {
    if (!trial.tracked) return;
    std::lock_guard lock(mutex_);
    if (trial.candidate_index >= states_.size()) return;
    auto& state = states_[trial.candidate_index];
    auto& pending = trial.warmup ? state.pending_warmups : state.pending_measurements;
    if (pending != 0) --pending;
}

Status WorkgroupAutotuner::observe(const Trial& trial, double duration_nanoseconds) {
    if (!trial.tracked || trial.candidate_index >= states_.size() || !std::isfinite(duration_nanoseconds) ||
        duration_nanoseconds <= 0.0)
        return Error(ErrorCode::InvalidArgument, "autotuner observation is not a valid measured trial");
    std::lock_guard lock(mutex_);
    auto& state = states_[trial.candidate_index];
    auto& pending = trial.warmup ? state.pending_warmups : state.pending_measurements;
    if (pending != 0) --pending;
    if (trial.warmup) {
        ++state.warmups;
        return {};
    }
    state.measurements.push_back(duration_nanoseconds);
    if (state.measurements.size() >= measurement_count_) {
        const auto center = median(state.measurements);
        const auto minimum = *std::min_element(state.measurements.begin(), state.measurements.end());
        const auto maximum = *std::max_element(state.measurements.begin(), state.measurements.end());
        state.score = center;
        state.stable = center > 0.0 && (maximum - minimum) / center <= maximum_dispersion_;
        state.evaluated = true;
        evaluate_locked();
    }
    return {};
}

Status WorkgroupAutotuner::restore(std::size_t candidate_index, const PerformanceStats& stats) {
    if (candidate_index >= states_.size())
        return Error(ErrorCode::InvalidArgument, "persisted autotune candidate index is out of range");
    if (!std::isfinite(stats.median) || !std::isfinite(stats.minimum) || !std::isfinite(stats.maximum) ||
        stats.median <= 0.0 || stats.minimum <= 0.0 || stats.maximum < stats.minimum)
        return Error(ErrorCode::InvalidArgument, "persisted autotune statistics are invalid");
    std::lock_guard lock(mutex_);
    if (stats.count < measurement_count_) return {};
    auto& state = states_[candidate_index];
    state.score = stats.median;
    state.stable = (stats.maximum - stats.minimum) / stats.median <= maximum_dispersion_;
    state.evaluated = true;
    state.warmups = warmup_count_;
    evaluate_locked();
    return {};
}

void WorkgroupAutotuner::evaluate_locked() {
    if (!std::all_of(states_.begin(), states_.end(), [](const auto& state) { return state.evaluated; }))
        return;
    for (std::size_t index = 0; index < states_.size(); ++index) {
        if (states_[index].stable && (!selected_ || states_[index].score < states_[*selected_].score))
            selected_ = index;
    }
}

std::array<std::uint32_t, 3> WorkgroupAutotuner::selected_local_size() const noexcept {
    std::lock_guard lock(mutex_);
    return candidates_[selected_.value_or(base_candidate_)].local_size;
}

std::optional<std::size_t> WorkgroupAutotuner::selected_index() const noexcept {
    std::lock_guard lock(mutex_);
    return selected_;
}

bool WorkgroupAutotuner::ready() const noexcept {
    std::lock_guard lock(mutex_);
    return std::all_of(states_.begin(), states_.end(), [](const auto& state) { return state.evaluated; });
}

}  // namespace kora::detail::kernel
