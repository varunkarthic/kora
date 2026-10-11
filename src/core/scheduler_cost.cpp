#include "core/scheduler_cost.hpp"

#include <algorithm>
#include <cmath>

namespace kora::detail {
namespace {
bool finite_nonnegative(double value) {
    return std::isfinite(value) && value >= 0;
}
}  // namespace

Result<ScoreBreakdown> evaluate_scheduler_cost(const SchedulerCostInput& input) {
    const double values[] = {input.dependency_ready_ns,     input.backend_available_ns,
                             input.input_transfer_ns,       input.execute_ns,
                             input.required_output_ns,      input.synchronization_ns,
                             input.predicted_usage_bytes,   input.effective_budget_bytes,
                             input.memory_penalty_scale_ns, input.uncertainty_sigma_ns,
                             input.uncertainty_lambda,      input.locality_saved_ns,
                             input.locality_cap_ns,         input.estimated_downstream_movement_ns,
                             input.lookahead_beta};
    for (const auto value : values)
        if (!finite_nonnegative(value))
            return Error(ErrorCode::InvalidArgument,
                         "scheduler costs and policy inputs must be finite and nonnegative");
    if (input.lookahead_beta > 1.0)
        return Error(ErrorCode::InvalidArgument, "scheduler look-ahead beta must be in [0, 1]");
    if (input.effective_budget_bytes == 0 && input.predicted_usage_bytes > 0)
        return Error(ErrorCode::InvalidArgument,
                     "positive predicted usage requires an effective memory budget");
    const double usage_ratio =
        input.effective_budget_bytes == 0 ? 0 : input.predicted_usage_bytes / input.effective_budget_bytes;
    if (!std::isfinite(usage_ratio) || usage_ratio >= 1.0)
        return Error(ErrorCode::InvalidArgument, "candidate exceeds effective memory budget");

    ScoreBreakdown score;
    score.ready_delay_ns = input.dependency_ready_ns;
    score.queue_delay_ns = std::max(0.0, input.backend_available_ns - input.dependency_ready_ns);
    score.input_transfer_ns = input.input_transfer_ns;
    score.execute_ns = input.execute_ns;
    score.required_output_ns = input.required_output_ns;
    score.synchronization_ns = input.synchronization_ns;
    if (usage_ratio > 0.70) {
        const double normalized = (usage_ratio - 0.70) / 0.30;
        score.memory_penalty_ns = input.memory_penalty_scale_ns * normalized * normalized;
    }
    score.uncertainty_penalty_ns = input.uncertainty_lambda * input.uncertainty_sigma_ns;
    score.locality_bonus_ns = std::min(input.locality_saved_ns, input.locality_cap_ns);
    score.downstream_movement_penalty_ns = input.lookahead_beta * input.estimated_downstream_movement_ns;
    score.predicted_finish_ns = score.ready_delay_ns + score.queue_delay_ns + score.input_transfer_ns +
                                score.execute_ns + score.required_output_ns + score.synchronization_ns +
                                score.memory_penalty_ns + score.uncertainty_penalty_ns -
                                score.locality_bonus_ns + score.downstream_movement_penalty_ns;
    if (!std::isfinite(score.predicted_finish_ns) || score.predicted_finish_ns < 0)
        return Error(ErrorCode::InvalidArgument,
                     "scheduler look-ahead produces a non-finite or negative predicted finish");
    return score;
}

Status QueueAvailability::add_estimated_work(std::uint64_t queue, double duration_ns) {
    if (!finite_nonnegative(duration_ns))
        return Error(ErrorCode::InvalidArgument, "queue duration must be finite and nonnegative");
    std::lock_guard lock(mutex_);
    const auto current = pending_ns_[queue];
    const auto updated = current + duration_ns;
    if (!std::isfinite(updated)) return Error(ErrorCode::Overflow, "queue estimate overflowed");
    pending_ns_[queue] = updated;
    return {};
}

Status QueueAvailability::observe_completion(std::uint64_t queue, double estimated_duration_ns,
                                             double actual_duration_ns, double correction_alpha) {
    if (!finite_nonnegative(estimated_duration_ns) || !finite_nonnegative(actual_duration_ns) ||
        !std::isfinite(correction_alpha) || correction_alpha <= 0 || correction_alpha > 1)
        return Error(ErrorCode::InvalidArgument, "queue completion sample or correction alpha is invalid");
    std::lock_guard lock(mutex_);
    const auto error = actual_duration_ns - estimated_duration_ns;
    auto [it, inserted] = correction_ns_.try_emplace(queue, error);
    if (!inserted) it->second += correction_alpha * (error - it->second);
    auto pending = pending_ns_.find(queue);
    if (pending != pending_ns_.end())
        pending->second = std::max(0.0, pending->second - estimated_duration_ns);
    return {};
}

Result<double> QueueAvailability::available_at_ns(std::uint64_t queue) const {
    std::lock_guard lock(mutex_);
    const auto pending = pending_ns_.find(queue);
    const auto correction = correction_ns_.find(queue);
    const auto estimate = (pending == pending_ns_.end() ? 0.0 : pending->second) +
                          (correction == correction_ns_.end() ? 0.0 : correction->second);
    return std::max(0.0, estimate);
}

}  // namespace kora::detail
