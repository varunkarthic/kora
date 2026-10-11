#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "kora/errors.hpp"

namespace kora::detail {

struct SchedulerCostInput {
    double dependency_ready_ns = 0;
    double backend_available_ns = 0;
    double input_transfer_ns = 0;
    double execute_ns = 0;
    double required_output_ns = 0;
    double synchronization_ns = 0;
    double predicted_usage_bytes = 0;
    double effective_budget_bytes = 0;
    double memory_penalty_scale_ns = 0;
    double uncertainty_sigma_ns = 0;
    double uncertainty_lambda = 1;
    double locality_saved_ns = 0;
    double locality_cap_ns = 0;
    /// Estimated movement downstream consumers incur if this candidate runs on this backend.
    double estimated_downstream_movement_ns = 0;
    /// Bounded-lookahead coefficient in [0, 1]; defaults to one quarter of estimated movement cost.
    double lookahead_beta = 0.25;
};

struct ScoreBreakdown {
    double ready_delay_ns = 0;
    double queue_delay_ns = 0;
    double input_transfer_ns = 0;
    double execute_ns = 0;
    double required_output_ns = 0;
    double synchronization_ns = 0;
    double memory_penalty_ns = 0;
    double uncertainty_penalty_ns = 0;
    double locality_bonus_ns = 0;
    /// Weighted downstream movement added by the graph-lookahead term.
    double downstream_movement_penalty_ns = 0;
    double predicted_finish_ns = 0;
};

Result<ScoreBreakdown> evaluate_scheduler_cost(const SchedulerCostInput& input);

/// Tracks estimated pending work and corrects it toward measured completion duration.
class QueueAvailability final {
public:
    Status add_estimated_work(std::uint64_t queue, double duration_ns);
    Status observe_completion(std::uint64_t queue, double estimated_duration_ns, double actual_duration_ns,
                              double correction_alpha = 0.25);
    Result<double> available_at_ns(std::uint64_t queue) const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, double> pending_ns_;
    std::unordered_map<std::uint64_t, double> correction_ns_;
};

}  // namespace kora::detail
