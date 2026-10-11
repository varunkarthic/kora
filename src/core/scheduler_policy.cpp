#include "core/scheduler_policy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace kora::detail {
namespace {

bool valid_nonnegative(double value) {
    return std::isfinite(value) && value >= 0;
}

bool feasible(const SchedulerCandidate& candidate) {
    return candidate.supported && candidate.healthy && candidate.memory_fits && candidate.allowed &&
           candidate.affinity_matches;
}

double priority_discount(SchedulingPriority priority) {
    switch (priority) {
        case SchedulingPriority::Low:
            return 3;
        case SchedulingPriority::Normal:
            return 2;
        case SchedulingPriority::High:
            return 1;
        case SchedulingPriority::RealtimeHint:
            return 0;
    }
    return 0;
}

}  // namespace

Result<SchedulerSelection> select_scheduler_candidate(const std::vector<SchedulerCandidate>& candidates,
                                                      const SchedulerPolicyOptions& options,
                                                      std::optional<std::uint64_t> current_candidate,
                                                      double exploration_draw) {
    const double option_values[] = {options.cold_start_estimate_ns,    options.low_confidence_exploration,
                                    options.stable_exploration,        options.absolute_hysteresis_ns,
                                    options.relative_hysteresis,       options.priority_aging_rate,
                                    options.maximum_aging_discount_ns, exploration_draw};
    for (const auto value : option_values)
        if (!valid_nonnegative(value))
            return Error(ErrorCode::InvalidArgument,
                         "scheduler policy values must be finite and nonnegative");
    if (options.low_confidence_exploration > 1 ||
        options.stable_exploration > options.low_confidence_exploration || options.relative_hysteresis > 1 ||
        exploration_draw >= 1 || options.stable_observations == 0)
        return Error(ErrorCode::InvalidArgument,
                     "scheduler policy probability, margin, or confidence threshold is invalid");

    std::vector<const SchedulerCandidate*> eligible;
    eligible.reserve(candidates.size());
    std::uint64_t min_observations = std::numeric_limits<std::uint64_t>::max();
    for (const auto& candidate : candidates) {
        if (!valid_nonnegative(candidate.predicted_finish_ns) || !valid_nonnegative(candidate.wait_age_ns))
            return Error(ErrorCode::InvalidArgument,
                         "candidate estimates and wait ages must be finite and nonnegative");
        if (feasible(candidate)) {
            eligible.push_back(&candidate);
            min_observations = std::min(min_observations, candidate.observations);
        }
    }
    if (eligible.empty())
        return Error(ErrorCode::NoEligibleDevice, "no candidate satisfies scheduler constraints");

    const auto effective_score = [&](const SchedulerCandidate& candidate) {
        const double estimate = candidate.observations == 0
                                    ? (candidate.predicted_finish_ns > 0 ? candidate.predicted_finish_ns
                                                                         : options.cold_start_estimate_ns)
                                    : candidate.predicted_finish_ns;
        const auto aging =
            std::min(options.maximum_aging_discount_ns, options.priority_aging_rate * candidate.wait_age_ns *
                                                            priority_discount(candidate.priority));
        return std::max(0.0, estimate - aging);
    };
    std::sort(eligible.begin(), eligible.end(), [&](const auto* left, const auto* right) {
        const auto left_score = effective_score(*left);
        const auto right_score = effective_score(*right);
        if (left_score != right_score) return left_score < right_score;
        if (left->observations == 0 && right->observations == 0 && left->known_safe != right->known_safe)
            return left->known_safe;
        return left->id < right->id;
    });

    const auto confidence = std::min(
        1.0, static_cast<double>(min_observations) / static_cast<double>(options.stable_observations));
    const auto exploration_probability =
        options.deterministic
            ? 0.0
            : options.low_confidence_exploration +
                  confidence * (options.stable_exploration - options.low_confidence_exploration);
    const SchedulerCandidate* selected = eligible.front();
    bool explored = false;
    if (!options.deterministic && eligible.size() > 1 && exploration_draw < exploration_probability) {
        const auto plausible_limit =
            effective_score(*eligible.front()) * (1.0 + options.relative_hysteresis) +
            options.absolute_hysteresis_ns;
        const auto plausible_end = std::find_if(eligible.begin(), eligible.end(), [&](const auto* candidate) {
            return effective_score(*candidate) > plausible_limit;
        });
        const auto plausible_count = static_cast<std::size_t>(plausible_end - eligible.begin());
        if (plausible_count > 1) {
            const auto scaled =
                exploration_draw / exploration_probability * static_cast<double>(plausible_count - 1);
            const auto pick = 1 + static_cast<std::size_t>(scaled);
            selected = eligible[pick];
            explored = true;
        }
    }

    bool retained = false;
    if (current_candidate && !explored) {
        const auto current = std::find_if(eligible.begin(), eligible.end(), [&](const auto* candidate) {
            return candidate->id == *current_candidate;
        });
        if (current != eligible.end() && selected->id != *current_candidate) {
            const auto current_score = effective_score(**current);
            const auto advantage = current_score - effective_score(*selected);
            if (advantage < options.absolute_hysteresis_ns ||
                advantage < current_score * options.relative_hysteresis) {
                selected = *current;
                retained = true;
            }
        }
    }

    return SchedulerSelection{selected->id, effective_score(*selected), exploration_probability, explored,
                              retained};
}

}  // namespace kora::detail
