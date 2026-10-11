#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "kora/errors.hpp"

namespace kora::detail {

enum class SchedulingPriority : std::uint8_t { Low, Normal, High, RealtimeHint };

struct SchedulerCandidate {
    std::uint64_t id = 0;
    double predicted_finish_ns = 0;
    std::uint64_t observations = 0;
    double wait_age_ns = 0;
    SchedulingPriority priority = SchedulingPriority::Normal;
    bool supported = true;
    bool healthy = true;
    bool memory_fits = true;
    bool allowed = true;
    bool affinity_matches = true;
    bool known_safe = false;
};

struct SchedulerPolicyOptions {
    bool deterministic = false;
    double cold_start_estimate_ns = 0;
    std::uint64_t stable_observations = 20;
    double low_confidence_exploration = 0.05;
    double stable_exploration = 0.01;
    double absolute_hysteresis_ns = 50'000;
    double relative_hysteresis = 0.10;
    double priority_aging_rate = 0;
    double maximum_aging_discount_ns = 0;
};

struct SchedulerSelection {
    std::uint64_t candidate_id = 0;
    double effective_finish_ns = 0;
    double exploration_probability = 0;
    bool explored = false;
    bool retained_by_hysteresis = false;
};

/// Selects among feasible candidates. exploration_draw is in [0,1) and is ignored in deterministic mode.
Result<SchedulerSelection> select_scheduler_candidate(const std::vector<SchedulerCandidate>& candidates,
                                                      const SchedulerPolicyOptions& options,
                                                      std::optional<std::uint64_t> current_candidate,
                                                      double exploration_draw);

}  // namespace kora::detail
