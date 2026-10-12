#include "core/scheduler_placement.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <unordered_map>

#include "core/calibration_cost.hpp"
#include "core/scheduler_cost.hpp"
#include "core/scheduler_policy.hpp"
#include "kora/version.hpp"

namespace kora::detail {
namespace {

bool eligible(const RuntimePlacementCandidate& candidate, const AdaptiveDispatchWorkload& workload) {
    const bool fits_memory =
        workload.required_device_memory_bytes == 0 ||
        (candidate.available_device_memory_bytes > 0 &&
         workload.required_device_memory_bytes < candidate.available_device_memory_bytes);
    return candidate.supported && candidate.healthy && candidate.allowed && fits_memory;
}

}  // namespace

Result<AdaptivePlacementDecision> select_adaptive_device(
    const std::vector<RuntimePlacementCandidate>& candidates, const DeviceId& current,
    const AdaptiveDispatchWorkload& workload) {
    if (workload.kernel_work == 0)
        return Error(ErrorCode::InvalidArgument, "adaptive dispatch kernel work must be nonzero");
    if (static_cast<std::uint8_t>(workload.kernel_kind) >
        static_cast<std::uint8_t>(AdaptiveKernelWorkKind::AluOperations))
        return Error(ErrorCode::InvalidArgument, "adaptive dispatch workload has an invalid kernel kind");
    const auto current_entry = std::find_if(candidates.begin(), candidates.end(),
                                            [&](const auto& candidate) { return candidate.id == current; });
    if (current_entry == candidates.end() || !eligible(*current_entry, workload))
        return Error(ErrorCode::NoEligibleDevice,
                     "the kernel's current device is not eligible for the adaptive workload");

    std::vector<SchedulerCandidate> scored;
    scored.reserve(candidates.size());
    std::unordered_map<std::uint64_t, ScoreBreakdown> breakdowns;
    breakdowns.reserve(candidates.size());
    bool found_current = false;
    for (const auto& candidate : candidates) {
        if (!eligible(candidate, workload)) continue;
        const auto current_candidate = candidate.id == current;
        found_current = found_current || current_candidate;
        CalibratedKernelKind kind = CalibratedKernelKind::TinyDispatch;
        switch (workload.kernel_kind) {
            case AdaptiveKernelWorkKind::TinyDispatch:
                kind = CalibratedKernelKind::TinyDispatch;
                break;
            case AdaptiveKernelWorkKind::StreamingBytes:
                kind = CalibratedKernelKind::Streaming;
                break;
            case AdaptiveKernelWorkKind::AluOperations:
                kind = CalibratedKernelKind::Alu;
                break;
        }
        const CalibrationCostWorkload cost_workload{
            .input_bytes = workload.input_bytes,
            .output_bytes = workload.output_bytes,
            .device_copy_bytes = workload.device_copy_bytes,
            .kernel_work = workload.kernel_work,
            .kernel_kind = kind,
        };
        auto measured = estimate_calibration_cost(candidate.calibration_record, candidate.id.uuid,
                                                  candidate.driver_version, cost_workload);
        if (!measured) return std::move(measured).error();
        if (!measured.value().complete_record) return AdaptivePlacementDecision{.device = current};

        SchedulerCostInput cost;
        cost.input_transfer_ns =
            measured.value().input_transfer_ns.value_or(0) + measured.value().device_copy_ns.value_or(0);
        cost.execute_ns = measured.value().kernel_ns.value_or(0);
        if (candidate.online_execution && candidate.online_execution->count >= 5 &&
            std::isfinite(candidate.online_execution->ewma) &&
            std::isfinite(candidate.online_execution->ew_variance) && candidate.online_execution->ewma > 0 &&
            candidate.online_execution->ew_variance >= 0) {
            const auto& observation = *candidate.online_execution;
            const auto confidence =
                static_cast<double>(observation.count) / (static_cast<double>(observation.count) + 5.0);
            const auto conservative_sample = observation.ewma + std::sqrt(observation.ew_variance);
            cost.execute_ns = (1.0 - confidence) * cost.execute_ns + confidence * conservative_sample;
        }
        cost.required_output_ns = measured.value().output_transfer_ns.value_or(0);
        cost.synchronization_ns = measured.value().submit_completion_ns.value_or(0);
        cost.predicted_usage_bytes = static_cast<double>(workload.required_device_memory_bytes);
        cost.effective_budget_bytes = static_cast<double>(candidate.available_device_memory_bytes);
        cost.memory_penalty_scale_ns = cost.execute_ns + cost.input_transfer_ns + cost.required_output_ns;
        auto score = evaluate_scheduler_cost(cost);
        if (!score) continue;
        breakdowns.insert_or_assign(candidate.id.index, score.value());
        scored.push_back(SchedulerCandidate{
            .id = candidate.id.index,
            .predicted_finish_ns = score.value().predicted_finish_ns,
            .observations = 1,
            .supported = candidate.supported,
            .healthy = candidate.healthy,
            .memory_fits = workload.required_device_memory_bytes == 0 ||
                           workload.required_device_memory_bytes < candidate.available_device_memory_bytes,
            .allowed = candidate.allowed,
            .affinity_matches = true,
            .known_safe = current_candidate,
        });
    }
    if (!found_current)
        return Error(ErrorCode::NoEligibleDevice,
                     "the kernel's current device is not eligible for the adaptive workload");
    SchedulerPolicyOptions policy;
    policy.deterministic = true;
    policy.absolute_hysteresis_ns = 0;
    policy.relative_hysteresis = 0;
    auto selected = select_scheduler_candidate(scored, policy, current.index, 0);
    if (!selected) return std::move(selected).error();
    const auto found = std::find_if(candidates.begin(), candidates.end(), [&](const auto& candidate) {
        return candidate.id.index == selected.value().candidate_id;
    });
    if (found == candidates.end())
        return Error(ErrorCode::InternalInvariantViolation, "scheduler selected an unknown device candidate");
    const auto score = breakdowns.find(found->id.index);
    if (score == breakdowns.end())
        return Error(ErrorCode::InternalInvariantViolation,
                     "scheduler selection is missing its score breakdown");
    return AdaptivePlacementDecision{.device = found->id, .score = score->second};
}

}  // namespace kora::detail
