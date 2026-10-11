#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "kora/errors.hpp"

namespace kora::detail {

struct AdmissionEstimate {
    std::uint64_t required_new_bytes = 0;
    std::uint64_t likely_eviction_bytes = 0;
    double eviction_cost_ms = 0.0;
    std::uint64_t budget_after_admission_bytes = 0;
    double fragmentation_risk = 0.0;
    bool admitted = false;
};

enum class RecoveryAction : std::uint8_t {
    Retry,
    SmallerBlock,
    RetrySmallerAllocation,
    Chunking,
    CpuFallback
};
struct RecoveryTrace {
    RecoveryAction action = RecoveryAction::Retry;
    bool recovered = false;
    std::vector<const char*> attempted_stages;
};

Result<AdmissionEstimate> estimate_admission(std::uint64_t available_bytes, std::uint64_t required_new_bytes,
                                             std::uint64_t likely_eviction_bytes, double eviction_cost_ms,
                                             double fragmentation_risk);

/// Each callback is invoked at most once. Preparation callbacks may fail without preventing later
/// recovery options; unsuccessful alternatives proceed to the next bounded stage.
struct OomRecoveryHooks {
    std::function<Status()> refresh_budget;
    std::function<Status()> reclaim_deferred;
    std::function<Status()> evict_eligible;
    std::function<Result<bool>()> retry_once;
    std::function<Result<bool>()> smaller_block;
    std::function<Result<bool>()> retry_smaller_allocation;
    std::function<Result<bool>()> chunking;
    std::function<Result<bool>()> cpu_fallback;
    /// If true, successful retry/smaller-block stages mean allocation succeeded. Higher-level
    /// recovery callers set false and continue through chunking/CPU fallback after an allocation retry.
    bool stop_after_allocation = true;
    /// Optional fallback error used when every configured action returns false.
    Error terminal_error{ErrorCode::OutOfMemoryBudget, "allocation recovery exhausted its bounded stages"};
    /// Receives the ordered stage names for an exhausted result.
    std::vector<const char*>* attempted_stages_output = nullptr;
};

Result<RecoveryTrace> recover_allocation_failure(const OomRecoveryHooks& hooks);

/// Executes the fixed preparation sequence, then each allocation alternative at most once.
/// The retry callback is one allocation attempt, not a recursive call to the coordinator.
Result<RecoveryTrace> recover_allocation_failure(
    const std::function<Result<bool>()>& refresh_budget,
    const std::function<Result<bool>()>& reclaim_deferred,
    const std::function<Result<bool>()>& evict_eligible, const std::function<Result<bool>()>& retry_once,
    const std::function<Result<bool>()>& smaller_block,
    const std::function<Result<bool>()>& retry_smaller_allocation = {},
    Error terminal_error = Error(ErrorCode::OutOfMemoryBudget,
                                 "allocation recovery exhausted its bounded stages"));

/// Task-level order, where allocation retries can lead into streaming/chunking and CPU fallback.
Result<RecoveryTrace> recover_task_allocation_failure(const OomRecoveryHooks& hooks);

}  // namespace kora::detail
