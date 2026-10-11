#include "memory/recovery.hpp"

#include <array>
#include <cmath>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "kora/detail/checked.hpp"

namespace kora::detail {
namespace {
constexpr const char* kRecoveryStageNames[] = {
    "refresh_budget", "reclaim_deferred",         "evict_eligible", "retry_once",
    "smaller_block",  "retry_smaller_allocation", "chunking",       "cpu_fallback"};
}

Result<AdmissionEstimate> estimate_admission(std::uint64_t available_bytes, std::uint64_t required_new_bytes,
                                             std::uint64_t likely_eviction_bytes, double eviction_cost_ms,
                                             double fragmentation_risk) {
    if (!std::isfinite(eviction_cost_ms) || eviction_cost_ms < 0.0 || !std::isfinite(fragmentation_risk) ||
        fragmentation_risk < 0.0 || fragmentation_risk > 1.0) {
        return Error(ErrorCode::InvalidArgument,
                     "admission eviction cost must be nonnegative and fragmentation risk must be in [0,1]");
    }
    const auto capacity = checked::add(available_bytes, likely_eviction_bytes);
    if (!capacity) return Error(ErrorCode::Overflow, "admission capacity arithmetic overflow");
    const auto admitted = *capacity >= required_new_bytes;
    const auto after = admitted ? *capacity - required_new_bytes : 0;
    return AdmissionEstimate{
        required_new_bytes, likely_eviction_bytes, eviction_cost_ms, after, fragmentation_risk, admitted};
}

Result<RecoveryTrace> recover_allocation_failure(const OomRecoveryHooks& hooks) {
    RecoveryTrace trace;
    const auto prepare = [&trace](const std::function<Status()>& callback, const char* name) -> Status {
        if (!callback) return ok_status();
        try {
            trace.attempted_stages.push_back(name);
            return callback();
        } catch (...) {
            return Error(ErrorCode::InternalInvariantViolation,
                         "allocation recovery preparation callback threw");
        }
    };
    const auto attempt = [&trace](const std::function<Result<bool>()>& callback, const char* name,
                                  RecoveryAction action) -> Result<bool> {
        if (!callback) return false;
        try {
            trace.attempted_stages.push_back(name);
            auto result = callback();
            if (!result)
                return Error(ErrorCode::InternalInvariantViolation,
                             "allocation recovery attempt callback returned an error")
                    .with_cause(std::move(result).error());
            if (result.value()) trace.action = action;
            return result;
        } catch (...) {
            return Error(ErrorCode::InternalInvariantViolation, "allocation recovery attempt callback threw");
        }
    };
    try {
        (void)prepare(hooks.refresh_budget, "refresh_budget");
        (void)prepare(hooks.reclaim_deferred, "reclaim_deferred");
        (void)prepare(hooks.evict_eligible, "evict_eligible");
        const std::array stages{
            std::tuple{&hooks.retry_once, kRecoveryStageNames[3], RecoveryAction::Retry},
            std::tuple{&hooks.smaller_block, kRecoveryStageNames[4], RecoveryAction::SmallerBlock},
            std::tuple{&hooks.retry_smaller_allocation, kRecoveryStageNames[5],
                       RecoveryAction::RetrySmallerAllocation},
            std::tuple{&hooks.chunking, kRecoveryStageNames[6], RecoveryAction::Chunking},
            std::tuple{&hooks.cpu_fallback, kRecoveryStageNames[7], RecoveryAction::CpuFallback}};
        for (const auto& [callback, name, action] : stages) {
            auto result = attempt(*callback, name, action);
            if (!result) return std::move(result).error();
            if (!result.value()) continue;
            if (result.value()) {
                trace.action = action;
                trace.recovered = true;
                if (hooks.stop_after_allocation) {
                    if (action == RecoveryAction::SmallerBlock) continue;
                    return trace;
                }
                if (action == RecoveryAction::Chunking || action == RecoveryAction::CpuFallback) return trace;
            }
        }
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to allocate bounded OOM recovery trace");
    } catch (const std::length_error&) {
        return Error(ErrorCode::Overflow, "OOM recovery trace exceeds container capacity");
    }
    std::string stages;
    try {
        for (const auto* stage : trace.attempted_stages) {
            if (!stages.empty()) stages += ", ";
            stages += stage;
        }
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to format bounded OOM recovery diagnostic");
    }
    auto error = hooks.terminal_error;
    if (error.code == ErrorCode::Ok) {
        error = Error(ErrorCode::OutOfMemoryBudget, "allocation recovery exhausted its bounded stages");
    }
    if (error.message.find("bounded recovery stages were exhausted") == std::string::npos)
        error.message += "; bounded recovery stages were exhausted";
    if (hooks.attempted_stages_output) *hooks.attempted_stages_output = trace.attempted_stages;
    if (!stages.empty()) error.with_recovery(std::move(stages));
    return error;
}

Result<RecoveryTrace> recover_task_allocation_failure(const OomRecoveryHooks& hooks) {
    auto task_hooks = hooks;
    task_hooks.stop_after_allocation = false;
    return recover_allocation_failure(task_hooks);
}

Result<RecoveryTrace> recover_allocation_failure(
    const std::function<Result<bool>()>& refresh_budget,
    const std::function<Result<bool>()>& reclaim_deferred,
    const std::function<Result<bool>()>& evict_eligible, const std::function<Result<bool>()>& retry_once,
    const std::function<Result<bool>()>& smaller_block,
    const std::function<Result<bool>()>& retry_smaller_allocation, Error terminal_error) {
    if (terminal_error.code == ErrorCode::Ok) {
        terminal_error =
            Error(ErrorCode::OutOfMemoryBudget, "allocation recovery exhausted its bounded stages");
    }
    OomRecoveryHooks hooks;
    const auto prepare = [](const std::function<Result<bool>()>& action) -> Status {
        if (!action) return ok_status();
        auto result = action();
        if (!result) return Status(std::move(result).error());
        return ok_status();
    };
    hooks.refresh_budget = [&] { return prepare(refresh_budget); };
    hooks.reclaim_deferred = [&] { return prepare(reclaim_deferred); };
    hooks.evict_eligible = [&] { return prepare(evict_eligible); };
    hooks.retry_once = retry_once;
    hooks.smaller_block = smaller_block;
    hooks.retry_smaller_allocation = retry_smaller_allocation;
    hooks.stop_after_allocation = true;
    hooks.terminal_error = std::move(terminal_error);
    return recover_allocation_failure(hooks);
}

}  // namespace kora::detail
