#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "memory/recovery.hpp"

namespace {
using kora::ErrorCode;
using kora::detail::AdmissionEstimate;
using kora::detail::estimate_admission;
using kora::detail::OomRecoveryHooks;
using kora::detail::recover_allocation_failure;
using kora::detail::recover_task_allocation_failure;
using kora::detail::RecoveryAction;

TEST(AdmissionEstimate, AccountsForEvictionAndReportsRemainingBudget) {
    const auto estimate = estimate_admission(40, 70, 50, 2.5, 0.2);
    ASSERT_TRUE(estimate);
    EXPECT_TRUE(estimate.value().admitted);
    EXPECT_EQ(estimate.value().budget_after_admission_bytes, 20U);
    EXPECT_DOUBLE_EQ(estimate.value().eviction_cost_ms, 2.5);
}

TEST(AdmissionEstimate, RejectsInvalidRiskAndDetectsOverflow) {
    EXPECT_EQ(estimate_admission(10, 1, 1, 0, 1.1).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(estimate_admission(UINT64_MAX, 1, 1, 0, 0).error().code, ErrorCode::Overflow);
    const auto insufficient = estimate_admission(10, 12, 1, 0, 0);
    ASSERT_TRUE(insufficient);
    EXPECT_FALSE(insufficient.value().admitted);
}

TEST(OomRecovery, ExecutesPreparationAndAlternativesOnceInRequiredOrder) {
    std::vector<int> order;
    OomRecoveryHooks hooks;
    hooks.refresh_budget = [&] {
        order.push_back(1);
        return kora::ok_status();
    };
    hooks.reclaim_deferred = [&] {
        order.push_back(2);
        return kora::ok_status();
    };
    hooks.evict_eligible = [&] {
        order.push_back(3);
        return kora::ok_status();
    };
    hooks.retry_once = [&]() -> kora::Result<bool> {
        order.push_back(4);
        return false;
    };
    hooks.smaller_block = [&]() -> kora::Result<bool> {
        order.push_back(5);
        return true;
    };
    hooks.retry_smaller_allocation = [&]() -> kora::Result<bool> {
        order.push_back(6);
        return false;
    };
    hooks.chunking = [&]() -> kora::Result<bool> {
        order.push_back(7);
        return true;
    };
    hooks.stop_after_allocation = false;
    const auto result = recover_allocation_failure(hooks);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.value().action, RecoveryAction::Chunking);
    EXPECT_EQ(order, (std::vector<int>{1, 2, 3, 4, 5, 6, 7}));
}

TEST(OomRecovery, SmallerBlockIsNotReportedAsAllocationSuccessAndRetryRunsOnce) {
    int smaller_block = 0;
    int retry = 0;
    OomRecoveryHooks hooks;
    hooks.retry_once = [&]() -> kora::Result<bool> {
        ++retry;
        return false;
    };
    hooks.smaller_block = [&]() -> kora::Result<bool> {
        ++smaller_block;
        return true;
    };
    hooks.retry_smaller_allocation = [&]() -> kora::Result<bool> {
        ++retry;
        return true;
    };
    const auto result = recover_allocation_failure(hooks);
    ASSERT_TRUE(result);
    EXPECT_TRUE(result.value().recovered);
    EXPECT_EQ(result.value().action, RecoveryAction::RetrySmallerAllocation);
    EXPECT_EQ(smaller_block, 1);
    EXPECT_EQ(retry, 2);
}

TEST(OomRecovery, PreparationFailuresDoNotSuppressLaterRecoveryOptions) {
    int refresh = 0;
    int reclaim = 0;
    int eviction = 0;
    int retry = 0;
    OomRecoveryHooks hooks;
    hooks.refresh_budget = [&] {
        ++refresh;
        return kora::Status(kora::Error(ErrorCode::TransferFailed, "refresh"));
    };
    hooks.reclaim_deferred = [&] {
        ++reclaim;
        return kora::ok_status();
    };
    hooks.evict_eligible = [&] {
        ++eviction;
        return kora::Status(kora::Error(ErrorCode::OutOfMemoryBudget, "evict"));
    };
    hooks.retry_once = [&]() -> kora::Result<bool> {
        ++retry;
        return true;
    };
    const auto result = recover_allocation_failure(hooks);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.value().action, RecoveryAction::Retry);
    EXPECT_EQ(refresh, 1);
    EXPECT_EQ(reclaim, 1);
    EXPECT_EQ(eviction, 1);
    EXPECT_EQ(retry, 1);
}

TEST(OomRecovery, TriesEachFallbackAtMostOnceAndReturnsPreciseExhaustion) {
    int count = 0;
    OomRecoveryHooks hooks;
    hooks.retry_once = [&]() -> kora::Result<bool> {
        ++count;
        return false;
    };
    hooks.smaller_block = [&]() -> kora::Result<bool> {
        ++count;
        return false;
    };
    hooks.retry_smaller_allocation = [&]() -> kora::Result<bool> {
        ++count;
        return false;
    };
    hooks.chunking = [&]() -> kora::Result<bool> {
        ++count;
        return false;
    };
    hooks.cpu_fallback = [&]() -> kora::Result<bool> {
        ++count;
        return false;
    };
    const auto result = recover_allocation_failure(hooks);
    ASSERT_FALSE(result);
    EXPECT_EQ(count, 5);
    EXPECT_EQ(result.error().code, ErrorCode::OutOfMemoryBudget);
    EXPECT_NE(result.error().recovery_action.find("cpu_fallback"), std::string::npos);
}

TEST(OomRecovery, ConvertsThrownHooksToStructuredErrors) {
    OomRecoveryHooks hooks;
    hooks.retry_once = []() -> kora::Result<bool> { throw std::runtime_error("injected"); };
    const auto result = recover_allocation_failure(hooks);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::InternalInvariantViolation);
}

TEST(OomRecovery, TaskPolicyContinuesToChunkingAndCpuFallbackAfterAllocationRetries) {
    int retry = 0;
    int smaller = 0;
    int chunk = 0;
    int cpu = 0;
    OomRecoveryHooks hooks;
    hooks.retry_once = [&]() -> kora::Result<bool> {
        ++retry;
        return true;
    };
    hooks.smaller_block = [&]() -> kora::Result<bool> {
        ++smaller;
        return false;
    };
    hooks.retry_smaller_allocation = [&]() -> kora::Result<bool> {
        ++smaller;
        return false;
    };
    hooks.chunking = [&]() -> kora::Result<bool> {
        ++chunk;
        return false;
    };
    hooks.cpu_fallback = [&]() -> kora::Result<bool> {
        ++cpu;
        return true;
    };
    const auto result = recover_task_allocation_failure(hooks);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.value().action, RecoveryAction::CpuFallback);
    EXPECT_EQ(retry, 1);
    EXPECT_EQ(smaller, 2);
    EXPECT_EQ(chunk, 1);
    EXPECT_EQ(cpu, 1);
}

TEST(OomRecovery, PreservesCallerSpecificTerminalErrorAndAddsStageTrace) {
    OomRecoveryHooks hooks;
    hooks.terminal_error =
        kora::Error(kora::ErrorCode::AllocationFailed, "vkAllocateMemory failed").with_vk(-2);
    hooks.retry_once = []() -> kora::Result<bool> { return false; };
    hooks.smaller_block = []() -> kora::Result<bool> { return false; };
    const auto result = recover_allocation_failure(hooks);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::AllocationFailed);
    EXPECT_EQ(result.error().vk_result, -2);
    EXPECT_NE(result.error().recovery_action.find("retry_once"), std::string::npos);
}
}  // namespace
