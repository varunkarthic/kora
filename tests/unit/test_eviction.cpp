#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "memory/eviction.hpp"

namespace {

using kora::BufferId;
using kora::ErrorCode;
using kora::detail::AccessHistory;
using kora::detail::EvictionAction;
using kora::detail::EvictionCandidate;
using kora::detail::EvictionPolicy;
using kora::detail::EvictionPolicyConfig;

EvictionCandidate candidate(std::uint32_t slot, std::uint64_t bytes, double probability = 0.5) {
    EvictionCandidate value;
    value.id = BufferId{slot, 1};
    value.resident_bytes = bytes;
    value.reuse_probability = probability;
    value.estimated_reload_cost_ms = 100.0;
    return value;
}

TEST(AccessHistory, RequiresTwoReferencesAndRejectsBackwardTime) {
    AccessHistory history;
    ASSERT_TRUE(history.record(2));
    EXPECT_EQ(history.size(), 1U);
    EXPECT_FALSE(history.kth_recent_age(10));
    EXPECT_EQ(history.record(1).error().code, ErrorCode::InvalidArgument);
    ASSERT_TRUE(history.record(4));
    ASSERT_TRUE(history.record(7));
    EXPECT_EQ(history.size(), AccessHistory::kDistance);
    EXPECT_EQ(history.most_recent_tick(), 7U);
    EXPECT_EQ(history.kth_recent_age(10), 6U);
}

TEST(EvictionPolicy, KeepValueMatchesWeightedFormulaAndLRUKRecency) {
    EvictionPolicy policy(
        EvictionPolicyConfig{.recent_access_weight = 2.0, .idle_age_penalty_per_tick = 0.5});
    const auto id = BufferId{3, 9};
    ASSERT_TRUE(policy.record_access(id, 1));
    ASSERT_TRUE(policy.record_access(id, 4));
    auto value = candidate(id.slot, 64, 0.5);
    value.id = id;
    value.last_access_tick = 0;
    value.estimated_reload_cost_ms = 20.0;
    value.user_priority_bonus = 1.0;
    value.expensive_recreation_bonus = 1.0;
    value.device_newer = true;
    value.estimated_writeback_cost_ms = 3.0;
    const auto keep = policy.keep_value(value, 6);
    ASSERT_TRUE(keep);
    EXPECT_NEAR(keep.value(), 10.0 + (2.0 / 6.0) + 1.0 + 1.0 + 3.0 - 1.0, 1e-9);

    EvictionPolicy one_touch(
        EvictionPolicyConfig{.recent_access_weight = 100.0, .idle_age_penalty_per_tick = 0.5});
    ASSERT_TRUE(one_touch.record_access(value.id, 5));
    const auto once = one_touch.keep_value(value, 6);
    ASSERT_TRUE(once);
    EXPECT_DOUBLE_EQ(once.value(), 10.0 + 1.0 + 1.0 + 3.0 - 0.5);
}

TEST(EvictionPolicy, SelectsLowestKeepValueUntilRequestedCapacityIsMet) {
    EvictionPolicy policy;
    std::array<EvictionCandidate, 3> candidates{candidate(4, 60, 0.9), candidate(2, 40, 0.1),
                                                candidate(3, 40, 0.2)};
    const auto plan = policy.plan(candidates, 20, 100, 0);
    ASSERT_TRUE(plan) << plan.error().describe();
    ASSERT_EQ(plan.value().decisions.size(), 2U);
    EXPECT_EQ(plan.value().decisions[0].id.slot, 2U);
    EXPECT_EQ(plan.value().decisions[1].id.slot, 3U);
    EXPECT_EQ(plan.value().bytes_to_reclaim, 80U);
}

TEST(EvictionPolicy, NeverSelectsPinnedInFlightOrBeingSubmittedBuffers) {
    auto pinned = candidate(1, 128);
    pinned.pin_count = 1;
    auto in_flight = candidate(2, 128);
    in_flight.in_flight = true;
    auto submitting = candidate(3, 128);
    submitting.being_submitted = true;
    auto eligible = candidate(4, 32, 0.0);
    std::array<EvictionCandidate, 4> candidates{pinned, in_flight, submitting, eligible};
    const auto plan = EvictionPolicy{}.plan(candidates, 0, 32, 0);
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan.value().decisions.size(), 1U);
    EXPECT_EQ(plan.value().decisions.front().id.slot, 4U);
}

TEST(EvictionPolicy, DeviceNewerRequiresWritebackUnlessDisposableOrReplicated) {
    auto dirty = candidate(1, 64);
    dirty.device_newer = true;
    dirty.estimated_writeback_cost_ms = 5.0;
    auto disposable = candidate(2, 64);
    disposable.device_newer = true;
    disposable.disposable = true;
    auto replicated = candidate(3, 64);
    replicated.device_newer = true;
    replicated.alternate_current_copy = true;
    std::array<EvictionCandidate, 3> candidates{dirty, disposable, replicated};
    const auto plan = EvictionPolicy{}.plan(candidates, 0, 192, 0);
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan.value().decisions.size(), 3U);
    EXPECT_EQ(plan.value().decisions[0].action, EvictionAction::Evict);
    EXPECT_EQ(plan.value().decisions[1].action, EvictionAction::Evict);
    EXPECT_EQ(plan.value().decisions[2].action, EvictionAction::WritebackThenEvict);
}

TEST(EvictionPolicy, LowBudgetStressNeverReturnsProtectedOrStaleGenerations) {
    EvictionPolicy policy;
    constexpr std::uint32_t count = 128;
    std::vector<EvictionCandidate> candidates;
    candidates.reserve(count);
    for (std::uint32_t slot = 0; slot < count; ++slot) {
        auto value = candidate(slot, 16, static_cast<double>(slot % 10) / 10.0);
        value.id.generation = slot + 1;
        value.last_access_tick = slot;
        if (slot % 17 == 0) value.pin_count = 1;
        if (slot % 19 == 0) value.in_flight = true;
        if (slot % 23 == 0) value.being_submitted = true;
        candidates.push_back(value);
        if (slot % 7 == 0) {
            ASSERT_TRUE(policy.record_access(value.id, slot));
            ASSERT_TRUE(policy.record_access(value.id, slot + count));
        }
    }
    const auto plan = policy.plan(candidates, 64, 512, count * 2);
    ASSERT_TRUE(plan) << plan.error().describe();
    EXPECT_GE(plan.value().bytes_to_reclaim, 448U);
    for (const auto& decision : plan.value().decisions) {
        const auto found = std::find_if(candidates.begin(), candidates.end(),
                                        [&](const auto& value) { return value.id == decision.id; });
        ASSERT_NE(found, candidates.end());
        EXPECT_EQ(found->pin_count, 0U);
        EXPECT_FALSE(found->in_flight);
        EXPECT_FALSE(found->being_submitted);
        EXPECT_EQ(found->id.generation, decision.id.generation);
    }
}

TEST(EvictionPolicy, InsufficientCapacityDoesNotReturnPartialPlan) {
    std::array<EvictionCandidate, 1> candidates{candidate(0, 32)};
    const auto plan = EvictionPolicy{}.plan(candidates, 0, 64, 0);
    ASSERT_FALSE(plan);
    EXPECT_EQ(plan.error().code, ErrorCode::OutOfMemoryBudget);
}

}  // namespace
