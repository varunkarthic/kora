#include <gtest/gtest.h>

#include "core/scheduler_policy.hpp"

namespace kora::detail {
namespace {

TEST(SchedulerPolicy, ExcludesUnsupportedUnhealthyAndConstrainedCandidates) {
    SchedulerCandidate unsupported{1, 1};
    unsupported.supported = false;
    SchedulerCandidate unhealthy{2, 2};
    unhealthy.healthy = false;
    SchedulerCandidate too_large{3, 3};
    too_large.memory_fits = false;
    SchedulerCandidate allowed{4, 4};
    const auto result =
        select_scheduler_candidate({unsupported, unhealthy, too_large, allowed}, {}, std::nullopt, 0.0);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.value().candidate_id, 4U);

    allowed.affinity_matches = false;
    EXPECT_EQ(select_scheduler_candidate({allowed}, {}, std::nullopt, 0.0).error().code,
              ErrorCode::NoEligibleDevice);
}

TEST(SchedulerPolicy, ColdStartPrefersKnownSafeOnTiesAndUsesCoarseEstimates) {
    SchedulerPolicyOptions options;
    options.cold_start_estimate_ns = 100;
    options.deterministic = true;
    SchedulerCandidate safe{9, 100};
    safe.known_safe = true;
    SchedulerCandidate unknown{1, 100};
    auto result = select_scheduler_candidate({unknown, safe}, options, std::nullopt, 0);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.value().candidate_id, 9U);
    unknown.predicted_finish_ns = 80;
    result = select_scheduler_candidate({unknown, safe}, options, std::nullopt, 0);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.value().candidate_id, 1U);
}

TEST(SchedulerPolicy, ExplorationIsBoundedConfidenceDecaysAndDeterminismDisablesIt) {
    SchedulerPolicyOptions options;
    options.absolute_hysteresis_ns = 1000;
    SchedulerCandidate a{1, 100, 0};
    SchedulerCandidate b{2, 101, 0};
    auto result = select_scheduler_candidate({a, b}, options, std::nullopt, 0.025);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.value().exploration_probability, 0.05);
    EXPECT_TRUE(result.value().explored);
    EXPECT_EQ(result.value().candidate_id, 2U);
    a.observations = 20;
    b.observations = 0;
    result = select_scheduler_candidate({a, b}, options, std::nullopt, 0.02);
    ASSERT_TRUE(result);
    EXPECT_DOUBLE_EQ(result.value().exploration_probability, 0.05);
    EXPECT_TRUE(result.value().explored);
    b.observations = 20;
    result = select_scheduler_candidate({a, b}, options, std::nullopt, 0.02);
    ASSERT_TRUE(result);
    EXPECT_NEAR(result.value().exploration_probability, 0.01, 1e-12);
    EXPECT_FALSE(result.value().explored);
    options.deterministic = true;
    result = select_scheduler_candidate({a, b}, options, std::nullopt, 0.0);
    ASSERT_TRUE(result);
    EXPECT_FALSE(result.value().explored);
    EXPECT_DOUBLE_EQ(result.value().exploration_probability, 0);
    EXPECT_EQ(result.value().candidate_id, 1U);
}

TEST(SchedulerPolicy, HysteresisRequiresAbsoluteAndRelativeAdvantage) {
    SchedulerPolicyOptions options;
    options.deterministic = true;
    options.absolute_hysteresis_ns = 10;
    options.relative_hysteresis = 0.10;
    SchedulerCandidate current{1, 100, 10};
    SchedulerCandidate faster{2, 95, 10};
    auto result = select_scheduler_candidate({current, faster}, options, 1, 0);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.value().candidate_id, 1U);
    EXPECT_TRUE(result.value().retained_by_hysteresis);
    faster.predicted_finish_ns = 80;
    result = select_scheduler_candidate({current, faster}, options, 1, 0);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.value().candidate_id, 2U);
    EXPECT_FALSE(result.value().retained_by_hysteresis);
}

TEST(SchedulerPolicy, PriorityAgingPreventsLowPriorityStarvationAndTiesAreStable) {
    SchedulerPolicyOptions options;
    options.deterministic = true;
    options.priority_aging_rate = 0.2;
    options.maximum_aging_discount_ns = 40;
    SchedulerCandidate normal{8, 100, 10};
    SchedulerCandidate aged_low{9, 130, 10};
    aged_low.priority = SchedulingPriority::Low;
    aged_low.wait_age_ns = 100;
    auto result = select_scheduler_candidate({normal, aged_low}, options, std::nullopt, 0);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.value().candidate_id, 9U);
    EXPECT_DOUBLE_EQ(result.value().effective_finish_ns, 90);

    SchedulerCandidate tie_a{4, 50, 2};
    SchedulerCandidate tie_b{3, 50, 2};
    result = select_scheduler_candidate({tie_a, tie_b}, options, std::nullopt, 0);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.value().candidate_id, 3U);
}

TEST(SchedulerPolicy, RejectsInvalidPolicyAndCandidateNumbers) {
    SchedulerPolicyOptions options;
    options.low_confidence_exploration = 1.1;
    EXPECT_FALSE(select_scheduler_candidate({SchedulerCandidate{1, 10}}, options, std::nullopt, 0));
    options = {};
    EXPECT_FALSE(select_scheduler_candidate({SchedulerCandidate{1, -1}}, options, std::nullopt, 0));
    EXPECT_FALSE(select_scheduler_candidate({SchedulerCandidate{1, 10}}, options, std::nullopt, 1));
}

}  // namespace
}  // namespace kora::detail
