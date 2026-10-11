#include <gtest/gtest.h>

#include "core/scheduler_cost.hpp"
#include "core/scheduler_policy.hpp"
#include "core/transfer_model.hpp"

namespace kora::detail {
namespace {

TEST(SchedulerCost, HandComputedBreakdownMatchesEftAndSumsToTotal) {
    SchedulerCostInput input;
    input.dependency_ready_ns = 100;
    input.backend_available_ns = 160;
    input.input_transfer_ns = 20;
    input.execute_ns = 30;
    input.required_output_ns = 10;
    input.synchronization_ns = 5;
    input.predicted_usage_bytes = 850;
    input.effective_budget_bytes = 1000;
    input.memory_penalty_scale_ns = 40;
    input.uncertainty_sigma_ns = 8;
    input.uncertainty_lambda = 0.5;
    input.locality_saved_ns = 50;
    input.locality_cap_ns = 25;
    input.estimated_downstream_movement_ns = 40;

    const auto result = evaluate_scheduler_cost(input);
    ASSERT_TRUE(result);
    const auto& score = result.value();
    EXPECT_DOUBLE_EQ(score.ready_delay_ns, 100);
    EXPECT_DOUBLE_EQ(score.queue_delay_ns, 60);
    EXPECT_DOUBLE_EQ(score.memory_penalty_ns, 10);
    EXPECT_DOUBLE_EQ(score.uncertainty_penalty_ns, 4);
    EXPECT_DOUBLE_EQ(score.locality_bonus_ns, 25);
    EXPECT_DOUBLE_EQ(score.downstream_movement_penalty_ns, 10);
    EXPECT_DOUBLE_EQ(score.predicted_finish_ns, 224);
    EXPECT_DOUBLE_EQ(score.ready_delay_ns + score.queue_delay_ns + score.input_transfer_ns +
                         score.execute_ns + score.required_output_ns + score.synchronization_ns +
                         score.memory_penalty_ns + score.uncertainty_penalty_ns - score.locality_bonus_ns +
                         score.downstream_movement_penalty_ns,
                     score.predicted_finish_ns);
}

TEST(SchedulerCost, LookaheadKeepsTwoGiBChainOnDownstreamPreferredDevice) {
    constexpr std::uint64_t kIntermediateBytes = 2ULL * 1024 * 1024 * 1024;
    constexpr double kCrossDeviceMovementNs = 200'000'000;
    TransferModelSet transfers;
    ASSERT_TRUE(
        transfers.observe(TransferPath::CrossDeviceViaHost, kIntermediateBytes, kCrossDeviceMovementNs));
    const auto movement = transfers.estimate_ns(TransferPath::CrossDeviceViaHost, kIntermediateBytes);
    ASSERT_TRUE(movement);
    EXPECT_DOUBLE_EQ(movement.value(), kCrossDeviceMovementNs);

    SchedulerCostInput gpu0;
    gpu0.execute_ns = 97'000'000;  // About 3% faster for the current node.
    gpu0.estimated_downstream_movement_ns = movement.value();
    SchedulerCostInput gpu1;
    gpu1.execute_ns = 100'000'000;
    const auto gpu0_score = evaluate_scheduler_cost(gpu0);
    const auto gpu1_score = evaluate_scheduler_cost(gpu1);
    ASSERT_TRUE(gpu0_score);
    ASSERT_TRUE(gpu1_score);
    EXPECT_DOUBLE_EQ(gpu0_score.value().downstream_movement_penalty_ns, 50'000'000);

    SchedulerPolicyOptions options;
    options.deterministic = true;
    const auto selected = select_scheduler_candidate(
        {{0, gpu0_score.value().predicted_finish_ns, 1}, {1, gpu1_score.value().predicted_finish_ns, 1}},
        options, std::nullopt, 0.5);
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected.value().candidate_id, 1U);
}

TEST(SchedulerCost, UsesBudgetKneeAndRejectsInfeasibleOrInvalidInputs) {
    SchedulerCostInput input;
    input.execute_ns = 100;
    input.effective_budget_bytes = 100;
    input.predicted_usage_bytes = 70;
    EXPECT_DOUBLE_EQ(evaluate_scheduler_cost(input).value().memory_penalty_ns, 0);
    input.predicted_usage_bytes = 85;
    input.memory_penalty_scale_ns = 100;
    EXPECT_NEAR(evaluate_scheduler_cost(input).value().memory_penalty_ns, 25.0, 1e-12);
    input.predicted_usage_bytes = 100;
    EXPECT_FALSE(evaluate_scheduler_cost(input));
    input.predicted_usage_bytes = 0;
    input.execute_ns = -1;
    EXPECT_FALSE(evaluate_scheduler_cost(input));
    input.execute_ns = 0;
    input.estimated_downstream_movement_ns = -1;
    EXPECT_FALSE(evaluate_scheduler_cost(input));
    input.estimated_downstream_movement_ns = 1;
    input.lookahead_beta = -0.1;
    EXPECT_FALSE(evaluate_scheduler_cost(input));
    input.lookahead_beta = 1.1;
    EXPECT_FALSE(evaluate_scheduler_cost(input));
}

TEST(QueueAvailability, TracksWorkAndCorrectsEstimatesFromRealCompletions) {
    QueueAvailability queues;
    ASSERT_TRUE(queues.add_estimated_work(7, 100));
    ASSERT_TRUE(queues.add_estimated_work(7, 50));
    EXPECT_DOUBLE_EQ(queues.available_at_ns(7).value(), 150);
    ASSERT_TRUE(queues.observe_completion(7, 100, 120));
    EXPECT_DOUBLE_EQ(queues.available_at_ns(7).value(), 70);
    ASSERT_TRUE(queues.observe_completion(7, 20, 0, 0.5));
    EXPECT_DOUBLE_EQ(queues.available_at_ns(7).value(), 30);
    EXPECT_DOUBLE_EQ(queues.available_at_ns(8).value(), 0);
    EXPECT_FALSE(queues.observe_completion(7, 10, 20, 0));
}

}  // namespace
}  // namespace kora::detail
