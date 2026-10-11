#include <gtest/gtest.h>

#include "core/scheduler_cost.hpp"

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

    const auto result = evaluate_scheduler_cost(input);
    ASSERT_TRUE(result);
    const auto& score = result.value();
    EXPECT_DOUBLE_EQ(score.ready_delay_ns, 100);
    EXPECT_DOUBLE_EQ(score.queue_delay_ns, 60);
    EXPECT_DOUBLE_EQ(score.memory_penalty_ns, 10);
    EXPECT_DOUBLE_EQ(score.uncertainty_penalty_ns, 4);
    EXPECT_DOUBLE_EQ(score.locality_bonus_ns, 25);
    EXPECT_DOUBLE_EQ(score.predicted_finish_ns, 214);
    EXPECT_DOUBLE_EQ(score.ready_delay_ns + score.queue_delay_ns + score.input_transfer_ns +
                         score.execute_ns + score.required_output_ns + score.synchronization_ns +
                         score.memory_penalty_ns + score.uncertainty_penalty_ns - score.locality_bonus_ns,
                     score.predicted_finish_ns);
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
