#include <gtest/gtest.h>

#include "kora/task_partition.hpp"

namespace kora {
namespace {

PartitionRequest request() {
    return {.total_work_units = 1000,
            .bytes_per_work_unit = 4,
            .granularity = 10,
            .minimum_chunk_units = 100,
            .estimated_unsplit_duration_ns = 1000,
            .partition_overhead_ns = 10,
            .minimum_gain_fraction = 0.05};
}

TEST(DynamicPartitioner, ProportionalWeightsCreateExactContiguousCoverage) {
    DynamicPartitioner partitioner;
    const DeviceId fast{0, "fast"};
    const DeviceId slow{1, "slow"};
    ASSERT_TRUE(partitioner.observe(fast, 200, 100));
    ASSERT_TRUE(partitioner.observe(slow, 100, 100));
    const auto result = partitioner.partition(request(), {{fast, 10000, 1}, {slow, 10000, 1}});
    ASSERT_TRUE(result);
    ASSERT_TRUE(result.value().split);
    ASSERT_EQ(result.value().slices.size(), 2U);
    EXPECT_NEAR(result.value().slices[0].weight, 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(result.value().slices[1].weight, 1.0 / 3.0, 1e-12);
    EXPECT_EQ(result.value().slices[0].offset, 0U);
    EXPECT_EQ(result.value().slices[1].offset, result.value().slices[0].work_units);
    EXPECT_EQ(result.value().slices[0].work_units + result.value().slices[1].work_units, 1000U);
    EXPECT_EQ(result.value().slices[0].work_units % 10, 0U);
    EXPECT_EQ(result.value().slices[1].work_units, 1000U - result.value().slices[0].work_units);
}

TEST(DynamicPartitioner, NewObservationsRebalanceByEwma) {
    DynamicPartitioner partitioner(1.0);
    const DeviceId a{0, "a"};
    const DeviceId b{1, "b"};
    ASSERT_TRUE(partitioner.observe(a, 100, 100));
    ASSERT_TRUE(partitioner.observe(b, 100, 100));
    auto first = partitioner.partition(request(), {{a, 10000, 1}, {b, 10000, 1}});
    ASSERT_TRUE(first);
    EXPECT_EQ(first.value().slices[0].work_units, first.value().slices[1].work_units);
    ASSERT_TRUE(partitioner.observe(a, 200, 100));
    auto second = partitioner.partition(request(), {{a, 10000, 1}, {b, 10000, 1}});
    ASSERT_TRUE(second);
    EXPECT_GT(second.value().slices[0].work_units, second.value().slices[1].work_units);
}

TEST(DynamicPartitioner, FallsBackWhenSplitIsUnprofitableOrOnlyOneDeviceFits) {
    DynamicPartitioner partitioner;
    const DeviceId a{0, "a"};
    const DeviceId b{1, "b"};
    PartitionRequest slow = request();
    slow.estimated_unsplit_duration_ns = 2000;
    slow.partition_overhead_ns = 5000;
    auto result = partitioner.partition(slow, {{a, 10000, 1}, {b, 10000, 1}});
    ASSERT_TRUE(result);
    EXPECT_FALSE(result.value().split);
    EXPECT_EQ(result.value().slices.size(), 1U);

    result = partitioner.partition(request(), {{a, 10000, 1}, {b, 100, 1}});
    ASSERT_TRUE(result);
    EXPECT_FALSE(result.value().split);
    EXPECT_EQ(result.value().slices.size(), 1U);
    EXPECT_EQ(result.value().slices.front().device, a);
}

TEST(DynamicPartitioner, EnforcesMemoryAndReportsImpossibleUnsplitTask) {
    DynamicPartitioner partitioner;
    const DeviceId a{0, "a"};
    const DeviceId b{1, "b"};
    auto result = partitioner.partition(request(), {{a, 2000, 1}, {b, 2000, 1}});
    ASSERT_TRUE(result);
    EXPECT_TRUE(result.value().split);
    EXPECT_LE(result.value().slices[0].work_units * 4, 2000U);
    EXPECT_LE(result.value().slices[1].work_units * 4, 2000U);
    EXPECT_EQ(partitioner.partition(request(), {{a, 100, 1}}).error().code, ErrorCode::OutOfMemoryBudget);
}

TEST(DynamicPartitioner, RejectsInvalidInputsAndObservations) {
    DynamicPartitioner partitioner;
    EXPECT_EQ(partitioner.observe({0, "a"}, 1, 0).error().code, ErrorCode::InvalidArgument);
    auto invalid = request();
    invalid.granularity = 0;
    EXPECT_EQ(partitioner.partition(invalid, {{DeviceId{0, "a"}, 10000, 1}}).error().code,
              ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace kora
