#include <gtest/gtest.h>

#include "core/scheduler_placement.hpp"

namespace kora::detail {
namespace {

RuntimePlacementCandidate candidate(std::uint32_t index, std::uint64_t rate) {
    RuntimePlacementCandidate value;
    value.id = DeviceId{index, std::string(32, static_cast<char>('0' + index))};
    value.driver_version = 3;
    value.available_device_memory_bytes = 1ULL << 30;
    value.calibration_record.device_uuid = value.id.uuid;
    value.calibration_record.driver_version = value.driver_version;
    value.calibration_record.submit_overhead_ns = 100;
    value.calibration_record.tiny_dispatch_ns = 100;
    value.calibration_record.h2d_bytes_per_second = rate;
    value.calibration_record.d2h_bytes_per_second = rate;
    value.calibration_record.device_copy_bytes_per_second = rate;
    value.calibration_record.streaming_bytes_per_second = rate;
    value.calibration_record.alu_operations_per_second = rate;
    return value;
}

AdaptiveDispatchWorkload workload() {
    return AdaptiveDispatchWorkload{
        .input_bytes = 1024,
        .output_bytes = 1024,
        .kernel_work = 1024,
        .required_device_memory_bytes = 4096,
        .kernel_kind = AdaptiveKernelWorkKind::StreamingBytes,
    };
}

TEST(SchedulerPlacement, ChoosesLowestMeasuredFinishTimeDeterministically) {
    const auto slow = candidate(0, 1'000'000);
    const auto fast = candidate(1, 10'000'000);
    const auto first = select_adaptive_device({slow, fast}, slow.id, workload());
    const auto second = select_adaptive_device({slow, fast}, slow.id, workload());
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(first.value(), fast.id);
    EXPECT_EQ(second.value(), first.value());
}

TEST(SchedulerPlacement, IncompleteCalibrationRetainsTheKernelBoundDevice) {
    const auto current = candidate(0, 1'000'000);
    auto incomplete = candidate(1, 10'000'000);
    incomplete.calibration_record.alu_operations_per_second = 0;
    const auto selected = select_adaptive_device({current, incomplete}, current.id, workload());
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected.value(), current.id);
}

TEST(SchedulerPlacement, FiltersUnsupportedUnhealthyAndUnderBudgetCandidates) {
    const auto current = candidate(0, 1'000'000);
    auto faster = candidate(1, 10'000'000);
    faster.available_device_memory_bytes = 1024;
    const auto selected = select_adaptive_device({current, faster}, current.id, workload());
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected.value(), current.id);

    auto unhealthy = candidate(1, 10'000'000);
    unhealthy.healthy = false;
    const auto healthy_selected = select_adaptive_device({current, unhealthy}, current.id, workload());
    ASSERT_TRUE(healthy_selected);
    EXPECT_EQ(healthy_selected.value(), current.id);

    auto unsupported = candidate(1, 10'000'000);
    unsupported.supported = false;
    const auto supported_selected = select_adaptive_device({current, unsupported}, current.id, workload());
    ASSERT_TRUE(supported_selected);
    EXPECT_EQ(supported_selected.value(), current.id);

    auto exact_limit = candidate(1, 10'000'000);
    exact_limit.available_device_memory_bytes = workload().required_device_memory_bytes;
    const auto limit_selected = select_adaptive_device({current, exact_limit}, current.id, workload());
    ASSERT_TRUE(limit_selected);
    EXPECT_EQ(limit_selected.value(), current.id);
}

TEST(SchedulerPlacement, RejectsInvalidWorkloadAndIneligibleCurrentDevice) {
    auto current = candidate(0, 1'000'000);
    auto invalid = workload();
    invalid.kernel_work = 0;
    EXPECT_FALSE(select_adaptive_device({current}, current.id, invalid));

    current.healthy = false;
    EXPECT_FALSE(select_adaptive_device({current}, current.id, workload()));
}

}  // namespace
}  // namespace kora::detail
