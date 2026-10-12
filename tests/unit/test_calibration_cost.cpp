#include <gtest/gtest.h>

#include "core/calibration_cost.hpp"

namespace kora::detail {
namespace {

calibration::Record complete_record() {
    calibration::Record record;
    record.device_uuid = "00112233445566778899aabbccddeeff";
    record.driver_version = 17;
    record.submit_overhead_ns = 1234;
    record.tiny_dispatch_ns = 4567;
    record.h2d_bytes_per_second = 1'000'000;
    record.d2h_bytes_per_second = 2'000'000;
    record.device_copy_bytes_per_second = 3'000'000;
    record.streaming_bytes_per_second = 4'000'000;
    record.alu_operations_per_second = 5'000'000;
    return record;
}

TEST(CalibrationCost, MapsWorkloadTermsToDistinctNanosecondComponents) {
    const auto record = complete_record();
    CalibrationCostWorkload workload;
    workload.input_bytes = 1'024;
    workload.output_bytes = 2'048;
    workload.device_copy_bytes = 3'072;
    workload.kernel_work = 10'000;
    workload.kernel_kind = CalibratedKernelKind::Streaming;

    const auto estimate =
        estimate_calibration_cost(record, record.device_uuid, record.driver_version, workload);
    ASSERT_TRUE(estimate);
    ASSERT_TRUE(estimate.value().complete_record);
    ASSERT_TRUE(estimate.value().input_transfer_ns);
    ASSERT_TRUE(estimate.value().output_transfer_ns);
    ASSERT_TRUE(estimate.value().device_copy_ns);
    ASSERT_TRUE(estimate.value().kernel_ns);
    EXPECT_FALSE(estimate.value().submit_completion_ns);
    EXPECT_DOUBLE_EQ(*estimate.value().input_transfer_ns, 1'024'000.0);
    EXPECT_DOUBLE_EQ(*estimate.value().output_transfer_ns, 1'024'000.0);
    EXPECT_NEAR(*estimate.value().device_copy_ns, 1'024'000.0, 1e-9);
    EXPECT_DOUBLE_EQ(*estimate.value().kernel_ns, 2'500'000.0);

    workload.input_bytes = 0;
    workload.output_bytes = 0;
    workload.device_copy_bytes = 0;
    workload.kernel_kind = CalibratedKernelKind::Alu;
    workload.kernel_work = 5'000;
    const auto alu = estimate_calibration_cost(record, record.device_uuid, record.driver_version, workload);
    ASSERT_TRUE(alu);
    ASSERT_TRUE(alu.value().kernel_ns);
    EXPECT_FALSE(alu.value().submit_completion_ns);
    EXPECT_DOUBLE_EQ(*alu.value().kernel_ns, 1'000'000.0);

    workload.kernel_kind = CalibratedKernelKind::TinyDispatch;
    workload.kernel_work = 3;
    const auto tiny = estimate_calibration_cost(record, record.device_uuid, record.driver_version, workload);
    ASSERT_TRUE(tiny);
    ASSERT_TRUE(tiny.value().kernel_ns);
    EXPECT_DOUBLE_EQ(*tiny.value().kernel_ns, 13'701.0);

    workload.kernel_kind = CalibratedKernelKind::None;
    workload.kernel_work = 0;
    const auto submit_only =
        estimate_calibration_cost(record, record.device_uuid, record.driver_version, workload);
    ASSERT_TRUE(submit_only);
    ASSERT_TRUE(submit_only.value().submit_completion_ns);
    EXPECT_DOUBLE_EQ(*submit_only.value().submit_completion_ns, 1'234.0);
}

TEST(CalibrationCost, IncompleteStaleOrMismatchedIdentityFallsBackWithoutZeroCost) {
    const auto record = complete_record();
    const CalibrationCostWorkload workload{.input_bytes = 1'000};

    const auto wrong_uuid =
        estimate_calibration_cost(record, "different-device", record.driver_version, workload);
    ASSERT_TRUE(wrong_uuid);
    EXPECT_FALSE(wrong_uuid.value().complete_record);
    EXPECT_FALSE(wrong_uuid.value().input_transfer_ns);

    const auto wrong_driver =
        estimate_calibration_cost(record, record.device_uuid, record.driver_version + 1, workload);
    ASSERT_TRUE(wrong_driver);
    EXPECT_FALSE(wrong_driver.value().complete_record);

    auto zero_metric = record;
    zero_metric.streaming_bytes_per_second = 0;
    const auto incomplete =
        estimate_calibration_cost(zero_metric, record.device_uuid, record.driver_version, workload);
    ASSERT_TRUE(incomplete);
    EXPECT_FALSE(incomplete.value().complete_record);
    EXPECT_FALSE(incomplete.value().input_transfer_ns);

    auto stale_schema = record;
    --stale_schema.schema;
    const auto stale =
        estimate_calibration_cost(stale_schema, record.device_uuid, record.driver_version, workload);
    ASSERT_TRUE(stale);
    EXPECT_FALSE(stale.value().complete_record);

    auto stale_kora_schema = record;
    ++stale_kora_schema.kora_schema;
    const auto stale_kora =
        estimate_calibration_cost(stale_kora_schema, record.device_uuid, record.driver_version, workload);
    ASSERT_TRUE(stale_kora);
    EXPECT_FALSE(stale_kora.value().complete_record);
}

TEST(CalibrationCost, RejectsAmbiguousKernelWorkload) {
    const auto record = complete_record();
    CalibrationCostWorkload workload;
    workload.kernel_work = 100;
    EXPECT_FALSE(estimate_calibration_cost(record, record.device_uuid, record.driver_version, workload));
    workload.kernel_kind = CalibratedKernelKind::Streaming;
    workload.kernel_work = 0;
    EXPECT_FALSE(estimate_calibration_cost(record, record.device_uuid, record.driver_version, workload));
}

TEST(CalibrationCost, AppliesMeasuredLatencyFloorsToSmallOperations) {
    const auto record = complete_record();
    CalibrationCostWorkload workload;
    workload.input_bytes = 1;
    workload.kernel_kind = CalibratedKernelKind::Streaming;
    workload.kernel_work = 1;

    const auto estimate =
        estimate_calibration_cost(record, record.device_uuid, record.driver_version, workload);
    ASSERT_TRUE(estimate);
    ASSERT_TRUE(estimate.value().input_transfer_ns);
    ASSERT_TRUE(estimate.value().kernel_ns);
    EXPECT_DOUBLE_EQ(*estimate.value().input_transfer_ns, static_cast<double>(record.submit_overhead_ns));
    EXPECT_DOUBLE_EQ(*estimate.value().kernel_ns, static_cast<double>(record.tiny_dispatch_ns));
}

}  // namespace
}  // namespace kora::detail
