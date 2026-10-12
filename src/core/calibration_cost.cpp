#include "core/calibration_cost.hpp"

#include <algorithm>
#include <cmath>

#include "kora/version.hpp"

namespace kora::detail {
namespace {

bool complete_and_compatible(const calibration::Record& record, std::string_view expected_uuid,
                             std::uint32_t expected_driver) {
    return !expected_uuid.empty() && record.schema == calibration::kSchemaVersion &&
           record.kora_schema == static_cast<std::uint32_t>(kora::kSchemaVersion) &&
           record.device_uuid == expected_uuid && record.driver_version == expected_driver &&
           record.submit_overhead_ns != 0 && record.tiny_dispatch_ns != 0 &&
           record.h2d_bytes_per_second != 0 && record.d2h_bytes_per_second != 0 &&
           record.device_copy_bytes_per_second != 0 && record.streaming_bytes_per_second != 0 &&
           record.alu_operations_per_second != 0;
}

double time_ns(std::uint64_t amount, std::uint64_t rate, std::uint64_t latency_floor_ns) {
    if (amount == 0) return 0;
    const auto estimate =
        static_cast<long double>(amount) * 1'000'000'000.0L / static_cast<long double>(rate);
    return static_cast<double>(std::max(estimate, static_cast<long double>(latency_floor_ns)));
}

}  // namespace

Result<CalibrationCostComponents> estimate_calibration_cost(const calibration::Record& record,
                                                            std::string_view expected_uuid,
                                                            std::uint32_t expected_driver,
                                                            const CalibrationCostWorkload& workload) {
    if (static_cast<std::uint8_t>(workload.kernel_kind) >
        static_cast<std::uint8_t>(CalibratedKernelKind::Alu))
        return Error(ErrorCode::InvalidArgument, "calibration cost workload has an invalid kernel kind");
    if ((workload.kernel_kind == CalibratedKernelKind::None) != (workload.kernel_work == 0))
        return Error(ErrorCode::InvalidArgument,
                     "calibration cost workload must pair a kernel kind with a nonzero work amount");

    CalibrationCostComponents result;
    if (!complete_and_compatible(record, expected_uuid, expected_driver)) return result;
    result.complete_record = true;
    result.input_transfer_ns =
        time_ns(workload.input_bytes, record.h2d_bytes_per_second, record.submit_overhead_ns);
    result.output_transfer_ns =
        time_ns(workload.output_bytes, record.d2h_bytes_per_second, record.submit_overhead_ns);
    result.device_copy_ns =
        time_ns(workload.device_copy_bytes, record.device_copy_bytes_per_second, record.submit_overhead_ns);
    const bool has_end_to_end_work = workload.input_bytes != 0 || workload.output_bytes != 0 ||
                                     workload.device_copy_bytes != 0 ||
                                     workload.kernel_kind != CalibratedKernelKind::None;
    if (!has_end_to_end_work) result.submit_completion_ns = static_cast<double>(record.submit_overhead_ns);
    switch (workload.kernel_kind) {
        case CalibratedKernelKind::None:
            break;
        case CalibratedKernelKind::TinyDispatch:
            result.kernel_ns =
                static_cast<double>(workload.kernel_work) * static_cast<double>(record.tiny_dispatch_ns);
            break;
        case CalibratedKernelKind::Streaming:
            result.kernel_ns =
                time_ns(workload.kernel_work, record.streaming_bytes_per_second, record.tiny_dispatch_ns);
            break;
        case CalibratedKernelKind::Alu:
            result.kernel_ns =
                time_ns(workload.kernel_work, record.alu_operations_per_second, record.tiny_dispatch_ns);
            break;
    }
    if (!std::isfinite(*result.input_transfer_ns) || !std::isfinite(*result.output_transfer_ns) ||
        !std::isfinite(*result.device_copy_ns) || (result.kernel_ns && !std::isfinite(*result.kernel_ns)))
        return Error(ErrorCode::Overflow, "calibration cost estimate exceeds finite nanosecond range");
    return result;
}

}  // namespace kora::detail
