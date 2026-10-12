#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "core/calibration.hpp"
#include "kora/errors.hpp"

namespace kora::detail {

enum class CalibratedKernelKind : std::uint8_t { None, TinyDispatch, Streaming, Alu };

struct CalibrationCostWorkload {
    std::uint64_t input_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t device_copy_bytes = 0;
    std::uint64_t kernel_work = 0;
    CalibratedKernelKind kernel_kind = CalibratedKernelKind::None;
};

/// Optional time estimates in nanoseconds. Empty estimates must use the caller's fallback model.
struct CalibrationCostComponents {
    bool complete_record = false;
    std::optional<double> input_transfer_ns;
    std::optional<double> output_transfer_ns;
    std::optional<double> device_copy_ns;
    std::optional<double> kernel_ns;
    std::optional<double> submit_completion_ns;
};

/// Converts a complete, identity-matched calibration record into workload-sized wall-time terms.
/// A false `complete_record` means callers should retain their static/online estimates unchanged.
Result<CalibrationCostComponents> estimate_calibration_cost(const calibration::Record& record,
                                                            std::string_view expected_uuid,
                                                            std::uint32_t expected_driver,
                                                            const CalibrationCostWorkload& workload);

}  // namespace kora::detail
