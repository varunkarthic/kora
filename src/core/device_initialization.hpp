#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include "kora/device_selection.hpp"

namespace kora::detail {

/// Initialize selected devices independently when the policy permits partial availability.
template <class Initialize, class Quarantine, class ReportFailure>
Status initialize_selected_devices(std::span<const DeviceId> selected, bool isolate_failures,
                                   Initialize&& initialize, Quarantine&& quarantine,
                                   ReportFailure&& report_failure) {
    std::optional<Error> first_error;
    std::size_t initialized_count = 0;
    for (const auto& id : selected) {
        auto result = initialize(id);
        if (result) {
            ++initialized_count;
            continue;
        }
        auto error = std::move(result).error();
        error.with_device("index " + std::to_string(id.index) +
                          (id.uuid.empty() ? std::string{} : " uuid " + id.uuid));
        if (!isolate_failures) return error;
        if (!first_error) first_error = error;
        quarantine(id);
        report_failure(id, error);
    }
    if (initialized_count != 0) return {};
    if (first_error) return std::move(*first_error);
    return Error(ErrorCode::NoEligibleDevice, "No selected device could be initialized");
}

}  // namespace kora::detail
