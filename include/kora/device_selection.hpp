#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kora/device_info.hpp"
#include "kora/errors.hpp"
#include "kora/options.hpp"

namespace kora {

enum class DeviceKind : std::uint8_t { Other, Integrated, Discrete, Virtual, Cpu };

struct DeviceId {
    std::uint32_t index = 0;
    std::string uuid;

    friend bool operator==(const DeviceId&, const DeviceId&) = default;
};

/// Selection-only facts. Startup calibration will add measured scores without changing policy semantics.
struct DeviceCandidate {
    DeviceId id;
    DeviceKind kind = DeviceKind::Other;
    CapabilityTier tier = CapabilityTier::A;
    std::uint64_t device_local_memory_bytes = 0;
    std::uint32_t compute_queue_count = 0;
    bool satisfies_requirements = true;
};

/// Select eligible device identifiers in deterministic preference order.
Result<std::vector<DeviceId>> select_devices(const ResolvedOptions& options,
                                             const std::vector<DeviceCandidate>& devices);

}  // namespace kora
