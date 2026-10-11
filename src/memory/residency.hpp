#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

#include "kora/errors.hpp"

namespace kora::detail {

using ResidencyDeviceId = std::uint32_t;

enum class ResidencyState : std::uint8_t {
    HostOnly,
    DeviceOnly,
    Shared,
    HostAndDeviceClean,
    HostNewer,
    DeviceNewer,
    Evicted,
    InTransfer,
    Disposed,
};

enum class ResidencyEvent : std::uint8_t {
    HostWrite,
    DeviceWrite,
    BeginTransfer,
    CompleteToHost,
    CompleteToDevice,
    EvictHost,
    EvictDevice,
    Dispose,
};

/// Pure policy transition. It validates state/event legality without performing I/O or mutation.
Result<ResidencyState> transition_residency(ResidencyState state, ResidencyEvent event);

struct ResidencyLocation {
    bool host = false;
    ResidencyDeviceId device = 0;

    static constexpr ResidencyLocation host_copy() noexcept { return {true, 0}; }
    static constexpr ResidencyLocation device_copy(ResidencyDeviceId id) noexcept { return {false, id}; }
    friend bool operator==(const ResidencyLocation&, const ResidencyLocation&) = default;
};

struct TransferPlan {
    bool required = false;
    ResidencyLocation source{};
    ResidencyLocation destination{};
    std::uint64_t version = 0;
    friend bool operator==(const TransferPlan&, const TransferPlan&) = default;
};

struct ResidencySnapshot {
    ResidencyState state = ResidencyState::Disposed;
    std::uint64_t current_version = 0;
    bool host_current = false;
    std::unordered_map<ResidencyDeviceId, bool> device_current;
    bool disposed = true;
};

/// Versioned per-copy validity and transfer planner for one logical allocation.
class ResidencyTracker final {
public:
    static ResidencyTracker host_initialized();
    static Result<ResidencyTracker> device_initialized(ResidencyDeviceId device);

    ResidencyState state() const noexcept { return state_; }
    std::uint64_t current_version() const noexcept { return current_version_; }
    Result<ResidencySnapshot> snapshot() const;

    Status write_host();
    Status write_device(ResidencyDeviceId device);
    Result<TransferPlan> plan_transfer(ResidencyLocation destination) const;
    Status begin_transfer(const TransferPlan& plan);
    Status complete_transfer();
    Status fail_transfer();
    Status evict_host();
    Status evict_device(ResidencyDeviceId device);
    Status dispose();

private:
    struct PendingTransfer {
        TransferPlan plan;
        ResidencyState previous_state = ResidencyState::HostOnly;
    };

    ResidencyState derive_state() const noexcept;
    bool current(ResidencyLocation location) const noexcept;
    Status advance_version();

    std::uint64_t current_version_ = 1;
    std::optional<std::uint64_t> host_version_ = 1;
    std::unordered_map<ResidencyDeviceId, std::uint64_t> device_versions_;
    ResidencyState state_ = ResidencyState::HostOnly;
    bool ever_had_device_copy_ = false;
    bool disposed_ = false;
    std::optional<PendingTransfer> pending_;
};

const char* to_string(ResidencyState state) noexcept;
const char* to_string(ResidencyEvent event) noexcept;

}  // namespace kora::detail
