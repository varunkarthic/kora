#include "memory/residency.hpp"

#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace kora::detail {
namespace {

Error illegal_transition(ResidencyState state, ResidencyEvent event) {
    return Error(ErrorCode::InvalidArgument,
                 std::string("illegal residency transition: ") + to_string(state) + " + " + to_string(event));
}

}  // namespace

Result<ResidencyState> transition_residency(ResidencyState state, ResidencyEvent event) {
    if (event == ResidencyEvent::Dispose) {
        if (state == ResidencyState::Disposed || state == ResidencyState::InTransfer)
            return illegal_transition(state, event);
        return ResidencyState::Disposed;
    }
    switch (state) {
        case ResidencyState::HostOnly:
            if (event == ResidencyEvent::HostWrite) return ResidencyState::HostOnly;
            if (event == ResidencyEvent::DeviceWrite) return ResidencyState::DeviceNewer;
            if (event == ResidencyEvent::BeginTransfer) return ResidencyState::InTransfer;
            return illegal_transition(state, event);
        case ResidencyState::DeviceOnly:
            if (event == ResidencyEvent::DeviceWrite) return ResidencyState::DeviceOnly;
            if (event == ResidencyEvent::HostWrite) return ResidencyState::HostNewer;
            if (event == ResidencyEvent::BeginTransfer) return ResidencyState::InTransfer;
            if (event == ResidencyEvent::EvictDevice) return illegal_transition(state, event);
            return illegal_transition(state, event);
        case ResidencyState::Shared:
        case ResidencyState::HostAndDeviceClean:
            if (event == ResidencyEvent::HostWrite) return ResidencyState::HostNewer;
            if (event == ResidencyEvent::DeviceWrite) return ResidencyState::DeviceNewer;
            if (event == ResidencyEvent::BeginTransfer) return ResidencyState::InTransfer;
            if (event == ResidencyEvent::EvictHost) return ResidencyState::DeviceOnly;
            if (event == ResidencyEvent::EvictDevice) return ResidencyState::Evicted;
            return illegal_transition(state, event);
        case ResidencyState::HostNewer:
            if (event == ResidencyEvent::HostWrite) return ResidencyState::HostNewer;
            if (event == ResidencyEvent::DeviceWrite) return ResidencyState::DeviceNewer;
            if (event == ResidencyEvent::BeginTransfer) return ResidencyState::InTransfer;
            if (event == ResidencyEvent::EvictDevice) return ResidencyState::Evicted;
            return illegal_transition(state, event);
        case ResidencyState::DeviceNewer:
            if (event == ResidencyEvent::HostWrite) return ResidencyState::HostNewer;
            if (event == ResidencyEvent::DeviceWrite) return ResidencyState::DeviceNewer;
            if (event == ResidencyEvent::BeginTransfer) return ResidencyState::InTransfer;
            if (event == ResidencyEvent::EvictHost) return ResidencyState::DeviceOnly;
            if (event == ResidencyEvent::EvictDevice) return ResidencyState::DeviceNewer;
            return illegal_transition(state, event);
        case ResidencyState::Evicted:
            if (event == ResidencyEvent::HostWrite) return ResidencyState::Evicted;
            if (event == ResidencyEvent::DeviceWrite) return ResidencyState::DeviceNewer;
            if (event == ResidencyEvent::BeginTransfer) return ResidencyState::InTransfer;
            return illegal_transition(state, event);
        case ResidencyState::InTransfer:
            if (event == ResidencyEvent::CompleteToHost) return ResidencyState::HostAndDeviceClean;
            if (event == ResidencyEvent::CompleteToDevice) return ResidencyState::Shared;
            return illegal_transition(state, event);
        case ResidencyState::Disposed:
            return illegal_transition(state, event);
    }
    return illegal_transition(state, event);
}

ResidencyTracker ResidencyTracker::host_initialized() {
    return {};
}

Result<ResidencyTracker> ResidencyTracker::device_initialized(ResidencyDeviceId device) {
    ResidencyTracker tracker;
    tracker.host_version_.reset();
    try {
        tracker.device_versions_.emplace(device, tracker.current_version_);
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to allocate initial device residency record");
    } catch (const std::length_error&) {
        return Error(ErrorCode::Overflow, "initial device residency metadata exceeds container capacity");
    }
    tracker.ever_had_device_copy_ = true;
    tracker.state_ = ResidencyState::DeviceOnly;
    return std::move(tracker);
}

Result<ResidencySnapshot> ResidencyTracker::snapshot() const {
    ResidencySnapshot result;
    result.state = state_;
    result.current_version = current_version_;
    result.host_current = host_version_ && *host_version_ == current_version_;
    try {
        result.device_current.reserve(device_versions_.size());
        for (const auto& [device, version] : device_versions_) {
            result.device_current.emplace(device, version == current_version_);
        }
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to create residency snapshot");
    } catch (const std::length_error&) {
        return Error(ErrorCode::Overflow, "residency snapshot exceeds container capacity");
    }
    result.disposed = disposed_;
    return result;
}

bool ResidencyTracker::current(ResidencyLocation location) const noexcept {
    if (location.host) return host_version_ && *host_version_ == current_version_;
    const auto found = device_versions_.find(location.device);
    return found != device_versions_.end() && found->second == current_version_;
}

ResidencyState ResidencyTracker::derive_state() const noexcept {
    if (disposed_) return ResidencyState::Disposed;
    if (pending_) return ResidencyState::InTransfer;
    const bool host_current = host_version_ && *host_version_ == current_version_;
    std::size_t current_devices = 0;
    for (const auto& [device, version] : device_versions_) {
        static_cast<void>(device);
        if (version == current_version_) ++current_devices;
    }
    if (current_devices > 1) return ResidencyState::Shared;
    if (host_current && current_devices == 1) {
        return ResidencyState::HostAndDeviceClean;
    }
    if (host_current) {
        if (!device_versions_.empty()) return ResidencyState::HostNewer;
        return ever_had_device_copy_ ? ResidencyState::Evicted : ResidencyState::HostOnly;
    }
    if (current_devices > 0) {
        if (host_version_) return ResidencyState::DeviceNewer;
        return current_devices > 1 ? ResidencyState::Shared : ResidencyState::DeviceOnly;
    }
    return ResidencyState::Disposed;  // An invariant violation is represented as terminal, never as valid
                                      // data.
}

Status ResidencyTracker::advance_version() {
    if (current_version_ == std::numeric_limits<std::uint64_t>::max())
        return Error(ErrorCode::Overflow, "residency content version overflow");
    ++current_version_;
    return ok_status();
}

Status ResidencyTracker::write_host() {
    auto next = transition_residency(state_, ResidencyEvent::HostWrite);
    if (!next) return next.error();
    auto advanced = advance_version();
    if (!advanced) return advanced;
    host_version_ = current_version_;
    state_ = derive_state();
    return ok_status();
}

Status ResidencyTracker::write_device(ResidencyDeviceId device) {
    auto next = transition_residency(state_, ResidencyEvent::DeviceWrite);
    if (!next) return next.error();
    auto advanced = advance_version();
    if (!advanced) return advanced;
    try {
        device_versions_[device] = current_version_;
    } catch (const std::bad_alloc&) {
        --current_version_;
        return Error(ErrorCode::AllocationFailed, "unable to record current device buffer copy");
    } catch (const std::length_error&) {
        --current_version_;
        return Error(ErrorCode::Overflow, "device residency metadata exceeds container capacity");
    }
    ever_had_device_copy_ = true;
    state_ = derive_state();
    return ok_status();
}

Result<TransferPlan> ResidencyTracker::plan_transfer(ResidencyLocation destination) const {
    if (disposed_) return Error(ErrorCode::StaleHandle, "disposed residency has no current copy to transfer");
    if (pending_)
        return Error(ErrorCode::InvalidArgument, "cannot plan another transfer while one is in flight");
    if (current(destination)) return TransferPlan{false, destination, destination, current_version_};
    ResidencyLocation source{};
    if (current(ResidencyLocation::host_copy())) {
        source = ResidencyLocation::host_copy();
    } else {
        bool found = false;
        for (const auto& [device, version] : device_versions_) {
            if (version == current_version_ && (destination.host || device != destination.device) &&
                (!found || device < source.device)) {
                source = ResidencyLocation::device_copy(device);
                found = true;
            }
        }
        if (!found) return Error(ErrorCode::InternalInvariantViolation, "no current residency copy exists");
    }
    if (source == destination) return TransferPlan{false, source, destination, current_version_};
    return TransferPlan{true, source, destination, current_version_};
}

Status ResidencyTracker::begin_transfer(const TransferPlan& plan) {
    if (!plan.required)
        return Error(ErrorCode::InvalidArgument, "cannot begin an unnecessary residency transfer");
    auto fresh_plan = plan_transfer(plan.destination);
    if (!fresh_plan) return fresh_plan.error();
    if (fresh_plan.value() != plan) return Error(ErrorCode::StaleHandle, "residency transfer plan is stale");
    auto next = transition_residency(state_, ResidencyEvent::BeginTransfer);
    if (!next) return next.error();
    pending_ = PendingTransfer{plan, state_};
    state_ = next.value();
    return ok_status();
}

Status ResidencyTracker::complete_transfer() {
    if (!pending_) return Error(ErrorCode::InvalidArgument, "no residency transfer is in flight");
    const auto completion =
        pending_->plan.destination.host ? ResidencyEvent::CompleteToHost : ResidencyEvent::CompleteToDevice;
    auto next = transition_residency(state_, completion);
    if (!next) return next.error();
    const auto transfer = pending_->plan;
    if (transfer.version != current_version_ || !current(transfer.source)) {
        return Error(ErrorCode::StaleHandle, "residency transfer source changed before completion");
    }
    try {
        if (transfer.destination.host) {
            host_version_ = transfer.version;
        } else {
            device_versions_[transfer.destination.device] = transfer.version;
            ever_had_device_copy_ = true;
        }
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to record completed residency transfer");
    } catch (const std::length_error&) {
        return Error(ErrorCode::Overflow, "device residency metadata exceeds container capacity");
    }
    pending_.reset();
    state_ = derive_state();
    return ok_status();
}

Status ResidencyTracker::fail_transfer() {
    if (!pending_) return Error(ErrorCode::InvalidArgument, "no residency transfer is in flight");
    state_ = pending_->previous_state;
    pending_.reset();
    return ok_status();
}

Status ResidencyTracker::evict_host() {
    auto next = transition_residency(state_, ResidencyEvent::EvictHost);
    if (!next) return next.error();
    bool current_device_exists = false;
    for (const auto& [device, version] : device_versions_) {
        static_cast<void>(device);
        current_device_exists |= version == current_version_;
    }
    if (!current_device_exists)
        return Error(ErrorCode::InvalidArgument, "cannot evict the last current copy of a buffer");
    host_version_.reset();
    state_ = derive_state();
    return ok_status();
}

Status ResidencyTracker::evict_device(ResidencyDeviceId device) {
    if (disposed_ || pending_)
        return Error(ErrorCode::InvalidArgument, "cannot evict a copy in this residency state");
    const auto found = device_versions_.find(device);
    if (found == device_versions_.end())
        return Error(ErrorCode::InvalidArgument, "device copy does not exist");
    const bool removed_current = found->second == current_version_;
    if (removed_current && !current(ResidencyLocation::host_copy())) {
        bool another_current_device = false;
        for (const auto& [other_device, version] : device_versions_) {
            another_current_device |= other_device != device && version == current_version_;
        }
        if (!another_current_device) {
            return Error(ErrorCode::InvalidArgument, "cannot evict the last current copy of a buffer");
        }
    }
    device_versions_.erase(found);
    state_ = derive_state();
    return ok_status();
}

Status ResidencyTracker::dispose() {
    auto next = transition_residency(state_, ResidencyEvent::Dispose);
    if (!next) return next.error();
    host_version_.reset();
    device_versions_.clear();
    pending_.reset();
    disposed_ = true;
    state_ = ResidencyState::Disposed;
    return ok_status();
}

const char* to_string(ResidencyState state) noexcept {
    switch (state) {
        case ResidencyState::HostOnly:
            return "HostOnly";
        case ResidencyState::DeviceOnly:
            return "DeviceOnly";
        case ResidencyState::Shared:
            return "Shared";
        case ResidencyState::HostAndDeviceClean:
            return "HostAndDeviceClean";
        case ResidencyState::HostNewer:
            return "HostNewer";
        case ResidencyState::DeviceNewer:
            return "DeviceNewer";
        case ResidencyState::Evicted:
            return "Evicted";
        case ResidencyState::InTransfer:
            return "InTransfer";
        case ResidencyState::Disposed:
            return "Disposed";
    }
    return "Unknown";
}

const char* to_string(ResidencyEvent event) noexcept {
    switch (event) {
        case ResidencyEvent::HostWrite:
            return "HostWrite";
        case ResidencyEvent::DeviceWrite:
            return "DeviceWrite";
        case ResidencyEvent::BeginTransfer:
            return "BeginTransfer";
        case ResidencyEvent::CompleteToHost:
            return "CompleteToHost";
        case ResidencyEvent::CompleteToDevice:
            return "CompleteToDevice";
        case ResidencyEvent::EvictHost:
            return "EvictHost";
        case ResidencyEvent::EvictDevice:
            return "EvictDevice";
        case ResidencyEvent::Dispose:
            return "Dispose";
    }
    return "Unknown";
}

}  // namespace kora::detail
