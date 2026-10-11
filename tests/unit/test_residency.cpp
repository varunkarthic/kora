#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "memory/residency.hpp"

namespace {

using kora::ErrorCode;
using kora::detail::ResidencyDeviceId;
using kora::detail::ResidencyEvent;
using kora::detail::ResidencyLocation;
using kora::detail::ResidencyState;
using kora::detail::ResidencyTracker;
using kora::detail::transition_residency;

struct TransitionCase {
    ResidencyState from;
    ResidencyEvent event;
    ResidencyState to;
};

constexpr std::array kTransitionCases{
    TransitionCase{ResidencyState::HostOnly, ResidencyEvent::HostWrite, ResidencyState::HostOnly},
    TransitionCase{ResidencyState::HostOnly, ResidencyEvent::DeviceWrite, ResidencyState::DeviceNewer},
    TransitionCase{ResidencyState::HostOnly, ResidencyEvent::BeginTransfer, ResidencyState::InTransfer},
    TransitionCase{ResidencyState::DeviceOnly, ResidencyEvent::HostWrite, ResidencyState::HostNewer},
    TransitionCase{ResidencyState::DeviceOnly, ResidencyEvent::DeviceWrite, ResidencyState::DeviceOnly},
    TransitionCase{ResidencyState::DeviceOnly, ResidencyEvent::BeginTransfer, ResidencyState::InTransfer},
    TransitionCase{ResidencyState::Shared, ResidencyEvent::HostWrite, ResidencyState::HostNewer},
    TransitionCase{ResidencyState::Shared, ResidencyEvent::DeviceWrite, ResidencyState::DeviceNewer},
    TransitionCase{ResidencyState::Shared, ResidencyEvent::BeginTransfer, ResidencyState::InTransfer},
    TransitionCase{ResidencyState::Shared, ResidencyEvent::EvictHost, ResidencyState::DeviceOnly},
    TransitionCase{ResidencyState::Shared, ResidencyEvent::EvictDevice, ResidencyState::Evicted},
    TransitionCase{ResidencyState::HostAndDeviceClean, ResidencyEvent::HostWrite, ResidencyState::HostNewer},
    TransitionCase{ResidencyState::HostAndDeviceClean, ResidencyEvent::DeviceWrite,
                   ResidencyState::DeviceNewer},
    TransitionCase{ResidencyState::HostAndDeviceClean, ResidencyEvent::BeginTransfer,
                   ResidencyState::InTransfer},
    TransitionCase{ResidencyState::HostAndDeviceClean, ResidencyEvent::EvictHost, ResidencyState::DeviceOnly},
    TransitionCase{ResidencyState::HostAndDeviceClean, ResidencyEvent::EvictDevice, ResidencyState::Evicted},
    TransitionCase{ResidencyState::HostNewer, ResidencyEvent::HostWrite, ResidencyState::HostNewer},
    TransitionCase{ResidencyState::HostNewer, ResidencyEvent::DeviceWrite, ResidencyState::DeviceNewer},
    TransitionCase{ResidencyState::HostNewer, ResidencyEvent::BeginTransfer, ResidencyState::InTransfer},
    TransitionCase{ResidencyState::HostNewer, ResidencyEvent::EvictDevice, ResidencyState::Evicted},
    TransitionCase{ResidencyState::DeviceNewer, ResidencyEvent::HostWrite, ResidencyState::HostNewer},
    TransitionCase{ResidencyState::DeviceNewer, ResidencyEvent::DeviceWrite, ResidencyState::DeviceNewer},
    TransitionCase{ResidencyState::DeviceNewer, ResidencyEvent::BeginTransfer, ResidencyState::InTransfer},
    TransitionCase{ResidencyState::DeviceNewer, ResidencyEvent::EvictHost, ResidencyState::DeviceOnly},
    TransitionCase{ResidencyState::DeviceNewer, ResidencyEvent::EvictDevice, ResidencyState::DeviceNewer},
    TransitionCase{ResidencyState::Evicted, ResidencyEvent::HostWrite, ResidencyState::Evicted},
    TransitionCase{ResidencyState::Evicted, ResidencyEvent::DeviceWrite, ResidencyState::DeviceNewer},
    TransitionCase{ResidencyState::Evicted, ResidencyEvent::BeginTransfer, ResidencyState::InTransfer},
    TransitionCase{ResidencyState::InTransfer, ResidencyEvent::CompleteToHost,
                   ResidencyState::HostAndDeviceClean},
    TransitionCase{ResidencyState::InTransfer, ResidencyEvent::CompleteToDevice, ResidencyState::Shared},
};

constexpr std::array kStates{ResidencyState::HostOnly,  ResidencyState::DeviceOnly,
                             ResidencyState::Shared,    ResidencyState::HostAndDeviceClean,
                             ResidencyState::HostNewer, ResidencyState::DeviceNewer,
                             ResidencyState::Evicted,   ResidencyState::InTransfer,
                             ResidencyState::Disposed};
constexpr std::array kEvents{ResidencyEvent::HostWrite,        ResidencyEvent::DeviceWrite,
                             ResidencyEvent::BeginTransfer,    ResidencyEvent::CompleteToHost,
                             ResidencyEvent::CompleteToDevice, ResidencyEvent::EvictHost,
                             ResidencyEvent::EvictDevice,      ResidencyEvent::Dispose};

void expect_current_copy_invariant(const ResidencyTracker& tracker) {
    const auto snapshot_result = tracker.snapshot();
    ASSERT_TRUE(snapshot_result);
    const auto& snapshot = snapshot_result.value();
    if (snapshot.disposed) {
        EXPECT_EQ(snapshot.state, ResidencyState::Disposed);
        EXPECT_FALSE(snapshot.host_current);
        for (const auto& [device, current] : snapshot.device_current) {
            static_cast<void>(device);
            EXPECT_FALSE(current);
        }
        return;
    }
    bool current_device = false;
    for (const auto& [device, current] : snapshot.device_current) {
        static_cast<void>(device);
        current_device |= current;
    }
    EXPECT_TRUE(snapshot.host_current || current_device);
}

TEST(ResidencyState, ExhaustiveStateEventTableAcceptsOnlyDocumentedTransitions) {
    for (const auto state : kStates) {
        for (const auto event : kEvents) {
            const auto expected = std::find_if(kTransitionCases.begin(), kTransitionCases.end(),
                                               [state, event](const TransitionCase& item) {
                                                   return item.from == state && item.event == event;
                                               });
            const bool disposal_is_legal = event == ResidencyEvent::Dispose &&
                                           state != ResidencyState::InTransfer &&
                                           state != ResidencyState::Disposed;
            const auto actual = transition_residency(state, event);
            if (expected != kTransitionCases.end()) {
                ASSERT_TRUE(actual) << kora::detail::to_string(state) << " + "
                                    << kora::detail::to_string(event);
                EXPECT_EQ(actual.value(), expected->to);
            } else if (disposal_is_legal) {
                ASSERT_TRUE(actual);
                EXPECT_EQ(actual.value(), ResidencyState::Disposed);
            } else {
                EXPECT_FALSE(actual) << kora::detail::to_string(state) << " + "
                                     << kora::detail::to_string(event);
                if (!actual) EXPECT_EQ(actual.error().code, ErrorCode::InvalidArgument);
            }
        }
    }
}

TEST(ResidencyTracker, PlansOnlyForAbsentOrStaleDestinationCopies) {
    auto tracker = ResidencyTracker::host_initialized();
    auto first_upload = tracker.plan_transfer(ResidencyLocation::device_copy(3));
    ASSERT_TRUE(first_upload);
    EXPECT_TRUE(first_upload.value().required);
    EXPECT_TRUE(first_upload.value().source.host);
    ASSERT_TRUE(tracker.begin_transfer(first_upload.value()));
    EXPECT_EQ(tracker.state(), ResidencyState::InTransfer);
    ASSERT_TRUE(tracker.complete_transfer());
    EXPECT_EQ(tracker.state(), ResidencyState::HostAndDeviceClean);

    const auto already_current = tracker.plan_transfer(ResidencyLocation::device_copy(3));
    ASSERT_TRUE(already_current);
    EXPECT_FALSE(already_current.value().required);

    auto second_upload = tracker.plan_transfer(ResidencyLocation::device_copy(8));
    ASSERT_TRUE(second_upload);
    EXPECT_TRUE(second_upload.value().required);
    ASSERT_TRUE(tracker.begin_transfer(second_upload.value()));
    ASSERT_TRUE(tracker.complete_transfer());
    EXPECT_EQ(tracker.state(), ResidencyState::Shared);

    ASSERT_TRUE(tracker.write_host());
    EXPECT_EQ(tracker.state(), ResidencyState::HostNewer);
    const auto stale_destination = tracker.plan_transfer(ResidencyLocation::device_copy(3));
    ASSERT_TRUE(stale_destination);
    EXPECT_TRUE(stale_destination.value().required);
    EXPECT_TRUE(stale_destination.value().source.host);
    ASSERT_TRUE(tracker.begin_transfer(stale_destination.value()));
    ASSERT_TRUE(tracker.complete_transfer());
    EXPECT_EQ(tracker.state(), ResidencyState::HostAndDeviceClean);
    expect_current_copy_invariant(tracker);
}

TEST(ResidencyTracker, DeviceWritesReadBackAndFailedTransfersPreserveCurrentCopy) {
    auto created = ResidencyTracker::device_initialized(5);
    ASSERT_TRUE(created);
    auto tracker = std::move(created).value();
    ASSERT_TRUE(tracker.write_host());
    EXPECT_EQ(tracker.state(), ResidencyState::HostNewer);
    ASSERT_TRUE(tracker.write_device(5));
    EXPECT_EQ(tracker.state(), ResidencyState::DeviceNewer);

    auto readback = tracker.plan_transfer(ResidencyLocation::host_copy());
    ASSERT_TRUE(readback);
    EXPECT_TRUE(readback.value().required);
    EXPECT_FALSE(readback.value().source.host);
    ASSERT_TRUE(tracker.begin_transfer(readback.value()));
    ASSERT_TRUE(tracker.fail_transfer());
    EXPECT_EQ(tracker.state(), ResidencyState::DeviceNewer);
    expect_current_copy_invariant(tracker);

    readback = tracker.plan_transfer(ResidencyLocation::host_copy());
    ASSERT_TRUE(readback);
    ASSERT_TRUE(tracker.begin_transfer(readback.value()));
    ASSERT_TRUE(tracker.complete_transfer());
    EXPECT_EQ(tracker.state(), ResidencyState::HostAndDeviceClean);
    const auto snapshot = tracker.snapshot();
    ASSERT_TRUE(snapshot);
    EXPECT_TRUE(snapshot.value().host_current);
    EXPECT_TRUE(snapshot.value().device_current.at(5));
}

TEST(ResidencyTracker, MultiDeviceCopiesRemainVersionedAndCannotEvictLastCurrentCopy) {
    auto tracker = ResidencyTracker::host_initialized();
    auto to_one = tracker.plan_transfer(ResidencyLocation::device_copy(1));
    ASSERT_TRUE(to_one && to_one.value().required);
    ASSERT_TRUE(tracker.begin_transfer(to_one.value()));
    ASSERT_TRUE(tracker.complete_transfer());
    auto to_two = tracker.plan_transfer(ResidencyLocation::device_copy(2));
    ASSERT_TRUE(to_two && to_two.value().required);
    ASSERT_TRUE(tracker.begin_transfer(to_two.value()));
    ASSERT_TRUE(tracker.complete_transfer());
    EXPECT_EQ(tracker.state(), ResidencyState::Shared);

    ASSERT_TRUE(tracker.write_device(1));
    EXPECT_EQ(tracker.state(), ResidencyState::DeviceNewer);
    const auto peer_copy = tracker.plan_transfer(ResidencyLocation::device_copy(2));
    ASSERT_TRUE(peer_copy);
    EXPECT_TRUE(peer_copy.value().required);
    EXPECT_EQ(peer_copy.value().source, ResidencyLocation::device_copy(1));

    EXPECT_FALSE(tracker.evict_device(1));  // It is the only current copy.
    EXPECT_EQ(tracker.state(), ResidencyState::DeviceNewer);
    ASSERT_TRUE(tracker.begin_transfer(peer_copy.value()));
    ASSERT_TRUE(tracker.complete_transfer());
    EXPECT_EQ(tracker.state(), ResidencyState::Shared);
    ASSERT_TRUE(tracker.evict_host());  // Stale host storage is unnecessary after peer replication.
    EXPECT_EQ(tracker.state(), ResidencyState::Shared);
    ASSERT_TRUE(tracker.evict_device(1));
    EXPECT_EQ(tracker.state(), ResidencyState::DeviceOnly);
    EXPECT_FALSE(tracker.evict_device(2));
    expect_current_copy_invariant(tracker);
}

TEST(ResidencyTracker, EvictionAndDisposalPreserveTheCurrentCopyInvariant) {
    auto tracker = ResidencyTracker::host_initialized();
    auto upload = tracker.plan_transfer(ResidencyLocation::device_copy(7));
    ASSERT_TRUE(upload && upload.value().required);
    ASSERT_TRUE(tracker.begin_transfer(upload.value()));
    ASSERT_TRUE(tracker.complete_transfer());
    ASSERT_TRUE(tracker.evict_host());
    EXPECT_EQ(tracker.state(), ResidencyState::DeviceOnly);
    EXPECT_FALSE(tracker.evict_device(7));
    expect_current_copy_invariant(tracker);

    auto readback = tracker.plan_transfer(ResidencyLocation::host_copy());
    ASSERT_TRUE(readback && readback.value().required);
    ASSERT_TRUE(tracker.begin_transfer(readback.value()));
    ASSERT_TRUE(tracker.complete_transfer());
    ASSERT_TRUE(tracker.evict_device(7));
    EXPECT_EQ(tracker.state(), ResidencyState::Evicted);
    expect_current_copy_invariant(tracker);
    ASSERT_TRUE(tracker.write_host());
    EXPECT_EQ(tracker.state(), ResidencyState::Evicted);
    ASSERT_TRUE(tracker.dispose());
    EXPECT_EQ(tracker.state(), ResidencyState::Disposed);
    expect_current_copy_invariant(tracker);
}

TEST(ResidencyTracker, DeterministicOperationSequenceAlwaysRetainsCurrentData) {
    auto tracker = ResidencyTracker::host_initialized();
    std::mt19937 generator(0x26A11U);
    for (std::size_t i = 0; i < 1000; ++i) {
        const auto device = static_cast<ResidencyDeviceId>(generator() % 4U);
        switch (generator() % 5U) {
            case 0:
                static_cast<void>(tracker.write_host());
                break;
            case 1:
                static_cast<void>(tracker.write_device(device));
                break;
            case 2: {
                const ResidencyLocation destination = (generator() & 1U) != 0
                                                          ? ResidencyLocation::host_copy()
                                                          : ResidencyLocation::device_copy(device);
                auto plan = tracker.plan_transfer(destination);
                if (plan && plan.value().required && tracker.begin_transfer(plan.value()))
                    static_cast<void>(tracker.complete_transfer());
                break;
            }
            case 3:
                static_cast<void>(tracker.evict_host());
                break;
            case 4:
                static_cast<void>(tracker.evict_device(device));
                break;
        }
        expect_current_copy_invariant(tracker);
    }
}

TEST(ResidencyTracker, RejectsStalePlansConcurrentTransfersAndDisposedCopies) {
    auto tracker = ResidencyTracker::host_initialized();
    auto plan = tracker.plan_transfer(ResidencyLocation::device_copy(4));
    ASSERT_TRUE(plan && plan.value().required);
    ASSERT_TRUE(tracker.write_host());
    EXPECT_EQ(tracker.begin_transfer(plan.value()).error().code, ErrorCode::StaleHandle);
    plan = tracker.plan_transfer(ResidencyLocation::device_copy(4));
    ASSERT_TRUE(plan);
    ASSERT_TRUE(tracker.begin_transfer(plan.value()));
    EXPECT_EQ(tracker.plan_transfer(ResidencyLocation::host_copy()).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(tracker.dispose().error().code, ErrorCode::InvalidArgument);
    ASSERT_TRUE(tracker.fail_transfer());
    ASSERT_TRUE(tracker.dispose());
    EXPECT_EQ(tracker.plan_transfer(ResidencyLocation::host_copy()).error().code, ErrorCode::StaleHandle);
    EXPECT_EQ(tracker.write_host().error().code, ErrorCode::InvalidArgument);
}

}  // namespace
