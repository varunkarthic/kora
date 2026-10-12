#include <gtest/gtest.h>

#include "kora/device_selection.hpp"

namespace {

using kora::CapabilityTier;
using kora::DeviceCandidate;
using kora::DeviceKind;
using kora::DevicePolicy;
using kora::ResolvedOptions;

DeviceCandidate candidate(std::uint32_t index, DeviceKind kind, CapabilityTier tier, std::uint64_t memory_gib,
                          std::uint32_t queues = 1, bool satisfies_requirements = true) {
    return {{index, std::string(32, static_cast<char>('0' + index))},
            kind,
            tier,
            memory_gib << 30U,
            queues,
            satisfies_requirements};
}

}  // namespace

TEST(DeviceSelection, AutoCanSelectDeviceAfterFirstEnumerationEntry) {
    ResolvedOptions options;
    const std::vector devices{candidate(0, DeviceKind::Discrete, CapabilityTier::A, 8),
                              candidate(1, DeviceKind::Integrated, CapabilityTier::C, 4)};
    auto selected = kora::select_devices(options, devices);
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected.value()[0].index, 1u);
    EXPECT_NE(selected.value()[0].index, devices.front().id.index);
}

TEST(DeviceSelection, ExplicitPreferencesWinAndFallBack) {
    const std::vector devices{candidate(0, DeviceKind::Integrated, CapabilityTier::D, 32),
                              candidate(1, DeviceKind::Discrete, CapabilityTier::A, 2)};
    ResolvedOptions options;
    options.device_policy = DevicePolicy::PreferDiscrete;
    ASSERT_TRUE(kora::select_devices(options, devices));
    EXPECT_EQ(kora::select_devices(options, devices).value()[0].index, 1u);
    options.device_policy = DevicePolicy::PreferIntegrated;
    EXPECT_EQ(kora::select_devices(options, devices).value()[0].index, 0u);

    const std::vector only_cpu{candidate(2, DeviceKind::Cpu, CapabilityTier::B, 1)};
    options.device_policy = DevicePolicy::PreferDiscrete;
    EXPECT_EQ(kora::select_devices(options, only_cpu).value()[0].index, 2u);
}

TEST(DeviceSelection, IndexAndUuidAreExactAndReportErrors) {
    const std::vector devices{candidate(0, DeviceKind::Integrated, CapabilityTier::A, 1),
                              candidate(1, DeviceKind::Discrete, CapabilityTier::A, 1, 0)};
    ResolvedOptions options;
    options.device_policy = DevicePolicy::ByIndex;
    options.device_index = 0;
    EXPECT_EQ(kora::select_devices(options, devices).value()[0].index, 0u);
    options.device_index = 7;
    EXPECT_EQ(kora::select_devices(options, devices).error().code, kora::ErrorCode::NoEligibleDevice);
    options.device_index = 1;
    EXPECT_EQ(kora::select_devices(options, devices).error().code, kora::ErrorCode::NoEligibleDevice);

    options.device_policy = DevicePolicy::ByUuid;
    options.device_uuid = std::string(32, '0');
    EXPECT_EQ(kora::select_devices(options, devices).value()[0].index, 0u);
    options.device_uuid = "bad";
    EXPECT_EQ(kora::select_devices(options, devices).error().code, kora::ErrorCode::InvalidArgument);
}

TEST(DeviceSelection, AllEligibleIsRankedAndTiesUseStableIndex) {
    ResolvedOptions options;
    options.device_policy = DevicePolicy::AllEligible;
    const std::vector devices{candidate(4, DeviceKind::Discrete, CapabilityTier::B, 4),
                              candidate(2, DeviceKind::Integrated, CapabilityTier::B, 4),
                              candidate(1, DeviceKind::Integrated, CapabilityTier::D, 8, 1, false)};
    auto selected = kora::select_devices(options, devices);
    ASSERT_TRUE(selected);
    ASSERT_EQ(selected.value().size(), 2u);
    EXPECT_EQ(selected.value()[0].index, 2u);
    EXPECT_EQ(selected.value()[1].index, 4u);
}

TEST(DeviceSelection, EmptyOrIneligibleSetReturnsNoEligibleDevice) {
    ResolvedOptions options;
    const std::vector<DeviceCandidate> none;
    auto selected = kora::select_devices(options, none);
    ASSERT_FALSE(selected);
    EXPECT_EQ(selected.error().code, kora::ErrorCode::NoEligibleDevice);
}
