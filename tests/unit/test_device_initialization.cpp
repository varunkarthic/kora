#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "core/device_initialization.hpp"

namespace kora::detail {
namespace {

TEST(DeviceInitialization, AllEligibleIsolatesFailuresAndKeepsHealthyDevices) {
    const std::array selected{DeviceId{0, "device-zero"}, DeviceId{1, "device-one"},
                              DeviceId{2, "device-two"}};
    std::vector<DeviceId> quarantined;
    std::vector<DeviceId> initialized;
    std::vector<Error> failures;
    auto result = initialize_selected_devices(
        selected, true,
        [&](const DeviceId& id) -> Status {
            if (id.index != 1) {
                initialized.push_back(id);
                return {};
            }
            return Error(ErrorCode::UnsupportedFeature, "injected logical-device initialization failure");
        },
        [&](const DeviceId& id) { quarantined.push_back(id); },
        [&](const DeviceId&, const Error& error) { failures.push_back(error); });

    ASSERT_TRUE(result) << result.error().describe();
    EXPECT_EQ(initialized, (std::vector<DeviceId>{selected[0], selected[2]}));
    EXPECT_EQ(quarantined, (std::vector<DeviceId>{selected[1]}));
    ASSERT_EQ(failures.size(), 1U);
    EXPECT_EQ(failures.front().code, ErrorCode::UnsupportedFeature);
    EXPECT_EQ(failures.front().device, "index 1 uuid device-one");
}

TEST(DeviceInitialization, ExplicitSelectionFailsFastWithoutQuarantining) {
    const std::array selected{DeviceId{4, "device-four"}, DeviceId{5, "device-five"}};
    std::vector<DeviceId> attempted;
    std::vector<DeviceId> quarantined;
    auto result = initialize_selected_devices(
        selected, false,
        [&](const DeviceId& id) -> Status {
            attempted.push_back(id);
            return Error(ErrorCode::AllocationFailed, "injected initialization failure");
        },
        [&](const DeviceId& id) { quarantined.push_back(id); }, [](const DeviceId&, const Error&) {});

    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::AllocationFailed);
    EXPECT_EQ(result.error().device, "index 4 uuid device-four");
    EXPECT_EQ(attempted, (std::vector<DeviceId>{selected[0]}));
    EXPECT_TRUE(quarantined.empty());
}

TEST(DeviceInitialization, AllFailuresReturnFirstErrorWithDeviceIdentity) {
    const std::array selected{DeviceId{2, "device-two"}, DeviceId{3, "device-three"}};
    std::vector<DeviceId> quarantined;
    auto result = initialize_selected_devices(
        selected, true,
        [](const DeviceId& id) -> Status {
            return Error(id.index == 2 ? ErrorCode::UnsupportedFeature : ErrorCode::AllocationFailed,
                         "injected failure");
        },
        [&](const DeviceId& id) { quarantined.push_back(id); }, [](const DeviceId&, const Error&) {});

    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::UnsupportedFeature);
    EXPECT_EQ(result.error().device, "index 2 uuid device-two");
    EXPECT_EQ(quarantined, (std::vector<DeviceId>{selected[0], selected[1]}));
}

}  // namespace
}  // namespace kora::detail
