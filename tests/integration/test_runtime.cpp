#include <gtest/gtest.h>

#include <cstdint>

#include "kora/runtime.hpp"
#include "support/vulkan_test.hpp"

TEST(Runtime, OwnsSelectedDeviceAndExposesMemoryStats) {
    auto runtime = kora::Runtime::create();
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    const auto devices = runtime.value().devices();
    ASSERT_FALSE(devices.empty());

    auto info = runtime.value().device_info(devices.front());
    ASSERT_TRUE(info) << info.error().describe();
    EXPECT_FALSE(info.value().name().empty());
    EXPECT_FALSE(info.value().memory_heaps().empty());
    if (info.value().api_version().major > 1 || info.value().api_version().minor >= 1) {
        EXPECT_EQ(devices.front().uuid.size(), 32U);
        kora::RuntimeOptions by_uuid;
        by_uuid.device_policy = kora::DevicePolicy::ByUuid;
        by_uuid.device_uuid = devices.front().uuid;
        auto uuid_runtime = kora::Runtime::create(by_uuid);
        ASSERT_TRUE(uuid_runtime) << uuid_runtime.error().describe();
        const auto uuid_devices = uuid_runtime.value().devices();
        ASSERT_EQ(uuid_devices.size(), 1U);
        EXPECT_EQ(uuid_devices.front().uuid, devices.front().uuid);
    }

    auto stats = runtime.value().memory_stats(devices.front());
    ASSERT_TRUE(stats) << stats.error().describe();
    EXPECT_EQ(stats.value().pools, 0U);
    EXPECT_EQ(stats.value().blocks, 0U);
    EXPECT_EQ(stats.value().live_allocations, 0U);
    ASSERT_EQ(stats.value().heaps.size(), info.value().memory_heaps().size());
    for (std::size_t index = 0; index < stats.value().heaps.size(); ++index) {
        EXPECT_EQ(stats.value().heaps[index].heap_index, index);
        EXPECT_LE(stats.value().heaps[index].reserved_bytes, stats.value().heaps[index].budget_bytes);
        EXPECT_EQ(stats.value().heaps[index].available_bytes,
                  stats.value().heaps[index].budget_bytes - stats.value().heaps[index].reserved_bytes);
    }

    EXPECT_EQ(runtime.value().memory_stats(kora::DeviceId{UINT32_MAX, {}}).error().code,
              kora::ErrorCode::InvalidArgument);
}

TEST(Runtime, HonorsExplicitDeviceIndexSelection) {
    kora::RuntimeOptions options;
    options.device_policy = kora::DevicePolicy::ByIndex;
    options.device_index = 0;
    auto runtime = kora::Runtime::create(options);
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    const auto devices = runtime.value().devices();
    ASSERT_EQ(devices.size(), 1U);
    EXPECT_EQ(devices.front().index, 0U);
}
