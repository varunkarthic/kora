#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "core/device_caps.hpp"
#include "memory/memory_manager.hpp"
#include "memory/memory_pool.hpp"
#include "memory/staging_ring.hpp"
#include "support/vulkan_test.hpp"
#include "vulkan/api.hpp"

namespace kora::detail {
namespace {

TEST(StagingRing, RandomRangesWaitForOldestTimelineAndNeverOverlapInFlight) {
    auto loader = vk::Loader::load();
    if (!loader) KORA_REQUIRE_VULKAN(loader);
    auto instance = loader.value().create_instance();
    if (!instance && instance.error().code == ErrorCode::NoVulkanImplementation) KORA_REQUIRE_VULKAN(false);
    ASSERT_TRUE(instance) << instance.error().describe();
    auto physical_devices = instance.value().physical_devices();
    ASSERT_TRUE(physical_devices) << physical_devices.error().describe();
    if (physical_devices.value().empty()) GTEST_SKIP() << "No Vulkan physical device is available";
    auto device = instance.value().create_device(physical_devices.value().front());
    ASSERT_TRUE(device) << device.error().describe();

    const auto info = DeviceCaps::from_vulkan(physical_devices.value().front()).to_device_info();
    auto type = MemoryManager::select_memory_type(
        info.memory_types(), info.memory_type_preferences(), MemoryPolicy::HostVisible,
        std::numeric_limits<std::uint32_t>::max(), MemoryTypeRequirements{.host_visible = true});
    if (!type) GTEST_SKIP() << "Device has no host-visible memory type";
    const auto& selected_type = info.memory_types()[type.value()];
    MemoryPoolConfig pool_config;
    pool_config.memory_type_index = type.value();
    pool_config.usage = MemoryUsageClass::Upload;
    pool_config.host_visible = true;
    pool_config.host_coherent = selected_type.host_coherent;
    pool_config.heap_size = info.memory_heaps()[selected_type.heap_index].size_bytes;
    pool_config.budget = pool_config.heap_size;
    pool_config.block_size_hint = 1024U * 1024U;
    auto pool_result = MemoryPool::create(device.value(), pool_config);
    if (!pool_result && pool_result.error().code == ErrorCode::AllocationFailed)
        GTEST_SKIP() << pool_result.error().describe();
    ASSERT_TRUE(pool_result) << pool_result.error().describe();

    StagingRing::Options options;
    options.initial_capacity = 4096;
    options.maximum_capacity = 4096;
    options.topology = info.memory_topology();
    options.measured_staging_benefit = true;  // Force the staging path on test hardware of any topology.
    auto ring_result = StagingRing::create(device.value(), *pool_result.value(), options);
    ASSERT_TRUE(ring_result) << ring_result.error().describe();
    auto& ring = *ring_result.value();
    std::uint64_t completed = 0;
    std::uint64_t next_timeline = 1;
    std::vector<std::pair<StagingSlice, std::uint64_t>> live;
    std::mt19937 random(0x53544147U);
    bool waited = false;
    for (std::size_t iteration = 0; iteration < 1000; ++iteration) {
        const auto size = static_cast<VkDeviceSize>(64 + random() % 384);
        auto slice = ring.reserve(size, 16, completed, [&](std::uint64_t value) {
            waited = true;
            completed = std::max(completed, value);
            return Result<std::uint64_t>(completed);
        });
        ASSERT_TRUE(slice) << slice.error().describe();
        live.erase(std::remove_if(live.begin(), live.end(),
                                  [completed](const auto& active) { return active.second <= completed; }),
                   live.end());
        EXPECT_NE(slice.value().mapped_data, nullptr);
        for (const auto& active : live) {
            const auto& other = active.first;
            if (other.buffer != slice.value().buffer) continue;
            EXPECT_TRUE(slice.value().offset + slice.value().size <= other.offset ||
                        other.offset + other.size <= slice.value().offset);
        }
        std::fill_n(slice.value().mapped_data, static_cast<std::size_t>(size), std::byte{0x6d});
        ASSERT_TRUE(ring.flush(slice.value()));
        ASSERT_TRUE(ring.retire(slice.value(), next_timeline));
        live.emplace_back(slice.value(), next_timeline);
        ++next_timeline;
    }
    EXPECT_TRUE(waited);
    EXPECT_EQ(ring.capacity(), options.maximum_capacity);

    auto full = ring.reserve(options.initial_capacity, 1, next_timeline, {});
    ASSERT_TRUE(full) << full.error().describe();
    bool waited_for_unsubmitted = false;
    auto exhausted = ring.reserve(1, 1, next_timeline, [&](std::uint64_t) {
        waited_for_unsubmitted = true;
        return Result<std::uint64_t>(next_timeline);
    });
    EXPECT_FALSE(exhausted);
    EXPECT_EQ(exhausted.error().code, ErrorCode::AllocationFailed);
    EXPECT_FALSE(waited_for_unsubmitted);
    ASSERT_TRUE(ring.cancel(full.value()));
    auto reusable = ring.reserve(1, 1, next_timeline, {});
    ASSERT_TRUE(reusable) << reusable.error().describe();
    EXPECT_TRUE(ring.cancel(reusable.value()));

    options.direction = StagingDirection::Readback;
    auto readback_result = StagingRing::create(device.value(), *pool_result.value(), options);
    ASSERT_TRUE(readback_result) << readback_result.error().describe();
    auto readback_slice = readback_result.value()->reserve(128, 16, next_timeline, {});
    ASSERT_TRUE(readback_slice) << readback_slice.error().describe();
    EXPECT_NE(readback_slice.value().mapped_data, nullptr);
    ASSERT_TRUE(readback_result.value()->retire(readback_slice.value(), next_timeline));
    EXPECT_TRUE(readback_result.value()->invalidate(readback_slice.value()));
}

}  // namespace
}  // namespace kora::detail
