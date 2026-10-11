#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include "core/device_caps.hpp"
#include "memory/memory_manager.hpp"
#include "memory/memory_pool.hpp"
#include "support/vulkan_test.hpp"
#include "vulkan/api.hpp"

namespace {

TEST(MemoryStress, RepeatedSmallAllocationsReportConsistentFragmentation) {
    auto loader = kora::vk::Loader::load();
    if (!loader) KORA_REQUIRE_VULKAN(loader);
    auto instance = loader.value().create_instance();
    if (!instance && instance.error().code == kora::ErrorCode::NoVulkanImplementation)
        KORA_REQUIRE_VULKAN(false);
    ASSERT_TRUE(instance) << instance.error().describe();
    auto physical_devices = instance.value().physical_devices();
    ASSERT_TRUE(physical_devices) << physical_devices.error().describe();

    const auto selected = std::find_if(
        physical_devices.value().begin(), physical_devices.value().end(), [](const auto& physical) {
            return std::any_of(physical.queue_family_properties.begin(),
                               physical.queue_family_properties.end(), [](const auto& family) {
                                   return family.queueCount > 0 &&
                                          (family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
                               });
        });
    if (selected == physical_devices.value().end()) GTEST_SKIP() << "No Vulkan compute-capable device";
    auto device = instance.value().create_device(*selected);
    ASSERT_TRUE(device) << device.error().describe();

    const auto info = kora::detail::DeviceCaps::from_vulkan(*selected).to_device_info();
    auto type = kora::detail::MemoryManager::select_memory_type(
        info.memory_types(), info.memory_type_preferences(), kora::detail::MemoryPolicy::HostVisible,
        std::numeric_limits<std::uint32_t>::max(),
        kora::detail::MemoryTypeRequirements{.host_visible = true});
    if (!type) GTEST_SKIP() << "Device has no host-visible memory type";
    const auto& memory_type = info.memory_types()[type.value()];
    kora::detail::MemoryPoolConfig config;
    config.memory_type_index = type.value();
    config.usage = kora::detail::MemoryUsageClass::Upload;
    config.host_visible = memory_type.host_visible;
    config.host_coherent = memory_type.host_coherent;
    config.heap_size = info.memory_heaps()[memory_type.heap_index].size_bytes;
    config.budget = config.heap_size;
    config.block_size_hint = 8U * 1024U * 1024U;
    auto pool_result = kora::detail::MemoryPool::create(device.value(), config);
    if (!pool_result && pool_result.error().code == kora::ErrorCode::AllocationFailed)
        GTEST_SKIP() << pool_result.error().describe();
    ASSERT_TRUE(pool_result) << pool_result.error().describe();
    auto& pool = *pool_result.value();

    std::mt19937_64 random(0x4b4f52415f3237ULL);
    for (std::size_t cycle = 0; cycle < 3; ++cycle) {
        std::vector<kora::detail::PoolAllocation> live;
        live.reserve(3000);
        for (std::size_t index = 0; index < 3000; ++index) {
            const auto size = 16U + random() % 1009U;
            const auto alignment = std::uint64_t{1} << (random() % 8U);
            auto allocation = pool.allocate({size, alignment, false, false});
            ASSERT_TRUE(allocation) << allocation.error().describe();
            EXPECT_EQ(allocation.value().offset % alignment, 0U);
            EXPECT_NE(allocation.value().mapped_data, nullptr);
            live.push_back(allocation.value());
        }

        const auto snapshot = pool.stats();
        EXPECT_EQ(snapshot.live_allocations, live.size());
        EXPECT_EQ(snapshot.allocated_bytes + snapshot.free_bytes, snapshot.reserved_bytes);
        EXPECT_EQ(snapshot.internal_fragmentation_bytes, snapshot.allocated_bytes - snapshot.requested_bytes);
        EXPECT_EQ(snapshot.external_fragmentation_bytes, snapshot.free_bytes - snapshot.largest_free_block);
        EXPECT_GE(snapshot.high_water_mark, snapshot.allocated_bytes);
        const auto total = kora::detail::MemoryManager::aggregate_stats(
            std::span<const kora::detail::MemoryPoolStats>(&snapshot, 1));
        ASSERT_TRUE(total) << total.error().describe();
        EXPECT_EQ(total.value().reserved_bytes, snapshot.reserved_bytes);
        EXPECT_EQ(total.value().allocated_bytes, snapshot.allocated_bytes);
        EXPECT_EQ(total.value().requested_bytes, snapshot.requested_bytes);
        EXPECT_EQ(total.value().live_allocations, snapshot.live_allocations);

        std::shuffle(live.begin(), live.end(), random);
        for (const auto& allocation : live) ASSERT_TRUE(pool.deallocate(allocation));
        const auto empty = pool.stats();
        EXPECT_EQ(empty.live_allocations, 0U);
        EXPECT_EQ(empty.allocated_bytes, 0U);
        EXPECT_EQ(empty.free_bytes, empty.reserved_bytes);
        EXPECT_EQ(empty.largest_free_block, empty.free_bytes);
        EXPECT_EQ(empty.external_fragmentation_bytes, 0U);
    }
}

}  // namespace
