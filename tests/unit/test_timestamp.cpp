#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "profiling/timestamp.hpp"

namespace {
using kora::detail::timestamp_delta;
using kora::detail::timestamp_nanoseconds;

struct QueryFixture {
    int reads = 0;
    VkQueryResultFlags flags = 0;
};

VkResult VKAPI_PTR create_query_pool(VkDevice, const VkQueryPoolCreateInfo*, const VkAllocationCallbacks*,
                                     VkQueryPool* pool) {
    *pool = reinterpret_cast<VkQueryPool>(std::uintptr_t{1});
    return VK_SUCCESS;
}

void VKAPI_PTR destroy_query_pool(VkDevice, VkQueryPool, const VkAllocationCallbacks*) {}

VkResult VKAPI_PTR get_query_pool_results(void* context, VkDevice, VkQueryPool, std::uint32_t,
                                          std::uint32_t count, std::size_t size, void* data,
                                          VkDeviceSize stride, VkQueryResultFlags flags) noexcept {
    auto& fixture = *static_cast<QueryFixture*>(context);
    ++fixture.reads;
    fixture.flags = flags;
    if (count != 2 || size != 2 * sizeof(std::uint64_t) || stride != sizeof(std::uint64_t))
        return VK_ERROR_UNKNOWN;
    auto* values = static_cast<std::uint64_t*>(data);
    values[0] = 100;
    values[1] = 130;
    return VK_SUCCESS;
}

void VKAPI_CALL cmd_write_timestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, std::uint32_t) {}
void VKAPI_CALL cmd_reset_query_pool(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {}

TEST(GpuTimestampConversion, HandlesWrapAtQueueFamilyValidBitWidth) {
    const auto delta = timestamp_delta(250, 5, 8);
    ASSERT_TRUE(delta);
    EXPECT_EQ(delta.value(), 11U);
    EXPECT_EQ(timestamp_delta(std::numeric_limits<std::uint64_t>::max() - 3, 2, 64).value(), 6U);
    EXPECT_FALSE(timestamp_delta(0, 1, 0));
    EXPECT_FALSE(timestamp_delta(0, 1, 65));
}

TEST(GpuTimestampConversion, ConvertsTicksUsingNanosecondPeriodAndChecksOverflow) {
    const auto duration = timestamp_nanoseconds(40, 2.5F);
    ASSERT_TRUE(duration);
    EXPECT_EQ(duration.value(), 100U);
    EXPECT_FALSE(timestamp_nanoseconds(1, 0.0F));
    EXPECT_FALSE(timestamp_nanoseconds(std::numeric_limits<std::uint64_t>::max(), 2.0F));
}

TEST(GpuTimestampRing, ReadsOnlyCompletedSlotsWithoutWaiting) {
    QueryFixture fixture;
    kora::vk::Api api;
    api.create_query_pool = create_query_pool;
    api.destroy_query_pool = destroy_query_pool;
    api.get_query_pool_results = {&fixture, get_query_pool_results};
    api.cmd_write_timestamp = cmd_write_timestamp;
    api.cmd_reset_query_pool = cmd_reset_query_pool;
    kora::detail::TimestampRing ring(api, reinterpret_cast<VkDevice>(std::uintptr_t{2}), 8, 2.5F);
    ASSERT_TRUE(ring.initialize());
    ASSERT_TRUE(ring.record_start(reinterpret_cast<VkCommandBuffer>(std::uintptr_t{3}), 7));
    ring.record_end(reinterpret_cast<VkCommandBuffer>(std::uintptr_t{3}), 7);
    ring.submitted(7, 12);

    const auto pending = ring.collect(11);
    ASSERT_TRUE(pending);
    EXPECT_TRUE(pending.value().empty());
    EXPECT_EQ(fixture.reads, 0);

    const auto completed = ring.collect(12);
    ASSERT_TRUE(completed);
    ASSERT_EQ(completed.value().size(), 1U);
    EXPECT_EQ(completed.value()[0].duration_nanoseconds, 75U);
    EXPECT_EQ(fixture.reads, 1);
    EXPECT_EQ(fixture.flags, VK_QUERY_RESULT_64_BIT);
    EXPECT_EQ(ring.samples_collected(), 1U);
    const auto history = ring.history();
    EXPECT_TRUE(history.empty());

    ring.enable_trace_history(true);

    for (std::uint64_t submission_id = 8; submission_id <= 4103; ++submission_id) {
        ASSERT_TRUE(ring.record_start(reinterpret_cast<VkCommandBuffer>(std::uintptr_t{3}), submission_id));
        ring.record_end(reinterpret_cast<VkCommandBuffer>(std::uintptr_t{3}), submission_id);
        ring.submitted(submission_id, submission_id);
        const auto collected = ring.collect(submission_id);
        ASSERT_TRUE(collected);
        ASSERT_EQ(collected.value().size(), 1U);
    }
    const auto bounded_history = ring.history();
    ASSERT_EQ(bounded_history.size(), 4096U);
    EXPECT_EQ(bounded_history.front().submission_id, 8U);
    EXPECT_EQ(bounded_history.back().submission_id, 4103U);
}
}  // namespace
