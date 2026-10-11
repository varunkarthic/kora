#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

#include "vulkan/command_pool_set.hpp"

namespace kora::vk {
namespace {
struct Mock {
    std::atomic<uint32_t> allocations{0};
    std::atomic<uint32_t> frees{0};
    std::atomic<uint32_t> resets{0};
    std::atomic<uint32_t> begins{0};
    std::atomic<uint32_t> ends{0};
    std::array<uintptr_t, 16> handles{};
};

VkResult VKAPI_PTR create_pool(void*, VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*,
                               VkCommandPool* pool) noexcept {
    *pool = reinterpret_cast<VkCommandPool>(uintptr_t{1});
    return VK_SUCCESS;
}
void VKAPI_PTR destroy_pool(void*, VkDevice, VkCommandPool, const VkAllocationCallbacks*) noexcept {}
VkResult VKAPI_PTR reset_pool(void* context, VkDevice, VkCommandPool, VkCommandPoolResetFlags) noexcept {
    ++static_cast<Mock*>(context)->resets;
    return VK_SUCCESS;
}
VkResult VKAPI_PTR allocate_buffers(void* context, VkDevice, const VkCommandBufferAllocateInfo* info,
                                    VkCommandBuffer* buffers) noexcept {
    auto& mock = *static_cast<Mock*>(context);
    ++mock.allocations;
    for (uint32_t i = 0; i < info->commandBufferCount; ++i) {
        mock.handles[i] = static_cast<uintptr_t>(i + 2);
        buffers[i] = reinterpret_cast<VkCommandBuffer>(mock.handles[i]);
    }
    return VK_SUCCESS;
}
void VKAPI_PTR free_buffers(void* context, VkDevice, VkCommandPool, uint32_t,
                            const VkCommandBuffer*) noexcept {
    ++static_cast<Mock*>(context)->frees;
}
VkResult VKAPI_PTR begin_buffer(void* context, VkCommandBuffer,
                                const VkCommandBufferBeginInfo* info) noexcept {
    EXPECT_NE(info->flags & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 0u);
    ++static_cast<Mock*>(context)->begins;
    return VK_SUCCESS;
}
VkResult VKAPI_PTR end_buffer(void* context, VkCommandBuffer) noexcept {
    ++static_cast<Mock*>(context)->ends;
    return VK_SUCCESS;
}

Api mock_api(Mock& mock) {
    Api api;
    api.create_command_pool = {&mock, create_pool};
    api.destroy_command_pool = {&mock, destroy_pool};
    api.reset_command_pool = {&mock, reset_pool};
    api.allocate_command_buffers = {&mock, allocate_buffers};
    api.free_command_buffers = {&mock, free_buffers};
    api.begin_command_buffer = {&mock, begin_buffer};
    api.end_command_buffer = {&mock, end_buffer};
    return api;
}

TEST(CommandPoolSet, ReusesRingWithoutSteadyStateAllocationOrFree) {
    Mock mock;
    CommandPoolSet::Options options;
    options.ring_size = 4;
    CommandPoolSet pools(mock_api(mock), reinterpret_cast<VkDevice>(uintptr_t{99}), options);
    ASSERT_EQ(pools.capacity(), 4u);
    ASSERT_EQ(pools.allocation_count(), 4u);
    ASSERT_EQ(mock.allocations.load(), 1u);

    for (uint64_t i = 1; i <= 10000; ++i) {
        const uint64_t done = i > 4 ? ((i - 1) / 4) * 4 : 0;
        auto buffer = pools.acquire(done);
        ASSERT_TRUE(buffer) << buffer.error().describe();
        ASSERT_TRUE(pools.end_recording(buffer.value()));
        ASSERT_TRUE(pools.submitted(buffer.value(), i));
    }
    EXPECT_EQ(mock.allocations.load(), 1u);
    EXPECT_EQ(mock.frees.load(), 0u);
    EXPECT_EQ(mock.resets.load(), 2499u);
    EXPECT_EQ(mock.begins.load(), 10000u);
    EXPECT_EQ(mock.ends.load(), 10000u);
}

TEST(CommandPoolSet, RejectsUnacquiredBufferAndZeroTimelineValue) {
    Mock mock;
    CommandPoolSet::Options options;
    options.ring_size = 1;
    CommandPoolSet pools(mock_api(mock), reinterpret_cast<VkDevice>(uintptr_t{99}), options);
    EXPECT_EQ(pools.submitted(reinterpret_cast<VkCommandBuffer>(uintptr_t{88}), 1).error().code,
              ErrorCode::InvalidArgument);
    auto buffer = pools.acquire(0);
    ASSERT_TRUE(buffer);
    EXPECT_EQ(pools.submitted(buffer.value(), 0).error().code, ErrorCode::InvalidArgument);
    ASSERT_TRUE(pools.end_recording(buffer.value()));
    ASSERT_TRUE(pools.submitted(buffer.value(), 1));
    EXPECT_EQ(pools.submitted(buffer.value(), 2).error().code, ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace kora::vk
