#include <gtest/gtest.h>

#include "vulkan/queue_family_transfer.hpp"

namespace kora::vk {
namespace {

TEST(QueueFamilyTransfer, SameFamilyUseRequiresNeitherBarriersNorSemaphore) {
    const auto pair = make_ownership_barrier_pair(
        reinterpret_cast<VkBuffer>(uintptr_t{7}), 4, 64,
        {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, true},
        {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, false}, 3, 3);
    EXPECT_FALSE(pair.has_value());
}

TEST(QueueFamilyTransfer, DifferentFamiliesGenerateMatchingReleaseAcquirePair) {
    const auto buffer = reinterpret_cast<VkBuffer>(uintptr_t{7});
    const auto pair = make_ownership_barrier_pair(
        buffer, 16, 128, {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, true},
        {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, false}, 2, 5);
    ASSERT_TRUE(pair.has_value());
    EXPECT_TRUE(pair->requires_semaphore_dependency);
    EXPECT_EQ(pair->source_family, 2U);
    EXPECT_EQ(pair->destination_family, 5U);
    for (const auto* barrier : {&pair->release, &pair->acquire}) {
        EXPECT_EQ(barrier->buffer, buffer);
        EXPECT_EQ(barrier->offset, 16U);
        EXPECT_EQ(barrier->size, 128U);
        EXPECT_EQ(barrier->srcQueueFamilyIndex, 2U);
        EXPECT_EQ(barrier->dstQueueFamilyIndex, 5U);
    }
    EXPECT_EQ(pair->release.srcStageMask, VK_PIPELINE_STAGE_TRANSFER_BIT);
    EXPECT_EQ(pair->release.srcAccessMask, VK_ACCESS_TRANSFER_WRITE_BIT);
    EXPECT_EQ(pair->release.dstStageMask, VK_PIPELINE_STAGE_2_NONE);
    EXPECT_EQ(pair->release.dstAccessMask, VK_ACCESS_2_NONE);
    EXPECT_EQ(pair->acquire.srcStageMask, VK_PIPELINE_STAGE_2_NONE);
    EXPECT_EQ(pair->acquire.srcAccessMask, VK_ACCESS_2_NONE);
    EXPECT_EQ(pair->acquire.dstStageMask, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    EXPECT_EQ(pair->acquire.dstAccessMask, VK_ACCESS_SHADER_READ_BIT);
    EXPECT_EQ(pair->release_legacy.srcQueueFamilyIndex, 2U);
    EXPECT_EQ(pair->release_legacy.dstQueueFamilyIndex, 5U);
    EXPECT_EQ(pair->release_legacy.srcAccessMask, VK_ACCESS_TRANSFER_WRITE_BIT);
    EXPECT_EQ(pair->release_legacy.dstAccessMask, 0U);
    EXPECT_EQ(pair->release_legacy_source_stage, VK_PIPELINE_STAGE_TRANSFER_BIT);
    EXPECT_EQ(pair->acquire_legacy.srcQueueFamilyIndex, 2U);
    EXPECT_EQ(pair->acquire_legacy.dstQueueFamilyIndex, 5U);
    EXPECT_EQ(pair->acquire_legacy.srcAccessMask, 0U);
    EXPECT_EQ(pair->acquire_legacy.dstAccessMask, VK_ACCESS_SHADER_READ_BIT);
    EXPECT_EQ(pair->acquire_legacy_destination_stage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

TEST(QueueFamilyTransfer, InvalidOrEmptyRangesDoNotProduceBarriers) {
    const auto scope = AccessScope{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, true};
    EXPECT_FALSE(make_ownership_barrier_pair(VK_NULL_HANDLE, 0, 4, scope, scope, 1, 2));
    EXPECT_FALSE(
        make_ownership_barrier_pair(reinterpret_cast<VkBuffer>(uintptr_t{7}), 0, 0, scope, scope, 1, 2));
    EXPECT_FALSE(make_ownership_barrier_pair(reinterpret_cast<VkBuffer>(uintptr_t{7}), 0, 4, scope, scope,
                                             VK_QUEUE_FAMILY_IGNORED, 2));
}

TEST(QueueFamilyTransfer, CachedPolicyOnlyRequiresTransferForDistinctFamilies) {
    EXPECT_FALSE((QueueFamilyPolicy{4, 4}).requires_ownership_transfer());
    EXPECT_FALSE((QueueFamilyPolicy{1, 6}).requires_ownership_transfer());
    EXPECT_TRUE((QueueFamilyPolicy{1, 6, false}).requires_ownership_transfer());
    EXPECT_TRUE((QueueFamilyPolicy{1, 6}).uses_concurrent_sharing());
    EXPECT_FALSE((QueueFamilyPolicy{VK_QUEUE_FAMILY_IGNORED, 6}).requires_ownership_transfer());
}

}  // namespace
}  // namespace kora::vk
