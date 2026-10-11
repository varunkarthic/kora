#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "vulkan/hazard_tracker.hpp"

namespace {

using kora::vk::accesses_conflict;
using kora::vk::AccessKind;
using kora::vk::AccessScope;
using kora::vk::BarrierPlan;
using kora::vk::HazardTracker;

TEST(HazardTracker, GoldenTableCoversRequiredAccessTransitions) {
    struct Case {
        AccessKind previous;
        AccessKind next;
        bool barrier;
        VkPipelineStageFlags source_stage;
        VkAccessFlags source_access;
        VkPipelineStageFlags destination_stage;
        VkAccessFlags destination_access;
    };
    constexpr std::array cases{
        Case{AccessKind::TransferWrite, AccessKind::ComputeRead, true, VK_PIPELINE_STAGE_TRANSFER_BIT,
             VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT},
        Case{AccessKind::ComputeWrite, AccessKind::ComputeRead, true, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
             VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT},
        Case{AccessKind::ComputeWrite, AccessKind::TransferRead, true, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
             VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT},
        Case{AccessKind::ComputeWrite, AccessKind::HostRead, true, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
             VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT},
        Case{AccessKind::HostWrite, AccessKind::ComputeRead, true, VK_PIPELINE_STAGE_HOST_BIT,
             VK_ACCESS_HOST_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT},
        Case{AccessKind::ComputeRead, AccessKind::ComputeRead, false, 0, 0, 0, 0},
    };

    for (const auto& test : cases) {
        SCOPED_TRACE(static_cast<int>(test.previous));
        HazardTracker tracker;
        (void)tracker.access(test.previous);
        const auto barriers = tracker.access(test.next);
        ASSERT_EQ(barriers.size(), test.barrier ? 1U : 0U);
        if (test.barrier) {
            EXPECT_EQ(barriers.front(), (BarrierPlan{test.source_stage, test.source_access,
                                                     test.destination_stage, test.destination_access}));
            EXPECT_NE(barriers.front().source_stage, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            EXPECT_NE(barriers.front().destination_stage, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
    }
}

TEST(HazardTracker, WriterOrdersEveryReaderAndReadersDoNotOrderEachOther) {
    HazardTracker tracker;
    EXPECT_TRUE(tracker.access(AccessKind::ComputeWrite).empty());
    EXPECT_EQ(tracker.access(AccessKind::ComputeRead).size(), 1U);
    EXPECT_EQ(tracker.access(AccessKind::TransferRead).size(), 1U);
    const auto barriers = tracker.access(AccessKind::HostWrite);
    ASSERT_EQ(barriers.size(), 1U);
    EXPECT_EQ(barriers[0].source_access,
              VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
}

TEST(HazardTracker, RandomSequencesPreserveBruteForceConflictingOrderings) {
    constexpr std::array kinds{AccessKind::HostRead,      AccessKind::HostWrite,   AccessKind::TransferRead,
                               AccessKind::TransferWrite, AccessKind::ComputeRead, AccessKind::ComputeWrite};
    std::mt19937 random(0x4b4f5241U);
    for (int sequence = 0; sequence < 256; ++sequence) {
        HazardTracker tracker;
        bool has_writer = false;
        AccessScope writer{};
        std::vector<AccessScope> readers;
        for (int index = 0; index < 128; ++index) {
            const auto next = kinds[random() % kinds.size()];
            const auto scope = kora::vk::access_scope(next);
            VkPipelineStageFlags expected_stages = 0;
            VkAccessFlags expected_access = 0;
            if (has_writer && accesses_conflict(writer, scope)) {
                expected_stages |= writer.stage;
                expected_access |= writer.access;
            }
            if (scope.writes) {
                for (const auto reader : readers) {
                    if (!accesses_conflict(reader, scope)) continue;
                    expected_stages |= reader.stage;
                    expected_access |= reader.access;
                }
            }
            const auto emitted = tracker.access(next);
            ASSERT_EQ(emitted.size(), expected_stages == 0 ? 0U : 1U) << "operation " << index;
            if (!emitted.empty()) {
                EXPECT_EQ(emitted.front().source_stage, expected_stages) << "operation " << index;
                EXPECT_EQ(emitted.front().source_access, expected_access) << "operation " << index;
                EXPECT_EQ(emitted.front().destination_stage, scope.stage) << "operation " << index;
                EXPECT_EQ(emitted.front().destination_access, scope.access) << "operation " << index;
            }
            if (scope.writes) {
                writer = scope;
                has_writer = true;
                readers.clear();
            } else {
                readers.push_back(scope);
            }
        }
    }
}

}  // namespace
