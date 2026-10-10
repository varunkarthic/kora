#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "memory/suballocator.hpp"

namespace {

using kora::ErrorCode;
using kora::detail::Suballocation;
using kora::detail::Suballocator;

Suballocation allocate(Suballocator& allocator, std::uint64_t size, std::uint64_t alignment = 1) {
    auto result = allocator.allocate(size, alignment);
    EXPECT_TRUE(result.ok()) << result.error().describe();
    return result.value();
}

TEST(Suballocator, SplitsAndTracksFragmentationStatistics) {
    auto created = Suballocator::create(256);
    ASSERT_TRUE(created);
    auto allocator = std::move(created).value();

    const auto first = allocate(allocator, 17, 1);
    const auto second = allocate(allocator, 32, 64);
    EXPECT_EQ(first.offset, 0u);
    EXPECT_EQ(first.allocated_size, 32u);
    EXPECT_EQ(second.offset, 64u);
    EXPECT_EQ(second.allocated_size, 32u);

    const auto stats = allocator.stats();
    EXPECT_EQ(stats.total_bytes, 256u);
    EXPECT_EQ(stats.allocated_bytes, 64u);
    EXPECT_EQ(stats.requested_bytes, 49u);
    EXPECT_EQ(stats.internal_fragmentation_bytes, 15u);
    EXPECT_EQ(stats.free_bytes, 192u);
    EXPECT_EQ(stats.largest_free_block, 160u);
    EXPECT_EQ(stats.external_fragmentation_bytes, 32u);
    EXPECT_EQ(stats.live_allocations, 2u);
    EXPECT_EQ(stats.high_water_mark, 64u);
    EXPECT_TRUE(allocator.validate_invariants());
}

TEST(Suballocator, CoalescesNoFreeNeighbours) {
    auto allocator = std::move(Suballocator::create(96)).value();
    const auto left = allocate(allocator, 32);
    const auto middle = allocate(allocator, 32);
    const auto right = allocate(allocator, 32);
    ASSERT_TRUE(allocator.deallocate(middle));
    EXPECT_EQ(allocator.stats().largest_free_block, 32u);
    EXPECT_EQ(allocator.stats().live_allocations, 2u);
    EXPECT_TRUE(allocator.validate_invariants());
    EXPECT_TRUE(allocator.deallocate(left));
    EXPECT_TRUE(allocator.deallocate(right));
}

TEST(Suballocator, CoalescesLeftNeighbour) {
    auto allocator = std::move(Suballocator::create(96)).value();
    const auto left = allocate(allocator, 32);
    const auto middle = allocate(allocator, 32);
    const auto right = allocate(allocator, 32);
    ASSERT_TRUE(allocator.deallocate(left));
    ASSERT_TRUE(allocator.deallocate(middle));
    EXPECT_EQ(allocator.stats().largest_free_block, 64u);
    EXPECT_TRUE(allocator.deallocate(right));
    EXPECT_EQ(allocator.stats().largest_free_block, 96u);
}

TEST(Suballocator, CoalescesRightNeighbour) {
    auto allocator = std::move(Suballocator::create(96)).value();
    const auto left = allocate(allocator, 32);
    const auto middle = allocate(allocator, 32);
    const auto right = allocate(allocator, 32);
    ASSERT_TRUE(allocator.deallocate(right));
    ASSERT_TRUE(allocator.deallocate(middle));
    EXPECT_EQ(allocator.stats().largest_free_block, 64u);
    EXPECT_TRUE(allocator.deallocate(left));
    EXPECT_EQ(allocator.stats().largest_free_block, 96u);
}

TEST(Suballocator, CoalescesBothNeighbours) {
    auto allocator = std::move(Suballocator::create(128)).value();
    const auto first = allocate(allocator, 32);
    const auto middle = allocate(allocator, 32);
    const auto third = allocate(allocator, 32);
    const auto fourth = allocate(allocator, 32);
    ASSERT_TRUE(allocator.deallocate(first));
    ASSERT_TRUE(allocator.deallocate(third));
    ASSERT_TRUE(allocator.deallocate(middle));
    EXPECT_EQ(allocator.stats().largest_free_block, 96u);
    EXPECT_TRUE(allocator.deallocate(fourth));
    EXPECT_EQ(allocator.stats().largest_free_block, 128u);
    EXPECT_TRUE(allocator.validate_invariants());
}

TEST(Suballocator, HonorsPowerOfTwoAlignmentAndReportsBadRequests) {
    EXPECT_EQ(Suballocator::create(0).error().code, ErrorCode::InvalidArgument);
    auto allocator = std::move(Suballocator::create(1024)).value();
    for (const std::uint64_t alignment : {1u, 2u, 8u, 64u, 256u}) {
        const auto allocation = allocate(allocator, 1, alignment);
        EXPECT_EQ(allocation.offset % alignment, 0u);
    }
    EXPECT_EQ(allocator.allocate(0, 1).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(allocator.allocate(1, 0).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(allocator.allocate(1, 3).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(allocator.allocate(std::numeric_limits<std::uint64_t>::max(), 1).error().code,
              ErrorCode::Overflow);
}

TEST(Suballocator, ExhaustionAndInvalidFreeAreRecoverableErrors) {
    auto allocator = std::move(Suballocator::create(64)).value();
    const auto all = allocate(allocator, 64);
    EXPECT_EQ(allocator.allocate(1, 1).error().code, ErrorCode::AllocationFailed);
    auto wrong = all;
    wrong.requested_size = 63;
    EXPECT_EQ(allocator.deallocate(wrong).error().code, ErrorCode::InvalidArgument);
    ASSERT_TRUE(allocator.deallocate(all));
    EXPECT_EQ(allocator.deallocate(all).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(allocator.stats().largest_free_block, 64u);
}

TEST(Suballocator, RandomizedOperationsMatchIntervalOracle) {
    constexpr std::uint64_t capacity = 8192;
    auto allocator = std::move(Suballocator::create(capacity)).value();
    std::vector<Suballocation> live;
    std::mt19937_64 random(0x4b4f5241U);

    for (int operation = 0; operation < 5000; ++operation) {
        const bool should_allocate = live.empty() || (live.size() < 80 && (random() % 100) < 65);
        if (should_allocate) {
            const std::uint64_t size = 1 + random() % 257;
            const std::uint64_t alignment = std::uint64_t{1} << (random() % 9);
            auto result = allocator.allocate(size, alignment);
            if (result) {
                const auto candidate = result.value();
                EXPECT_EQ(candidate.offset % alignment, 0u);
                EXPECT_LE(candidate.offset + candidate.allocated_size, capacity);
                for (const auto& existing : live) {
                    EXPECT_TRUE(candidate.offset + candidate.allocated_size <= existing.offset ||
                                existing.offset + existing.allocated_size <= candidate.offset);
                }
                live.push_back(candidate);
            } else {
                EXPECT_EQ(result.error().code, ErrorCode::AllocationFailed);
            }
        } else {
            const auto position = static_cast<std::size_t>(random() % live.size());
            ASSERT_TRUE(allocator.deallocate(live[position]));
            live.erase(live.begin() + static_cast<std::ptrdiff_t>(position));
        }
        ASSERT_TRUE(allocator.validate_invariants());
        const auto stats = allocator.stats();
        EXPECT_EQ(stats.allocated_bytes + stats.free_bytes, capacity);
        EXPECT_EQ(stats.live_allocations, live.size());
    }

    for (const auto& allocation : live) ASSERT_TRUE(allocator.deallocate(allocation));
    const auto final = allocator.stats();
    EXPECT_EQ(final.free_bytes, capacity);
    EXPECT_EQ(final.largest_free_block, capacity);
    EXPECT_EQ(final.external_fragmentation_bytes, 0u);
    EXPECT_EQ(final.live_allocations, 0u);
    EXPECT_TRUE(allocator.validate_invariants());
}

}  // namespace
