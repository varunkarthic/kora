#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

#include "memory/dirty_ranges.hpp"

namespace {

using kora::DirtyRange;
using kora::ErrorCode;
using kora::detail::DirtyRangeSet;

struct FakeMappedMemory {
    std::vector<VkMappedMemoryRange> flushed;
    std::vector<VkMappedMemoryRange> invalidated;
    VkResult flush_result = VK_SUCCESS;
    VkResult invalidate_result = VK_SUCCESS;
    std::uint32_t flush_calls = 0;
    std::uint32_t invalidate_calls = 0;
};

template <class Handle>
Handle fake_handle(std::uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<Handle>(value);
    else
        return static_cast<Handle>(value);
}

VkResult VKAPI_PTR flush_ranges(void* context, VkDevice, std::uint32_t count,
                                const VkMappedMemoryRange* ranges) noexcept {
    auto& fake = *static_cast<FakeMappedMemory*>(context);
    ++fake.flush_calls;
    fake.flushed.assign(ranges, ranges + count);
    return fake.flush_result;
}

VkResult VKAPI_PTR invalidate_ranges(void* context, VkDevice, std::uint32_t count,
                                     const VkMappedMemoryRange* ranges) noexcept {
    auto& fake = *static_cast<FakeMappedMemory*>(context);
    ++fake.invalidate_calls;
    fake.invalidated.assign(ranges, ranges + count);
    return fake.invalidate_result;
}

kora::vk::Api fake_api(FakeMappedMemory& fake) {
    kora::vk::Api api;
    api.flush_mapped_memory_ranges = {&fake, flush_ranges};
    api.invalidate_mapped_memory_ranges = {&fake, invalidate_ranges};
    return api;
}

TEST(DirtyRangeSet, MergesOverlappingAndAdjacentRangesInSortedOrder) {
    DirtyRangeSet dirty;
    ASSERT_TRUE(dirty.add({24, 8}));
    ASSERT_TRUE(dirty.add({0, 8}));
    ASSERT_TRUE(dirty.add({8, 16}));  // Adjacent intervals coalesce.
    ASSERT_TRUE(dirty.add({64, 4}));
    ASSERT_TRUE(dirty.add({66, 8}));
    ASSERT_EQ(dirty.ranges().size(), 2U);
    EXPECT_EQ(dirty.ranges()[0].offset, 0U);
    EXPECT_EQ(dirty.ranges()[0].size, 32U);
    EXPECT_EQ(dirty.ranges()[1].offset, 64U);
    EXPECT_EQ(dirty.ranges()[1].size, 10U);
    EXPECT_TRUE(dirty.add({100, 0}));
    EXPECT_EQ(dirty.ranges().size(), 2U);
}

TEST(DirtyRangeSet, AlignsToAtomBoundariesAndClampsAllocationEnd) {
    DirtyRangeSet dirty;
    ASSERT_TRUE(dirty.add({3, 5}));
    ASSERT_TRUE(dirty.add({70, 10}));
    auto aligned = dirty.aligned(100, 64);
    ASSERT_TRUE(aligned);
    ASSERT_EQ(aligned.value().size(), 1U);  // Aligned intervals touch at atom 64.
    EXPECT_EQ(aligned.value()[0].offset, 0U);
    EXPECT_EQ(aligned.value()[0].size, 100U);

    DirtyRangeSet tail;
    ASSERT_TRUE(tail.add({96, 4}));
    auto tail_aligned = tail.aligned(100, 64);
    ASSERT_TRUE(tail_aligned);
    ASSERT_EQ(tail_aligned.value().size(), 1U);
    EXPECT_EQ(tail_aligned.value()[0].offset, 64U);
    EXPECT_EQ(tail_aligned.value()[0].size, 36U);
}

TEST(DirtyRangeSet, RejectsInvalidAlignmentOutOfBoundsAndOverflow) {
    DirtyRangeSet dirty;
    EXPECT_EQ(dirty.add({UINT64_MAX, 1}).error().code, ErrorCode::Overflow);
    ASSERT_TRUE(dirty.add({10, 2}));
    EXPECT_EQ(dirty.aligned(11, 64).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(dirty.aligned(12, 0).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(dirty.aligned(12, 3).error().code, ErrorCode::InvalidArgument);
}

TEST(DirtyRangeSet, DeterministicRandomIntervalsMatchIndependentByteOracle) {
    constexpr std::size_t kCapacity = 2048;
    std::array<bool, kCapacity> touched{};
    DirtyRangeSet dirty;
    std::mt19937 generator(0x25D17U);
    for (std::size_t i = 0; i < 1500; ++i) {
        const auto offset = static_cast<std::size_t>(generator() % kCapacity);
        const auto size = std::size_t{1} + generator() % (kCapacity - offset);
        ASSERT_TRUE(dirty.add({offset, size}));
        std::fill(touched.begin() + static_cast<std::ptrdiff_t>(offset),
                  touched.begin() + static_cast<std::ptrdiff_t>(offset + size), true);
    }
    std::size_t index = 0;
    std::size_t range_index = 0;
    while (index < kCapacity) {
        if (!touched[index]) {
            ++index;
            continue;
        }
        const auto start = index;
        while (index < kCapacity && touched[index]) ++index;
        ASSERT_LT(range_index, dirty.ranges().size());
        EXPECT_EQ(dirty.ranges()[range_index].offset, start);
        EXPECT_EQ(dirty.ranges()[range_index].size, index - start);
        ++range_index;
    }
    EXPECT_EQ(range_index, dirty.ranges().size());
}

TEST(DirtyRangeSet, RandomAlignedRangesCoverEveryDirtyByteWithinAllocation) {
    constexpr std::uint64_t kCapacity = 4099;
    constexpr std::uint64_t kAtom = 64;
    std::array<bool, kCapacity> touched{};
    DirtyRangeSet dirty;
    std::mt19937 generator(0xA11E25U);
    for (std::size_t i = 0; i < 700; ++i) {
        const auto offset = static_cast<std::size_t>(generator() % kCapacity);
        const auto size = std::size_t{1} + generator() % (kCapacity - offset);
        ASSERT_TRUE(dirty.add({offset, size}));
        std::fill(touched.begin() + static_cast<std::ptrdiff_t>(offset),
                  touched.begin() + static_cast<std::ptrdiff_t>(offset + size), true);
    }

    const auto aligned = dirty.aligned(kCapacity, kAtom);
    ASSERT_TRUE(aligned);
    std::uint64_t prior_end = 0;
    for (const auto range : aligned.value()) {
        EXPECT_EQ(range.offset % kAtom, 0U);
        EXPECT_GE(range.offset, prior_end);
        EXPECT_LE(range.offset + range.size, kCapacity);
        const auto end = range.offset + range.size;
        EXPECT_TRUE(end == kCapacity || range.size % kAtom == 0);
        prior_end = end;
    }
    for (std::size_t byte = 0; byte < kCapacity; ++byte) {
        if (!touched[byte]) continue;
        EXPECT_TRUE(std::any_of(aligned.value().begin(), aligned.value().end(), [byte](DirtyRange range) {
            return byte >= range.offset && byte < range.offset + range.size;
        }));
    }
}

TEST(DirtyRangeSet, FlushesAndInvalidatesMergedAlignedRangesAndSkipsCoherentMemory) {
    FakeMappedMemory fake;
    const auto api = fake_api(fake);
    const std::array ranges{DirtyRange{3, 5}, DirtyRange{70, 10}};
    const auto device = fake_handle<VkDevice>(1);
    const auto memory = fake_handle<VkDeviceMemory>(2);

    ASSERT_TRUE(kora::detail::flush_mapped_ranges(api, device, memory, 100, 64, ranges, false));
    ASSERT_EQ(fake.flush_calls, 1U);
    ASSERT_EQ(fake.flushed.size(), 1U);
    EXPECT_EQ(fake.flushed[0].offset, 0U);
    EXPECT_EQ(fake.flushed[0].size, VK_WHOLE_SIZE);
    ASSERT_TRUE(kora::detail::invalidate_mapped_ranges(api, device, memory, 100, 64, ranges, false));
    EXPECT_EQ(fake.invalidate_calls, 1U);
    EXPECT_EQ(fake.invalidated.size(), 1U);
    EXPECT_EQ(fake.invalidated[0].size, VK_WHOLE_SIZE);

    ASSERT_TRUE(kora::detail::flush_mapped_ranges(api, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, 0, ranges, true));
    ASSERT_TRUE(
        kora::detail::invalidate_mapped_ranges(api, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, 0, ranges, true));
    EXPECT_EQ(fake.flush_calls, 1U);
    EXPECT_EQ(fake.invalidate_calls, 1U);
}

TEST(DirtyRangeSet, PropagatesVulkanFailuresAndRejectsMissingDispatch) {
    FakeMappedMemory fake;
    fake.flush_result = VK_ERROR_DEVICE_LOST;
    const auto api = fake_api(fake);
    const std::array ranges{DirtyRange{16, 8}};
    auto flushed = kora::detail::flush_mapped_ranges(api, fake_handle<VkDevice>(1),
                                                     fake_handle<VkDeviceMemory>(2), 128, 64, ranges, false);
    ASSERT_FALSE(flushed);
    EXPECT_EQ(flushed.error().code, ErrorCode::TransferFailed);
    EXPECT_EQ(flushed.error().vk_result, VK_ERROR_DEVICE_LOST);

    const kora::vk::Api empty_api;
    auto invalidated = kora::detail::invalidate_mapped_ranges(
        empty_api, fake_handle<VkDevice>(1), fake_handle<VkDeviceMemory>(2), 128, 64, ranges, false);
    ASSERT_FALSE(invalidated);
    EXPECT_EQ(invalidated.error().code, ErrorCode::UnsupportedFeature);
}

}  // namespace
