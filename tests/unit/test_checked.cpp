#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "kora/detail/checked.hpp"

using namespace kora::checked;
constexpr auto kMax = std::numeric_limits<std::uint64_t>::max();

TEST(Checked, AddBoundaries) {
    EXPECT_EQ(add<std::uint64_t>(1, 2), 3u);
    EXPECT_EQ(add<std::uint64_t>(kMax, 0), kMax);
    EXPECT_FALSE(add<std::uint64_t>(kMax, 1));
    EXPECT_FALSE(add<std::uint64_t>(kMax / 2 + 1, kMax / 2 + 1));
}

TEST(Checked, MulBoundaries) {
    EXPECT_EQ(mul<std::uint64_t>(0, kMax), 0u);
    EXPECT_EQ(mul<std::uint64_t>(kMax, 1), kMax);
    EXPECT_FALSE(mul<std::uint64_t>(kMax, 2));
    EXPECT_FALSE(mul<std::uint64_t>(1ull << 32, 1ull << 32));
    EXPECT_EQ(mul<std::uint64_t>(1ull << 31, 1ull << 32), 1ull << 63);
    EXPECT_FALSE(mul<std::uint32_t>(65536u, 65536u));
}

TEST(Checked, AlignUp) {
    EXPECT_EQ(align_up(0, 16), 0u);
    EXPECT_EQ(align_up(1, 16), 16u);
    EXPECT_EQ(align_up(16, 16), 16u);
    EXPECT_EQ(align_up(17, 16), 32u);
    EXPECT_FALSE(align_up(1, 0));
    EXPECT_FALSE(align_up(1, 12));     // not a power of two
    EXPECT_FALSE(align_up(kMax, 16));  // overflow
    EXPECT_EQ(align_up(kMax - 15, 16), kMax - 15);
}

TEST(Checked, AlignDownAndPow2) {
    EXPECT_EQ(align_down(31, 16), 16u);
    EXPECT_FALSE(align_down(31, 0));
    EXPECT_FALSE(align_down(31, 6));
    EXPECT_TRUE(is_pow2(1));
    EXPECT_FALSE(is_pow2(0));
    EXPECT_FALSE(is_pow2(6));
}

TEST(Checked, InRange) {
    EXPECT_TRUE(in_range(0, 10, 10));
    EXPECT_FALSE(in_range(1, 10, 10));
    EXPECT_FALSE(in_range(kMax, 2, kMax));  // offset+size overflows
    EXPECT_TRUE(in_range(10, 0, 10));
}

TEST(Checked, ResultVariants) {
    EXPECT_EQ(mul_r(3, 4, "x").value(), 12u);
    EXPECT_EQ(mul_r(kMax, 2, "count*size").error().code, kora::ErrorCode::Overflow);
    EXPECT_EQ(add_r(kMax, 1, "off+size").error().code, kora::ErrorCode::Overflow);
}

TEST(Checked, ByteSize) {
    EXPECT_EQ(byte_size(4, 8, "elements").value(), 32u);
    EXPECT_EQ(byte_size<std::uint32_t>(4, "uint32 elements").value(), 16u);
    EXPECT_EQ(byte_size(0, kMax, "empty").value(), 0u);
    EXPECT_EQ(byte_size(kMax, 2, "elements").error().code, kora::ErrorCode::Overflow);
}

TEST(Checked, Volume) {
    EXPECT_EQ(volume(2, 3, 4, "extent").value(), 24u);
    EXPECT_EQ(volume(0, kMax, kMax, "empty extent").value(), 0u);
    EXPECT_EQ(volume(kMax, 2, 1, "width*height").error().code, kora::ErrorCode::Overflow);
    EXPECT_EQ(volume(kMax / 2, 2, 2, "area*depth").error().code, kora::ErrorCode::Overflow);
}

TEST(Checked, RangeEnd) {
    EXPECT_EQ(range_end(4, 6, 10, "upload").value(), 10u);
    EXPECT_EQ(range_end(10, 0, 10, "empty tail").value(), 10u);
    EXPECT_EQ(range_end(kMax, 1, kMax, "overflow").error().code, kora::ErrorCode::Overflow);
    EXPECT_EQ(range_end(4, 7, 10, "outside").error().code, kora::ErrorCode::InvalidArgument);
}

TEST(Checked, AlignUpResult) {
    EXPECT_EQ(align_up_r(17, 16, "offset").value(), 32u);
    EXPECT_EQ(align_up_r(17, 0, "offset").error().code, kora::ErrorCode::InvalidArgument);
    EXPECT_EQ(align_up_r(17, 12, "offset").error().code, kora::ErrorCode::InvalidArgument);
    EXPECT_EQ(align_up_r(kMax, 16, "offset").error().code, kora::ErrorCode::Overflow);
}

static_assert(add<std::uint32_t>(1, 2).value() == 3);
static_assert(!add<std::uint32_t>(std::numeric_limits<std::uint32_t>::max(), 1));
static_assert(mul<std::uint32_t>(3, 4).value() == 12);
static_assert(!mul<std::uint32_t>(65536, 65536));
