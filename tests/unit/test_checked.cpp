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
