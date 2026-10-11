#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

#include "kora/verification.hpp"

namespace kora {
namespace {

TEST(Verification, ExactAndToleranceComparatorsHandleFloatingPointValues) {
    const std::array<float, 2> expected{1.0F, -2.0F};
    const std::array<float, 2> exact{1.0F, -2.0F};
    const std::array<float, 2> close{1.0001F, -2.0001F};
    VerificationPolicy policy;
    ASSERT_TRUE(policy.compare<float>(expected, exact).value());
    EXPECT_FALSE(policy.compare<float>(expected, close).value());

    policy.mode = VerificationMode::AbsoluteTolerance;
    policy.tolerance = 0.001;
    EXPECT_TRUE(policy.compare<float>(expected, close).value());
    policy.mode = VerificationMode::RelativeTolerance;
    policy.tolerance = 0.0002;
    EXPECT_TRUE(policy.compare<float>(expected, close).value());
}

TEST(Verification, ULPComparatorCountsAdjacentRepresentableValuesAndRejectsNaN) {
    const std::array<float, 1> expected{1.0F};
    const std::array<float, 1> adjacent{std::nextafter(1.0F, 2.0F)};
    const std::array<float, 1> distant{2.0F};
    VerificationPolicy policy;
    policy.mode = VerificationMode::ULP;
    policy.max_ulps = 1;
    EXPECT_TRUE(policy.compare<float>(expected, adjacent).value());
    EXPECT_FALSE(policy.compare<float>(expected, distant).value());
    const std::array<float, 1> nan{std::numeric_limits<float>::quiet_NaN()};
    EXPECT_FALSE(policy.compare<float>(nan, nan).value());
}

TEST(Verification, CustomComparatorReceivesWholeByteSpansAndValidatesInputs) {
    const std::array<float, 2> expected{1.0F, 2.0F};
    const std::array<float, 2> actual{1.0F, 3.0F};
    VerificationPolicy policy;
    policy.mode = VerificationMode::Custom;
    policy.custom = [](std::span<const std::byte> left, std::span<const std::byte> right) {
        return left.size() == right.size() && left.size() == 2 * sizeof(float) &&
               std::to_integer<unsigned char>(left[0]) == std::to_integer<unsigned char>(right[0]);
    };
    EXPECT_TRUE(policy.compare<float>(expected, actual).value());

    policy.custom = {};
    const auto missing = policy.compare<float>(expected, actual);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().code, ErrorCode::InvalidArgument);
    policy.mode = VerificationMode::AbsoluteTolerance;
    policy.tolerance = -1.0;
    const auto invalid = policy.compare<float>(expected, actual);
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.error().code, ErrorCode::InvalidArgument);
    policy.mode = VerificationMode::Custom;
    policy.custom = [](auto, auto) -> bool { throw std::runtime_error("comparator failure"); };
    const auto thrown = policy.compare<float>(expected, actual);
    ASSERT_FALSE(thrown);
    EXPECT_EQ(thrown.error().code, ErrorCode::InternalInvariantViolation);
}

TEST(Verification, SizeMismatchIsNotEqual) {
    const std::array<float, 1> one{1.0F};
    const std::array<float, 2> two{1.0F, 1.0F};
    EXPECT_FALSE(VerificationPolicy{}.compare<float>(one, two).value());
}

}  // namespace
}  // namespace kora
