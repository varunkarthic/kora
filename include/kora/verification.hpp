#pragma once
/// @file verification.hpp
/// Comparator policies for checking duplicate floating-point results.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <type_traits>

#include "kora/errors.hpp"

namespace kora {

enum class VerificationMode : std::uint8_t { Exact, AbsoluteTolerance, RelativeTolerance, ULP, Custom };

/// Comparison settings for duplicate numeric output. Custom callbacks receive the complete byte spans.
struct VerificationPolicy {
    VerificationMode mode = VerificationMode::Exact;
    double tolerance = 0.0;
    std::uint64_t max_ulps = 0;
    std::function<bool(std::span<const std::byte>, std::span<const std::byte>)> custom;

    template <class T>
    Result<bool> compare(std::span<const T> expected, std::span<const T> actual) const {
        static_assert(std::is_floating_point_v<T>, "verification supports floating-point values");
        if (expected.size() != actual.size()) return false;
        if ((mode == VerificationMode::AbsoluteTolerance || mode == VerificationMode::RelativeTolerance) &&
            (!std::isfinite(tolerance) || tolerance < 0.0))
            return Error(ErrorCode::InvalidArgument, "verification tolerance must be finite and nonnegative");
        if (mode == VerificationMode::Custom) {
            if (!custom) return Error(ErrorCode::InvalidArgument, "custom verification comparator is empty");
            try {
                return custom(std::as_bytes(expected), std::as_bytes(actual));
            } catch (...) {
                return Error(ErrorCode::InternalInvariantViolation, "custom verification comparator threw");
            }
        }
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const auto left = expected[index];
            const auto right = actual[index];
            if (left == right) continue;
            if (mode == VerificationMode::Exact) {
                return false;
            } else if (mode == VerificationMode::AbsoluteTolerance) {
                if (!(std::abs(static_cast<double>(left) - static_cast<double>(right)) <= tolerance))
                    return false;
            } else if (mode == VerificationMode::RelativeTolerance) {
                const auto difference = std::abs(static_cast<double>(left) - static_cast<double>(right));
                const auto scale =
                    std::max(std::abs(static_cast<double>(left)), std::abs(static_cast<double>(right)));
                if (!(difference <= tolerance * scale)) return false;
            } else if (mode == VerificationMode::ULP) {
                if (!within_ulps(left, right, max_ulps)) return false;
            } else {
                return Error(ErrorCode::InvalidArgument, "unknown verification comparator mode");
            }
        }
        return true;
    }

private:
    template <class T>
    static bool within_ulps(T left, T right, std::uint64_t limit) {
        static_assert(sizeof(T) == sizeof(std::uint32_t) || sizeof(T) == sizeof(std::uint64_t));
        using UInt = std::conditional_t<sizeof(T) == sizeof(std::uint32_t), std::uint32_t, std::uint64_t>;
        if (std::isnan(left) || std::isnan(right)) return false;
        if (left == right) return true;
        const auto ordered = [](T value) {
            const UInt bits = std::bit_cast<UInt>(value);
            constexpr UInt sign = UInt{1} << (sizeof(UInt) * 8U - 1U);
            return (bits & sign) != 0 ? static_cast<UInt>(~bits + UInt{1}) : static_cast<UInt>(bits | sign);
        };
        const auto a = ordered(left);
        const auto b = ordered(right);
        const auto distance = a > b ? a - b : b - a;
        return distance <= limit;
    }
};

}  // namespace kora
