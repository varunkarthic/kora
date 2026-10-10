#pragma once
/// @file checked.hpp
/// Overflow-safe size arithmetic. Every public size calculation (count*sizeof, w*h*d, offset+size,
/// alignment padding) must go through these helpers so overflow becomes an Error before any Vulkan call.
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>

#include "kora/errors.hpp"

namespace kora::checked {

template <class T>
inline constexpr bool is_size_v = std::is_unsigned_v<T> && !std::is_same_v<T, bool>;

/// a + b, or nullopt on overflow.
template <class T>
constexpr std::optional<T> add(T a, T b) noexcept {
    static_assert(is_size_v<T>);
#if defined(__GNUC__) || defined(__clang__)
    if (!std::is_constant_evaluated()) {
        T result{};
        if (__builtin_add_overflow(a, b, &result)) return std::nullopt;
        return result;
    }
#endif
    if (b > std::numeric_limits<T>::max() - a) return std::nullopt;
    return static_cast<T>(a + b);
}

/// a * b, or nullopt on overflow.
template <class T>
constexpr std::optional<T> mul(T a, T b) noexcept {
    static_assert(is_size_v<T>);
#if defined(__GNUC__) || defined(__clang__)
    if (!std::is_constant_evaluated()) {
        T result{};
        if (__builtin_mul_overflow(a, b, &result)) return std::nullopt;
        return result;
    }
#endif
    if (a != 0 && b > std::numeric_limits<T>::max() / a) return std::nullopt;
    return static_cast<T>(a * b);
}

constexpr bool is_pow2(std::uint64_t x) noexcept {
    return x != 0 && (x & (x - 1)) == 0;
}

/// Round v up to a multiple of `align` (power of two). nullopt on overflow or bad alignment.
constexpr std::optional<std::uint64_t> align_up(std::uint64_t v, std::uint64_t align) noexcept {
    if (!is_pow2(align)) return std::nullopt;
    auto s = add<std::uint64_t>(v, align - 1);
    if (!s) return std::nullopt;
    return *s & ~(align - 1);
}

/// Round v down to a multiple of `align`. nullopt for zero or non-power-of-two alignment.
constexpr std::optional<std::uint64_t> align_down(std::uint64_t v, std::uint64_t align) noexcept {
    if (!is_pow2(align)) return std::nullopt;
    return v & ~(align - 1);
}

/// True iff [offset, offset+size) lies inside [0, capacity) with no overflow.
constexpr bool in_range(std::uint64_t offset, std::uint64_t size, std::uint64_t capacity) noexcept {
    auto end = add(offset, size);
    return end && *end <= capacity;
}

/// Result-returning variants for public APIs.
inline Result<std::uint64_t> mul_r(std::uint64_t a, std::uint64_t b, std::string_view what) {
    if (auto r = mul(a, b)) return *r;
    return Error(ErrorCode::Overflow, std::string("size overflow computing ") + std::string(what));
}
inline Result<std::uint64_t> add_r(std::uint64_t a, std::uint64_t b, std::string_view what) {
    if (auto r = add(a, b)) return *r;
    return Error(ErrorCode::Overflow, std::string("size overflow computing ") + std::string(what));
}

/// count * element_size, with a contextual Overflow error.
inline Result<std::uint64_t> byte_size(std::uint64_t count, std::uint64_t element_size,
                                       std::string_view what) {
    return mul_r(count, element_size, what);
}

/// count * sizeof(T), with a contextual Overflow error.
template <class T>
Result<std::uint64_t> byte_size(std::uint64_t count, std::string_view what) {
    return byte_size(count, sizeof(T), what);
}

/// width * height * depth, with overflow checked after each multiplication.
inline Result<std::uint64_t> volume(std::uint64_t width, std::uint64_t height, std::uint64_t depth,
                                    std::string_view what) {
    auto area = mul(width, height);
    if (!area) {
        return Error(ErrorCode::Overflow, std::string("size overflow computing ") + std::string(what));
    }
    return mul_r(*area, depth, what);
}

/// Return offset + size when the half-open range fits in capacity.
inline Result<std::uint64_t> range_end(std::uint64_t offset, std::uint64_t size, std::uint64_t capacity,
                                       std::string_view what) {
    auto end = add(offset, size);
    if (!end) {
        return Error(ErrorCode::Overflow, std::string("size overflow computing ") + std::string(what));
    }
    if (*end > capacity) {
        return Error(ErrorCode::InvalidArgument,
                     std::string("range exceeds capacity computing ") + std::string(what));
    }
    return *end;
}

/// Result-returning alignment for API boundaries.
inline Result<std::uint64_t> align_up_r(std::uint64_t value, std::uint64_t alignment, std::string_view what) {
    if (!is_pow2(alignment)) {
        return Error(ErrorCode::InvalidArgument,
                     std::string("alignment must be a non-zero power of two computing ") + std::string(what));
    }
    if (auto result = align_up(value, alignment)) return *result;
    return Error(ErrorCode::Overflow, std::string("size overflow computing ") + std::string(what));
}

}  // namespace kora::checked
