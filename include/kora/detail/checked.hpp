#pragma once
/// @file checked.hpp
/// Overflow-safe size arithmetic. Every public size calculation (count*sizeof, w*h*d, offset+size,
/// alignment padding) must go through these helpers so overflow becomes an Error before any Vulkan call.
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include "kora/errors.hpp"

namespace kora::checked {

template <class T>
inline constexpr bool is_size_v = std::is_unsigned_v<T> && !std::is_same_v<T, bool>;

/// a + b, or nullopt on overflow.
template <class T>
constexpr std::optional<T> add(T a, T b) noexcept {
    static_assert(is_size_v<T>);
    if (b > std::numeric_limits<T>::max() - a) return std::nullopt;
    return static_cast<T>(a + b);
}

/// a * b, or nullopt on overflow.
template <class T>
constexpr std::optional<T> mul(T a, T b) noexcept {
    static_assert(is_size_v<T>);
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

/// Round v down to a multiple of `align` (power of two).
constexpr std::uint64_t align_down(std::uint64_t v, std::uint64_t align) noexcept {
    return v & ~(align - 1);
}

/// True iff [offset, offset+size) lies inside [0, capacity) with no overflow.
constexpr bool in_range(std::uint64_t offset, std::uint64_t size, std::uint64_t capacity) noexcept {
    auto end = add(offset, size);
    return end && *end <= capacity;
}

/// Result-returning variants for public APIs.
inline Result<std::uint64_t> mul_r(std::uint64_t a, std::uint64_t b, const char* what) {
    if (auto r = mul(a, b)) return *r;
    return Error(ErrorCode::Overflow, std::string("size overflow computing ") + what);
}
inline Result<std::uint64_t> add_r(std::uint64_t a, std::uint64_t b, const char* what) {
    if (auto r = add(a, b)) return *r;
    return Error(ErrorCode::Overflow, std::string("size overflow computing ") + what);
}

}  // namespace kora::checked
