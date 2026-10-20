#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace kora::detail::remote {

using Digest = std::array<std::uint8_t, 32>;

Digest sha256(std::span<const std::uint8_t> data);
/// HMAC-SHA256 (RFC 2104) used for the worker handshake.
Digest hmac_sha256(std::span<const std::uint8_t> key, std::span<const std::uint8_t> message);
/// Compares digests without data-dependent early exit.
bool constant_time_equal(const Digest& left, const Digest& right) noexcept;

}  // namespace kora::detail::remote
