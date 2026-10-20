#include "remote/protocol.hpp"

#include <array>
#include <cstring>

#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#endif

namespace kora::detail::remote {
namespace {
// CRC32C (Castagnoli) tables for slicing-by-8 on targets without CRC instructions.
std::array<std::array<std::uint32_t, 256>, 8> make_crc_tables() noexcept {
    std::array<std::array<std::uint32_t, 256>, 8> tables{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        auto value = i;
        for (int bit = 0; bit < 8; ++bit) value = (value & 1U) ? 0x82F63B78U ^ (value >> 1U) : value >> 1U;
        tables[0][i] = value;
    }
    for (std::size_t slice = 1; slice < 8; ++slice)
        for (std::size_t i = 0; i < 256; ++i)
            tables[slice][i] = (tables[slice - 1][i] >> 8U) ^ tables[0][tables[slice - 1][i] & 0xFFU];
    return tables;
}

std::uint32_t crc32c_software(std::uint32_t crc, const std::uint8_t* data, std::size_t size) noexcept {
    static const auto tables = make_crc_tables();
    while (size >= 8) {
        std::uint64_t word = 0;
        std::memcpy(&word, data, 8);
        word ^= crc;  // Little-endian load, as on every supported target.
        crc = tables[7][word & 0xFFU] ^ tables[6][(word >> 8U) & 0xFFU] ^ tables[5][(word >> 16U) & 0xFFU] ^
              tables[4][(word >> 24U) & 0xFFU] ^ tables[3][(word >> 32U) & 0xFFU] ^
              tables[2][(word >> 40U) & 0xFFU] ^ tables[1][(word >> 48U) & 0xFFU] ^ tables[0][word >> 56U];
        data += 8;
        size -= 8;
    }
    while (size-- != 0) crc = tables[0][(crc ^ *data++) & 0xFFU] ^ (crc >> 8U);
    return crc;
}

void put(std::byte* out, std::uint64_t value, std::size_t width) noexcept {
    for (std::size_t i = 0; i < width; ++i) out[i] = static_cast<std::byte>((value >> (8U * i)) & 0xFFU);
}

std::uint64_t get(const std::byte* in, std::size_t width) noexcept {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < width; ++i)
        value |= std::uint64_t{std::to_integer<std::uint8_t>(in[i])} << (8U * i);
    return value;
}

Error protocol_error(const std::string& what) {
    return Error(ErrorCode::IoError, "worker protocol: " + what);
}
}  // namespace

std::uint32_t crc32c(std::span<const std::byte> bytes) noexcept {
    const auto* data = reinterpret_cast<const std::uint8_t*>(bytes.data());
    std::size_t size = bytes.size();
    std::uint32_t crc = 0xFFFFFFFFU;
#if defined(__ARM_FEATURE_CRC32)
    for (; size >= 8; data += 8, size -= 8) {
        std::uint64_t word = 0;
        std::memcpy(&word, data, 8);
        crc = __crc32cd(crc, word);
    }
    for (; size != 0; ++data, --size) crc = __crc32cb(crc, *data);
#elif defined(__SSE4_2__)
    std::uint64_t wide = crc;
    for (; size >= 8; data += 8, size -= 8) {
        std::uint64_t word = 0;
        std::memcpy(&word, data, 8);
        wide = _mm_crc32_u64(wide, word);
    }
    crc = static_cast<std::uint32_t>(wide);
    for (; size != 0; ++data, --size) crc = _mm_crc32_u8(crc, *data);
#else
    crc = crc32c_software(crc, data, size);
#endif
    return crc ^ 0xFFFFFFFFU;
}

std::uint32_t crc32c_portable(std::span<const std::byte> bytes) noexcept {
    return crc32c_software(0xFFFFFFFFU, reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()) ^
           0xFFFFFFFFU;
}

Status send_message(Socket& socket, MessageType type, std::uint64_t request_id,
                    std::span<const std::byte> payload, std::chrono::milliseconds timeout) {
    std::array<std::byte, kHeaderSize> header{};
    put(header.data(), kMagic, 4);
    put(header.data() + 4, kProtocolVersion, 2);
    put(header.data() + 6, static_cast<std::uint16_t>(type), 2);
    put(header.data() + 8, request_id, 8);
    put(header.data() + 16, payload.size(), 8);
    put(header.data() + 24, crc32c(payload), 4);
    KORA_TRY_STATUS(socket.send_all(header, timeout));
    return socket.send_all(payload, timeout);
}

Result<Message> receive_message(Socket& socket, std::uint64_t max_payload,
                                std::chrono::milliseconds timeout) {
    std::array<std::byte, kHeaderSize> header{};
    KORA_TRY_STATUS(socket.receive_all(header, timeout));
    if (get(header.data(), 4) != kMagic) return protocol_error("bad frame magic");
    if (get(header.data() + 4, 2) != kProtocolVersion) return protocol_error("unsupported protocol version");
    Message message;
    message.type = static_cast<MessageType>(get(header.data() + 6, 2));
    message.request_id = get(header.data() + 8, 8);
    const auto size = get(header.data() + 16, 8);
    if (size > max_payload) return protocol_error("frame exceeds the configured size limit");
    message.payload.resize(static_cast<std::size_t>(size));
    KORA_TRY_STATUS(socket.receive_all(message.payload, timeout));
    if (crc32c(message.payload) != static_cast<std::uint32_t>(get(header.data() + 24, 4)))
        return protocol_error("frame checksum mismatch");
    return message;
}

void Writer::u32(std::uint32_t value) {
    const auto offset = bytes_.size();
    bytes_.resize(offset + 4);
    put(bytes_.data() + offset, value, 4);
}

void Writer::u64(std::uint64_t value) {
    const auto offset = bytes_.size();
    bytes_.resize(offset + 8);
    put(bytes_.data() + offset, value, 8);
}

void Writer::bytes(std::span<const std::byte> value) {
    u64(value.size());
    bytes_.insert(bytes_.end(), value.begin(), value.end());
}

void Writer::string(std::string_view value) {
    bytes(std::as_bytes(std::span(value.data(), value.size())));
}

Result<std::span<const std::byte>> Reader::take(std::size_t count) {
    if (count > bytes_.size() - offset_) return protocol_error("truncated payload");
    auto view = bytes_.subspan(offset_, count);
    offset_ += count;
    return view;
}

Result<std::uint8_t> Reader::u8() {
    auto view = take(1);
    if (!view) return std::move(view).error();
    return std::to_integer<std::uint8_t>(view.value()[0]);
}

Result<std::uint32_t> Reader::u32() {
    auto view = take(4);
    if (!view) return std::move(view).error();
    return static_cast<std::uint32_t>(get(view.value().data(), 4));
}

Result<std::uint64_t> Reader::u64() {
    auto view = take(8);
    if (!view) return std::move(view).error();
    return get(view.value().data(), 8);
}

Result<std::span<const std::byte>> Reader::bytes() {
    auto size = u64();
    if (!size) return std::move(size).error();
    if (size.value() > bytes_.size() - offset_) return protocol_error("truncated payload");
    return take(static_cast<std::size_t>(size.value()));
}

Result<std::string> Reader::string() {
    auto view = bytes();
    if (!view) return std::move(view).error();
    return std::string(reinterpret_cast<const char*>(view.value().data()), view.value().size());
}

}  // namespace kora::detail::remote
