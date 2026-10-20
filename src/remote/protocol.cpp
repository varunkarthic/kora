#include "remote/protocol.hpp"

#include <array>
#include <cstring>

namespace kora::detail::remote {
namespace {
std::array<std::uint32_t, 256> make_crc_table() noexcept {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        auto value = i;
        for (int bit = 0; bit < 8; ++bit) value = (value & 1U) ? 0xEDB88320U ^ (value >> 1U) : value >> 1U;
        table[i] = value;
    }
    return table;
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

std::uint32_t crc32(std::span<const std::byte> bytes) noexcept {
    static const auto table = make_crc_table();
    std::uint32_t crc = 0xFFFFFFFFU;
    for (const auto byte : bytes)
        crc = table[(crc ^ std::to_integer<std::uint8_t>(byte)) & 0xFFU] ^ (crc >> 8U);
    return crc ^ 0xFFFFFFFFU;
}

Status send_message(Socket& socket, MessageType type, std::uint64_t request_id,
                    std::span<const std::byte> payload, std::chrono::milliseconds timeout) {
    std::array<std::byte, kHeaderSize> header{};
    put(header.data(), kMagic, 4);
    put(header.data() + 4, kProtocolVersion, 2);
    put(header.data() + 6, static_cast<std::uint16_t>(type), 2);
    put(header.data() + 8, request_id, 8);
    put(header.data() + 16, payload.size(), 8);
    put(header.data() + 24, crc32(payload), 4);
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
    if (crc32(message.payload) != static_cast<std::uint32_t>(get(header.data() + 24, 4)))
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
