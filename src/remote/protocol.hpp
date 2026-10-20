#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "kora/errors.hpp"
#include "remote/socket.hpp"

namespace kora::detail::remote {

// Wire format: 32-byte little-endian header followed by `payload_size` bytes.
//   u32 magic 'KORA' | u16 version | u16 type | u64 request_id | u64 payload_size | u32 crc32 | u32 reserved
// A bad magic, version, size, or CRC is a link failure: the connection is dropped and the caller
// re-executes outstanding work locally. Payload fields use the Writer/Reader encodings below.
inline constexpr std::uint32_t kMagic = 0x41524F4BU;
inline constexpr std::uint16_t kProtocolVersion = 1;
inline constexpr std::size_t kHeaderSize = 32;
inline constexpr std::string_view kHandshakeContext = "kora-worker/1";

enum class MessageType : std::uint16_t {
    Challenge = 1,      // worker -> client: bytes nonce
    Auth = 2,           // client -> worker: bytes hmac
    Welcome = 3,        // worker -> client: u64 session, string name, string device, u64 max_message
    Reject = 4,         // either direction: string reason
    LoadKernel = 10,    // client -> worker: u64 key, string metadata_json, string entry, bytes words
    KernelLoaded = 11,  // worker -> client
    Failure = 12,       // worker -> client: string message
    Dispatch = 20,      // client -> worker: see cluster.cpp encode_dispatch()
    Outputs = 21,       // worker -> client: u32 count, (u32 binding, bytes data)*
    NeedData = 22,      // worker -> client: u32 count, u32 binding*
    Release = 30,       // client -> worker: u32 count, u64 buffer_key*
    Ping = 40,
    Pong = 41,
};

struct Message {
    MessageType type = MessageType::Ping;
    std::uint64_t request_id = 0;
    std::vector<std::byte> payload;
};

std::uint32_t crc32(std::span<const std::byte> bytes) noexcept;

Status send_message(Socket& socket, MessageType type, std::uint64_t request_id,
                    std::span<const std::byte> payload, std::chrono::milliseconds timeout);
Result<Message> receive_message(Socket& socket, std::uint64_t max_payload, std::chrono::milliseconds timeout);

class Writer final {
public:
    void u8(std::uint8_t value) { bytes_.push_back(static_cast<std::byte>(value)); }
    void u32(std::uint32_t value);
    void u64(std::uint64_t value);
    void bytes(std::span<const std::byte> value);
    void string(std::string_view value);
    std::span<const std::byte> view() const noexcept { return bytes_; }
    std::vector<std::byte>& buffer() noexcept { return bytes_; }

private:
    std::vector<std::byte> bytes_;
};

/// Bounds-checked payload reader; every accessor fails instead of reading past the end.
class Reader final {
public:
    explicit Reader(std::span<const std::byte> bytes) noexcept : bytes_(bytes) {}
    Result<std::uint8_t> u8();
    Result<std::uint32_t> u32();
    Result<std::uint64_t> u64();
    Result<std::span<const std::byte>> bytes();
    Result<std::string> string();
    bool done() const noexcept { return offset_ == bytes_.size(); }

private:
    Result<std::span<const std::byte>> take(std::size_t count);
    std::span<const std::byte> bytes_;
    std::size_t offset_ = 0;
};

}  // namespace kora::detail::remote
