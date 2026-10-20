#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "kora/errors.hpp"

namespace kora::detail::remote {

struct Endpoint {
    std::string host;
    std::uint16_t port = 0;
};

/// Parses "host:port" or "[ipv6]:port".
Result<Endpoint> parse_endpoint(const std::string& text);

/// Move-only TCP socket. Every blocking call takes a deadline so a vanished peer never hangs a thread.
class Socket final {
public:
    Socket() = default;
    ~Socket();
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& other) noexcept;
    Socket& operator=(Socket&& other) noexcept;

    static Result<Socket> connect(const Endpoint& endpoint, std::chrono::milliseconds timeout);
    static Result<Socket> listen(const Endpoint& endpoint);
    /// Waits up to `timeout` for a connection; returns an invalid socket when none arrived.
    Result<Socket> accept(std::chrono::milliseconds timeout);

    Status send_all(std::span<const std::byte> bytes, std::chrono::milliseconds timeout);
    Status receive_all(std::span<std::byte> bytes, std::chrono::milliseconds timeout);
    /// True when data (or an orderly close) is ready to read within `timeout`.
    Result<bool> readable(std::chrono::milliseconds timeout);
    /// Closes the socket. Only the thread that owns and uses the socket may call this.
    void close() noexcept;
    /// Wakes a thread blocked on this socket (its I/O then fails) without closing it, so the owning
    /// thread can still close it safely. Safe to call from any thread.
    void interrupt() noexcept;

    bool valid() const noexcept { return handle_.load(std::memory_order_acquire) != kInvalid; }
    bool peer_is_loopback() const noexcept { return peer_loopback_; }
    std::uint16_t local_port() const noexcept;

private:
#ifdef _WIN32
    using Handle = std::uintptr_t;
    static constexpr Handle kInvalid = ~Handle{0};
#else
    using Handle = int;
    static constexpr Handle kInvalid = -1;
#endif
    explicit Socket(Handle handle, bool peer_loopback = false) noexcept
        : handle_(handle), peer_loopback_(peer_loopback) {}
    std::atomic<Handle> handle_{kInvalid};
    bool peer_loopback_ = false;
};

}  // namespace kora::detail::remote
