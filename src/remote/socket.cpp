#include "remote/socket.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <utility>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace kora::detail::remote {
namespace {
#ifdef _WIN32
using NativeHandle = SOCKET;
int last_error() {
    return WSAGetLastError();
}
bool would_block(int error) {
    return error == WSAEWOULDBLOCK || error == WSAEINPROGRESS;
}
int poll_handles(pollfd* fds, unsigned long count, int timeout) {
    return WSAPoll(fds, count, timeout);
}
void close_native(NativeHandle handle) {
    closesocket(handle);
}
Status ensure_winsock() {
    static std::once_flag once;
    static int result = 0;
    std::call_once(once, [] {
        WSADATA data{};
        result = WSAStartup(MAKEWORD(2, 2), &data);
    });
    if (result != 0) return Error(ErrorCode::IoError, "WSAStartup failed");
    return ok_status();
}
#else
using NativeHandle = int;
int last_error() {
    return errno;
}
bool would_block(int error) {
    return error == EINPROGRESS || error == EAGAIN || error == EWOULDBLOCK;
}
int poll_handles(pollfd* fds, nfds_t count, int timeout) {
    return ::poll(fds, count, timeout);
}
void close_native(NativeHandle handle) {
    ::close(handle);
}
Status ensure_winsock() {
    return ok_status();
}
#endif

Error io_error(const std::string& what) {
    return Error(ErrorCode::IoError, what + " (socket error " + std::to_string(last_error()) + ")");
}

void configure(NativeHandle handle) {
    int one = 1;
    setsockopt(handle, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
#ifdef SO_NOSIGPIPE
    setsockopt(handle, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
#ifdef _WIN32
    u_long nonblocking = 1;
    ioctlsocket(handle, FIONBIO, &nonblocking);
#else
    fcntl(handle, F_SETFL, fcntl(handle, F_GETFL, 0) | O_NONBLOCK);
#endif
}

bool is_loopback(const sockaddr_storage& address) {
    if (address.ss_family == AF_INET) {
        const auto* ipv4 = reinterpret_cast<const sockaddr_in*>(&address);
        return (ntohl(ipv4->sin_addr.s_addr) >> 24U) == 127U;
    }
    if (address.ss_family == AF_INET6) {
        const auto* ipv6 = reinterpret_cast<const sockaddr_in6*>(&address);
        static const in6_addr loopback = IN6ADDR_LOOPBACK_INIT;
        return std::memcmp(&ipv6->sin6_addr, &loopback, sizeof(loopback)) == 0;
    }
    return false;
}

int milliseconds(std::chrono::milliseconds timeout) {
    return static_cast<int>(std::clamp<std::int64_t>(timeout.count(), 0, 1'000'000));
}

// Waits for `events` until `deadline`, retrying interrupted waits.
Result<bool> wait_for(NativeHandle handle, short events, std::chrono::steady_clock::time_point deadline) {
    for (;;) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        pollfd fd{};
        fd.fd = handle;
        fd.events = events;
        const int ready =
            poll_handles(&fd, 1, milliseconds(std::max(remaining, std::chrono::milliseconds(0))));
        if (ready > 0) return true;
        if (ready == 0) return false;
#ifndef _WIN32
        if (errno == EINTR) continue;
#endif
        return io_error("poll failed");
    }
}

#ifndef MSG_NOSIGNAL
constexpr int kSendFlags = 0;
#else
constexpr int kSendFlags = MSG_NOSIGNAL;
#endif
}  // namespace

Result<Endpoint> parse_endpoint(const std::string& text) {
    std::string host;
    std::string port;
    if (!text.empty() && text.front() == '[') {
        const auto close = text.find(']');
        if (close == std::string::npos || close + 1 >= text.size() || text[close + 1] != ':')
            return Error(ErrorCode::InvalidArgument, "endpoint must be [ipv6]:port");
        host = text.substr(1, close - 1);
        port = text.substr(close + 2);
    } else {
        const auto colon = text.rfind(':');
        if (colon == std::string::npos || colon == 0)
            return Error(ErrorCode::InvalidArgument, "endpoint must be host:port: " + text);
        host = text.substr(0, colon);
        port = text.substr(colon + 1);
    }
    if (port.empty() || port.size() > 5 || port.find_first_not_of("0123456789") != std::string::npos)
        return Error(ErrorCode::InvalidArgument, "endpoint port is not a number: " + text);
    const auto value = std::stoul(port);
    if (value > 65535) return Error(ErrorCode::InvalidArgument, "endpoint port exceeds 65535: " + text);
    return Endpoint{host, static_cast<std::uint16_t>(value)};
}

Socket::~Socket() {
    close();
}
Socket::Socket(Socket&& other) noexcept
    : handle_(other.handle_.exchange(kInvalid)), peer_loopback_(other.peer_loopback_) {}
Socket& Socket::operator=(Socket&& other) noexcept {
    if (this != &other) {
        close();
        handle_.store(other.handle_.exchange(kInvalid));
        peer_loopback_ = other.peer_loopback_;
    }
    return *this;
}

void Socket::interrupt() noexcept {
    const auto handle = handle_.load(std::memory_order_acquire);
    if (handle == kInvalid) return;
#ifdef _WIN32
    ::shutdown(static_cast<NativeHandle>(handle), SD_BOTH);
#else
    ::shutdown(static_cast<NativeHandle>(handle), SHUT_RDWR);
#endif
}

void Socket::close() noexcept {
    const auto stored = handle_.exchange(kInvalid);
    if (stored == kInvalid) return;
    const auto handle = static_cast<NativeHandle>(stored);
#ifdef _WIN32
    ::shutdown(handle, SD_BOTH);
#else
    ::shutdown(handle, SHUT_RDWR);
#endif
    close_native(handle);
}

Result<Socket> Socket::connect(const Endpoint& endpoint, std::chrono::milliseconds timeout) {
    KORA_TRY_STATUS(ensure_winsock());
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* results = nullptr;
    const auto port = std::to_string(endpoint.port);
    if (getaddrinfo(endpoint.host.c_str(), port.c_str(), &hints, &results) != 0 || results == nullptr)
        return Error(ErrorCode::IoError, "cannot resolve worker host " + endpoint.host);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    Error failure(ErrorCode::IoError, "cannot connect to " + endpoint.host + ":" + port);
    for (auto* candidate = results; candidate != nullptr; candidate = candidate->ai_next) {
        const auto handle = ::socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol);
        if (handle == static_cast<NativeHandle>(kInvalid)) continue;
        configure(handle);
        Socket socket(static_cast<Handle>(handle),
                      is_loopback(*reinterpret_cast<sockaddr_storage*>(candidate->ai_addr)));
        if (::connect(handle, candidate->ai_addr, static_cast<socklen_t>(candidate->ai_addrlen)) != 0 &&
            !would_block(last_error()))
            continue;
        auto ready = wait_for(handle, POLLOUT, deadline);
        if (!ready || !ready.value()) continue;
        int error = 0;
        socklen_t length = sizeof(error);
        getsockopt(handle, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&error), &length);
        if (error != 0) continue;
        freeaddrinfo(results);
        return socket;
    }
    freeaddrinfo(results);
    return failure;
}

Result<Socket> Socket::listen(const Endpoint& endpoint) {
    KORA_TRY_STATUS(ensure_winsock());
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* results = nullptr;
    const auto port = std::to_string(endpoint.port);
    const char* host = endpoint.host.empty() ? nullptr : endpoint.host.c_str();
    if (getaddrinfo(host, port.c_str(), &hints, &results) != 0 || results == nullptr)
        return Error(ErrorCode::IoError, "cannot resolve listen address " + endpoint.host);
    for (auto* candidate = results; candidate != nullptr; candidate = candidate->ai_next) {
        const auto handle = ::socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol);
        if (handle == static_cast<NativeHandle>(kInvalid)) continue;
        int one = 1;
        setsockopt(handle, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
        Socket socket(static_cast<Handle>(handle));
        if (::bind(handle, candidate->ai_addr, static_cast<socklen_t>(candidate->ai_addrlen)) != 0) continue;
        if (::listen(handle, 8) != 0) continue;
        configure(handle);
        freeaddrinfo(results);
        return socket;
    }
    freeaddrinfo(results);
    return io_error("cannot listen on " + endpoint.host + ":" + port);
}

Result<Socket> Socket::accept(std::chrono::milliseconds timeout) {
    if (!valid()) return Error(ErrorCode::InvalidArgument, "listening socket is closed");
    auto ready = wait_for(static_cast<NativeHandle>(handle_.load()), POLLIN,
                          std::chrono::steady_clock::now() + timeout);
    if (!ready) return std::move(ready).error();
    if (!ready.value()) return Socket{};
    sockaddr_storage address{};
    socklen_t length = sizeof(address);
    const auto handle =
        ::accept(static_cast<NativeHandle>(handle_.load()), reinterpret_cast<sockaddr*>(&address), &length);
    if (handle == static_cast<NativeHandle>(kInvalid)) {
        if (would_block(last_error())) return Socket{};
        return io_error("accept failed");
    }
    configure(handle);
    return Socket(static_cast<Handle>(handle), is_loopback(address));
}

Status Socket::send_all(std::span<const std::byte> bytes, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::size_t sent = 0;
    while (sent < bytes.size()) {
        if (!valid()) return Error(ErrorCode::IoError, "connection closed");
        const auto chunk = std::min<std::size_t>(bytes.size() - sent, 1U << 30U);
        const auto result =
            ::send(static_cast<NativeHandle>(handle_.load()),
                   reinterpret_cast<const char*>(bytes.data() + sent), static_cast<int>(chunk), kSendFlags);
        if (result > 0) {
            sent += static_cast<std::size_t>(result);
            continue;
        }
        if (result < 0 && would_block(last_error())) {
            auto ready = wait_for(static_cast<NativeHandle>(handle_.load()), POLLOUT, deadline);
            if (!ready) return std::move(ready).error();
            if (!ready.value()) return Error(ErrorCode::Timeout, "send timed out");
            continue;
        }
#ifndef _WIN32
        if (result < 0 && errno == EINTR) continue;
#endif
        return io_error("send failed");
    }
    return ok_status();
}

Status Socket::receive_all(std::span<std::byte> bytes, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::size_t received = 0;
    while (received < bytes.size()) {
        if (!valid()) return Error(ErrorCode::IoError, "connection closed");
        const auto chunk = std::min<std::size_t>(bytes.size() - received, 1U << 30U);
        const auto result =
            ::recv(static_cast<NativeHandle>(handle_.load()),
                   reinterpret_cast<char*>(bytes.data() + received), static_cast<int>(chunk), 0);
        if (result > 0) {
            received += static_cast<std::size_t>(result);
            continue;
        }
        if (result == 0) return Error(ErrorCode::IoError, "connection closed by peer");
        if (would_block(last_error())) {
            auto ready = wait_for(static_cast<NativeHandle>(handle_.load()), POLLIN, deadline);
            if (!ready) return std::move(ready).error();
            if (!ready.value()) return Error(ErrorCode::Timeout, "receive timed out");
            continue;
        }
#ifndef _WIN32
        if (errno == EINTR) continue;
#endif
        return io_error("receive failed");
    }
    return ok_status();
}

Result<bool> Socket::readable(std::chrono::milliseconds timeout) {
    if (!valid()) return Error(ErrorCode::IoError, "connection closed");
    return wait_for(static_cast<NativeHandle>(handle_.load()), POLLIN,
                    std::chrono::steady_clock::now() + timeout);
}

std::uint16_t Socket::local_port() const noexcept {
    if (!valid()) return 0;
    sockaddr_storage address{};
    socklen_t length = sizeof(address);
    if (getsockname(static_cast<NativeHandle>(handle_.load()), reinterpret_cast<sockaddr*>(&address),
                    &length) != 0)
        return 0;
    if (address.ss_family == AF_INET) return ntohs(reinterpret_cast<sockaddr_in*>(&address)->sin_port);
    if (address.ss_family == AF_INET6) return ntohs(reinterpret_cast<sockaddr_in6*>(&address)->sin6_port);
    return 0;
}

}  // namespace kora::detail::remote
