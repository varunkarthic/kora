#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "kora/errors.hpp"
#include "kora/options.hpp"

namespace kora {

/// Configuration for a process that executes KORA dispatches on behalf of a remote `Cluster`.
struct WorkerOptions {
    /// Address to listen on, as "host:port" or "[ipv6]:port". Port 0 picks a free port.
    std::string listen = "127.0.0.1:7400";
    /// Shared secret for the HMAC-SHA256 handshake. Empty uses KORA_WORKER_TOKEN; when that is also
    /// empty, only loopback clients are accepted.
    std::string token;
    /// Options for the worker's own Runtime (device selection, telemetry, ...).
    RuntimeOptions runtime;
    /// Name reported to clients; empty reports the selected device name.
    std::string name;
    /// Upper bound on buffer bytes cached for a client; least-recently-used entries are evicted.
    std::uint64_t max_cache_bytes = 1ULL << 30U;
    /// Largest accepted protocol frame.
    std::uint64_t max_message_bytes = 1ULL << 30U;
    /// Deadline for any single send or receive once a request has started.
    std::chrono::milliseconds io_timeout{30'000};
};

struct WorkerStats {
    std::uint64_t sessions = 0;
    std::uint64_t rejected_connections = 0;
    std::uint64_t dispatches = 0;
    std::uint64_t kernels_loaded = 0;
    std::uint64_t cached_bytes = 0;
    bool session_active = false;
};

/// Serves one authenticated client at a time; a newly authenticated client replaces an idle or
/// vanished one, so a host that reconnects after a link drop is never locked out.
class WorkerServer final {
public:
    static Result<std::unique_ptr<WorkerServer>> start(WorkerOptions options);
    ~WorkerServer();
    WorkerServer(const WorkerServer&) = delete;
    WorkerServer& operator=(const WorkerServer&) = delete;

    /// Port actually bound (useful with port 0).
    std::uint16_t port() const noexcept;
    /// Closes the listener and any active session; outstanding client work fails over to the client.
    void stop() noexcept;
    WorkerStats stats() const;

    struct Impl;

private:
    explicit WorkerServer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace kora
