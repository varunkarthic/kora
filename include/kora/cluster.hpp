#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "kora/buffer.hpp"
#include "kora/errors.hpp"
#include "kora/event.hpp"
#include "kora/kernel_metadata.hpp"
#include "kora/options.hpp"
#include "kora/runtime.hpp"

namespace kora {

/// Where a `Cluster` may run each dispatch.
enum class OffloadPolicy {
    /// Pick the target with the lowest measured latency for this kernel and data size, exploring
    /// unmeasured connected workers occasionally.
    Auto,
    /// Never offload; workers stay connected but idle.
    LocalOnly,
    /// Offload to a connected worker whenever one can run the kernel.
    PreferRemote,
};

struct ClusterOptions {
    /// Options for the local Runtime, which also owns every buffer's authoritative host copy.
    RuntimeOptions local;
    /// Worker endpoints ("host:port"). KORA_WORKERS (comma-separated) adds more. Absent workers are
    /// retried in the background, so they may be started before or after the Cluster.
    std::vector<std::string> workers;
    /// Handshake secret; empty uses KORA_WORKER_TOKEN.
    std::string token;
    OffloadPolicy policy = OffloadPolicy::Auto;
    std::chrono::milliseconds connect_timeout{1'000};
    std::chrono::milliseconds reconnect_interval{500};
    /// Deadline for one remote request; exceeding it drops the link and re-runs the job locally.
    std::chrono::milliseconds request_timeout{30'000};
    std::chrono::milliseconds heartbeat_interval{1'000};
    std::uint64_t max_message_bytes = 1ULL << 30U;
};

struct WorkerStatus {
    std::string endpoint;
    bool connected = false;
    std::string name;
    std::string device;
    std::uint64_t dispatches = 0;
    std::uint64_t failovers = 0;
    std::uint64_t connects = 0;
    std::uint64_t disconnects = 0;
};

struct ClusterStats {
    std::uint64_t local_dispatches = 0;
    std::uint64_t remote_dispatches = 0;
    /// Remote jobs re-executed locally after a link failure, timeout, or worker error.
    std::uint64_t failovers = 0;
    /// Local device losses recovered by recreating the local Runtime.
    std::uint64_t local_recoveries = 0;
};

namespace detail {
struct ClusterImpl;
struct ClusterKernelState;
}  // namespace detail

/// A kernel loaded once and runnable on the local Runtime or any compatible worker.
class ClusterKernel final {
public:
    ClusterKernel() = default;
    bool valid() const noexcept { return state_ != nullptr; }

private:
    friend class Cluster;
    friend class ClusterDispatch;
    explicit ClusterKernel(std::shared_ptr<detail::ClusterKernelState> state) : state_(std::move(state)) {}
    std::shared_ptr<detail::ClusterKernelState> state_;
};

class ClusterDispatch final {
public:
    ClusterDispatch& global(std::uint64_t x, std::uint64_t y = 1, std::uint64_t z = 1);
    ClusterDispatch& read(const BufferHandle& buffer);
    ClusterDispatch& write(const BufferHandle& buffer);
    ClusterDispatch& read_write(const BufferHandle& buffer);
    template <class T>
    ClusterDispatch& read(const Buffer<T>& buffer) {
        return read(buffer.handle());
    }
    template <class T>
    ClusterDispatch& write(const Buffer<T>& buffer) {
        return write(buffer.handle());
    }
    template <class T>
    ClusterDispatch& read_write(const Buffer<T>& buffer) {
        return read_write(buffer.handle());
    }
    ClusterDispatch& push(std::span<const std::byte> bytes);
    template <class T>
        requires std::is_trivially_copyable_v<T>
    ClusterDispatch& push(const T& value) {
        return push(std::as_bytes(std::span<const T>(&value, 1)));
    }
    /// Enqueues the dispatch. The Event completes after outputs are in the host copies.
    Result<Event> submit();

private:
    friend class Cluster;
    ClusterDispatch(std::shared_ptr<detail::ClusterImpl> impl,
                    std::shared_ptr<detail::ClusterKernelState> kernel)
        : impl_(std::move(impl)), kernel_(std::move(kernel)) {}
    std::shared_ptr<detail::ClusterImpl> impl_;
    std::shared_ptr<detail::ClusterKernelState> kernel_;
    std::array<std::uint64_t, 3> global_{1, 1, 1};
    std::vector<BufferHandle> buffers_;
    std::vector<KernelAccess> accesses_;
    std::vector<std::byte> push_;
};

/// One host Runtime plus any number of hot-swappable remote KORA workers. Buffers always live
/// on the host; each dispatch runs where it is expected to finish first, and any remote failure
/// transparently re-runs it locally.
class Cluster final {
public:
    static Result<Cluster> create(ClusterOptions options = {});
    Cluster(Cluster&&) noexcept;
    Cluster& operator=(Cluster&&) noexcept;
    ~Cluster();

    /// The local Runtime. Do not call recover() on it directly; the Cluster recovers device loss.
    Runtime& runtime() noexcept;
    template <class T>
    Result<Buffer<T>> create_buffer(std::uint64_t count) {
        return runtime().create_buffer<T>(count);
    }
    Result<ClusterKernel> load_kernel(const std::filesystem::path& spirv, std::string entry_point,
                                      KernelMetadata metadata);
    ClusterDispatch dispatch(const ClusterKernel& kernel);

    /// Adds a worker endpoint at runtime; it is connected in the background.
    Status attach(const std::string& endpoint);
    /// Removes a worker; its in-flight jobs finish locally.
    Status detach(const std::string& endpoint);
    /// Waits for every submitted dispatch to complete.
    Status wait_idle();
    std::vector<WorkerStatus> workers() const;
    ClusterStats stats() const;

private:
    explicit Cluster(std::shared_ptr<detail::ClusterImpl> impl);
    std::shared_ptr<detail::ClusterImpl> impl_;
};

}  // namespace kora
