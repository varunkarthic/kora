#include "kora/cluster.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "core/buffer_manager_access.hpp"
#include "core/event_state.hpp"
#include "kernel/metadata_json.hpp"
#include "remote/protocol.hpp"
#include "remote/sha256.hpp"

namespace kora {
namespace remote = detail::remote;

namespace detail {

struct ClusterKernelState {
    std::uint64_t key = 0;
    std::vector<std::uint32_t> words;
    std::string entry_point;
    KernelMetadata metadata;
    std::string metadata_json;
    std::mutex mutex;  // Guards the local kernel, which is reloaded after local device recovery.
    Kernel local;
    std::uint64_t local_generation = 0;
};

namespace {
using Clock = std::chrono::steady_clock;

std::uint64_t buffer_key(BufferId id) noexcept {
    return std::uint64_t{id.slot} | (id.generation << 32U);
}

struct IdHash {
    std::size_t operator()(BufferId id) const noexcept { return std::hash<std::uint64_t>{}(buffer_key(id)); }
};

std::uint64_t fnv1a(std::span<const std::byte> bytes, std::uint64_t hash = 1469598103934665603ULL) noexcept {
    for (const auto byte : bytes) {
        hash ^= std::to_integer<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct Job {
    std::shared_ptr<ClusterKernelState> kernel;
    std::array<std::uint64_t, 3> global{1, 1, 1};
    std::vector<BufferHandle> buffers;
    std::vector<KernelAccess> accesses;
    std::vector<std::byte> push;
    EventPromise promise;
    std::vector<Event> dependencies;
    std::uint64_t bytes = 0;
    bool force_local = false;
    bool local_retried = false;
    std::uint64_t local_generation = 0;
};

// Outcome of one remote attempt. Only Link failures drop the connection.
enum class RemoteOutcome { Done, Rejected, Link };
}  // namespace

struct WorkerLink {
    std::string endpoint;
    remote::Endpoint address;
    std::thread thread;
    std::atomic<bool> stop{false};
    std::atomic<bool> connected{false};
    std::atomic<std::uint32_t> inflight{0};

    std::mutex mutex;  // Guards everything below except session-only fields.
    std::condition_variable cv;
    std::deque<std::shared_ptr<Job>> jobs;
    std::vector<std::uint64_t> releases;
    std::shared_ptr<remote::Socket> socket;
    std::unordered_set<std::uint64_t> unsupported_kernels;
    WorkerStatus status;

    // Session state, used only by the link thread and reset on every connection.
    std::unordered_set<std::uint64_t> loaded_kernels;
    std::unordered_map<std::uint64_t, std::uint64_t> worker_versions;
    std::uint64_t next_request = 1;
    std::uint64_t max_message = 0;
};

struct ClusterImpl : std::enable_shared_from_this<ClusterImpl> {
    ClusterOptions options;
    Runtime runtime;
    std::shared_mutex runtime_mutex;  // Shared for local work; exclusive while recovering.
    std::atomic<std::uint64_t> local_generation{1};

    std::mutex submit_mutex;  // Guards dependency tracking and outstanding events.
    struct Access {
        Event last_write;
        std::vector<Event> reads;
    };
    std::unordered_map<BufferId, Access, IdHash> access;
    std::vector<Event> outstanding;

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<std::shared_ptr<Job>> queue;
    std::vector<BufferId> released;
    bool stopping = false;
    bool shut_down = false;
    std::thread dispatcher;

    mutable std::mutex workers_mutex;
    std::vector<std::shared_ptr<WorkerLink>> workers;

    std::atomic<std::uint64_t> local_dispatches{0};
    std::atomic<std::uint64_t> remote_dispatches{0};
    std::atomic<std::uint64_t> failovers{0};
    std::atomic<std::uint64_t> local_recoveries{0};
    std::atomic<std::uint32_t> local_inflight{0};

    std::mutex model_mutex;
    std::unordered_map<std::string, double> latency_ema;
    std::uint64_t explore_counter = 0;

    void dispatcher_loop();
    void enqueue(std::shared_ptr<Job> job, bool front);
    void run_local(const std::shared_ptr<Job>& job);
    void recover_local(std::uint64_t observed_generation);
    std::shared_ptr<WorkerLink> choose_worker(const Job& job);
    void link_loop(const std::shared_ptr<WorkerLink>& link);
    Status handshake(WorkerLink& link, remote::Socket& socket);
    RemoteOutcome run_remote(WorkerLink& link, remote::Socket& socket, Job& job);
    void fail_over(const std::shared_ptr<Job>& job, WorkerLink* link);
    void record(const std::string& target, const Job& job, Clock::duration elapsed);
    std::optional<double> predicted(const std::string& target, const Job& job);
    Status start_link(const std::string& endpoint);
    void shutdown();
};

namespace {
std::string model_key(const std::string& target, const Job& job) {
    const auto bucket = job.bytes == 0 ? 0 : static_cast<int>(std::log2(static_cast<double>(job.bytes)));
    return target + "|" + std::to_string(job.kernel->key) + "|" + std::to_string(bucket);
}
}  // namespace

void ClusterImpl::record(const std::string& target, const Job& job, Clock::duration elapsed) {
    const auto ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    std::lock_guard lock(model_mutex);
    auto [entry, inserted] = latency_ema.try_emplace(model_key(target, job), ns);
    if (!inserted) entry->second = 0.8 * entry->second + 0.2 * ns;
}

std::optional<double> ClusterImpl::predicted(const std::string& target, const Job& job) {
    std::lock_guard lock(model_mutex);
    const auto found = latency_ema.find(model_key(target, job));
    if (found == latency_ema.end()) return std::nullopt;
    return found->second;
}

void ClusterImpl::enqueue(std::shared_ptr<Job> job, bool front) {
    {
        std::lock_guard lock(queue_mutex);
        if (front)
            queue.push_front(std::move(job));
        else
            queue.push_back(std::move(job));
    }
    queue_cv.notify_one();
}

void ClusterImpl::fail_over(const std::shared_ptr<Job>& job, WorkerLink* link) {
    failovers.fetch_add(1, std::memory_order_relaxed);
    if (link) {
        std::lock_guard lock(link->mutex);
        ++link->status.failovers;
    }
    job->force_local = true;
    enqueue(job, true);  // Its dependents are already ordered behind its Event.
}

std::shared_ptr<WorkerLink> ClusterImpl::choose_worker(const Job& job) {
    if (job.force_local || options.policy == OffloadPolicy::LocalOnly) return nullptr;
    std::vector<std::shared_ptr<WorkerLink>> candidates;
    {
        std::lock_guard lock(workers_mutex);
        for (const auto& link : workers) {
            if (!link->connected.load(std::memory_order_acquire)) continue;
            std::lock_guard link_lock(link->mutex);
            if (!link->unsupported_kernels.contains(job.kernel->key)) candidates.push_back(link);
        }
    }
    if (candidates.empty()) return nullptr;
    const auto load = [](std::uint32_t inflight, double latency) { return latency * (1.0 + inflight); };
    if (options.policy == OffloadPolicy::PreferRemote)
        return *std::min_element(candidates.begin(), candidates.end(),
                                 [](const auto& left, const auto& right) {
                                     return left->inflight.load() < right->inflight.load();
                                 });
    // Auto: measure locally first, explore each unmeasured worker occasionally, then pick the
    // lowest expected completion time including work already queued on each target.
    const auto local = predicted("local", job);
    if (!local) return nullptr;
    std::shared_ptr<WorkerLink> best;
    double best_cost = load(local_inflight.load(), *local);
    for (const auto& link : candidates) {
        const auto latency = predicted(link->endpoint, job);
        if (!latency) {
            std::lock_guard lock(model_mutex);
            if (explore_counter++ % 8 == 0) return link;
            continue;
        }
        const auto cost = load(link->inflight.load(), *latency);
        if (cost < best_cost) {
            best_cost = cost;
            best = link;
        }
    }
    return best;
}

void ClusterImpl::dispatcher_loop() {
    for (;;) {
        std::shared_ptr<Job> job;
        std::vector<BufferId> released_now;
        {
            std::unique_lock lock(queue_mutex);
            queue_cv.wait(lock, [&] { return stopping || !queue.empty() || !released.empty(); });
            released_now.swap(released);
            if (!queue.empty()) {
                job = std::move(queue.front());
                queue.pop_front();
            } else if (stopping && released_now.empty()) {
                return;
            }
        }
        if (!released_now.empty()) {
            {
                std::lock_guard lock(submit_mutex);
                for (const auto id : released_now) access.erase(id);
            }
            std::lock_guard lock(workers_mutex);
            for (const auto& link : workers) {
                std::lock_guard link_lock(link->mutex);
                for (const auto id : released_now) link->releases.push_back(buffer_key(id));
                link->cv.notify_one();
            }
        }
        if (!job) continue;
        // Never block this thread on dependencies: a dependency may be a remote job that fails
        // over and needs this thread to run it locally. Re-queue the job once they complete.
        std::erase_if(job->dependencies, [](const Event& item) { return item.done(); });
        if (!job->dependencies.empty()) {
            auto all = Event::when_all(job->dependencies);
            job->dependencies.clear();
            if (all) {
                std::weak_ptr<ClusterImpl> weak = weak_from_this();
                const auto registered = all.value().on_complete([weak, job](Status) {
                    if (auto self = weak.lock()) self->enqueue(job, false);
                });
                if (registered) continue;
            }
        }
        if (job->local_generation != 0 && job->local_generation == local_generation.load()) {
            recover_local(job->local_generation);  // A local attempt observed device loss.
        }
        if (auto link = choose_worker(*job)) {
            link->inflight.fetch_add(1);
            std::lock_guard lock(link->mutex);
            link->jobs.push_back(job);
            link->cv.notify_one();
            continue;
        }
        run_local(job);
    }
}

void ClusterImpl::recover_local(std::uint64_t observed_generation) {
    std::unique_lock lock(runtime_mutex);
    if (local_generation.load() != observed_generation) return;  // Another job already recovered.
    if (runtime.recover()) {
        local_generation.fetch_add(1);
        local_recoveries.fetch_add(1, std::memory_order_relaxed);
    }
}

void ClusterImpl::run_local(const std::shared_ptr<Job>& job) {
    const auto started = Clock::now();
    Result<Event> submitted = Error(ErrorCode::InvalidArgument, "local dispatch was not submitted");
    const auto generation = local_generation.load();
    {
        std::shared_lock lock(runtime_mutex);
        auto& kernel = *job->kernel;
        std::lock_guard kernel_lock(kernel.mutex);
        if (kernel.local_generation != generation) {
            const auto devices = runtime.devices();
            auto loaded =
                devices.empty()
                    ? Result<Kernel>(Error(ErrorCode::NoEligibleDevice, "local runtime has no device"))
                    : runtime.load_kernel(kernel.words, kernel.entry_point, kernel.metadata, devices.front());
            if (!loaded) {
                job->promise.complete(Status(std::move(loaded).error()));
                return;
            }
            kernel.local = std::move(loaded).value();
            kernel.local_generation = generation;
        }
        auto builder = runtime.dispatch(kernel.local);
        builder.global(job->global[0], job->global[1], job->global[2]);
        for (std::size_t i = 0; i < job->buffers.size(); ++i) {
            switch (job->accesses[i]) {
                case KernelAccess::Read:
                    builder.read(job->buffers[i]);
                    break;
                case KernelAccess::Write:
                    builder.write(job->buffers[i]);
                    break;
                case KernelAccess::ReadWrite:
                    builder.read_write(job->buffers[i]);
                    break;
            }
        }
        builder.push(std::span<const std::byte>(job->push));
        submitted = builder.submit();
    }
    const auto retry_after_loss = [this, job, generation](const Error& error) {
        if (error.code != ErrorCode::DeviceLost || job->local_retried) return false;
        job->local_retried = true;
        job->local_generation = generation;
        enqueue(job, true);
        return true;
    };
    if (!submitted) {
        if (!retry_after_loss(submitted.error())) job->promise.complete(Status(std::move(submitted).error()));
        return;
    }
    local_inflight.fetch_add(1);
    std::weak_ptr<ClusterImpl> weak = weak_from_this();
    (void)submitted.value().on_complete([weak, job, started, retry_after_loss](Status status) {
        auto self = weak.lock();
        if (!self) {
            job->promise.complete(std::move(status));
            return;
        }
        self->local_inflight.fetch_sub(1);
        if (status) {
            self->local_dispatches.fetch_add(1, std::memory_order_relaxed);
            self->record("local", *job, Clock::now() - started);
        } else if (retry_after_loss(status.error())) {
            return;
        }
        job->promise.complete(std::move(status));
    });
}

Status ClusterImpl::handshake(WorkerLink& link, remote::Socket& socket) {
    const auto timeout = options.connect_timeout + std::chrono::milliseconds(4'000);
    auto challenge = remote::receive_message(socket, 1024, timeout);
    if (!challenge) return std::move(challenge).error();
    if (challenge.value().type != remote::MessageType::Challenge)
        return Error(ErrorCode::IoError, "worker did not send a challenge");
    remote::Reader reader(challenge.value().payload);
    auto nonce = reader.bytes();
    if (!nonce) return std::move(nonce).error();
    std::vector<std::uint8_t> message(nonce.value().size() + remote::kHandshakeContext.size());
    for (std::size_t i = 0; i < nonce.value().size(); ++i)
        message[i] = std::to_integer<std::uint8_t>(nonce.value()[i]);
    std::copy(remote::kHandshakeContext.begin(), remote::kHandshakeContext.end(),
              message.begin() + nonce.value().size());
    const auto mac = remote::hmac_sha256(
        std::span(reinterpret_cast<const std::uint8_t*>(options.token.data()), options.token.size()),
        message);
    remote::Writer auth;
    auth.bytes(std::as_bytes(std::span(mac)));
    KORA_TRY_STATUS(remote::send_message(socket, remote::MessageType::Auth, 0, auth.view(), timeout));
    auto welcome = remote::receive_message(socket, 1U << 20U, timeout);
    if (!welcome) return std::move(welcome).error();
    remote::Reader fields(welcome.value().payload);
    if (welcome.value().type == remote::MessageType::Reject) {
        auto reason = fields.string();
        return Error(ErrorCode::IoError, "worker rejected connection: " + (reason ? reason.value() : ""));
    }
    if (welcome.value().type != remote::MessageType::Welcome)
        return Error(ErrorCode::IoError, "unexpected handshake reply");
    auto session = fields.u64();
    auto name = fields.string();
    auto device = fields.string();
    auto max_message = fields.u64();
    if (!session || !name || !device || !max_message)
        return Error(ErrorCode::IoError, "malformed worker welcome");
    std::lock_guard lock(link.mutex);
    link.status.name = name.value();
    link.status.device = device.value();
    link.max_message = std::min(max_message.value(), options.max_message_bytes);
    return ok_status();
}

RemoteOutcome ClusterImpl::run_remote(WorkerLink& link, remote::Socket& socket, Job& job) {
    const auto timeout = options.request_timeout;
    const auto request = [&](remote::MessageType type,
                             std::span<const std::byte> payload) -> Result<remote::Message> {
        const auto id = link.next_request++;
        KORA_TRY_STATUS(remote::send_message(socket, type, id, payload, timeout));
        auto reply = remote::receive_message(socket, link.max_message, timeout);
        if (reply && reply.value().request_id != id)
            return Error(ErrorCode::IoError, "worker reply is out of order");
        return reply;
    };
    auto& kernel = *job.kernel;
    if (!link.loaded_kernels.contains(kernel.key)) {
        remote::Writer load;
        load.u64(kernel.key);
        load.string(kernel.metadata_json);
        load.string(kernel.entry_point);
        load.bytes(std::as_bytes(std::span(kernel.words)));
        auto reply = request(remote::MessageType::LoadKernel, load.view());
        if (!reply) return RemoteOutcome::Link;
        if (reply.value().type == remote::MessageType::Failure) {
            std::lock_guard lock(link.mutex);
            link.unsupported_kernels.insert(kernel.key);  // e.g. missing device features
            return RemoteOutcome::Rejected;
        }
        if (reply.value().type != remote::MessageType::KernelLoaded) return RemoteOutcome::Link;
        link.loaded_kernels.insert(kernel.key);
    }

    // Host copies are authoritative: bring back anything newer on the local device, then capture
    // each buffer's version so a concurrent host write is never overwritten by a stale result.
    std::vector<std::uint64_t> versions(job.buffers.size());
    for (std::size_t i = 0; i < job.buffers.size(); ++i) {
        if (job.accesses[i] != KernelAccess::Write) {
            std::shared_lock lock(runtime_mutex);
            auto readback = runtime.readback(job.buffers[i]);
            if (!readback || !readback.value().wait()) return RemoteOutcome::Rejected;
        }
        auto version = job.buffers[i].host_version();
        if (!version) return RemoteOutcome::Rejected;
        versions[i] = version.value();
    }
    const auto encode = [&](const std::vector<bool>& force_data) -> Result<remote::Writer> {
        remote::Writer writer;
        writer.u64(kernel.key);
        for (const auto value : job.global) writer.u64(value);
        writer.bytes(job.push);
        writer.u32(static_cast<std::uint32_t>(job.buffers.size()));
        for (std::size_t i = 0; i < job.buffers.size(); ++i) {
            const auto key = buffer_key(job.buffers[i].id());
            writer.u64(key);
            writer.u64(versions[i]);
            writer.u64(job.buffers[i].size_bytes());
            writer.u8(static_cast<std::uint8_t>(job.accesses[i]));
            writer.u64(versions[i] + 1);  // Host version after this result commits.
            const auto known = link.worker_versions.find(key);
            const bool send =
                job.accesses[i] != KernelAccess::Write &&
                (force_data[i] || known == link.worker_versions.end() || known->second != versions[i]);
            writer.u8(send ? 1 : 0);
            if (!send) continue;
            auto mapped = job.buffers[i].map(MapAccess::Read);
            if (!mapped) return std::move(mapped).error();
            writer.bytes(std::span<const std::byte>(mapped.value().data(), mapped.value().size()));
        }
        return writer;
    };
    std::vector<bool> force(job.buffers.size(), false);
    auto payload = encode(force);
    if (!payload) return RemoteOutcome::Rejected;
    auto reply = request(remote::MessageType::Dispatch, payload.value().view());
    if (reply && reply.value().type == remote::MessageType::NeedData) {
        remote::Reader reader(reply.value().payload);
        auto count = reader.u32();
        if (!count) return RemoteOutcome::Link;
        for (std::uint32_t i = 0; i < count.value(); ++i) {
            auto index = reader.u32();
            if (!index || index.value() >= force.size()) return RemoteOutcome::Link;
            force[index.value()] = true;
            link.worker_versions.erase(buffer_key(job.buffers[index.value()].id()));
        }
        payload = encode(force);
        if (!payload) return RemoteOutcome::Rejected;
        reply = request(remote::MessageType::Dispatch, payload.value().view());
    }
    if (!reply) return RemoteOutcome::Link;
    if (reply.value().type == remote::MessageType::Failure) return RemoteOutcome::Rejected;
    if (reply.value().type != remote::MessageType::Outputs) return RemoteOutcome::Link;

    // Validate the whole reply before committing anything, so a bad reply changes no buffer.
    remote::Reader reader(reply.value().payload);
    auto count = reader.u32();
    if (!count) return RemoteOutcome::Link;
    std::vector<std::pair<std::uint32_t, std::span<const std::byte>>> outputs;
    for (std::uint32_t i = 0; i < count.value(); ++i) {
        auto index = reader.u32();
        auto bytes = reader.bytes();
        if (!index || !bytes || index.value() >= job.buffers.size() ||
            job.accesses[index.value()] == KernelAccess::Read ||
            bytes.value().size() != job.buffers[index.value()].size_bytes())
            return RemoteOutcome::Link;
        outputs.emplace_back(index.value(), bytes.value());
    }
    const auto expected_outputs = static_cast<std::size_t>(std::count_if(
        job.accesses.begin(), job.accesses.end(), [](auto a) { return a != KernelAccess::Read; }));
    if (!reader.done() || outputs.size() != expected_outputs) return RemoteOutcome::Link;
    for (std::size_t i = 0; i < job.buffers.size(); ++i)
        if (job.accesses[i] != KernelAccess::Write)
            link.worker_versions[buffer_key(job.buffers[i].id())] = versions[i];
    for (const auto& [index, bytes] : outputs) {
        const auto key = buffer_key(job.buffers[index].id());
        auto committed = job.buffers[index].write_bytes_if_version(bytes, versions[index]);
        if (committed && committed.value().written && committed.value().host_version == versions[index] + 1)
            link.worker_versions[key] = versions[index] + 1;
        else
            link.worker_versions.erase(key);  // Host changed meanwhile; the host copy wins.
    }
    return RemoteOutcome::Done;
}

void ClusterImpl::link_loop(const std::shared_ptr<WorkerLink>& link) {
    const auto wait_or_stop = [&](std::chrono::milliseconds duration) {
        std::unique_lock lock(link->mutex);
        link->cv.wait_for(lock, duration, [&] { return link->stop.load(); });
    };
    const auto drain_to_local = [&] {
        std::deque<std::shared_ptr<Job>> pending;
        {
            std::lock_guard lock(link->mutex);
            pending.swap(link->jobs);
            link->releases.clear();
        }
        for (auto& job : pending) {
            link->inflight.fetch_sub(1);
            fail_over(job, link.get());
        }
    };
    while (!link->stop.load()) {
        auto connected = remote::Socket::connect(link->address, options.connect_timeout);
        if (!connected) {
            wait_or_stop(options.reconnect_interval);
            continue;
        }
        auto socket = std::make_shared<remote::Socket>(std::move(connected).value());
        {
            std::lock_guard lock(link->mutex);
            link->socket = socket;
        }
        if (!handshake(*link, *socket)) {
            socket->close();
            wait_or_stop(options.reconnect_interval);
            continue;
        }
        link->loaded_kernels.clear();
        link->worker_versions.clear();
        {
            std::lock_guard lock(link->mutex);
            ++link->status.connects;
            link->releases.clear();  // A new session starts with an empty worker cache.
        }
        link->connected.store(true, std::memory_order_release);
        auto last_activity = Clock::now();
        while (!link->stop.load()) {
            std::shared_ptr<Job> job;
            std::vector<std::uint64_t> releases;
            {
                std::unique_lock lock(link->mutex);
                link->cv.wait_for(lock, options.heartbeat_interval, [&] {
                    return link->stop.load() || !link->jobs.empty() || !link->releases.empty();
                });
                releases.swap(link->releases);
                if (!link->jobs.empty()) {
                    job = std::move(link->jobs.front());
                    link->jobs.pop_front();
                }
            }
            if (link->stop.load()) {
                if (job) {
                    link->inflight.fetch_sub(1);
                    fail_over(job, link.get());
                }
                break;
            }
            if (!releases.empty()) {
                remote::Writer writer;
                writer.u32(static_cast<std::uint32_t>(releases.size()));
                for (const auto key : releases) {
                    writer.u64(key);
                    link->worker_versions.erase(key);
                }
                if (!remote::send_message(*socket, remote::MessageType::Release, 0, writer.view(),
                                          options.request_timeout)) {
                    if (job) {
                        link->inflight.fetch_sub(1);
                        fail_over(job, link.get());
                    }
                    break;
                }
            }
            if (job) {
                const auto started = Clock::now();
                const auto outcome = run_remote(*link, *socket, *job);
                link->inflight.fetch_sub(1);
                if (outcome == RemoteOutcome::Done) {
                    record(link->endpoint, *job, Clock::now() - started);
                    remote_dispatches.fetch_add(1, std::memory_order_relaxed);
                    {
                        std::lock_guard lock(link->mutex);
                        ++link->status.dispatches;
                    }
                    job->promise.complete(ok_status());
                    last_activity = Clock::now();
                    continue;
                }
                fail_over(job, link.get());
                if (outcome == RemoteOutcome::Link) break;
                continue;
            }
            if (Clock::now() - last_activity >= options.heartbeat_interval) {
                const auto id = link->next_request++;
                if (!remote::send_message(*socket, remote::MessageType::Ping, id, {},
                                          options.request_timeout))
                    break;
                auto pong = remote::receive_message(*socket, 1024, options.request_timeout);
                if (!pong || pong.value().type != remote::MessageType::Pong) break;
                last_activity = Clock::now();
            }
        }
        link->connected.store(false, std::memory_order_release);
        socket->close();
        {
            std::lock_guard lock(link->mutex);
            ++link->status.disconnects;
            link->socket.reset();
        }
        drain_to_local();
        if (!link->stop.load()) wait_or_stop(options.reconnect_interval);
    }
    link->connected.store(false, std::memory_order_release);
    drain_to_local();
}

Status ClusterImpl::start_link(const std::string& endpoint) {
    auto address = remote::parse_endpoint(endpoint);
    if (!address) return std::move(address).error();
    auto link = std::make_shared<WorkerLink>();
    link->endpoint = endpoint;
    link->address = address.value();
    link->status.endpoint = endpoint;
    std::lock_guard lock(workers_mutex);
    for (const auto& existing : workers)
        if (existing->endpoint == endpoint)
            return Error(ErrorCode::InvalidArgument, "worker is already attached");
    link->thread = std::thread([this, link] { link_loop(link); });
    workers.push_back(std::move(link));
    return ok_status();
}

void ClusterImpl::shutdown() {
    {
        std::lock_guard lock(queue_mutex);
        if (shut_down) return;
        shut_down = true;
    }
    std::vector<std::shared_ptr<WorkerLink>> links;
    {
        std::lock_guard lock(workers_mutex);
        links.swap(workers);
    }
    for (const auto& link : links) {
        {
            std::lock_guard lock(link->mutex);
            link->stop.store(true);
            if (link->socket) link->socket->interrupt();
        }
        link->cv.notify_all();
    }
    for (const auto& link : links)
        if (link->thread.joinable()) link->thread.join();
    std::vector<Event> pending;
    {
        std::lock_guard lock(submit_mutex);
        pending = outstanding;
    }
    for (const auto& event : pending) (void)event.wait();  // Failed-over jobs still run locally.
    {
        std::lock_guard lock(queue_mutex);
        stopping = true;
    }
    queue_cv.notify_all();
    if (dispatcher.joinable()) dispatcher.join();
}

}  // namespace detail

Result<Cluster> Cluster::create(ClusterOptions options) {
    if (options.token.empty())
        if (const char* token = std::getenv("KORA_WORKER_TOKEN")) options.token = token;
    if (const char* extra = std::getenv("KORA_WORKERS")) {
        std::stringstream stream(extra);
        std::string endpoint;
        while (std::getline(stream, endpoint, ','))
            if (!endpoint.empty()) options.workers.push_back(endpoint);
    }
    auto runtime = Runtime::create(options.local);
    if (!runtime) return std::move(runtime).error();
    auto impl = std::make_shared<detail::ClusterImpl>();
    impl->options = options;
    impl->runtime = std::move(runtime).value();
    std::weak_ptr<detail::ClusterImpl> weak = impl;
    detail::BufferManagerAccess::add_release_listener(detail::BufferManagerAccess::buffers(impl->runtime),
                                                      weak, [weak](BufferId id) {
                                                          auto self = weak.lock();
                                                          if (!self) return;
                                                          {
                                                              std::lock_guard lock(self->queue_mutex);
                                                              self->released.push_back(id);
                                                          }
                                                          self->queue_cv.notify_one();
                                                      });
    impl->dispatcher = std::thread([raw = impl.get()] { raw->dispatcher_loop(); });
    for (const auto& endpoint : options.workers) {
        auto started = impl->start_link(endpoint);
        if (!started) {
            impl->shutdown();
            return std::move(started).error();
        }
    }
    return Cluster(std::move(impl));
}

Cluster::Cluster(std::shared_ptr<detail::ClusterImpl> impl) : impl_(std::move(impl)) {}
Cluster::Cluster(Cluster&&) noexcept = default;
Cluster& Cluster::operator=(Cluster&& other) noexcept {
    if (this != &other) {
        if (impl_) impl_->shutdown();
        impl_ = std::move(other.impl_);
    }
    return *this;
}
Cluster::~Cluster() {
    if (impl_) impl_->shutdown();
}

Runtime& Cluster::runtime() noexcept {
    return impl_->runtime;
}

Result<ClusterKernel> Cluster::load_kernel(const std::filesystem::path& spirv, std::string entry_point,
                                           KernelMetadata metadata) {
    std::ifstream input(spirv, std::ios::binary | std::ios::ate);
    if (!input) return Error(ErrorCode::InvalidKernel, "cannot open SPIR-V module: " + spirv.string());
    const auto end = input.tellg();
    const auto size = end <= 0 ? std::uint64_t{0} : static_cast<std::uint64_t>(end);
    if (size == 0 || size > 16U * 1024U * 1024U || size % 4 != 0)
        return Error(ErrorCode::InvalidKernel, "SPIR-V module is empty, misaligned, or exceeds 16 MiB");
    auto state = std::make_shared<detail::ClusterKernelState>();
    state->words.resize(static_cast<std::size_t>(size / 4));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(state->words.data()), static_cast<std::streamsize>(size));
    if (!input) return Error(ErrorCode::IoError, "failed to read complete SPIR-V module");
    state->entry_point = std::move(entry_point);
    state->metadata = std::move(metadata);
    state->metadata_json = detail::serialize_kernel_metadata(state->metadata);
    state->key = detail::fnv1a(std::as_bytes(std::span(state->words)));
    state->key = detail::fnv1a(
        std::as_bytes(std::span(state->metadata_json.data(), state->metadata_json.size())), state->key);
    state->key = detail::fnv1a(std::as_bytes(std::span(state->entry_point.data(), state->entry_point.size())),
                               state->key);
    // Validate once on the local device so errors surface here, not at the first dispatch.
    std::shared_lock lock(impl_->runtime_mutex);
    const auto devices = impl_->runtime.devices();
    if (devices.empty()) return Error(ErrorCode::NoEligibleDevice, "local runtime has no device");
    auto local =
        impl_->runtime.load_kernel(state->words, state->entry_point, state->metadata, devices.front());
    if (!local) return std::move(local).error();
    state->local = std::move(local).value();
    state->local_generation = impl_->local_generation.load();
    return ClusterKernel(std::move(state));
}

ClusterDispatch Cluster::dispatch(const ClusterKernel& kernel) {
    return ClusterDispatch(impl_, kernel.state_);
}

Status Cluster::attach(const std::string& endpoint) {
    return impl_->start_link(endpoint);
}

Status Cluster::detach(const std::string& endpoint) {
    std::shared_ptr<detail::WorkerLink> link;
    {
        std::lock_guard lock(impl_->workers_mutex);
        const auto found = std::find_if(impl_->workers.begin(), impl_->workers.end(),
                                        [&](const auto& item) { return item->endpoint == endpoint; });
        if (found == impl_->workers.end()) return Error(ErrorCode::InvalidArgument, "worker is not attached");
        link = *found;
        impl_->workers.erase(found);
    }
    {
        std::lock_guard lock(link->mutex);
        link->stop.store(true);
        if (link->socket) link->socket->interrupt();
    }
    link->cv.notify_all();
    if (link->thread.joinable()) link->thread.join();
    return ok_status();
}

Status Cluster::wait_idle() {
    std::vector<Event> pending;
    {
        std::lock_guard lock(impl_->submit_mutex);
        pending = impl_->outstanding;
    }
    for (const auto& event : pending) KORA_TRY_STATUS(event.wait());
    return ok_status();
}

std::vector<WorkerStatus> Cluster::workers() const {
    std::vector<WorkerStatus> result;
    std::lock_guard lock(impl_->workers_mutex);
    for (const auto& link : impl_->workers) {
        std::lock_guard link_lock(link->mutex);
        auto status = link->status;
        status.connected = link->connected.load();
        result.push_back(std::move(status));
    }
    return result;
}

ClusterStats Cluster::stats() const {
    return {impl_->local_dispatches.load(), impl_->remote_dispatches.load(), impl_->failovers.load(),
            impl_->local_recoveries.load()};
}

ClusterDispatch& ClusterDispatch::global(std::uint64_t x, std::uint64_t y, std::uint64_t z) {
    global_ = {x, y, z};
    return *this;
}
ClusterDispatch& ClusterDispatch::read(const BufferHandle& buffer) {
    buffers_.push_back(buffer);
    accesses_.push_back(KernelAccess::Read);
    return *this;
}
ClusterDispatch& ClusterDispatch::write(const BufferHandle& buffer) {
    buffers_.push_back(buffer);
    accesses_.push_back(KernelAccess::Write);
    return *this;
}
ClusterDispatch& ClusterDispatch::read_write(const BufferHandle& buffer) {
    buffers_.push_back(buffer);
    accesses_.push_back(KernelAccess::ReadWrite);
    return *this;
}
ClusterDispatch& ClusterDispatch::push(std::span<const std::byte> bytes) {
    push_.assign(bytes.begin(), bytes.end());
    return *this;
}

Result<Event> ClusterDispatch::submit() {
    if (!impl_ || !kernel_) return Error(ErrorCode::InvalidArgument, "cluster dispatch has no kernel");
    if (buffers_.size() != kernel_->metadata.bindings.size())
        return Error(ErrorCode::InvalidKernelBinding, "dispatch buffer count does not match kernel metadata");
    if (push_.size() != kernel_->metadata.push_constant_size)
        return Error(ErrorCode::InvalidArgument, "push-constant byte size does not match kernel metadata");
    for (const auto& buffer : buffers_)
        if (!buffer) return Error(ErrorCode::InvalidArgument, "cluster dispatch has an empty buffer handle");
    auto job = std::make_shared<detail::Job>();
    job->kernel = kernel_;
    job->global = global_;
    job->buffers = buffers_;
    job->accesses = accesses_;
    job->push = push_;
    for (const auto& buffer : buffers_) job->bytes += buffer.size_bytes();
    auto event = job->promise.event();
    {
        std::lock_guard lock(impl_->queue_mutex);
        if (impl_->shut_down) return Error(ErrorCode::InvalidArgument, "cluster is shut down");
    }
    {
        // Order this job after earlier jobs that touch the same buffers (RAW, WAR, WAW).
        std::lock_guard lock(impl_->submit_mutex);
        std::erase_if(impl_->outstanding, [](const Event& item) { return item.done(); });
        impl_->outstanding.push_back(event);
        // Collect every dependency before recording this job, so a buffer bound twice never makes
        // the job wait on itself.
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            auto& state = impl_->access[buffers_[i].id()];
            std::erase_if(state.reads, [](const Event& item) { return item.done(); });
            if (state.last_write.valid() && !state.last_write.done())
                job->dependencies.push_back(state.last_write);
            if (accesses_[i] != KernelAccess::Read)
                job->dependencies.insert(job->dependencies.end(), state.reads.begin(), state.reads.end());
        }
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            auto& state = impl_->access[buffers_[i].id()];
            if (accesses_[i] == KernelAccess::Read) {
                state.reads.push_back(event);
            } else {
                state.reads.clear();
                state.last_write = event;
            }
        }
    }
    impl_->enqueue(std::move(job), false);
    return event;
}

}  // namespace kora
