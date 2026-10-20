#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <unordered_map>

#include "kora/runtime.hpp"
#include "kora/worker.hpp"
#include "remote/protocol.hpp"
#include "remote/sha256.hpp"

namespace kora {
namespace remote = detail::remote;

namespace {
constexpr auto kHandshakeTimeout = std::chrono::milliseconds(5'000);
constexpr auto kAcceptPoll = std::chrono::milliseconds(100);
// ponytail: fixed handshake timeout and accept poll; make configurable if slow links need it.

std::vector<std::byte> random_nonce() {
    std::random_device device;
    std::vector<std::byte> nonce(32);
    for (auto& byte : nonce) byte = static_cast<std::byte>(device() & 0xFFU);
    return nonce;
}

remote::Digest handshake_mac(const std::string& token, std::span<const std::byte> nonce) {
    std::vector<std::uint8_t> message(nonce.size() + remote::kHandshakeContext.size());
    for (std::size_t i = 0; i < nonce.size(); ++i) message[i] = std::to_integer<std::uint8_t>(nonce[i]);
    std::copy(remote::kHandshakeContext.begin(), remote::kHandshakeContext.end(),
              message.begin() + nonce.size());
    return remote::hmac_sha256(std::span(reinterpret_cast<const std::uint8_t*>(token.data()), token.size()),
                               message);
}

struct CachedBuffer {
    BufferHandle handle;
    std::uint64_t version = 0;
    std::uint64_t size = 0;
    std::list<std::uint64_t>::iterator lru;
};
}  // namespace

struct WorkerServer::Impl {
    WorkerOptions options;
    Runtime runtime;
    remote::Socket listener;
    std::uint16_t bound_port = 0;
    std::atomic<bool> stopping{false};
    std::thread accept_thread;

    std::mutex session_mutex;  // Guards session_socket and session_thread handoff.
    std::shared_ptr<remote::Socket> session_socket;
    std::thread session_thread;

    mutable std::mutex stats_mutex;
    WorkerStats stats;

    // Session state; touched only by the single active session thread.
    std::unordered_map<std::uint64_t, Kernel> kernels;
    std::unordered_map<std::uint64_t, CachedBuffer> buffers;
    std::list<std::uint64_t> lru;
    std::uint64_t cached_bytes = 0;

    void accept_loop();
    void serve(std::shared_ptr<remote::Socket> socket);
    Status handle(remote::Socket& socket, const remote::Message& message);
    Status load_kernel(remote::Socket& socket, const remote::Message& message);
    Status dispatch(remote::Socket& socket, const remote::Message& message);
    void reset_session();
    void evict_until(std::uint64_t needed, const std::vector<std::uint64_t>& keep);
    void count(std::uint64_t WorkerStats::* field) {
        std::lock_guard lock(stats_mutex);
        ++(stats.*field);
    }
    Status reply_failure(remote::Socket& socket, std::uint64_t request, const std::string& message) {
        remote::Writer writer;
        writer.string(message);
        return remote::send_message(socket, remote::MessageType::Failure, request, writer.view(),
                                    options.io_timeout);
    }
};

Result<std::unique_ptr<WorkerServer>> WorkerServer::start(WorkerOptions options) {
    if (options.token.empty())
        if (const char* token = std::getenv("KORA_WORKER_TOKEN")) options.token = token;
    auto endpoint = remote::parse_endpoint(options.listen);
    if (!endpoint) return std::move(endpoint).error();
    auto runtime = Runtime::create(options.runtime);
    if (!runtime) return std::move(runtime).error();
    auto listener = remote::Socket::listen(endpoint.value());
    if (!listener) return std::move(listener).error();
    auto impl = std::make_unique<Impl>();
    impl->options = std::move(options);
    impl->runtime = std::move(runtime).value();
    impl->listener = std::move(listener).value();
    impl->bound_port = impl->listener.local_port();
    if (impl->options.name.empty()) {
        const auto devices = impl->runtime.devices();
        auto info = devices.empty() ? Result<DeviceInfo>(Error(ErrorCode::NoEligibleDevice, "no device"))
                                    : impl->runtime.device_info(devices.front());
        impl->options.name = info ? info.value().name() : "kora-worker";
    }
    auto* raw = impl.get();
    raw->accept_thread = std::thread([raw] { raw->accept_loop(); });
    return std::unique_ptr<WorkerServer>(new WorkerServer(std::move(impl)));
}

WorkerServer::WorkerServer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
WorkerServer::~WorkerServer() {
    stop();
}
std::uint16_t WorkerServer::port() const noexcept {
    return impl_ ? impl_->bound_port : 0;
}

WorkerStats WorkerServer::stats() const {
    std::lock_guard lock(impl_->stats_mutex);
    return impl_->stats;
}

void WorkerServer::stop() noexcept {
    if (!impl_ || impl_->stopping.exchange(true)) return;
    if (impl_->accept_thread.joinable()) impl_->accept_thread.join();
    std::thread session;
    {
        std::lock_guard lock(impl_->session_mutex);
        if (impl_->session_socket) impl_->session_socket->close();
        session = std::move(impl_->session_thread);
    }
    if (session.joinable()) session.join();
    impl_->listener.close();
}

void WorkerServer::Impl::accept_loop() {
    while (!stopping.load(std::memory_order_acquire)) {
        auto accepted = listener.accept(kAcceptPoll);
        if (!accepted || !accepted.value().valid()) continue;
        auto socket = std::make_shared<remote::Socket>(std::move(accepted).value());
        // Authenticate before touching the active session, so an unauthenticated peer cannot
        // displace a working client.
        const auto nonce = random_nonce();
        remote::Writer challenge;
        challenge.bytes(nonce);
        auto sent = remote::send_message(*socket, remote::MessageType::Challenge, 0, challenge.view(),
                                         kHandshakeTimeout);
        auto reply = sent ? remote::receive_message(*socket, 1024, kHandshakeTimeout)
                          : Result<remote::Message>(std::move(sent).error());
        bool accepted_peer = false;
        if (reply && reply.value().type == remote::MessageType::Auth) {
            remote::Reader reader(reply.value().payload);
            auto mac = reader.bytes();
            if (options.token.empty()) {
                accepted_peer = socket->peer_is_loopback();
            } else if (mac && mac.value().size() == 32) {
                remote::Digest received{};
                for (std::size_t i = 0; i < 32; ++i)
                    received[i] = std::to_integer<std::uint8_t>(mac.value()[i]);
                accepted_peer = remote::constant_time_equal(received, handshake_mac(options.token, nonce));
            }
        }
        if (!accepted_peer) {
            remote::Writer reason;
            reason.string(options.token.empty() ? "worker accepts only loopback clients without a token"
                                                : "worker authentication failed");
            (void)remote::send_message(*socket, remote::MessageType::Reject, 0, reason.view(),
                                       kHandshakeTimeout);
            count(&WorkerStats::rejected_connections);
            continue;
        }
        std::thread previous;
        {
            std::lock_guard lock(session_mutex);
            if (session_socket) session_socket->close();  // Replace a vanished or idle client.
            previous = std::move(session_thread);
        }
        if (previous.joinable()) previous.join();
        std::lock_guard lock(session_mutex);
        if (stopping.load(std::memory_order_acquire)) break;
        session_socket = socket;
        session_thread = std::thread([this, socket] { serve(socket); });
    }
}

void WorkerServer::Impl::reset_session() {
    kernels.clear();
    buffers.clear();
    lru.clear();
    cached_bytes = 0;
    std::lock_guard lock(stats_mutex);
    stats.cached_bytes = 0;
}

void WorkerServer::Impl::serve(std::shared_ptr<remote::Socket> socket) {
    reset_session();
    {
        std::lock_guard lock(stats_mutex);
        ++stats.sessions;
        stats.session_active = true;
    }
    static std::atomic<std::uint64_t> next_session{1};
    remote::Writer welcome;
    welcome.u64(next_session.fetch_add(1));
    welcome.string(options.name);
    const auto devices = runtime.devices();
    auto info = devices.empty() ? Result<DeviceInfo>(Error(ErrorCode::NoEligibleDevice, "no device"))
                                : runtime.device_info(devices.front());
    welcome.string(info ? info.value().name() : "unknown");
    welcome.u64(options.max_message_bytes);
    if (remote::send_message(*socket, remote::MessageType::Welcome, 0, welcome.view(), kHandshakeTimeout)) {
        while (!stopping.load(std::memory_order_acquire)) {
            // Idle clients send heartbeats; silence beyond the I/O timeout means the peer is gone.
            auto message = remote::receive_message(*socket, options.max_message_bytes, options.io_timeout);
            if (!message) break;
            if (!handle(*socket, message.value())) break;
        }
    }
    socket->close();
    reset_session();
    std::lock_guard lock(stats_mutex);
    stats.session_active = false;
}

Status WorkerServer::Impl::handle(remote::Socket& socket, const remote::Message& message) {
    switch (message.type) {
        case remote::MessageType::Ping:
            return remote::send_message(socket, remote::MessageType::Pong, message.request_id, {},
                                        options.io_timeout);
        case remote::MessageType::LoadKernel:
            return load_kernel(socket, message);
        case remote::MessageType::Dispatch:
            return dispatch(socket, message);
        case remote::MessageType::Release: {
            remote::Reader reader(message.payload);
            auto count_field = reader.u32();
            if (!count_field) return std::move(count_field).error();
            for (std::uint32_t i = 0; i < count_field.value(); ++i) {
                auto key = reader.u64();
                if (!key) return std::move(key).error();
                const auto found = buffers.find(key.value());
                if (found == buffers.end()) continue;
                cached_bytes -= found->second.size;
                lru.erase(found->second.lru);
                buffers.erase(found);
            }
            return ok_status();
        }
        default:
            return Error(ErrorCode::IoError, "unexpected worker message");
    }
}

Status WorkerServer::Impl::load_kernel(remote::Socket& socket, const remote::Message& message) {
    remote::Reader reader(message.payload);
    auto key = reader.u64();
    auto metadata_json = key ? reader.string() : Result<std::string>(Error(ErrorCode::IoError, "truncated"));
    auto entry =
        metadata_json ? reader.string() : Result<std::string>(Error(ErrorCode::IoError, "truncated"));
    auto words =
        entry ? reader.bytes() : Result<std::span<const std::byte>>(Error(ErrorCode::IoError, "truncated"));
    if (!words || words.value().size() % 4 != 0)
        return Error(ErrorCode::IoError, "malformed kernel load request");
    auto metadata = parse_kernel_metadata(metadata_json.value());
    if (!metadata) return reply_failure(socket, message.request_id, metadata.error().describe());
    std::vector<std::uint32_t> module(words.value().size() / 4);
    std::memcpy(module.data(), words.value().data(), words.value().size());
    const auto devices = runtime.devices();
    if (devices.empty()) return reply_failure(socket, message.request_id, "worker has no device");
    auto kernel = runtime.load_kernel(module, entry.value(), std::move(metadata).value(), devices.front());
    if (!kernel) return reply_failure(socket, message.request_id, kernel.error().describe());
    kernels.insert_or_assign(key.value(), std::move(kernel).value());
    count(&WorkerStats::kernels_loaded);
    return remote::send_message(socket, remote::MessageType::KernelLoaded, message.request_id, {},
                                options.io_timeout);
}

void WorkerServer::Impl::evict_until(std::uint64_t needed, const std::vector<std::uint64_t>& keep) {
    for (auto it = lru.begin(); it != lru.end() && cached_bytes + needed > options.max_cache_bytes;) {
        const auto key = *it;
        if (std::find(keep.begin(), keep.end(), key) != keep.end()) {
            ++it;
            continue;
        }
        it = lru.erase(it);
        const auto found = buffers.find(key);
        if (found == buffers.end()) continue;
        cached_bytes -= found->second.size;
        buffers.erase(found);
    }
}

Status WorkerServer::Impl::dispatch(remote::Socket& socket, const remote::Message& message) {
    struct Binding {
        std::uint64_t key = 0;
        std::uint64_t version = 0;
        std::uint64_t size = 0;
        KernelAccess access = KernelAccess::Read;
        std::uint64_t result_version = 0;
        std::optional<std::span<const std::byte>> data;
    };
    remote::Reader reader(message.payload);
    const auto malformed = Error(ErrorCode::IoError, "malformed dispatch request");
    auto kernel_key = reader.u64();
    std::array<std::uint64_t, 3> global{};
    for (auto& value : global) {
        auto field = reader.u64();
        if (!field) return malformed;
        value = field.value();
    }
    auto push = reader.bytes();
    auto binding_count = reader.u32();
    if (!kernel_key || !push || !binding_count || binding_count.value() > 1024) return malformed;
    std::vector<Binding> bindings(binding_count.value());
    for (auto& binding : bindings) {
        auto key = reader.u64();
        auto version = reader.u64();
        auto size = reader.u64();
        auto access = reader.u8();
        auto result_version = reader.u64();
        auto has_data = reader.u8();
        if (!key || !version || !size || !access || !result_version || !has_data || access.value() > 2)
            return malformed;
        binding = {key.value(),
                   version.value(),
                   size.value(),
                   static_cast<KernelAccess>(access.value()),
                   result_version.value(),
                   std::nullopt};
        if (has_data.value() != 0) {
            auto data = reader.bytes();
            if (!data || data.value().size() != binding.size) return malformed;
            binding.data = data.value();
        }
        if (binding.size == 0 || binding.size > options.max_message_bytes) return malformed;
    }
    if (!reader.done()) return malformed;
    const auto kernel = kernels.find(kernel_key.value());
    if (kernel == kernels.end())
        return reply_failure(socket, message.request_id, "kernel is not loaded on this worker");

    // Ask for any input whose current contents this worker does not hold.
    remote::Writer missing;
    std::uint32_t missing_count = 0;
    std::vector<std::uint64_t> keys;
    std::uint64_t incoming = 0;
    for (std::uint32_t i = 0; i < bindings.size(); ++i) {
        const auto& binding = bindings[i];
        keys.push_back(binding.key);
        const auto cached = buffers.find(binding.key);
        const bool current = cached != buffers.end() && cached->second.size == binding.size &&
                             cached->second.version == binding.version;
        if (cached == buffers.end() || cached->second.size != binding.size) incoming += binding.size;
        if (binding.access != KernelAccess::Write && !binding.data && !current) {
            missing.u32(i);
            ++missing_count;
        }
    }
    if (missing_count != 0) {
        remote::Writer reply;
        reply.u32(missing_count);
        reply.buffer().insert(reply.buffer().end(), missing.view().begin(), missing.view().end());
        return remote::send_message(socket, remote::MessageType::NeedData, message.request_id, reply.view(),
                                    options.io_timeout);
    }
    evict_until(incoming, keys);
    if (cached_bytes + incoming > options.max_cache_bytes)
        return reply_failure(socket, message.request_id, "dispatch exceeds the worker cache limit");

    auto builder = runtime.dispatch(kernel->second);
    builder.global(global[0], global[1], global[2]);
    std::vector<BufferHandle> handles;
    for (const auto& binding : bindings) {
        auto cached = buffers.find(binding.key);
        if (cached == buffers.end() || cached->second.size != binding.size) {
            if (cached != buffers.end()) {
                cached_bytes -= cached->second.size;
                lru.erase(cached->second.lru);
                buffers.erase(cached);
            }
            auto created = runtime.create_buffer(binding.size, BufferUsage::Storage);
            if (!created) return reply_failure(socket, message.request_id, created.error().describe());
            lru.push_back(binding.key);
            cached = buffers
                         .emplace(binding.key, CachedBuffer{std::move(created).value(), 0, binding.size,
                                                            std::prev(lru.end())})
                         .first;
            cached_bytes += binding.size;
        } else {
            lru.splice(lru.end(), lru, cached->second.lru);
        }
        if (binding.data) {
            auto written = cached->second.handle.write_bytes(*binding.data);
            if (!written) return reply_failure(socket, message.request_id, written.error().describe());
            cached->second.version = binding.version;
        }
        handles.push_back(cached->second.handle);
        switch (binding.access) {
            case KernelAccess::Read:
                builder.read(cached->second.handle);
                break;
            case KernelAccess::Write:
                builder.write(cached->second.handle);
                break;
            case KernelAccess::ReadWrite:
                builder.read_write(cached->second.handle);
                break;
        }
    }
    {
        std::lock_guard lock(stats_mutex);
        stats.cached_bytes = cached_bytes;
    }
    builder.push(push.value());
    auto event = builder.submit();
    Status completed = event ? event.value().wait() : Status(std::move(event).error());
    if (!completed) {
        // The worker's own device may have been lost; drop state so the next session starts clean.
        const auto description = completed.error().describe();
        if (completed.error().code == ErrorCode::DeviceLost) {
            reset_session();
            (void)runtime.recover();
        }
        return reply_failure(socket, message.request_id, description);
    }
    remote::Writer outputs;
    std::uint32_t output_count = 0;
    for (const auto& binding : bindings) output_count += binding.access != KernelAccess::Read ? 1U : 0U;
    outputs.u32(output_count);
    for (std::uint32_t i = 0; i < bindings.size(); ++i) {
        if (bindings[i].access == KernelAccess::Read) continue;
        auto bytes = handles[i].read_bytes();
        if (!bytes) return reply_failure(socket, message.request_id, bytes.error().describe());
        outputs.u32(i);
        outputs.bytes(bytes.value());
        buffers.at(bindings[i].key).version = bindings[i].result_version;
    }
    count(&WorkerStats::dispatches);
    return remote::send_message(socket, remote::MessageType::Outputs, message.request_id, outputs.view(),
                                options.io_timeout);
}

}  // namespace kora
