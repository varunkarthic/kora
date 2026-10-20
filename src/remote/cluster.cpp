#include "kora/cluster.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "core/buffer_manager_access.hpp"
#include "core/event_state.hpp"
#include "core/runtime_replica_access.hpp"
#include "kernel/metadata_json.hpp"
#include "remote/protocol.hpp"
#include "remote/sha256.hpp"
#include "vulkan/api.hpp"

namespace kora {
namespace remote = detail::remote;

namespace detail {

namespace {
using Clock = std::chrono::steady_clock;

std::string device_key(const DeviceId& id) {
    return "local:" + std::to_string(id.index) + ":" + id.uuid;
}
}  // namespace

struct ClusterKernelState {
    std::uint64_t key = 0;
    std::vector<std::uint32_t> words;
    std::string entry_point;
    KernelMetadata metadata;
    std::string metadata_json;
    std::optional<std::uint32_t> base_offset;  // Push-constant offset of the `base` field, if any.
    std::mutex mutex;  // Guards the per-device kernels, reloaded after the local Runtime changes.
    std::uint64_t local_generation = 0;
    std::unordered_map<std::string, Kernel> local;
    std::unordered_set<std::string> unsupported_local;
};

namespace {
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

struct SplitState;

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
    std::uint32_t split_element_bytes = 0;  // Non-zero when the caller allowed splitting.
    // Set on a slice of a split dispatch.
    std::shared_ptr<SplitState> split;
    std::uint64_t slice_base = 0;
    std::uint64_t slice_count = 0;
    std::optional<DeviceId> assigned_device;
    Clock::time_point started{};
};

// Outcome of one remote attempt. Only Link failures drop the connection.
enum class RemoteOutcome { Done, Rejected, Link };
}  // namespace

struct WorkerLink;

namespace {
// Collects the slices of one split dispatch and commits the merged outputs once.
struct SplitState {
    std::shared_ptr<Job> parent;
    std::mutex mutex;
    std::size_t remaining = 0;
    std::optional<Error> error;
    std::vector<std::size_t> outputs;               // Indices of written bindings.
    std::vector<std::vector<std::byte>> assembled;  // Pre-dispatch contents, overwritten per slice.
    std::vector<std::uint64_t> versions;            // Host versions the commit expects.
    std::uint32_t element_bytes = 0;
    Clock::time_point started;
};

// Byte range of a slice within one written buffer.
std::pair<std::size_t, std::size_t> slice_bytes(const Job& slice, std::uint64_t buffer_size) {
    const auto element_bytes = slice.split->element_bytes;
    const auto begin = std::min<std::uint64_t>(slice.slice_base * element_bytes, buffer_size);
    const auto end =
        std::min<std::uint64_t>((slice.slice_base + slice.slice_count) * element_bytes, buffer_size);
    return {static_cast<std::size_t>(begin), static_cast<std::size_t>(end - begin)};
}
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

// A place a job can run: one local device, or one connected worker.
struct Target {
    std::shared_ptr<WorkerLink> link;
    std::optional<DeviceId> device;
    std::string key;
};

struct ClusterImpl : std::enable_shared_from_this<ClusterImpl> {
    ClusterOptions options;
    bool all_local_devices = false;
    Runtime runtime;
    std::shared_mutex runtime_mutex;  // Shared for local work; exclusive while rebuilding.
    std::atomic<std::uint64_t> local_generation{1};

    struct LocalTarget {
        DeviceId id;
        std::string key;
        std::string name;
    };
    std::vector<LocalTarget> local_targets;  // Guarded by runtime_mutex.

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
    bool rescan_requested = false;
    std::thread dispatcher;
    std::thread monitor;
    std::condition_variable
        monitor_cv;                  // Separate from queue_cv so notify_one always reaches the dispatcher.
    std::string device_fingerprint;  // Used only by the monitor thread.

    mutable std::mutex workers_mutex;
    std::vector<std::shared_ptr<WorkerLink>> workers;

    std::atomic<std::uint64_t> local_dispatches{0};
    std::atomic<std::uint64_t> remote_dispatches{0};
    std::atomic<std::uint64_t> failovers{0};
    std::atomic<std::uint64_t> local_recoveries{0};
    std::atomic<std::uint64_t> split_dispatches{0};
    std::atomic<std::uint64_t> slices{0};
    std::atomic<std::uint64_t> device_rescans{0};
    std::atomic<std::uint32_t> local_inflight{0};

    struct DeviceCounters {
        std::uint64_t dispatches = 0;
        std::uint64_t slices = 0;
        std::uint32_t inflight = 0;
    };
    mutable std::mutex model_mutex;                          // Guards the models and per-device counters.
    std::unordered_map<std::string, double> latency_ema;     // ns per dispatch
    std::unordered_map<std::string, double> throughput_ema;  // elements per ns
    std::unordered_map<std::string, DeviceCounters> device_counters;
    std::uint64_t explore_counter = 0;
    std::unordered_map<std::string, std::uint32_t> split_declines;  // Per split model key.

    void dispatcher_loop();
    void monitor_loop();
    void enqueue(std::shared_ptr<Job> job, bool front);
    void refresh_local_targets();
    void rebuild_local(bool after_loss, std::uint64_t observed_generation);
    Result<Kernel> local_kernel(ClusterKernelState& kernel, const DeviceId& device);
    Target choose_target(const Job& job);
    std::vector<Target> candidates(const Job& job);
    bool try_split(const std::shared_ptr<Job>& job);
    void start_on(const std::shared_ptr<Job>& job, const Target& target);
    void run_local(const std::shared_ptr<Job>& job, const DeviceId& device);
    void finish_slice(const std::shared_ptr<Job>& slice, Status status,
                      const std::vector<std::span<const std::byte>>& ranges, const std::string& target_key);
    void ensure_host_current(const Job& job);
    void link_loop(const std::shared_ptr<WorkerLink>& link);
    Status handshake(WorkerLink& link, remote::Socket& socket);
    RemoteOutcome run_remote(WorkerLink& link, remote::Socket& socket, Job& job,
                             std::vector<std::vector<std::byte>>* collected);
    void fail_over(const std::shared_ptr<Job>& job, WorkerLink* link);
    void record_latency(const std::string& target, const Job& job, Clock::duration elapsed);
    void record_throughput(const std::string& target, const Job& job, Clock::duration elapsed);
    std::optional<double> predicted(const std::string& target, const Job& job);
    std::optional<double> throughput(const std::string& target, const Job& job);
    void count_device(const std::string& key, bool slice, int inflight_delta);
    Status start_link(const std::string& endpoint);
    void shutdown();
};

namespace {
std::string model_key(const std::string& target, const Job& job) {
    const auto bucket = job.bytes == 0 ? 0 : static_cast<int>(std::log2(static_cast<double>(job.bytes)));
    return target + "|" + std::to_string(job.kernel->key) + "|" + std::to_string(bucket);
}

std::string hex(const std::uint8_t* bytes, std::size_t count) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(digits[bytes[i] >> 4U]);
        out.push_back(digits[bytes[i] & 0xFU]);
    }
    return out;
}

// Identity of the physical devices present now; it changes when a GPU appears or disappears.
std::string physical_device_fingerprint() {
    auto loader = vk::Loader::load();
    if (!loader) return {};
    auto instance = loader.value().create_instance();
    if (!instance) return {};
    auto physical = instance.value().physical_devices();
    if (!physical) return {};
    std::vector<std::string> ids;
    for (const auto& device : physical.value())
        ids.push_back(hex(device.id_properties.deviceUUID, VK_UUID_SIZE) + "/" +
                      device.properties.deviceName);
    std::sort(ids.begin(), ids.end());
    std::string fingerprint;
    for (const auto& id : ids) fingerprint += id + ";";
    return fingerprint;
}
}  // namespace

void ClusterImpl::record_latency(const std::string& target, const Job& job, Clock::duration elapsed) {
    const auto ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    std::lock_guard lock(model_mutex);
    auto [entry, inserted] = latency_ema.try_emplace(model_key(target, job), ns);
    if (!inserted) entry->second = 0.8 * entry->second + 0.2 * ns;
}

void ClusterImpl::record_throughput(const std::string& target, const Job& job, Clock::duration elapsed) {
    const auto ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    if (ns <= 0 || job.slice_count == 0) return;
    const auto rate = static_cast<double>(job.slice_count) / ns;
    std::lock_guard lock(model_mutex);
    auto [entry, inserted] = throughput_ema.try_emplace(target + "|" + std::to_string(job.kernel->key), rate);
    if (!inserted) entry->second = 0.7 * entry->second + 0.3 * rate;
}

std::optional<double> ClusterImpl::predicted(const std::string& target, const Job& job) {
    std::lock_guard lock(model_mutex);
    const auto found = latency_ema.find(model_key(target, job));
    if (found == latency_ema.end()) return std::nullopt;
    return found->second;
}

std::optional<double> ClusterImpl::throughput(const std::string& target, const Job& job) {
    std::lock_guard lock(model_mutex);
    const auto found = throughput_ema.find(target + "|" + std::to_string(job.kernel->key));
    if (found == throughput_ema.end()) return std::nullopt;
    return found->second;
}

void ClusterImpl::count_device(const std::string& key, bool slice, int inflight_delta) {
    std::lock_guard lock(model_mutex);
    auto& counters = device_counters[key];
    if (inflight_delta > 0) ++counters.inflight;
    if (inflight_delta < 0 && counters.inflight > 0) --counters.inflight;
    if (inflight_delta < 0) ++(slice ? counters.slices : counters.dispatches);
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
    job->assigned_device.reset();
    enqueue(job, true);  // Its dependents are already ordered behind its Event.
}

void ClusterImpl::refresh_local_targets() {
    local_targets.clear();
    for (const auto& id : runtime.devices()) {
        auto info = runtime.device_info(id);
        local_targets.push_back(
            {id, device_key(id), info ? info.value().name() : "device " + std::to_string(id.index)});
    }
}

Result<Kernel> ClusterImpl::local_kernel(ClusterKernelState& kernel, const DeviceId& device) {
    const auto generation = local_generation.load();
    const auto key = device_key(device);
    std::lock_guard lock(kernel.mutex);
    if (kernel.local_generation != generation) {
        kernel.local.clear();
        kernel.unsupported_local.clear();
        kernel.local_generation = generation;
    }
    if (const auto found = kernel.local.find(key); found != kernel.local.end()) return found->second;
    if (kernel.unsupported_local.contains(key))
        return Error(ErrorCode::UnsupportedFeature, "kernel is not supported on this local device");
    auto loaded = runtime.load_kernel(kernel.words, kernel.entry_point, kernel.metadata, device);
    if (!loaded) {
        kernel.unsupported_local.insert(key);
        return std::move(loaded).error();
    }
    kernel.local.emplace(key, loaded.value());
    return std::move(loaded).value();
}

std::vector<Target> ClusterImpl::candidates(const Job& job) {
    std::vector<Target> result;
    {
        std::shared_lock lock(runtime_mutex);
        for (const auto& target : local_targets) {
            std::lock_guard kernel_lock(job.kernel->mutex);
            if (job.kernel->local_generation == local_generation.load() &&
                job.kernel->unsupported_local.contains(target.key))
                continue;
            result.push_back({nullptr, target.id, target.key});
        }
    }
    if (job.force_local || options.policy == OffloadPolicy::LocalOnly) return result;
    std::lock_guard lock(workers_mutex);
    for (const auto& link : workers) {
        if (!link->connected.load(std::memory_order_acquire)) continue;
        std::lock_guard link_lock(link->mutex);
        if (!link->unsupported_kernels.contains(job.kernel->key))
            result.push_back({link, std::nullopt, link->endpoint});
    }
    return result;
}

Target ClusterImpl::choose_target(const Job& job) {
    auto options_list = candidates(job);
    if (options_list.empty()) return {};
    const auto inflight = [&](const Target& target) -> double {
        if (target.link) return target.link->inflight.load();
        std::lock_guard lock(model_mutex);
        const auto found = device_counters.find(target.key);
        return found == device_counters.end() ? 0.0 : found->second.inflight;
    };
    if (options.policy == OffloadPolicy::PreferRemote) {
        std::optional<Target> best_remote;
        for (const auto& target : options_list)
            if (target.link && (!best_remote || inflight(target) < inflight(*best_remote)))
                best_remote = target;
        if (best_remote) return *best_remote;
    }
    // Measure every local device once, explore each unmeasured worker occasionally, then pick the
    // lowest expected completion time including work already queued on each target.
    for (const auto& target : options_list)
        if (!target.link && !predicted(target.key, job)) return target;
    std::optional<Target> best;
    double best_cost = 0;
    for (const auto& target : options_list) {
        const auto latency = predicted(target.key, job);
        if (!latency) {
            if (options.policy == OffloadPolicy::PreferRemote) return target;
            std::lock_guard lock(model_mutex);
            if (explore_counter++ % 8 == 0) return target;
            continue;
        }
        const auto cost = *latency * (1.0 + inflight(target));
        if (!best || cost < best_cost) {
            best = target;
            best_cost = cost;
        }
    }
    return best ? *best : options_list.front();
}

void ClusterImpl::ensure_host_current(const Job& job) {
    std::shared_lock lock(runtime_mutex);
    for (const auto& buffer : job.buffers)
        for (const auto& target : local_targets) {
            auto readback = runtime.readback(buffer, target.id);
            if (readback) (void)readback.value().wait();
        }
}

bool ClusterImpl::try_split(const std::shared_ptr<Job>& job) {
    if (job->split_element_bytes == 0 || job->split || job->force_local) return false;
    const auto count = job->global[0];
    if (count < options.split_min_elements || job->global[1] != 1 || job->global[2] != 1) return false;
    auto targets = candidates(*job);
    if (targets.size() < 2) return false;

    // Weight targets by measured throughput; unmeasured ones get the average so they are explored.
    std::vector<double> rates;
    double known_sum = 0;
    std::size_t known = 0;
    for (const auto& target : targets) {
        const auto rate = throughput(target.key, *job);
        rates.push_back(rate.value_or(0));
        if (rate) {
            known_sum += *rate;
            ++known;
        }
    }
    const double fallback = known == 0 ? 1.0 : known_sum / static_cast<double>(known);
    for (auto& rate : rates)
        if (rate <= 0) rate = fallback;
    std::vector<std::size_t> order(targets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto left, auto right) { return rates[left] > rates[right]; });
    // Use the k fastest targets that minimize the estimated makespan, if that beats one target.
    // ponytail: linear model (count / summed rate + fixed overhead per extra slice); replace with
    // measured per-target fixed latency if slices of very different sizes mispredict.
    const auto overhead = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(options.split_overhead).count());
    std::size_t best_k = 1;
    double rate_sum = 0;
    double best_time = static_cast<double>(count) / rates[order[0]];
    const double single = best_time;
    for (std::size_t k = 1; k <= order.size(); ++k) {
        rate_sum += rates[order[k - 1]];
        const double time = static_cast<double>(count) / rate_sum + overhead * static_cast<double>(k - 1);
        if (time < best_time) {
            best_time = time;
            best_k = k;
        }
    }
    if (best_k < 2 || best_time > 0.95 * single) return false;
    // The model assumes devices run independently; two drivers of one GPU, or a CPU renderer
    // sharing memory bandwidth, do not. Split only after one device has been measured alone, and
    // stop once finished splits measure no faster, re-probing every 16th time in case load changes.
    std::optional<double> alone;
    for (const auto& target : targets)
        if (const auto latency = predicted(target.key, *job); latency && (!alone || *latency < *alone))
            alone = latency;
    if (!alone) return false;
    if (const auto measured = predicted("split", *job); measured && *measured >= 0.95 * *alone) {
        std::lock_guard lock(model_mutex);
        if (++split_declines[model_key("split", *job)] % 16 != 0) return false;
    }

    // Slice boundaries are multiples of the workgroup size so no two devices write one element.
    const std::uint64_t granule = std::max<std::uint32_t>(job->kernel->metadata.local_size[0], 1);
    std::vector<std::pair<std::size_t, std::uint64_t>> plan;  // (target, element count)
    double used_rate = 0;
    for (std::size_t k = 0; k < best_k; ++k) used_rate += rates[order[k]];
    std::uint64_t assigned = 0;
    for (std::size_t k = 0; k < best_k; ++k) {
        std::uint64_t share =
            k + 1 == best_k
                ? count - assigned
                : static_cast<std::uint64_t>(static_cast<double>(count) * rates[order[k]] / used_rate) /
                      granule * granule;
        share = std::min(share, count - assigned);
        if (share == 0) continue;
        plan.emplace_back(order[k], share);
        assigned += share;
    }
    if (plan.size() < 2) return false;

    ensure_host_current(*job);
    auto state = std::make_shared<SplitState>();
    state->parent = job;
    state->element_bytes = job->split_element_bytes;
    state->started = Clock::now();
    for (std::size_t i = 0; i < job->buffers.size(); ++i) {
        if (job->accesses[i] == KernelAccess::Read) continue;
        auto version = job->buffers[i].host_version();
        auto contents = job->buffers[i].read_bytes();
        if (!version || !contents) return false;
        state->outputs.push_back(i);
        state->versions.push_back(version.value());
        state->assembled.push_back(std::move(contents).value());
    }
    state->remaining = plan.size();
    split_dispatches.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t base = 0;
    for (const auto& [target_index, share] : plan) {
        auto slice = std::make_shared<Job>(*job);
        slice->promise = EventPromise();
        slice->dependencies.clear();
        slice->split = state;
        slice->slice_base = base;
        slice->slice_count = share;
        slice->global = {share, 1, 1};
        std::memcpy(slice->push.data() + *job->kernel->base_offset, &base, sizeof(std::uint32_t));
        base += share;
        start_on(slice, targets[target_index]);
    }
    return true;
}

void ClusterImpl::start_on(const std::shared_ptr<Job>& job, const Target& target) {
    job->started = Clock::now();
    if (target.link) {
        target.link->inflight.fetch_add(1);
        std::lock_guard lock(target.link->mutex);
        target.link->jobs.push_back(job);
        target.link->cv.notify_one();
        return;
    }
    if (target.device) {
        run_local(job, *target.device);
        return;
    }
    auto error = Error(ErrorCode::NoEligibleDevice, "no local or remote target can run this kernel");
    if (job->split)
        finish_slice(job, error, {}, {});
    else
        job->promise.complete(error);
}

void ClusterImpl::finish_slice(const std::shared_ptr<Job>& slice, Status status,
                               const std::vector<std::span<const std::byte>>& ranges,
                               const std::string& target_key) {
    auto& state = *slice->split;
    if (status) {
        slices.fetch_add(1, std::memory_order_relaxed);
        if (!target_key.empty()) record_throughput(target_key, *slice, Clock::now() - slice->started);
    }
    std::unique_lock lock(state.mutex);
    if (status) {
        for (std::size_t k = 0; k < state.outputs.size() && k < ranges.size(); ++k) {
            const auto [offset, size] = slice_bytes(*slice, state.assembled[k].size());
            std::memcpy(state.assembled[k].data() + offset, ranges[k].data(),
                        std::min(size, ranges[k].size()));
        }
    } else if (!state.error) {
        state.error = status.error();
    }
    if (--state.remaining != 0) return;
    lock.unlock();
    // Every slice reported: commit the merged outputs once, or nothing at all.
    if (state.error) {
        state.parent->promise.complete(Status(*state.error));
        return;
    }
    for (std::size_t k = 0; k < state.outputs.size(); ++k)
        (void)state.parent->buffers[state.outputs[k]].write_bytes_if_version(state.assembled[k],
                                                                             state.versions[k]);
    record_latency("split", *state.parent, Clock::now() - state.started);
    state.parent->promise.complete(ok_status());
}

void ClusterImpl::dispatcher_loop() {
    for (;;) {
        std::shared_ptr<Job> job;
        std::vector<BufferId> released_now;
        bool rebuild = false;
        {
            std::unique_lock lock(queue_mutex);
            queue_cv.wait_for(lock, std::chrono::milliseconds(rescan_requested ? 20 : 1'000), [&] {
                return stopping || !queue.empty() || !released.empty() ||
                       (rescan_requested && local_inflight.load() == 0);
            });
            released_now.swap(released);
            if (rescan_requested && local_inflight.load() == 0) {
                rescan_requested = false;
                rebuild = true;
            } else if (!queue.empty()) {
                job = std::move(queue.front());
                queue.pop_front();
            } else if (stopping && released_now.empty()) {
                return;
            }
        }
        if (rebuild) rebuild_local(false, 0);
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
        if (job->local_generation != 0 && job->local_generation == local_generation.load())
            rebuild_local(true, job->local_generation);  // A local attempt observed device loss.
        if (try_split(job)) continue;
        Target target = choose_target(*job);
        if (job->force_local && job->assigned_device)
            target = {nullptr, job->assigned_device, device_key(*job->assigned_device)};
        start_on(job, target);
    }
}

void ClusterImpl::rebuild_local(bool after_loss, std::uint64_t observed_generation) {
    std::unique_lock lock(runtime_mutex);
    if (after_loss && local_generation.load() != observed_generation) return;  // Already recovered.
    const auto rebuilt = after_loss ? runtime.recover() : runtime.rebuild_devices();
    if (!rebuilt) return;
    local_generation.fetch_add(1);
    refresh_local_targets();
    (after_loss ? local_recoveries : device_rescans).fetch_add(1, std::memory_order_relaxed);
}

void ClusterImpl::monitor_loop() {
    device_fingerprint = physical_device_fingerprint();
    for (;;) {
        {
            std::unique_lock lock(queue_mutex);
            monitor_cv.wait_for(lock, options.device_rescan_interval, [&] { return stopping || shut_down; });
            if (stopping || shut_down) return;
        }
        auto now = physical_device_fingerprint();
        if (now.empty() || now == device_fingerprint) continue;
        device_fingerprint = std::move(now);
        {
            std::lock_guard lock(queue_mutex);
            rescan_requested = true;
        }
        queue_cv.notify_all();
    }
}

void ClusterImpl::run_local(const std::shared_ptr<Job>& job, const DeviceId& device) {
    const auto started = Clock::now();
    const auto generation = local_generation.load();
    const auto key = device_key(device);
    const bool slice = job->split != nullptr;
    Result<Event> submitted = Error(ErrorCode::InvalidArgument, "local dispatch was not submitted");
    {
        std::shared_lock lock(runtime_mutex);
        auto kernel = local_kernel(*job->kernel, device);
        if (!kernel) {
            submitted = std::move(kernel).error();
        } else {
            auto builder = runtime.dispatch(kernel.value());
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
            // A slice leaves its outputs on the device; only its own range is read back below.
            if (slice) builder.defer_readback();
            submitted = builder.submit();
        }
    }
    const auto retry_after_loss = [this, job, generation](const Error& error) {
        if (error.code != ErrorCode::DeviceLost || job->local_retried) return false;
        job->local_retried = true;
        job->local_generation = generation;
        job->force_local = true;
        job->assigned_device.reset();
        enqueue(job, true);
        return true;
    };
    const auto fail = [this, job](Status status) {
        if (job->split)
            finish_slice(job, std::move(status), {}, {});
        else
            job->promise.complete(std::move(status));
    };
    if (!submitted) {
        if (!retry_after_loss(submitted.error())) fail(Status(std::move(submitted).error()));
        return;
    }
    local_inflight.fetch_add(1);
    count_device(key, slice, +1);
    std::weak_ptr<ClusterImpl> weak = weak_from_this();
    (void)submitted.value().on_complete(
        [weak, job, device, key, slice, started, retry_after_loss, fail](Status status) {
            auto self = weak.lock();
            if (!self) {
                job->promise.complete(std::move(status));
                return;
            }
            const auto done_locally = [self, key, slice] {
                self->local_inflight.fetch_sub(1);
                self->count_device(key, slice, -1);
                self->queue_cv.notify_all();  // A pending rescan may now proceed.
            };
            if (!status) {
                done_locally();
                if (!retry_after_loss(status.error())) fail(std::move(status));
                return;
            }
            if (!slice) {
                done_locally();
                self->local_dispatches.fetch_add(1, std::memory_order_relaxed);
                self->record_latency(key, *job, Clock::now() - started);
                job->promise.complete(ok_status());
                return;
            }
            // Read back just this slice's range of every written buffer, without waiting here.
            auto ranges = std::make_shared<std::vector<std::vector<std::byte>>>();
            std::vector<Event> reads;
            Status read_status = ok_status();
            {
                std::shared_lock lock(self->runtime_mutex);
                for (const auto index : job->split->outputs) {
                    const auto [offset, size] = slice_bytes(*job, job->buffers[index].size_bytes());
                    ranges->emplace_back(size);
                    auto read = RuntimeReplicaAccess::read_replica(self->runtime, job->buffers[index], device,
                                                                   offset, ranges->back());
                    if (!read) {
                        read_status = Status(std::move(read).error());
                        break;
                    }
                    reads.push_back(std::move(read).value());
                }
            }
            auto all = read_status ? Event::when_all(reads) : Result<Event>(read_status.error());
            if (!all) {
                done_locally();
                fail(Status(std::move(all).error()));
                return;
            }
            (void)all.value().on_complete([weak, job, key, ranges, done_locally, fail](Status read) {
                auto inner = weak.lock();
                done_locally();
                if (!inner || !read) {
                    fail(read ? Status(Error(ErrorCode::Cancelled, "cluster is shutting down"))
                              : std::move(read));
                    return;
                }
                std::vector<std::span<const std::byte>> views(ranges->begin(), ranges->end());
                inner->finish_slice(job, ok_status(), views, key);
            });
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

RemoteOutcome ClusterImpl::run_remote(WorkerLink& link, remote::Socket& socket, Job& job,
                                      std::vector<std::vector<std::byte>>* collected) {
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
    if (!job.split) ensure_host_current(job);  // A split dispatch did this before slicing.
    for (std::size_t i = 0; i < job.buffers.size(); ++i) {
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
    if (collected) {
        // A slice returns only its own range; the worker's copy now differs from the host's.
        for (const auto output : job.split->outputs) {
            const auto found = std::find_if(outputs.begin(), outputs.end(),
                                            [&](const auto& item) { return item.first == output; });
            if (found == outputs.end()) return RemoteOutcome::Link;
            const auto [offset, size] = slice_bytes(job, found->second.size());
            collected->emplace_back(found->second.begin() + static_cast<std::ptrdiff_t>(offset),
                                    found->second.begin() + static_cast<std::ptrdiff_t>(offset + size));
            link.worker_versions.erase(buffer_key(job.buffers[output].id()));
        }
        return RemoteOutcome::Done;
    }
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
                std::vector<std::vector<std::byte>> collected;
                const auto outcome = run_remote(*link, *socket, *job, job->split ? &collected : nullptr);
                link->inflight.fetch_sub(1);
                if (outcome == RemoteOutcome::Done) {
                    {
                        std::lock_guard lock(link->mutex);
                        ++link->status.dispatches;
                    }
                    last_activity = Clock::now();
                    if (job->split) {
                        std::vector<std::span<const std::byte>> views(collected.begin(), collected.end());
                        finish_slice(job, ok_status(), views, link->endpoint);
                        continue;
                    }
                    record_latency(link->endpoint, *job, Clock::now() - started);
                    remote_dispatches.fetch_add(1, std::memory_order_relaxed);
                    job->promise.complete(ok_status());
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
    monitor_cv.notify_all();  // Wakes the device monitor, which exits once shut_down is set.
    if (monitor.joinable()) monitor.join();
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
    // Use every eligible local device unless the caller (or KORA_DEVICE) chose one.
    const bool all_local_devices = !options.local.device_policy && std::getenv("KORA_DEVICE") == nullptr;
    if (all_local_devices) options.local.device_policy = DevicePolicy::AllEligible;
    auto runtime = Runtime::create(options.local);
    if (!runtime) return std::move(runtime).error();
    auto impl = std::make_shared<detail::ClusterImpl>();
    impl->options = options;
    impl->all_local_devices = all_local_devices;
    impl->runtime = std::move(runtime).value();
    impl->refresh_local_targets();
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
    if (all_local_devices && options.device_rescan_interval.count() > 0)
        impl->monitor = std::thread([raw = impl.get()] { raw->monitor_loop(); });
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
    for (const auto& field : state->metadata.push_constants)
        if (field.name == "base" && field.type == "u32" && field.size == 4) state->base_offset = field.offset;
    // Validate on the first local device so errors surface here, not at the first dispatch; other
    // devices load the kernel when they are first chosen.
    std::shared_lock lock(impl_->runtime_mutex);
    if (impl_->local_targets.empty())
        return Error(ErrorCode::NoEligibleDevice, "local runtime has no device");
    auto local = impl_->local_kernel(*state, impl_->local_targets.front().id);
    if (!local) return std::move(local).error();
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

std::vector<LocalDeviceStatus> Cluster::local_devices() const {
    std::vector<LocalDeviceStatus> result;
    std::shared_lock lock(impl_->runtime_mutex);
    std::lock_guard model_lock(impl_->model_mutex);
    for (const auto& target : impl_->local_targets) {
        LocalDeviceStatus status{target.id, target.name};
        if (const auto found = impl_->device_counters.find(target.key);
            found != impl_->device_counters.end()) {
            status.dispatches = found->second.dispatches;
            status.slices = found->second.slices;
            status.inflight = found->second.inflight;
        }
        result.push_back(std::move(status));
    }
    return result;
}

ClusterStats Cluster::stats() const {
    ClusterStats stats;
    stats.local_dispatches = impl_->local_dispatches.load();
    stats.remote_dispatches = impl_->remote_dispatches.load();
    stats.failovers = impl_->failovers.load();
    stats.local_recoveries = impl_->local_recoveries.load();
    stats.split_dispatches = impl_->split_dispatches.load();
    stats.slices = impl_->slices.load();
    stats.device_rescans = impl_->device_rescans.load();
    return stats;
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
ClusterDispatch& ClusterDispatch::split(std::uint32_t output_element_bytes) {
    split_element_bytes_ = output_element_bytes;
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
    if (split_element_bytes_ != 0 && !kernel_->base_offset)
        return Error(ErrorCode::InvalidArgument, "split() requires a u32 push-constant field named 'base'");
    if (split_element_bytes_ != 0 && global_[0] > std::numeric_limits<std::uint32_t>::max())
        return Error(ErrorCode::InvalidArgument, "split() supports at most 2^32 elements");
    auto job = std::make_shared<detail::Job>();
    job->kernel = kernel_;
    job->global = global_;
    job->buffers = buffers_;
    job->accesses = accesses_;
    job->push = push_;
    job->split_element_bytes = split_element_bytes_;
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
