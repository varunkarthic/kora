#include "kora/cpu_executor.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "core/event_state.hpp"

namespace kora {
namespace {

constexpr std::size_t kPriorityLaneCount = 4;
constexpr std::size_t kMaximumWorkerCount = 1024;

thread_local CpuExecutor* active_executor = nullptr;
thread_local std::size_t active_worker_index = 0;

std::size_t lane_index(CpuTaskPriority priority) {
    return static_cast<std::size_t>(priority);
}

std::size_t default_worker_count(std::size_t reserved_host_threads) noexcept {
    const auto hardware = static_cast<std::size_t>(std::thread::hardware_concurrency());
    if (hardware == 0 || hardware <= reserved_host_threads) return 1;
    return std::max<std::size_t>(1, hardware - reserved_host_threads);
}

Error exception_error() {
    try {
        throw;
    } catch (const std::exception& error) {
        return Error(ErrorCode::InternalInvariantViolation,
                     std::string("CPU task threw an exception: ") + error.what());
    } catch (...) {
        return Error(ErrorCode::InternalInvariantViolation, "CPU task threw a non-standard exception");
    }
}

}  // namespace

struct CpuExecutor::WorkItem {
    Task task;
    std::shared_ptr<detail::EventPromise> promise;
    CpuTaskPriority priority = CpuTaskPriority::Normal;
};

struct CpuExecutor::State {
    struct Worker {
        std::mutex mutex;
        std::array<std::deque<WorkItem>, kPriorityLaneCount> lanes;
    };

    explicit State(std::size_t count) {
        workers.reserve(count);
        for (std::size_t index = 0; index < count; ++index) workers.push_back(std::make_unique<Worker>());
        threads.reserve(count);
    }

    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<std::thread> threads;
    std::atomic<std::size_t> queued{0};
    std::atomic<std::size_t> outstanding{0};
    std::atomic<std::size_t> next_worker{0};
    std::mutex lifecycle_mutex;
    std::mutex shutdown_mutex;
    std::condition_variable changed;
    bool accepting = true;
    bool stopping = false;
};

CpuExecutor::CpuExecutor(CpuExecutorOptions options) : options_(options) {
    const auto count = options_.worker_count == 0 ? default_worker_count(options_.reserved_host_threads)
                                                  : options_.worker_count;
    options_.worker_count = count;
    state_ = std::make_unique<State>(count);
}

Result<std::unique_ptr<CpuExecutor>> CpuExecutor::create(const CpuExecutorOptions& options) {
    const auto count = options.worker_count == 0 ? default_worker_count(options.reserved_host_threads)
                                                 : options.worker_count;
    if (count == 0 || count > kMaximumWorkerCount)
        return Error(ErrorCode::InvalidArgument, "CPU executor worker count is outside the supported range");

    std::unique_ptr<CpuExecutor> executor;
    try {
        executor.reset(new CpuExecutor(options));
        for (std::size_t index = 0; index < count; ++index) {
            executor->state_->threads.emplace_back(
                [instance = executor.get(), index] { instance->worker_loop(index); });
        }
    } catch (const std::bad_alloc&) {
        if (executor) (void)executor->shutdown();
        return Error(ErrorCode::AllocationFailed, "unable to allocate CPU executor worker state");
    } catch (const std::system_error& error) {
        if (executor) (void)executor->shutdown();
        return Error(ErrorCode::AllocationFailed,
                     std::string("unable to start CPU executor worker: ") + error.what());
    }
    return executor;
}

CpuExecutor::~CpuExecutor() {
    (void)shutdown();
}

std::size_t CpuExecutor::worker_count() const noexcept {
    return state_ ? state_->workers.size() : 0;
}

Result<Event> CpuExecutor::submit(Task task, CpuTaskPriority priority) {
    if (!state_) return Error(ErrorCode::InvalidArgument, "CPU executor is not initialized");
    if (!task) return Error(ErrorCode::InvalidArgument, "CPU task callback is empty");
    if (lane_index(priority) >= kPriorityLaneCount)
        return Error(ErrorCode::InvalidArgument, "CPU task priority is invalid");

    std::shared_ptr<detail::EventPromise> promise;
    try {
        promise = std::make_shared<detail::EventPromise>();
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to allocate CPU task event state");
    }
    const auto event = promise->event();

    {
        std::lock_guard lifecycle_lock(state_->lifecycle_mutex);
        const bool nested_submission = active_executor == this;
        if (!state_->accepting && !nested_submission)
            return Error(ErrorCode::Cancelled, "CPU executor is shutting down and rejects new work");
        const auto index = nested_submission ? active_worker_index
                                             : state_->next_worker.fetch_add(1, std::memory_order_relaxed) %
                                                   state_->workers.size();
        auto& worker = *state_->workers[index];
        try {
            std::lock_guard worker_lock(worker.mutex);
            worker.lanes[lane_index(priority)].push_back(WorkItem{std::move(task), promise, priority});
        } catch (const std::bad_alloc&) {
            return Error(ErrorCode::AllocationFailed, "unable to enqueue CPU task");
        } catch (const std::length_error&) {
            return Error(ErrorCode::Overflow, "CPU executor queue exceeds its container limit");
        }
        state_->queued.fetch_add(1, std::memory_order_release);
        state_->outstanding.fetch_add(1, std::memory_order_release);
    }
    state_->changed.notify_one();
    return event;
}

bool CpuExecutor::run_one(std::size_t worker_index) {
    if (!state_ || worker_index >= state_->workers.size()) return false;
    WorkItem item;
    bool found = false;
    auto& local = *state_->workers[worker_index];
    {
        std::lock_guard lock(local.mutex);
        for (std::size_t lane = kPriorityLaneCount; lane > 0; --lane) {
            auto& queue = local.lanes[lane - 1];
            if (queue.empty()) continue;
            item = std::move(queue.back());
            queue.pop_back();
            found = true;
            break;
        }
    }
    if (!found) {
        for (std::size_t offset = 1; offset < state_->workers.size() && !found; ++offset) {
            auto& victim = *state_->workers[(worker_index + offset) % state_->workers.size()];
            std::lock_guard lock(victim.mutex);
            for (std::size_t lane = kPriorityLaneCount; lane > 0; --lane) {
                auto& queue = victim.lanes[lane - 1];
                if (queue.empty()) continue;
                item = std::move(queue.front());
                queue.pop_front();
                found = true;
                break;
            }
        }
    }
    if (!found) return false;
    state_->queued.fetch_sub(1, std::memory_order_acq_rel);
    execute_task(std::move(item));
    return true;
}

void CpuExecutor::execute_task(WorkItem&& item) noexcept {
    Status status;
    try {
        status = item.task();
    } catch (...) {
        status = exception_error();
    }
    item.promise->complete(std::move(status));
    state_->outstanding.fetch_sub(1, std::memory_order_acq_rel);
    state_->changed.notify_all();
}

void CpuExecutor::worker_loop(std::size_t worker_index) noexcept {
    auto* previous_executor = active_executor;
    const auto previous_index = active_worker_index;
    active_executor = this;
    active_worker_index = worker_index;
    for (;;) {
        if (run_one(worker_index)) continue;
        std::unique_lock lock(state_->lifecycle_mutex);
        state_->changed.wait(lock, [this] {
            return state_->queued.load(std::memory_order_acquire) != 0 ||
                   (state_->stopping && state_->outstanding.load(std::memory_order_acquire) == 0);
        });
        if (state_->stopping && state_->outstanding.load(std::memory_order_acquire) == 0) break;
    }
    active_executor = previous_executor;
    active_worker_index = previous_index;
}

Status CpuExecutor::wait(const Event& event) {
    if (!event.valid()) return Error(ErrorCode::InvalidArgument, "cannot wait on an invalid Event");
    if (active_executor != this) return event.wait();
    while (!event.done()) {
        if (run_one(active_worker_index)) continue;
        std::unique_lock lock(state_->lifecycle_mutex);
        state_->changed.wait_for(lock, std::chrono::milliseconds(1), [this, &event] {
            return event.done() || state_->queued.load(std::memory_order_acquire) != 0 ||
                   (state_->stopping && state_->outstanding.load(std::memory_order_acquire) == 0);
        });
    }
    return event.wait();
}

Status CpuExecutor::shutdown() {
    if (!state_) return ok_status();
    if (active_executor == this)
        return Error(ErrorCode::InvalidArgument, "CPU executor cannot be shut down by one of its tasks");
    std::lock_guard shutdown_lock(state_->shutdown_mutex);
    {
        std::lock_guard lock(state_->lifecycle_mutex);
        state_->accepting = false;
        state_->stopping = true;
    }
    state_->changed.notify_all();
    for (auto& thread : state_->threads) {
        if (thread.joinable()) thread.join();
    }
    return ok_status();
}

}  // namespace kora
