#pragma once
/// @file cpu_executor.hpp
/// Fixed-size CPU worker pool with priority lanes and cooperative work stealing.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "kora/event.hpp"
#include "kora/options.hpp"

namespace kora {

enum class CpuTaskPriority : std::uint8_t { Low, Normal, High, Critical };

struct CpuExecutorOptions {
    /// Zero selects max(1, hardware_concurrency - reserved_host_threads).
    std::size_t worker_count = 0;
    std::size_t reserved_host_threads = 1;
};

/// Fixed-size CPU task pool. Local mutex-protected deques provide bounded, simple work stealing.
/// Workers pop locally from the back and steal from other workers from the front. Submissions do
/// not block on queue capacity. Worker-side wait() cooperatively executes queued work to avoid
/// recursive submit-and-wait deadlocks, including with one worker.
class CpuExecutor final {
public:
    using Task = std::function<Status()>;

    static Result<std::unique_ptr<CpuExecutor>> create(const CpuExecutorOptions& options = {});
    ~CpuExecutor();
    CpuExecutor(const CpuExecutor&) = delete;
    CpuExecutor& operator=(const CpuExecutor&) = delete;
    CpuExecutor(CpuExecutor&&) = delete;
    CpuExecutor& operator=(CpuExecutor&&) = delete;

    /// Enqueue work and return an Event completed with its Status or captured exception.
    Result<Event> submit(Task task, CpuTaskPriority priority = CpuTaskPriority::Normal);
    /// A worker waiting on this executor helps run work; other callers block on the Event directly.
    Status wait(const Event& event);
    /// Stop accepting work, then either drain queued tasks (Graceful) or cancel queued tasks
    /// (CancelPending), and join workers. Active tasks are never interrupted. Calling shutdown from
    /// one of this executor's tasks is rejected.
    Status shutdown(ShutdownMode mode = ShutdownMode::Graceful);

    std::size_t worker_count() const noexcept;

private:
    struct State;
    struct WorkItem;
    explicit CpuExecutor(CpuExecutorOptions options);
    bool run_one(std::size_t worker_index);
    void worker_loop(std::size_t worker_index) noexcept;
    void execute_task(WorkItem&& item) noexcept;

    CpuExecutorOptions options_;
    std::unique_ptr<State> state_;
};

}  // namespace kora
