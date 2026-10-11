# CPU executor

`kora::CpuExecutor` is a fixed-size C++ task pool for CPU-native work. It returns the same `Event`
abstraction used by other asynchronous operations, so task failures and exceptions can be observed
through `wait()` without escaping worker threads. This issue establishes the executor primitive;
kernel fallback registration and CPU/GPU graph scheduling are separate follow-up work.

## Scheduling and worker count

Each worker owns four mutex-protected priority deques. A worker pops its own newest item from the
highest non-empty priority lane; when idle it steals the oldest item from another worker's highest
non-empty lane. This lock-based design is intentionally simpler than a lock-free Chase-Lev deque and
has bounded, explicit synchronization costs. External submissions are distributed round-robin.
Priority lanes are strict, so sustained high-priority traffic can delay lower-priority work.

When `worker_count` is zero, the pool uses
`max(1, hardware_concurrency - reserved_host_threads)`. The default reserves one hardware thread for
the application/runtime when the reported processor count permits it. An explicit worker count is
accepted from 1 through 1024. The executor does not set CPU affinity or silently grow its pool.

## Lifecycle and recursion

`submit()` enqueues a callback and returns its `Event`. Callback `Status` failures are forwarded;
thrown standard and non-standard exceptions become `InternalInvariantViolation` statuses. `shutdown()`
stops accepting external submissions, drains queued work (including recursive submissions made by
already-running tasks), and joins every worker. The destructor performs the same graceful drain.

Calling `executor.wait(event)` from one of this executor's worker tasks cooperatively runs queued work
while waiting. This prevents the classic fixed-pool deadlock where all workers block on child tasks
that remain queued. Calls to `Event::wait()` do not perform this helping behavior; recursive worker
tasks that synchronously wait for children must use the executor's `wait()` method. Calling
`shutdown()` from one of the executor's own callbacks is rejected because joining the current worker
would deadlock.

## Function reference

| API | Contract |
| --- | --- |
| `CpuExecutor::create(options)` | Creates the fixed worker set; invalid worker counts or thread/allocation failures return `Result` errors. |
| `CpuExecutor::submit(task, priority)` | Enqueues a non-empty callback and returns an `Event`; submissions after shutdown are rejected with `Cancelled`. |
| `CpuExecutor::wait(event)` | Waits for an event, helping execute queued tasks when called by a worker of this pool. |
| `CpuExecutor::shutdown()` | Stops external admission, drains outstanding callbacks, and joins workers; repeated calls are safe. |
| `CpuExecutor::worker_count()` | Returns the configured fixed worker count. |

Unit tests cover priority ordering, exception conversion, single-worker recursive submit-and-wait,
shutdown draining, and one million tiny tasks with work stealing. The executor currently runs explicit
native callbacks only; it does not yet register kernel fallbacks or schedule `TaskGraph` nodes.
