# CPU executor

`kora::CpuExecutor` is a fixed-size C++ task pool for CPU-native work. It returns the same `Event`
abstraction used by other asynchronous operations, so task failures and exceptions can be observed
through `wait()` without escaping worker threads. `Runtime::register_cpu_fallback()` associates a
stable semantic key with a native implementation. `Runtime::run_cpu_fallback()` dispatches that
implementation through the same `Event` completion model as GPU work. This is explicit fallback
dispatch; automatic scheduler selection and `TaskGraph` execution remain separate work.

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
| `Runtime::register_cpu_fallback(key, fn)` | Registers or replaces a non-empty semantic key and callable for this Runtime. |
| `Runtime::run_cpu_fallback(key, buffers, accesses, push)` | Makes read inputs host-visible, maps declared buffers, invokes the callback on the CPU pool, and returns an `Event` for completion/failure. |
| `CpuTaskContext::read(index)` | Returns a read-only byte span only for `Read` or `ReadWrite` declarations. |
| `CpuTaskContext::write(index)` | Returns a writable byte span only for `Write` or `ReadWrite` declarations. |
| `CpuTaskContext::read_write(index)` | Returns a writable byte span only for `ReadWrite` declarations. |
| `CpuTaskContext::push_constants()` | Returns the push-constant bytes supplied to the fallback invocation. |

The fallback bridge calls `Runtime::readback()` for read and read-write inputs before mapping them, so
device-newer contents are downloaded and invalidated non-coherent ranges are handled by the transfer
engine. Writable mappings are released before the returned Event completes; their host versions then
force the next GPU consumer to upload the updated shadow. The callback owns no mapping beyond its call,
and spans must not escape it. Its successful or failed `Status` becomes the fallback Event status;
exceptions are caught and reported as `InternalInvariantViolation`. Runtime shutdown drains the CPU pool.

Integration coverage runs a vector-add GPU dispatch, a registered CPU mutation, and two more GPU
dispatches, verifying final contents and transfer counts. This verifies the common completion path and
host/device visibility for that chain. Registration is per Runtime, not serialized, and callers declare
CPU access modes explicitly; the implementation does not infer them from kernel metadata. The callback
registry currently provides explicit keyed dispatch only, not automatic scheduler selection or
`TaskGraph` execution.
