# Completion processor

`vk::QueueContext` owns a dedicated completion processor for one Vulkan queue. The submission worker
signals a private monotonically increasing timeline value for every successful queue submission and
enqueues its completion callback. The processor waits for those values in queue order and invokes each
callback only after the GPU has reached the associated value.

## Lifetime and shutdown invariants

- A successful completion callback means the private queue timeline reached the submission's value;
  enqueue acceptance and `vkQueueSubmit` success alone are not completion.
- Each timeline wait uses a finite polling interval. `VK_TIMEOUT` only ends a polling interval: the
  processor queries the counter and continues waiting if the target is not complete. It never treats
  timeout as success or releases callback-owned resources on timeout.
- Callback delivery remains ordered by queue submission. Exceptions are contained so one callback
  cannot terminate processing of later completions.
- The queue context owns the private timeline semaphore and joins the completion processor before
  destroying that semaphore. `Graceful` shutdown first joins the submitter, which marks submission
  production finished; the processor drains every queued completion before returning.
- `CancelPending` cancels requests that have not reached queue submission. Already-submitted work is
  still awaited and receives its completion callback. Cancellation does not reclaim GPU-visible data
  early.
- The Vulkan device and queue must outlive the context and its worker threads. Callers must not destroy
  the context from either worker callback; shutdown detects this and returns `InvalidArgument`.

## Completion flow

```text
producer -> accepted submission -> serialized submit + timeline signal
                                    -> completion queue
                                    -> timed timeline wait / counter poll
                                    -> callback -> release deferred resources
shutdown -> reject new work -> stop submitter -> drain completions -> join -> destroy semaphore
```

## API reference

| Function | Contract |
| --- | --- |
| `QueueContext::completed_value()` | Samples the private timeline counter; Vulkan query failures are returned as structured errors. |
| `QueueContext::wait_for_value(value, timeout)` | Waits for one private timeline threshold; returns `Timeout` if the caller's deadline expires. Value zero is already complete. |
| `QueueContext::shutdown(Graceful)` | Stops acceptance, submits accepted requests, waits for all successful GPU work, delivers callbacks, and joins workers. |
| `QueueContext::shutdown(CancelPending)` | Stops acceptance, cancels queued requests, then drains callbacks for work already submitted to Vulkan. |

The finite processor poll interval is internal and does not expose periodic progress notifications. A
device-lost or other Vulkan wait/query error is delivered to the affected callback as a failure; callers
must still follow Vulkan device-loss teardown rules.

## Validation

Mock-dispatch tests cover worker shutdown races, callback serialization, queue-submit failure, and
pending cancellation. The integration suite submits 1,024 concurrent real Vulkan operations and
repeatedly creates, submits to, and gracefully destroys queue contexts, asserting each callback ran
before its context's destruction completed. This verifies lifecycle behavior on the available test
device; other driver/device behavior is not implied.
