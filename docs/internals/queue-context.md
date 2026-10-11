# QueueContext

`vk::QueueContext` is the sole host-side submission owner for one `VkQueue`. Any API thread may
enqueue an owned submission record; one service thread dequeues and calls `vkQueueSubmit2` serially
when synchronization2 is enabled, otherwise it uses `vkQueueSubmit`. A
second worker waits on a private timeline semaphore and delivers completion callbacks after the GPU
has completed each successful submission.

See {doc}`completion` for the processor's bounded wait/poll behavior, resource-lifetime rules, and
shutdown validation.

## Rationale and invariants

Vulkan queue submission requires external synchronization. Producers never call `vkQueueSubmit`
directly through this service: they transfer owned arrays and a callback to the MPSC ready queue. The
service serializes requests in its accepted order and appends an internal monotonic timeline signal to
each Vulkan submit. The completion worker waits for each signal in order, so callbacks mean GPU
completion, not just successful enqueue or `vkQueueSubmit` return.

The preferred context mode uses a private timeline semaphore. If timeline semaphores are unavailable or
the runtime's `force_legacy_sync` validation option is set, it uses a bounded pool of eight reusable
fences and a software completion counter. Queue handles passed to it must be exclusively owned
by this context for the context lifetime; callers must not submit directly to that queue or create a
second service for it. Submission records own their wait semaphores, stage masks, command-buffer
handles, signal semaphores, timeline values, optional fence handle, and callback. Underlying Vulkan
objects (especially command buffers and semaphores) must remain valid until their completion callback.

Wait stage count must equal wait semaphore count. Optional timeline value arrays, when supplied, must
match the corresponding semaphore arrays; omitted arrays mean zero values (appropriate for binary
semaphores and a zero-valued timeline wait). For a timeline signal semaphore, callers must provide a
valid increasing value. In fence fallback mode, only binary external semaphores (zero values) are
accepted. The service does not merge arbitrary extension `pNext` chains; this keeps the
owned submission contract explicit and guarantees it can attach timeline metadata for its own signal.

Callbacks run serially on the completion thread, in queue order. Callback exceptions are contained and
do not terminate the worker. Callbacks may enqueue more work while the context is accepting, but must
not destroy the context or call `shutdown()` from a QueueContext worker callback.

## Shutdown

`Graceful` atomically stops acceptance and submits every request already accepted. It then waits for
all successful submissions and invokes their completion callbacks before returning. `CancelPending`
stops acceptance, cancels jobs still in the ready queue with `ErrorCode::Cancelled`, but does not
interrupt a queue submit already in progress or GPU work already accepted; those still reach their
completion callback after safe completion. Destruction uses graceful shutdown and does not throw.

## API reference

| Function | Contract |
| --- | --- |
| `QueueContext::create(api, device, queue)` | Creates a private timeline semaphore and starts the submission and completion workers. Missing handles/dispatch return `InvalidArgument`; unavailable timeline support returns `UnsupportedFeature`. |
| `submit(Submission)` | Validates array counts, copies ownership into the MPSC queue, and returns a context-local ID. Once shutdown begins, returns `Cancelled`. |
| `shutdown(Graceful)` | Rejects future submissions, drains accepted requests, waits for GPU completion, and joins both workers. |
| `shutdown(CancelPending)` | Rejects future submissions, cancels requests not yet submitted, and still waits for in-flight/submitted work to complete safely. |

`Submission::on_complete(id, status)` is called exactly once for every accepted request. A successful
status means its internal timeline signal completed. A failed `vkQueueSubmit` result is returned as a
Vulkan-bearing error; a request cancelled before submission receives `ErrorCode::Cancelled`.

## Validation

Mock-dispatch stress submits 2,048 requests concurrently from 16 producer threads and checks that no
two `vkQueueSubmit` calls overlap, internal timeline metadata is well-formed, callbacks all complete,
and graceful shutdown rejects later work. A deterministic blocked-submit test verifies CancelPending
cancels queued jobs while allowing the in-flight operation to finish; a concurrent-shutdown test
checks that multiple callers safely join the workers. Lavapipe additionally processes 1,024 real queue
submissions from 16 producers and waits for all GPU completion callbacks.
