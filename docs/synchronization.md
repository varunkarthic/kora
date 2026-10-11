# Queue synchronization and completion

KORA's timeline path uses one `vk::QueueContext` per logical queue. The queue context is the sole host
submission owner for that queue and creates a private timeline semaphore. Every successful queue
submission signals the next monotonically increasing value; failed recording or queue submission does
not report a successful completion value.

## Completion API

| Function | Contract |
| --- | --- |
| `QueueContext::submit(Submission)` | Enqueues an owned request and returns a context-local request ID. The worker assigns its internal timeline value when it records the request. |
| `QueueContext::completed_value()` | Queries the private semaphore counter. The value is a completion threshold, not a count of callbacks. |
| `QueueContext::wait_for_value(value, timeout_ns)` | Waits for a private timeline threshold; zero is already complete, and an expired timeout returns `ErrorCode::Timeout`. |
| `QueueContext::shutdown(Graceful)` | Stops acceptance, submits all accepted work, drains completion callbacks, then joins workers. |
| `QueueContext::shutdown(CancelPending)` | Cancels work not yet submitted, but still awaits already-submitted GPU work. |

Submission IDs and timeline values currently advance in the same order, but callers should treat IDs as
opaque request identifiers and use timeline helper values only for GPU completion. The internal signal
is appended by QueueContext so external semaphore arrays remain owned by the submission. External
timeline wait/signal arrays must have matching value arrays; binary semaphores use zero values.

## Resource lifetime integration

`BufferManager` records the final queue timeline use for each logical handle and defers reclamation until
that value completes. `StagingRing` tags reservations with their release value; on pressure it reclaims
completed ranges or waits for the oldest in-flight value rather than overwriting it. `CommandPoolSet`
reuses command buffers only after their associated completion values are reached. These paths use the
same helper API instead of allocating one fence per dispatch.

## Fence fallback

When timeline semaphores are unavailable, or `RuntimeOptions::force_legacy_sync` is enabled for
validation, QueueContext uses a bounded pool of eight reusable fences. The submission worker acquires
a fence before queue submission; the completion worker waits for that fence, advances a software
monotonic completion value, resets the fence, and returns it to the pool. Backpressure waits for a
pooled fence rather than allocating a fence for every request. `completed_value()` and
`wait_for_value()` retain the same threshold semantics in both modes.

Fallback submissions use binary semaphore waits/signals only. Omitted or all-zero semaphore value
arrays are accepted; a nonzero timeline value is rejected with `UnsupportedFeature`. The force-fallback
option disables timeline semaphore and synchronization2 feature enablement during logical-device
creation so this path is exercised even on capable Vulkan implementations. It does not change the
queried physical-device capability snapshot.

## Invariants

- Queue handles are externally synchronized by one QueueContext; do not submit directly while it owns
  the queue.
- The timeline semaphore and all referenced Vulkan objects outlive each in-flight submission.
- A timeout is not completion and never permits early callback delivery or resource reuse.
- Completion callbacks are ordered and run on a separate worker. They may enqueue work but must not
  destroy or shut down their own QueueContext.
- Timeline and synchronization2 features are enabled only when queried capabilities permit them and
  the force-fallback option is off. The fence path preserves event and resource-retirement semantics.

See [QueueContext internals](internals/queue-context.md), [completion lifecycle](internals/completion.md),
and [transfer integration](internals/transfers.md) for implementation details.
