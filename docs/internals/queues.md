# Queues and transfers

## QueueContext

Each device queue has a `QueueContext` with a submission thread and a completion thread.

- **Submission** collects recorded work, optionally coalesces independent submissions (same key, no
  dependencies) into one `vkQueueSubmit`, and assigns monotonically increasing completion values.
- **Completion** waits on a timeline semaphore when the device supports it, or on pooled fences that
  emulate completion values, then runs each submission's callback in order.
- **Device loss** is recorded once in a fault state shared by the device's queues; queued work fails
  with `DeviceLost`, and later submissions are rejected before reaching Vulkan.
- **Shutdown** drains (or cancels) queued work; callbacks may submit follow-up work while draining.

## Hazards and barriers

The hazard tracker records, per replica, the last access (read or write) and the queue value that
made it. A new access emits a barrier only for RAW, WAR, or WAW conflicts, choosing stages and access
masks precisely; synchronization2 is used when enabled. Transfers between queue families emit paired
release/acquire ownership barriers.

## Command buffers

`CommandPoolSet` keeps a ring of command buffers per thread-owned pool and reuses each only after its
completion value has been reached. A full ring reports `OutOfMemoryBudget` until work completes.

## Timestamps

`TimestampRing` hands out query pairs for sampled dispatches and collects results without stalling
the completion thread; values are masked to the queue's valid bits and scaled by the device's
timestamp period.
