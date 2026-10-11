# Command pools and reusable command buffers

KORA's internal `vk::CommandPoolSet` provides a bounded ring of primary command buffers for one
submission thread and one Vulkan queue family. The caller owns submission ordering and completion
timeline sampling; the class owns pool and command-buffer lifetime, ring availability, begin/end
recording, and safe reuse.

## Rationale and invariants

Vulkan externally synchronizes command-pool operations. A pool must not be concurrently recorded,
allocated, reset, or destroyed by unrelated threads. KORA therefore captures the constructing
thread ID and, by default, asserts on cross-thread use in Debug builds. Create a separate pool set for
each submission worker/queue-family pair.

Construction allocates the configured number of primary command buffers once. `acquire(completed)`
returns the next ring slot. Once every submission associated with the pool has completed, the next
acquire performs one batch `vkResetCommandPool` and begins cycling the full ring again. It never resets
a pool while any command buffer can still be pending. Buffers begin with
`VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT`. If the ring is full before the whole batch completes,
acquire returns `OutOfMemoryBudget`; callers should wait for queue progress and retry.
`end_recording(buffer)` ends recording before the caller submits the buffer. Only after queue submission
succeeds should `submitted(buffer, signal_value)` release its ring slot and associate its completion
threshold with the same queue timeline. Values must be nonzero and strictly increasing within this set.

The caller must call `submitted` only after command recording succeeds and must submit the returned
buffer on the queue corresponding to the pool's queue family. If queue submission fails after
`submitted`, the ring must not be told that the signal value completed; device/queue recovery must
resolve that slot before reuse. Destroy the set only after all command buffers it owns are no longer
pending, and before destroying its logical device.

The current class does not submit work, create timeline semaphores, wait, or sample completion. It
does not provide simultaneous-use buffers or cross-thread access. Scheduler/`QueueContext`
integration is intentionally a later layer.

## API reference

| Function | Contract |
| --- | --- |
| `CommandPoolSet(api, device, options)` | Creates a transient/resettable pool and preallocates `ring_size` primary buffers. Invalid options or Vulkan allocation failures leave the set unavailable (`capacity() == 0`). |
| `acquire(completed_value)` | Batch-resets the pool only after all slots complete, then begins the next one-time slot. Returns an error if the set is unavailable, called from the wrong thread, or the ring is full. |
| `end_recording(buffer)` | Ends a currently acquired buffer so it can be passed to queue submission. |
| `submitted(buffer, signal_value)` | After successful queue submission, records the completion threshold and releases the slot. Zero/non-increasing values and unknown/not-ended buffers are rejected. |
| `capacity()` | Returns the ring's configured/preallocated slot count. |
| `allocation_count()` | Returns the number of command buffers allocated at construction; steady-state reuse does not change it. |

Vulkan calls go through `vk::Api`, preserving the loader lifetime and fault-injection boundary. The
ring has no public-header dependency: it is an internal backend component until the submission API is
integrated.

## Validation

The fault-injection test uses a mock dispatch table to verify one allocation batch, zero command
buffer frees during 10,000 acquire/end/reuse cycles, `ONE_TIME_SUBMIT`, and invalid state handling.
The lavapipe integration test records and submits 10,000 empty command buffers in batches, waits on a
timeline semaphore before reusing slots, and confirms the ring capacity remains fixed. Without a
Vulkan implementation or timeline support, that integration test is explicitly skipped.
