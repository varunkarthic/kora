# Transfer engine

`detail::TransferEngine` implements the internal buffer data movement used by later runtime layers.
It shares the selected transfer `vk::QueueContext` for serialized queue submission and timeline
completion, and records one-time command buffers from a thread-owned `vk::CommandPoolSet`. The engine is intentionally
internal: public buffer ownership and scheduling are separate layers and are not implied by these
low-level operations.

## Ownership and lifetime invariants

- The `vk::Device`, `vk::QueueContext`, upload/readback `MemoryPool`s, and buffers supplied to an
  operation must outlive the engine and every returned `Event`.
- The queue context must submit to the queue family supplied in `Options::queue_family`, belong to the
  same device, and be the exclusive host submission path for that `VkQueue`. Bypassing it invalidates
  timeline ordering and synchronization assumptions.
- Upload copies the caller's source bytes into owned or staging memory before returning. Download
  writes into the caller's destination span only from the completion callback; that span must remain
  valid and exclusively writable until its event completes.
- A successful call means work was accepted by `QueueContext`, not that the GPU has completed it.
  `Event::wait()` reports completion status. A timeout from `wait_for()` does not cancel work.
- Staging slices remain reserved until the queue timeline reaches their release value. The engine
  never reuses a slice based only on host submission order.

## Transfer paths and synchronization

`StagingRing::should_use_staging()` selects the path using memory topology and measured policy data,
not vendor IDs. Discrete memory stages through persistent host-visible upload/readback rings. UMA
uses direct mapped host-visible memory when the provided `TransferBuffer` includes a mapping; the
engine flushes non-coherent writes before submission and invalidates non-coherent readback memory
after completion. A policy-selected direct path without a usable mapping returns
`UnsupportedFeature` instead of silently changing policy.

Each transfer describes its buffer accesses to the internal `vk::HazardTracker`, which produces
legacy `vkCmdPipelineBarrier` stage/access scopes for RAW, WAR, and WAW dependencies. Upload makes
host writes visible before transfer reads; compute-to-transfer and transfer-to-compute use is ordered
through queue timeline semaphores when runtime buffers are concurrently shared across distinct
families. Download orders prior device writes before transfer or host reads; copy and fill declare
source/destination accesses separately. Normal plans use precise host, transfer, and compute stages
rather than `VK_PIPELINE_STAGE_ALL_COMMANDS_BIT`. Explicit runtime task access declarations and
automatic exclusive-buffer ownership management remain separate mechanisms. When Runtime uses
distinct families, it supplies concurrent buffer sharing and an explicit timeline wait at the
consuming queue.

## Operations

| Operation | Contract |
| --- | --- |
| `upload(destination, offset, bytes)` | Validates the destination range, snapshots source bytes, flushes mapped/staging memory as needed, then copies or publishes the host writes. Empty input returns an already-complete event. |
| `download(source, offset, bytes)` | Validates the source range, copies to a readback slice or orders direct host access, invalidates non-coherent memory after GPU completion, then copies to the caller's span. |
| `copy(source, source_offset, destination, destination_offset, size)` | Records `vkCmdCopyBuffer`; ranges may have arbitrary byte lengths and offsets. Overlapping ranges in the same `VkBuffer` are rejected. Zero size is a completed no-op. |
| `fill(destination, offset, size, value)` | Records `vkCmdFillBuffer`; size must be nonzero and offset/size must be multiples of four, as required by Vulkan. |

All range checks use checked arithmetic before recording. Vulkan dispatch failures become `Status`/`Event`
errors with the associated `VkResult`. Queue recording failures discard the unsubmitted command
buffer and cancel its unsubmitted staging reservation.

## Event reference

`kora::Event` is a copyable handle to shared completion state:

- `valid()` distinguishes an empty handle; waiting or querying an empty handle returns
  `InvalidArgument`.
- `done()` is a non-blocking completion query.
- `wait()` blocks until success or failure and returns that status.
- `wait_for(duration)` returns `Timeout` if the deadline expires; the operation continues.
- `status()` returns `Timeout` while pending and the final operation status after completion.

The handle does not own the referenced Vulkan resources or downloaded byte span. Those lifetimes remain
the caller's responsibility. Dependency composition and aggregate waits are separate follow-up work.

## Validation

The lavapipe integration exercises every discovered local device with deterministic random payloads at
1, 2, 3, 5, 17, 4,093, 65,539, 1 MiB + 3, 8 MiB + 1, and 64 MiB. It tests upload/download and
device-copy/download at odd offsets, then validates a fill pattern. Separate queue and command-pool
tests cover submission-thread recording, timeline completion, failed recording, cancellation, and
command-buffer reuse. The integration proves behavior on the devices available to that test host; it is
not a claim about untested discrete GPU hardware or other Vulkan drivers.
