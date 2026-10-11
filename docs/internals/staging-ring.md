# Staging rings

`detail::StagingRing` owns one or more persistently mapped `VkBuffer` blocks for upload or readback.
It suballocates byte ranges and associates each submitted range with a release value from the
timeline governing its queue. A reservation remains unavailable until that timeline value completes.
Create one ring for uploads and another for readbacks when the transfer policy selects staging.

## Policy and ownership

`should_use_staging()` uses device memory topology, never vendor identifiers. Discrete memory uses
staging by default; UMA selects direct host-visible access. `DiscreteWithBar` and unknown topology use
staging only when a measured transfer policy reports a benefit. A positive measurement can select
staging on any topology. `create()` enforces this decision.
The integration test can explicitly set the measured-benefit override to exercise staging on any test
device.

The ring requires a host-visible `MemoryPool`. The pool and logical device must outlive the ring and
must not be destroyed while any slice is in use. The pool should be dedicated to serialized ring
allocation or otherwise externally synchronized. Each Vulkan ring block is a transfer-source buffer
for upload or transfer-destination buffer for readback, persistently mapped through its pool
allocation. Additional buffer usage flags are opt-in.

`StagingSlice::mapped_data` addresses the CPU mapping at the returned buffer-relative offset. For
non-coherent pool memory, call `flush(slice)` after upload writes and before queue consumption, then
call `invalidate(slice)` for readback data after GPU completion. The ring pads reservations to the
non-coherent atom size and requires the matching atom size in its options. The ring does not submit
copies or infer timeline values; callers retire each submitted slice with the release value of the
queue that accesses its buffer. A slice may instead be cancelled only before submission.

## Allocation lifecycle and exhaustion

Reservations are aligned, non-overlapping ranges within persistent blocks. The ring reclaims only
retired segments whose release value is at or below the supplied completed value. It first searches
existing blocks, then grows by a bounded block while total logical capacity remains below
`maximum_capacity`. If allocation remains exhausted, it waits for the smallest release value among
submitted segments and retries. If only unsubmitted reservations remain, or no wait callback is
provided, allocation fails rather than waiting on work that cannot complete or overwriting it.

## API reference

| Function | Contract |
| --- | --- |
| `StagingRing::create(device, host_pool, options)` | Validates topology policy and capacity bounds, creates the initial transfer buffer, allocates compatible host-visible memory, binds it, and maps through the pool. |
| `reserve(size, alignment, completed, wait_for)` | Reclaims completed ranges and returns a mapped slice, growing or waiting for the oldest submitted range as needed. |
| `retire(slice, release_value)` | Tags a live reservation with the queue timeline value after which it may be reused. |
| `cancel(slice)` | Releases a reservation that has not been submitted; submitted slices cannot be cancelled. |
| `flush(slice)` / `invalidate(slice)` | Performs the required mapped-memory cache operation for non-coherent pool memory; coherent memory is a validated no-op. |
| `should_use_staging(topology, measured_benefit)` | Returns the vendor-neutral topology/measurement policy used by `create()`. |

## Validation

The policy unit test covers discrete, UMA, discrete-with-BAR, and unknown topology decisions. The
lavapipe integration creates real Vulkan upload buffers, performs 1,000 deterministic random-size
reservations using a fake timeline/wait callback, checks that in-flight ranges never overlap, forces
the tiny-ring wait path, and verifies that exhaustion containing only unsubmitted work fails without
calling the wait function. It also creates and reserves a readback buffer. Actual copy commands are
owned by the transfer-engine layer.
