# Segregated-fit suballocator

KORA's internal `Suballocator` manages byte offsets inside one future Vulkan memory block. It is a
CPU-only component: it neither owns memory nor calls Vulkan. Pool and memory-type policy belong to
the later memory-manager layers.

## Design

Free regions are indexed by 64 first-level power-of-two classes and four subdivisions per class.
Two bitmaps locate the next non-empty class without walking empty classes. A short search within a
candidate class handles alignment padding, making ordinary lookup constant or near-constant time.

Physical neighbours and free-list links are indices into a side vector. No allocator metadata is
written into the managed byte range. Allocation performs checked arithmetic, splits alignment
prefixes and unused suffixes, and rounds requested sizes to a 16-byte quantum. Freeing immediately
coalesces the left neighbour, the right neighbour, or both when they are free. Retired metadata
slots are reused.

This allocator serves buffers only. Vulkan `bufferImageGranularity` is therefore intentionally not
part of its contract; mixed buffer/image pools must add that policy above this layer.

## API reference

| Function | Contract |
| --- | --- |
| `Suballocator::create(capacity)` | Creates one free region; zero capacity returns `InvalidArgument`. |
| `allocate(size, alignment)` | Returns an aligned offset, requested size, and rounded occupied size. Size must be nonzero and alignment a power of two. Exhaustion returns `AllocationFailed`; arithmetic overflow returns `Overflow`. |
| `deallocate(allocation)` | Frees the exact live handle and coalesces neighbours. Unknown, duplicate, or altered handles return `InvalidArgument`. |
| `stats()` | Returns a point-in-time telemetry snapshot. |
| `validate_invariants()` | Checks physical coverage, adjacency, free-list membership, bitmaps, and the live-allocation index. |

The type is not thread-safe. A memory pool must serialize access or give each allocator a single
owner.

## Telemetry definitions

- `total_bytes`: capacity of the managed block.
- `allocated_bytes`: bytes occupied after 16-byte allocation rounding.
- `requested_bytes`: bytes requested by live callers.
- `free_bytes`: `total_bytes - allocated_bytes`.
- `internal_fragmentation_bytes`: `allocated_bytes - requested_bytes`.
- `largest_free_block`: largest contiguous free region.
- `external_fragmentation_bytes`: free bytes outside the largest free region.
- `live_allocations`: number of currently valid handles.
- `high_water_mark`: maximum simultaneous `allocated_bytes` observed.

These byte counts avoid unstable percentage conventions. A caller can derive ratios using its own
denominator while handling zero capacity explicitly.

## Invariants and validation

After every mutation, Debug builds assert that:

- active physical regions cover `[0, capacity)` exactly with no gaps or overlap;
- physical previous/next links are reciprocal and adjacent free regions never remain separate;
- every free region appears exactly once in its size-class list and bitmap state matches list state;
- allocated regions do not appear in free lists and each has exactly one live-handle entry;
- the sum of free and allocated region sizes equals capacity.

Unit tests exercise aligned splitting, exhaustion, invalid frees, each of the four neighbour states
(neither, left, right, and both), telemetry, and 5,000 deterministic randomized operations checked
against an independent interval-overlap oracle.
