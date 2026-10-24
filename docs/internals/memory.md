# Memory subsystem

| Component | Role |
| --- | --- |
| `Suballocator` | TLSF-style allocator over one block: segregated free lists, coalescing, O(1) running statistics. |
| `MemoryPool` | Device-memory blocks of one memory type and usage class. Tries an allocation first; on failure refreshes the budget and retries once. Keeps one spare empty block and frees the rest. Large requests get dedicated allocations. |
| `MemoryManager` | Memory-type selection (policy order intersected with `memoryTypeBits` and required flags) and the per-heap reservation ledger. |
| `StagingRing` | Host-visible ring for staged copies. Reservations are timeline-gated; reclaimed ranges return through a min-heap; a cancelled newest reservation hands its space straight back. Slices keep the ring's mapping alive. |
| `TransferEngine` | Uploads, downloads, and copies, direct or staged by topology. Staged transfers larger than the ring are split into pipelined chunks; producers wait only for pending reservations to be submitted, not for earlier transfers to finish. |
| `DirtyRangeSet` | Coalesced host-written ranges, flushed or invalidated per non-coherent atom. |
| Residency and eviction models | Version-based freshness tracking and LRU-2 eviction planning, used by tests and planners. |

## Replicas

A buffer stores its replicas next to its host copy, keyed by device context. A replica records the
host version it holds, whether it holds results newer than the host (`device_newer`), its last queue
use, and its outstanding access for barriers. Submissions hold `Arc`s to the replicas they use, so a
replica is freed when both the buffer and every in-flight submission let go of it.

Split-dispatch slices in a `Cluster` use *partial outputs*: their written replicas are never marked
as the buffer's newest contents, and the Cluster reads back exactly its slice's byte range.

## Budgets

The ledger reserves backing blocks (not suballocations) against each heap's soft limit: the
configured fraction of `heapBudget − heapUsage` from `VK_EXT_memory_budget`, or of the heap size. A
driver refusal lowers the effective limit by 10%, never below live reservations. Refreshes happen at
most once per second before new allocations and on `memory_stats`.
