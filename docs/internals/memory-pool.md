# Memory pools and blocks

`MemoryPool` owns large `VkDeviceMemory` blocks and uses the internal segregated-fit allocator to
serve many small buffer allocations. A pool represents one device, memory type, usage class,
host-coherency mode, and device-address mode; callers must not mix incompatible requirements.

## Block policy

The initial target size follows the runtime design heuristic:

- device-local: `clamp(available / 64, 16 MiB, 256 MiB)`;
- host-visible: `clamp(available / 128, 8 MiB, 64 MiB)`.

`available` is the smaller of heap size and a nonzero queried budget. Tests and higher policy layers
may supply an explicit block-size hint. A failed `vkAllocateMemory` halves the next target, down to
1 MiB, providing a conservative adaptation hook for later budget management.

Requests larger than half the current block size bypass pooling. A request also bypasses pooling
when Vulkan dedicated requirements mark it required or preferred. Dedicated allocations retain the
same error and mapping semantics but have their own `VkDeviceMemory` lifetime.

## Persistent mapping

Host-visible blocks are mapped once across their full lifetime. Each returned allocation exposes a
pointer advanced by its suballocation offset. The block is unmapped exactly once immediately before
`vkFreeMemory`; individual frees never map or unmap. Non-host-visible pools always return a null
mapped pointer. Coherency is recorded in the pool key. Non-coherent callers use the dirty-range
helpers to atom-align and dispatch flush/invalidate calls; the pool does not silently flush on every
map or suballocation free.

## API reference

| Function | Contract |
| --- | --- |
| `MemoryPool::create(device, config)` | Validates a live device and nonzero heap, then computes the initial block size without allocating eagerly. The device must outlive the pool. |
| `allocate(request)` | Reuses an existing block, creates one block, or takes the dedicated path. Size must be nonzero and alignment a power of two. |
| `deallocate(allocation)` | Returns a pooled range for immediate coalescing or destroys a dedicated allocation. Altered, unknown, and duplicate handles fail. |
| `stats()` | Reports reserved, occupied, and requested bytes; block/dedicated/live counts; and actual `vkAllocateMemory` calls. |

All Vulkan allocation and mapping failures use `ErrorCode::AllocationFailed` and retain the original
`VkResult`. The pool is single-owner and not thread-safe; the future memory manager supplies
synchronization.

## Invariants and test evidence

- Small requests never allocate one Vulkan object per buffer; blocks remain stable and are reused.
- Every pooled handle names its owning block and exact suballocation.
- Every dedicated handle has a unique live token.
- Host mappings and device allocations have one owning pool and one destruction path.
- The integration test allocates 10,000 aligned 256-byte ranges against a real Vulkan device and
  asserts at most two `vkAllocateMemory` calls, then verifies dedicated bypass and complete release.
