# Dirty ranges and non-coherent mapped memory

KORA tracks writes to mapped allocations as half-open byte intervals `[offset, offset + size)`.
Intervals are sorted, non-empty, and merged whenever they overlap or touch. Keeping the set compact
reduces Vulkan call count and makes flush/invalidate behavior independent of the order in which
callers mapped and wrote buffer regions.

## Range-set contract

`kora::detail::DirtyRangeSet::add(range)` validates the endpoint with checked 64-bit arithmetic and
coalesces the new interval with every overlapping or adjacent interval. A zero-sized interval is a
no-op. The set itself is allocation-agnostic; allocation bounds are checked when ranges are aligned.

`aligned(allocation_size, non_coherent_atom_size)` expands the start downward and the end upward to
the device's atom size. Atom size must be a non-zero power of two. Every input range must fit wholly
within the allocation. An expanded final endpoint is clamped to `allocation_size`, as permitted for
the final mapped-memory range. Expanded intervals that touch are merged again. The resulting ranges
cover every dirty byte, remain within the allocation, and satisfy atom alignment except where the
allocation's final endpoint is not atom-aligned.

The helper reports `InvalidArgument` for invalid alignment or out-of-bounds intervals and `Overflow`
for endpoint arithmetic overflow. No caller should round or add endpoints without checked helpers.

## Vulkan operations

| Function | Contract |
| --- | --- |
| `flush_mapped_ranges(api, device, memory, allocation_size, atom_size, ranges, coherent)` | No-op for coherent memory or an empty range set. Otherwise validates handles, coalesces and atom-aligns the intervals, then calls the device-scoped `vkFlushMappedMemoryRanges` dispatch entry. |
| `invalidate_mapped_ranges(...)` | Same validation and alignment contract, then calls `vkInvalidateMappedMemoryRanges`; use after GPU writes and before host reads. |
| `DirtyRangeSet::clear()` | Discard tracked writes only after the caller has successfully handled the corresponding flush. A failed Vulkan flush leaves the source set unchanged. |

When an aligned range reaches the allocation end, KORA passes `VK_WHOLE_SIZE`; this expresses the
permitted final partial atom without pretending bytes exist beyond the allocation. Other ranges use
their exact atom-multiple size. Missing dispatch entries produce `UnsupportedFeature`; Vulkan errors
are returned as `TransferFailed` with the original `VkResult` retained.

Writable mappings in the host-shadow `BufferManager` feed this same interval-set implementation.
The manager has not yet allocated Vulkan memory, so its dirty intervals remain declarations; the
memory-pool integration will use the helpers when real mapped allocations are attached. Coherent
memory skips both Vulkan calls but still retains dirty metadata until the caller clears it.

## Required ordering and invariants

- Flush host writes before submitting GPU work that consumes them.
- Invalidate GPU-written memory after completion and before reading it on the host.
- Do not clear dirty metadata when flush fails.
- Keep allocation size and atom size associated with the allocation/device that produced the range.
- Never combine intervals from different allocations or memory objects in one set.

`tests/unit/test_dirty_ranges.cpp` tests adjacent/overlap merging, invalid inputs, allocation-tail
clamping, deterministic randomized coverage against a byte-level oracle, coherent-memory no-ops, and
flush/invalidate dispatch success and error propagation. These tests use a fake dispatch table and
do not require a Vulkan device.
