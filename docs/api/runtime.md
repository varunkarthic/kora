# Runtime API

Header: `include/kora/runtime.hpp`

## Construction and ownership

`Runtime::create(options)` resolves the provided `RuntimeOptions`, loads Vulkan, creates one instance,
selects eligible compute devices with the existing device-selection policy, creates their logical
devices, and constructs a memory manager for each selected device. Expected initialization failures
are returned as `Result<Runtime>`. The move-only Runtime owns the loader, instance, logical devices, and
device memory-manager state; destroying it releases logical devices before the instance and loader.

`devices()` returns the selected `DeviceId` values. Each ID uses the Vulkan enumeration index and, when
available from Vulkan 1.1 physical-device properties, a lowercase 32-character device UUID. IDs are
valid only for the owning Runtime's lifetime. `device_info(id)` returns the immutable public capability
snapshot for that selected device. `memory_stats(id)` rejects IDs not owned by the Runtime with
`InvalidArgument`.

## Memory snapshots

`MemoryStats` is Vulkan-header-free. Its pool byte counters are checked sums over tracked pools;
`largest_free_block` is the maximum contiguous free region across those pools. `heaps` reports the
per-heap budget limit, KORA reservations, remaining reservation allowance, external usage when Vulkan
reports it, and whether the budget extension was queried. If no pool allocations are registered yet,
pool counters are zero while heap budget information is still available. The current buffer façade is
not yet backed by Vulkan memory pools, so this is a diagnostic foundation for that integration.

The snapshot is taken while holding the Runtime's per-device pool-registry lock. It is not a promise
that a later Vulkan allocation will succeed: the operating system and other processes may consume
memory between the snapshot and allocation.

## Example

```cpp
#include <kora/runtime.hpp>

auto runtime = kora::Runtime::create();
if (!runtime) {
    // Handle runtime.error() without throwing.
}
for (const auto& device : runtime.value().devices()) {
    auto stats = runtime.value().memory_stats(device);
    if (!stats) {
        // Handle a stale or foreign DeviceId.
        continue;
    }
    // Inspect stats.value().allocated_bytes and per-heap budgets.
}
```

## Current limitations

- UUID selection requires UUID properties; Vulkan 1.0 devices without the properties2 identity query do
  not produce a UUID and therefore cannot match `ByUuid` selection.
- The Runtime currently creates the selected device and memory manager, but the existing host-shadow
  `BufferManager` has not yet been connected to Vulkan pools. Consequently pool counters remain zero
  until a pool-allocating runtime subsystem registers pools.
- `DevicePolicy::AllEligible` creates all selected logical devices; callers should account for the
  associated driver resources.
