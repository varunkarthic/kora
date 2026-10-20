# Memory model

KORA records Vulkan memory heaps and types in `kora::DeviceInfo`, then classifies the visible memory
layout without relying on vendor IDs or device names. The topology is a heuristic for later
allocation policy; it does not allocate memory, guarantee bandwidth, or exclude any memory type.

## Topology classification

`DeviceCaps::classify_memory_properties()` examines each memory type together with its heap flags.
A type is treated as device-local when its type advertises `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` or
its referenced heap advertises `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`. Invalid heap references are not
dereferenced.

| `MemoryTopology` | Heuristic | Benchmark flag |
| --- | --- | --- |
| `Discrete` | At least one device-local type is not host-visible, and no device-local host-visible type is reported | No |
| `Uma` | Device-local host-visible types exist, and either no device-local host-invisible type exists or every heap is device-local | No |
| `DiscreteWithBar` | Both device-local host-visible and device-local host-invisible types exist, and a separate non-device-local (system-memory) heap exists | Yes |
| `Unknown` | The properties do not provide evidence for one of the layouts above | No |

`DiscreteWithBar` is deliberately marked `memory_topology_needs_benchmark()`: the
`DEVICE_LOCAL | HOST_VISIBLE` flags alone do not establish that direct host access is faster than
staging. The heuristic reports the observed flags and leaves that performance decision to measured
behavior. A mixed topology and every memory type remain visible to callers.

A discrete GPU always exposes a system-memory heap alongside its device-local heap, even when a
resizable BAR adds a host-visible device-local type. A device whose only heaps are device-local
(for example, an integrated GPU that advertises host-visible and host-invisible types in one shared
heap) is therefore classified `Uma`.

### Unified-memory direct access

On `Uma` devices, runtime buffers use the `Auto` policy, which prefers host-visible device-local
memory. A dispatch then writes a changed host shadow directly into the buffer's mapping (with a
flush for non-coherent memory) instead of staging it through a transfer submission, and a dispatch
with default readback copies its writable outputs from the mapping after completion. Each runtime
buffer counts the GPU operations that are queued or executing against it; the direct path is used
only while that count is zero, so a host write can never change bytes that queued work will read.
Otherwise, and on all other topologies, KORA keeps the queue-ordered staging path. Only default
readback dispatches record the shader-write to host-read barrier that direct reads require;
`Runtime::readback()` after a deferred dispatch uses a transfer download.

## Memory heaps and types

`DeviceInfo::memory_heaps()` preserves each heap's byte size, device-local flag, and multi-instance
flag. `DeviceInfo::memory_types()` preserves each type's heap index and device-local, host-visible,
host-coherent, host-cached, lazily-allocated, and protected flags. Heap sizes and type properties
must be considered together; a type index refers to its `heap_index`.

## Preferred memory-type order

`DeviceInfo::memory_type_preferences()` returns a stable ordering of memory-type indices for each
policy category: `automatic`, `device_preferred`, `host_visible`, `read_mostly`, `write_mostly`,
`streaming`, and `readback`. These are preference rankings only. They contain every reported memory
type, preserve index order for ties, and do not apply Vulkan's per-resource `memoryTypeBits` mask.
An allocator must intersect candidates with that mask and enforce any required property flags
before allocation.

The order prefers, respectively:

- `automatic`: topology-appropriate device-local or shared memory; on discrete layouts, dedicated
  device-local memory ranks ahead of host-visible BAR memory.
- `device_preferred`: device-local memory, preferring a non-host-visible type before a host-visible
  device-local type.
- `host_visible`: host-visible, coherent, then cached memory.
- `read_mostly`: device-local, host-cached, then host-visible memory.
- `write_mostly`: device-local, host-coherent, then host-visible memory.
- `streaming`: host-visible, host-coherent, then device-local memory.
- `readback`: host-visible, host-cached, then host-coherent memory.

These rankings are deterministic hints, not final type selection. In particular, the `read_mostly`
and `write_mostly` rankings do not claim workload-specific performance.

## Allocation-time policy and budget accounting

The internal `kora::detail::MemoryManager` turns those rankings into a legal candidate for a
particular resource. `select_memory_type(policy, compatible_type_bits, required)` walks the selected
policy order and returns the first index that is both present in the Vulkan `memoryTypeBits` mask and
has every requested property (`device_local`, `host_visible`, `host_coherent`, and/or `host_cached`).
The mask is authoritative: a preferred type that is not permitted for the resource is skipped. An
empty mask is `InvalidArgument`; if no candidate satisfies the mask and property requirements, the
operation returns `UnsupportedFeature` rather than weakening the requirements.

The seven policy values map directly to the corresponding `DeviceInfo::MemoryTypePreferences` order:
`Auto` to `automatic`, `DevicePreferred` to `device_preferred`, and the remaining policies to their
same-named list. These are hints rather than strict residency requirements; callers express required
properties separately. Topology affects selection through the rankings created by `DeviceCaps`, not
through vendor or device identifiers.

`MemoryManager::create()` establishes one reservation ledger per reported heap. With `VK_EXT_memory_budget`, the initial soft limit is the configured fraction (80% by default) of `max(heapBudget - heapUsage, 0)`. Without the extension, it applies the same configurable fraction (accepted range 0 through 1) to physical heap size. Limits are clamped to heap size. The fallback is conservative because heap size does not report current OS or other-process usage. `available_bytes` means the KORA soft limit minus KORA reservations; it is not a promise that Vulkan allocation will succeed.

`reserve(heap_index, bytes)` uses checked addition and changes the ledger only if the result is
within its limit. Arithmetic overflow returns `Overflow`; a valid sum over the limit returns
`OutOfMemoryBudget`. `release(heap_index, bytes)` rejects a release greater than the current
reservation with `InternalInvariantViolation`, preserving state on failure. Debug builds assert
after each successful mutation that reserved bytes do not exceed the limit. Invalid heap indices return `InvalidArgument`. The ledger tracks backing `VkDeviceMemory` allocations: each pooled block is reserved once, not once per suballocation, and released when the block or dedicated allocation is freed. Failed Vulkan allocations release their provisional reservation and lower the effective heap limit by 20%, without shrinking below live reservations.

`MemoryManager::refresh_budget(physical_device, fallback_fraction)` replaces the observed heap snapshot. Runtime refreshes at most once per second before creating a new device buffer and on explicit `Runtime::memory_stats()` reads. Extension-based refresh accounts for KORA's own reservations separately from external usage, then reapplies the configured soft-limit fraction to remaining capacity. Without live extension telemetry, refresh preserves allocation-failure reductions. If a new observation is lower than current reservations, those allocations remain valid: the effective limit is held at the reserved amount, available bytes become zero, and no new reservations are admitted until capacity returns or allocations are freed. Available-byte calculation is underflow-safe, and accepted limits cannot exceed physical heap size. Vulkan allocation errors remain authoritative.

## Logical buffer residency

The internal `ResidencyTracker` models which copies contain the current logical buffer contents.
It is separate from Vulkan handles: device IDs are opaque runtime indices, and content versions—not
device type, vendor ID, or memory flags—determine freshness. A new host-initialized buffer starts at
version 1 in `HostOnly`; a device-initialized buffer has one current device replica in `DeviceOnly`.
Every host or device write advances the version, making all other copies stale without deleting
their allocations.

| State | Meaning |
| --- | --- |
| `HostOnly` | Current data exists only in the host copy; no device copy has been created. |
| `DeviceOnly` | Current data exists on one device and there is no host copy. |
| `Shared` | Current data exists in multiple replicas, including multiple current device copies. |
| `HostAndDeviceClean` | The host and exactly one device copy hold the current version. |
| `HostNewer` | Host data is current; any extant device copies are stale. |
| `DeviceNewer` | A device copy is current; the host copy, if present, is stale. |
| `Evicted` | Host data remains current after all device copies were removed. |
| `InTransfer` | A checked copy is in progress; its source remains current until completion. |
| `Disposed` | No copy is current; this is the sole state exempt from the current-copy invariant. |

The pure `transition_residency(state, event)` function rejects illegal state/event pairs. The tracker
adds the per-device version map and resolves summary states for cases (such as multi-device copies)
that a single enum cannot distinguish. Its legal transition graph is:

```{mermaid}
stateDiagram-v2
    [*] --> HostOnly: host_initialized
    [*] --> DeviceOnly: device_initialized
    HostOnly --> HostOnly: HostWrite
    HostOnly --> DeviceNewer: DeviceWrite
    HostOnly --> InTransfer: BeginTransfer
    DeviceOnly --> HostNewer: HostWrite
    DeviceOnly --> DeviceOnly: DeviceWrite
    DeviceOnly --> InTransfer: BeginTransfer
    Shared --> HostNewer: HostWrite
    Shared --> DeviceNewer: DeviceWrite
    Shared --> InTransfer: BeginTransfer
    HostAndDeviceClean --> HostNewer: HostWrite
    HostAndDeviceClean --> DeviceNewer: DeviceWrite
    HostAndDeviceClean --> InTransfer: BeginTransfer
    HostNewer --> HostNewer: HostWrite
    HostNewer --> DeviceNewer: DeviceWrite
    HostNewer --> InTransfer: BeginTransfer
    HostNewer --> Evicted: EvictDevice
    DeviceNewer --> HostNewer: HostWrite
    DeviceNewer --> DeviceNewer: DeviceWrite
    DeviceNewer --> InTransfer: BeginTransfer
    DeviceNewer --> DeviceOnly: EvictHost
    DeviceNewer --> DeviceNewer: EvictDevice
    Evicted --> Evicted: HostWrite
    Evicted --> DeviceNewer: DeviceWrite
    Evicted --> InTransfer: BeginTransfer
    InTransfer --> HostAndDeviceClean: CompleteToHost
    InTransfer --> Shared: CompleteToDevice
    Shared --> DeviceOnly: EvictHost
    Shared --> Evicted: EvictDevice
    HostAndDeviceClean --> DeviceOnly: EvictHost
    HostAndDeviceClean --> Evicted: EvictDevice
    HostOnly --> Disposed: Dispose
    DeviceOnly --> Disposed: Dispose
    Shared --> Disposed: Dispose
    HostAndDeviceClean --> Disposed: Dispose
    HostNewer --> Disposed: Dispose
    DeviceNewer --> Disposed: Dispose
    Evicted --> Disposed: Dispose
```

### API reference and transfer policy

Memory-budget function contracts:

| Function | Contract |
| --- | --- |
| `MemoryManager::create(physical_device, device_info, fallback_heap_fraction)` | Establishes per-heap soft limits from extension-reported free budget or, without the extension, the configured heap-size fraction. |
| `MemoryManager::refresh_budget(physical_device, fallback_heap_fraction)` | Refreshes telemetry while retaining reservations and preserving failure shrinkage when telemetry is unavailable. |
| `MemoryManager::reserve(heap_index, bytes)` / `release(heap_index, bytes)` | Accounts for backing Vulkan allocations exactly once and rejects over-budget or underflow mutations. |
| `MemoryManager::record_allocation_failure(heap_index, shrink_factor)` | Reduces future admission capacity while preserving live reservations. |
| `Runtime::memory_stats(device)` | Refreshes memory budget and returns pool and per-heap snapshots. |

Residency function contracts:

| Function | Contract |
| --- | --- |
| `transition_residency(state, event)` | Pure transition validation; illegal pairs return `InvalidArgument`. |
| `write_host()` / `write_device(device)` | Advance the content version and mark only the written copy current. Version overflow is rejected without mutation. |
| `plan_transfer(destination)` | Return a no-op when the destination already holds the current version; otherwise select a current host/device source and capture the version. |
| `begin_transfer(plan)` | Revalidate the captured plan, then retain its source as current while entering `InTransfer`. Stale or duplicate plans fail. |
| `complete_transfer()` / `fail_transfer()` | Mark the destination current only on success; failure restores the pre-transfer state and retains the source copy. |
| `evict_host()` / `evict_device(device)` | Remove a copy only when another current copy remains. Stale copies may be discarded without losing current data. |
| `dispose()` | Explicitly invalidate all copies. Other operations on a disposed tracker fail. |

Transfers therefore happen only for absent or stale destinations. Copy freshness is determined by
version equality; stale allocations are never considered valid sources. `begin_transfer` compares a
plan with a fresh query so a plan cannot be submitted after an intervening write. The tracker is a
policy/model layer; runtime dispatch applies the same freshness principle per buffer using a monotonic
host-shadow version and a per-device cache. A cached device version is reused for subsequent
read/read-write bindings, including intermediate outputs marked `DeviceNewer`, while host writes and
writable mappings advance the host version and force a new upload.

By default, dispatch completion preserves the existing host-readable behavior by downloading every
written binding. `DispatchBuilder::defer_readback()` keeps outputs device-resident for a later dispatch;
`Runtime::readback(buffer)` explicitly requests host freshness after the producing dispatch completes.
`Runtime::transfer_stats()` reports upload and download operation counts, allowing chain tests to
assert that only initial inputs are uploaded and only the requested final output is downloaded.
Device copies live only as long as their Runtime context and are not shared across Runtime instances.
The tracker remains single-owner and not thread-safe; callers must serialize writes, submissions, and
readback requests for a logical allocation.

| Runtime API | Residency contract |
| --- | --- |
| `BufferHandle::host_version()` | Returns a monotonic host-shadow generation, advanced by writes and writable mappings. |
| `DispatchBuilder::defer_readback()` | Leaves write/read-write outputs device-newer when dispatch completes instead of scheduling host copies. |
| `Runtime::readback(buffer)` | Schedules a device-to-host copy for a deferred device-newer buffer; wait for the returned event before reading the host shadow. Already host-current buffers return a completed event. |
| `Runtime::transfer_stats()` | Reports upload/download operation counts issued by the runtime for verification and diagnostics. |

For every non-disposed tracker, at least one host or device copy must equal `current_version`. An
eviction that would remove the last such copy returns `InvalidArgument` and leaves the tracker
unchanged. `Disposed` is the explicit exception. The exhaustive state/event table, per-device stale
copy cases, deterministic randomized invariant sequence, and transfer no-op/stale-destination
coverage live in `tests/unit/test_residency.cpp`.

## Tests

Synthetic memory-property tests cover discrete, UMA, BAR-like, and unknown layouts, including the
BAR benchmark flag and stable per-policy orderings. Memory-manager tests cover memoryTypeBits and
required-property filtering, topology-dependent candidate ordering, budget-extension and fallback
limits, over-budget and overflow reservations, invalid indices, and release underflow. The
`DeviceInfo` mapping test also verifies heap size and per-type flags survive conversion from Vulkan
properties.

Residency tests exhaust every state/event pair, check host/device freshness for single and multiple
devices, verify stale or absent destinations trigger plans while current destinations do not, and
exercise transfer failure, eviction, disposal, and a deterministic operation sequence against the
current-copy invariant. Runtime integration tests execute a two-dispatch chain, assert exact upload
and download counters, and verify explicit readback returns the final result.

Run the focused tests with:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug --output-on-failure -R 'DeviceCaps|MemoryManager|BudgetLedger|Residency'
```
