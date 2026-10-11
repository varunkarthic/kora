# Runtime API

Header: `include/kora/runtime.hpp`

## Construction and ownership

`Runtime::create(options)` resolves the provided `RuntimeOptions`, loads Vulkan, creates one instance,
selects eligible compute devices with the existing device-selection policy, creates their logical
devices, constructs a memory manager and a pool for each reported Vulkan memory type, and initializes
the queue/transfer services needed by the dispatch API. Expected initialization failures
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
reports it, and whether the budget extension was queried. Runtime-owned transfer staging buffers are
allocated during initialization, and lazily created storage buffers are reflected in pool counters.

The snapshot is taken while holding the Runtime's per-device pool-registry lock. It is not a promise
that a later Vulkan allocation will succeed: the operating system and other processes may consume
memory between the snapshot and allocation.

## Buffers, kernels, and dispatch

`Runtime::create_buffer<T>(count)` creates typed host-shadow storage. `Buffer<T>::write(span)` copies
exactly `count` trivially-copyable values into that shadow; `read()` returns a copy. The runtime creates
Vulkan storage lazily, uploads a read/read-write shadow when its host content version is not already
resident on the selected device, and reuses current device copies across dispatches. By default,
write/read-write outputs are downloaded before the returned `Event` completes, preserving the original
host-readable behavior. `DispatchBuilder::defer_readback()` opts out for a chain intermediate;
`Runtime::readback(buffer)` explicitly schedules a download when host access is later needed. Wait for
the producing dispatch event before requesting an explicit readback. `transfer_stats()` reports upload
and download operation counts issued by the Runtime.

`load_kernel(path, entry_point, metadata)` reads and structurally validates a bounded SPIR-V module.
The metadata is checked against reflected descriptor bindings, push-constant size, local size, and
specialization constants when the pipeline is created. `dispatch(kernel)` returns a fluent
`DispatchBuilder`; each `.read`, `.write`, or `.read_write` call corresponds to the next metadata
binding in vector order. `global(x[, y, z])` specifies invocation extents (not work-group counts), and
the runtime rounds each dimension up by the reflected local size. A bounds check in the shader remains
the caller's responsibility for non-multiple extents. `.push(value)` copies a trivially-copyable
value's bytes; its total size must exactly match metadata and fit the selected device's push-constant
limit. `submit()` returns an `Event` whose `wait()` covers compute completion and, unless deferred,
output readback. Host writes and writable mappings advance the buffer's host content version and
invalidate the device-copy cache on its next use.

This first dispatch version supports storage-buffer descriptors only. Other descriptor kinds return
`UnsupportedFeature`. Invalid dimensions, binding counts/access modes, push-constant sizes, and device
limits are rejected before pipeline/dispatch Vulkan calls. A runtime, its `Kernel`, and builders should
not be used concurrently with or after destruction of that runtime.

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
- `DevicePolicy::AllEligible` initializes queue and transfer services for every selected device;
  callers should account for the associated driver resources and staging allocations.
- `BufferHandle::read()` reads the host shadow only. After a deferred dispatch output, call and wait for
  `Runtime::readback()` before reading that buffer on the host. Device-resident copies are currently
  cached per Runtime-selected device and are discarded with that Runtime.
- Startup calibration currently stores only a bounded host-side overhead probe. Tiny GPU dispatch,
  H2D/D2H, device-copy, streaming-kernel, and ALU measurements are recorded as unavailable and do not
  influence device selection. See [Startup calibration](../calibration.md).
