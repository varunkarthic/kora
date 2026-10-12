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

### Shutdown

`Runtime` destruction is `noexcept`. `RuntimeOptions::shutdown_mode` selects the teardown policy and
defaults to `ShutdownMode::Graceful`. Graceful shutdown first closes Runtime dispatch admission, then
joins each device's compute and transfer submission/completion workers, then drains and joins the CPU
fallback executor. Work already accepted is allowed to finish, including completion callbacks and
CPU fallback descendants. Queue workers and device-owned callback dependencies are gone before
buffers, pipelines, logical devices, the Vulkan instance, and loader are released.

`ShutdownMode::CancelPending` uses the same ownership ordering but cancels queued, not-yet-submitted
Vulkan submissions and queued CPU tasks. Their Events complete with `ErrorCode::Cancelled`; work
already executing or submitted to Vulkan is not forcibly interrupted and must finish before teardown
returns. Neither mode can safely abort a hung driver call or a user callback that never returns.
Shutdown must not be initiated from a Runtime-owned queue callback or CPU executor task; those
lower-level shutdown calls reject self-join attempts. Destroying a Runtime concurrently with calls
through that Runtime, its kernels, or dispatch builders is unsupported.

After an event reports `DeviceLost`, `Runtime::recover()` transactionally creates replacement logical
devices with the Runtime's resolved options. Existing host-backed `BufferHandle`s remain valid, and
authoritative host contents upload lazily on their next use. Outputs of in-flight writes return
`DataLoss` until replaced with a complete `write_bytes()` call. Existing `Kernel` values are bound to
the old Runtime and must be loaded again. Recovery drains old queue callbacks before returning; if
replacement creation fails, the original failed Runtime remains intact.
`device_fault_report(id)` returns best-effort extension records when `VK_EXT_device_fault` is
supported; KORA bounds the report to 256 address and vendor records and never persists or logs these
diagnostic details. Query the report before `recover()`, which replaces the per-device fault state.
`calibrate()` bypasses cached records and reruns the bounded best-effort calibration for every
selected device. It can be used when startup calibration was disabled; unavailable or failed probes
remain zero and do not block Runtime use.
For `DevicePolicy::AllEligible`, devices that fail logical-device initialization are skipped while
other selected devices continue; `quarantined_devices()` lists those identities. Creation fails if
none initialize. Other selection policies remain fail-fast.
Call `DispatchBuilder::verify_with(policy, type)` before `submit()` to run the same dispatch on every
initialized, non-quarantined device and compare writable outputs. Matching bytes are published to the
host shadow. Any mismatch returns `ValidationError`, marks outputs `DataLoss`, and quarantines every
participant because the Runtime cannot establish a trustworthy result. See
[`api/verification`](verification) for comparator modes, data types, and limitations. Verification
requires at least two devices that can run the kernel.

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
`Runtime::readback(buffer)` explicitly schedules a download from the first selected device when host
access is later needed; `Runtime::readback(buffer, device)` selects the producing device explicitly.
Wait for the producing dispatch event before requesting an explicit readback. A selected-device
readback is also the host-staging step when a different device will consume the output; that device
uploads the current host version to its own replica on dispatch. `transfer_stats()` reports upload
and download operation counts aggregated across the Runtime's selected device contexts.
`profiling_stats(device)` nonblockingly polls dispatch timestamp results already past the queue's
completion timeline. It reports cumulative collected/dropped counts and the most recently collected
duration; calibrated host time and deviation are available only when supported. Profiling is optional:
unsupported queues report `UnsupportedFeature` without preventing runtime creation or dispatch.

`load_kernel(path, entry_point, metadata)` reads and structurally validates a bounded SPIR-V module
for the first selected device. Pass a `DeviceId` as a fourth argument to validate and associate the
kernel with another selected device. `dispatch(kernel)` routes using that kernel's owning runtime and
device. The metadata is checked against reflected descriptor bindings, push-constant size, local size,
and specialization constants when the pipeline is created. It returns a fluent `DispatchBuilder`; each
`.read`, `.write`, or `.read_write` call corresponds to the next metadata
binding in vector order. `global(x[, y, z])` specifies invocation extents (not work-group counts), and
the runtime rounds each dimension up by the reflected local size. A bounds check in the shader remains
the caller's responsibility for non-multiple extents. `.push(value)` copies a trivially-copyable
value's bytes; its total size must exactly match metadata and fit the selected device's push-constant
limit. `submit()` returns an `Event` whose `wait()` covers compute completion and, unless deferred,
output readback. Host writes and writable mappings advance the buffer's host content version and
invalidate the device-copy cache on its next use.

`Runtime::dispatch_adaptive(kernel, workload)` is an explicit alternative for one dispatch. It chooses
among initialized devices that pass SPIR-V capability validation, health, and the caller-declared
device-memory admission estimate. Cross-device comparison requires `DevicePolicy::AllEligible`.
Every eligible candidate must have a complete calibration record matching its UUID, driver, and
schema; if any eligible record is absent or stale, the kernel's original device is retained. The
workload declares transfer byte totals, a kernel class and class-specific work count, and a conservative
device-memory requirement. The returned builder is ordinary and device-bound; buffers, push constants,
and invocation extents are still validated at submit time. Inspect `DispatchBuilder::selected_device()`
to observe the chosen target. At submission, if an input has a newer device-side copy on the
kernel-bound device, the builder returns to that device to avoid silently reading a stale host shadow.
If newer inputs reside on another device or are split across devices, submission returns
`UnsupportedFeature`; explicitly read them back before migration. Inspect `selected_device()` again
after submission to see the effective target. Adaptive dispatch cannot be combined with duplicate
verification. `LocalSizeId` autotuned kernels stay on their original device. This API is deterministic
and never changes the behavior of `dispatch()`.

Compute submissions with the same cached kernel pipeline may be coalesced by the Runtime for up to
100 microseconds (maximum batch of eight) when their buffer accesses are independent. Each dispatch
retains its own `Event`; shared writes, read/write conflicts, or differing external semaphore waits
are submitted separately. Use
`.priority(DispatchPriority::High)` or `.priority(DispatchPriority::RealtimeHint)` to bypass the
coalescing wait. Priorities are hints, not hard real-time guarantees.

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
  `Runtime::readback()` before reading that buffer on the host. Device-resident copies are cached
  separately per selected device and are discarded with that Runtime. Cross-device movement is
  explicit host staging; no peer/external-memory fast path is enabled.
- Startup calibration records bounded queue-submit/completion and private-scratch H2D, D2H, and
  device-copy measurements, plus correctness-checked tiny-dispatch, streaming, and integer-ALU probes
  when supported and completed within budget. Results are host-observed diagnostics, not pure
  hardware counters, and do not yet influence device selection. See
  [Startup calibration](../calibration.md).

For caller-defined range-aware work, see [multiple Vulkan devices](../multi-gpu.md) and
`kora/task_partition.hpp`. `DynamicPartitioner` returns device-specific contiguous ranges; the
caller owns dispatch, output merging, and deciding whether the operation's semantics permit splitting.
