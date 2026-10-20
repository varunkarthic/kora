# Multiple Vulkan devices

For automatic routing across every local GPU, splitting dispatches, and hot-plugged devices, see
[Multi-device, remote workers, and hot-swap](cluster.md). This page covers explicit control.

KORA can create a runtime that owns more than one eligible Vulkan device. Each selected device has
its own logical device, memory manager, pipeline cache, transfer engine, compute/transfer queues,
completion timeline, timestamp state, and device-buffer replicas. Runtime selection and capability
checks remain vendor-neutral; device identity is based on Vulkan enumeration and UUID data, never
vendor IDs.

## Select and inspect devices

Use `RuntimeOptions.device_policy = DevicePolicy::ALL_ELIGIBLE` to initialize all eligible devices.
`runtime.devices` returns the stable `DeviceId` values owned by that runtime. Use
`runtime.device_info(device)` and `runtime.memory_stats(device)` in C++ to inspect a selected device.
Use `DevicePolicy::BY_INDEX` or `BY_UUID` when an application needs a single explicit device.

```cpp
kora::RuntimeOptions options;
options.device_policy = kora::DevicePolicy::AllEligible;
auto runtime = kora::Runtime::create(options);
if (!runtime) return runtime.error();

const auto devices = runtime.value().devices();
for (const auto& id : devices) {
    auto info = runtime.value().device_info(id);
    if (!info) return info.error();
    // Use id to load a device-specific Kernel and route dispatches.
}
```

## Device-specific kernels and dispatch

`Runtime::load_kernel(path, entry_point, metadata, device)` validates SPIR-V against the selected
device's queried limits/features and associates the returned `Kernel` with that device. Calling
`Runtime::dispatch(kernel)` routes the builder to the matching owned device context. A kernel from
another Runtime is rejected at submission. Pipeline caches, descriptors, queues, buffer replicas,
and timeline values are isolated per device.

Buffers are logical, host-shadow-backed objects shared by the Runtime. A dispatch creates or reuses
the selected device's replica. Host writes advance the host content version; the next device consumer
uploads the current host snapshot when its replica is stale.

## Cross-device staging through host memory

No Vulkan external-memory or peer-transfer path is enabled: the project has no evidence that such a
path is both supported and faster across all target implementations. The portable transfer contract
uses the shared host shadow as an explicit staging point:

1. Dispatch a producer kernel on device A and call `defer_readback()` to retain its output on A.
2. Wait for the producer Event, then call `Runtime::readback(buffer, device_a)` and wait for that Event.
3. Load the consumer kernel for device B and dispatch it with the staged buffer as input. Since B's
   replica is not current at the host version, KORA uploads the host snapshot to B before dispatch.

The readback overload selects the source replica and uses that device's own compute completion
timeline before its transfer queue downloads the bytes. Device-local last-use values are not
interchangeable between queues. The source readback updates the host shadow only if it has not changed
since the transfer was scheduled. The destination dispatch then observes the resulting content
version. This is a correctness-first host bounce; it is not presented as peer-to-peer transfer or as
an optimized cross-device scheduler.

```cpp
auto produced = runtime.dispatch(kernel_a)
    .global(count).read(input_a).read(input_b).write(shared).push(count)
    .defer_readback().submit();
if (!produced) return produced.error();
if (auto status = produced.value().wait(); !status) return status;

auto staged = runtime.readback(shared, device_a);
if (!staged) return staged.error();
if (auto status = staged.value().wait(); !status) return status;

auto consumed = runtime.dispatch(kernel_b)
    .global(count).read(shared).read(other_input).write(output).push(count).submit();
```

Without `defer_readback()`, output bindings are copied to the host as part of ordinary dispatch
completion. `readback(buffer, device)` is useful when the output was deferred or when choosing a
non-default source device explicitly. The legacy `readback(buffer)` overload continues to select the
runtime's first selected device.

## Python API

The optional `kora` package exposes `RuntimeOptions`, `DevicePolicy.ALL_ELIGIBLE`,
`Runtime.devices`, and `DeviceId`. Pass a returned ID to `runtime.load_kernel(..., device)` and
`runtime.readback(buffer, device)`. Each operation still uses the native C++ implementation and
returns the same awaitable `Event` described in [Python package](python-api.md).

## Validation and limitations

The integration test requests all eligible local devices and exercises each ordered source/destination
pair. It verifies that a vector produced on one device, staged through host memory, and consumed on
another matches the CPU expected values. It skips when Vulkan is unavailable or fewer than two
eligible devices exist, as in single-device CI environments. Local runs can cover MoltenVK, llvmpipe,
and other installed implementations; the exact available pair matrix is machine-dependent.

Current device selection initializes the chosen devices and lets the caller name a target explicitly.
It does not rewrite arbitrary kernels or automatically schedule a graph. A caller may opt a
range-aware operation into `DynamicPartitioner`: the policy returns non-overlapping contiguous
`PartitionSlice` ranges with a device assignment. The caller remains responsible for passing each
range to its kernel (typically as base/count parameters) and merging the per-device outputs. This
explicit contract avoids assuming that any arbitrary dispatch can be split safely.

## Partitionable task policy

`DynamicPartitioner::observe(device, work_units, duration_ns)` records completed-work throughput as
work-units per nanosecond and updates the per-device estimate with an EWMA. `partition(request,
devices)` uses those measured rates when available and otherwise each `PartitionDevice` cold-start
rate. The caller supplies a current per-task memory allowance for each device; a slice is never
assigned beyond that byte cap.

The planner assigns at least the configured minimum chunk to each participating device, distributes
granularity-sized chunks proportionally to throughput, and gives any sub-granularity tail to the final
slice. Offsets are ordered, contiguous, and cover the requested extent exactly. If only one device
fits, multiple minimum chunks cannot fit, or predicted makespan including partition overhead does
not improve the unsplit estimate by `minimum_gain_fraction`, the result is a single full-range slice
on the fastest device that can hold it. If no one device can hold the unsplit task, the planner
returns `OutOfMemoryBudget`; it does not silently produce an incomplete plan. The planner is a pure
advisory primitive and does not submit work, synchronize devices, or merge buffers.

| API | Contract |
| --- | --- |
| `PartitionDevice` | Stable `DeviceId`, task memory allowance, and cold-start throughput in work units/ns. |
| `PartitionRequest` | Extent, bytes per unit, chunk granularity/minimum, unsplit duration, split overhead, and required gain. |
| `DynamicPartitioner::observe()` | Updates one device's throughput EWMA from a positive completed-work sample. |
| `DynamicPartitioner::partition()` | Returns an exact contiguous range plan, a single-device fallback, or a structured error. |
| `PartitionSlice` | Device, zero-based offset, extent, and throughput share used as its weight. |

The unit suite checks proportional allocation, rebalance, granularity, memory limits, invalid input,
and unprofitable/one-device fallback. The Vulkan integration test runs disjoint vector-add ranges on
two or more local devices and compares their merged output to the unsplit CPU result. It skips when
the machine exposes fewer than two eligible Vulkan devices.

Cross-device input movement still uses the explicit host-shadow staging contract above; no
peer/external-memory fast path is enabled.
