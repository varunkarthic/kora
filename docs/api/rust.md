# Rust API

```toml
[dependencies]
kora = { git = "https://github.com/varunkarthic/kora" }
```

The generated reference for every public item is published with this site:
**[KORA rustdoc](https://kora.readthedocs.io/en/latest/rustdoc/kora/)**. Build it locally with:

```sh
cargo doc -p kora --no-deps --open
```

## Where things live

| Item | Purpose |
| --- | --- |
| `Runtime`, `RuntimeOptions` | Devices, buffers, kernels, dispatch, readback, recovery, telemetry |
| `Buffer<T>`, `BufferHandle` | Typed and untyped logical buffers; `map_read`/`map_write` guards |
| `Kernel`, `KernelMetadata`, `parse_kernel_metadata` | Validated kernels and their manifests |
| `DispatchBuilder`, `DispatchPriority` | Building and submitting dispatches |
| `Event`, `EventPromise` | Completion handles, callbacks, `when_all` |
| `Cluster`, `ClusterOptions`, `OffloadPolicy` | Local devices plus remote workers ([guide](../guides/cluster.md)) |
| `WorkerServer`, `WorkerOptions` | Embedding a worker |
| `AdaptiveDispatchWorkload` | [Adaptive placement](../guides/adaptive.md) |
| `TaskGraph`, `ExecutionPlan` | [Task graphs](../guides/task-graphs.md) |
| `CpuExecutor` | [Work-stealing thread pool](../guides/cpu.md) |
| `DeviceInfo`, `DeviceId`, `DevicePolicy` | Capabilities and [selection](../guides/devices.md) |
| `Error`, `ErrorCode`, `Result`, `Status` | [Errors](errors.md) |
| `kora::vulkan`, `kora::memory`, `kora::kernel` | Lower-level building blocks (instances, devices, queues, pools, staging, SPIR-V reflection) used by the runtime and tools |

## Conventions

- Fallible operations return `kora::Result<T>`; nothing panics on bad input.
- `Runtime`, `Buffer`, `Kernel`, `Event`, and `Cluster` are `Send + Sync`. Builders are consumed by
  `submit`.
- Buffers hold `T: bytemuck::Pod` values; `kora::bytemuck` is re-exported.
- Dropping a `Runtime` drains its queues; dropping a `Cluster` waits for submitted work and
  disconnects workers.
- The minimum supported Rust version is 1.88 (edition 2024).
