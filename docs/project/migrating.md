# Migrating from 0.x

KORA 1.0 replaces the C++20 runtime with a Rust core. The concepts, environment variables, kernel
manifest format, error codes, and tools carry over; the native API is now Rust or C.

## Choose your API

| 0.x | 1.0 |
| --- | --- |
| C++ headers (`#include <kora/runtime.hpp>`) and CMake `kora::kora` | The `kora` Rust crate, or the C ABI (`kora.h` + `libkora`) from C or C++ |
| pybind11 module built with scikit-build | PyO3 abi3 wheel built with maturin (same `import kora`) |
| `kora-info`, `kora-top`, `kora-bench`, `kora-worker` (C++) | Same tools in the `kora-tools` crate |

## C++ to Rust

| C++ 0.x | Rust 1.0 |
| --- | --- |
| `Runtime::create(options)` → `Result<Runtime>` | `Runtime::new(&options)` → `Result<Runtime>` |
| `runtime.create_buffer<float>(n)` | `runtime.create_buffer::<f32>(n)` |
| `runtime.load_kernel(path, entry, metadata[, device])` | `load_kernel_file(path, entry, metadata)`, `load_kernel_on(&device, &words, entry, metadata)` |
| `runtime.dispatch(k).global(x, y, z)` | `runtime.dispatch(&k).global_3d(x, y, z)` (`global(x)` for 1-D) |
| `.push(value)` | `.push(&value)` with `T: bytemuck::Pod` |
| `builder.defer_readback()` (mutating) | `.defer_readback()` (consuming builder) |
| `event.wait_for(duration)` | `event.wait_timeout(duration)` |
| `runtime.readback(buffer, device)` | `readback_from(&buffer, &device)` |
| `Cluster::create(options)` | `Cluster::new(options)` |
| `Result<T>` / `Status` | `kora::Result<T>` / `kora::Status` (`std::result::Result`) |

## Behavior changes

- Buffer versions are unique across all buffers, and mappings are exclusive: a thread that touches a
  buffer it has mapped gets `InvalidArgument` instead of deadlocking.
- Dependent dispatches now get a memory barrier between them (0.x recorded none), and staged inputs no
  longer block `submit`.
- Results that only a lost device held are marked lost on recovery instead of reverting to older host
  contents.
- Using a deferred result on a different device before reading it back is an error instead of a
  stale read.
- The worker protocol is version 2; 0.x and 1.0 workers and clients reject each other.
- `kora-bench` writes report schema 4 (adds four host families and `implementation`).

## Python

- Requires Python 3.11 or newer (one abi3 wheel).
- `RuntimeOptions` fields default to `None`, meaning "use the environment and built-in defaults".
- `MappedView` belongs to the thread that created it; `release()` ends a mapping early.
- A `DispatchBuilder` can be submitted once.
- New: `Runtime.create_buffer_from`, `Buffer.to_numpy`, SPIR-V bytes in `load_kernel`, `Event.done`,
  `Buffer.id`, `KernelMetadata.from_json`/`to_json`.

## Unchanged

Environment variables (`KORA_DEVICE`, `KORA_PROFILE`, …), the kernel manifest format, error code
names and values, cache contents (rebuilt automatically if incompatible), and `kora-info --json`
(schema 2).
