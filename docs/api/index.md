# API

KORA's core is a Rust crate. The C ABI and the Python package are thin layers over it with the same
concepts and error codes.

| | Rust | C | Python |
| --- | --- | --- | --- |
| Package | `kora` crate | `libkora` + `kora.h` | `kora-runtime` wheel (`import kora`) |
| Errors | `kora::Result<T>` | `KoraStatus` + `kora_last_error_message()` | `KoraError` with `.code` |
| Runtime | `Runtime::new` | `kora_runtime_create` | `Runtime.create` |
| Buffers | `create_buffer::<T>` | `kora_buffer_create` | `create_buffer`, `create_buffer_from(numpy)` |
| Kernels | `load_kernel`, `load_kernel_file`, `load_kernel_on` | `kora_kernel_create[_from_file]` | `load_kernel(path or bytes)` |
| Dispatch | `dispatch(&k).global(n).read(..)...submit()` | `kora_dispatch_*` | `dispatch(k).global_size(n).read(..)...submit()` |
| Wait | `Event::wait` | `kora_event_wait` | `event.wait()`, `await event` |
| Cluster, adaptive dispatch, task graphs | yes | — | task graphs |

<div class="grid cards" markdown>

-   __[Rust](rust.md)__ — the crate's modules and where to find the generated reference.
-   __[C](c.md)__ — every function in `kora.h`, ownership, and threading rules.
-   __[Python](python.md)__ — classes, NumPy interop, asyncio, and differences from 0.x.
-   __[Errors](errors.md)__ — the stable error codes shared by all three.

</div>
