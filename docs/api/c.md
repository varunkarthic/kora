# C API

`libkora` exposes a stable C ABI generated from Rust by cbindgen. The header is
[`crates/kora-capi/include/kora.h`](https://github.com/varunkarthic/kora/blob/main/crates/kora-capi/include/kora.h).

```sh
cargo build --release -p kora-capi
cc -I crates/kora-capi/include app.c -L target/release -lkora -o app
```

## Rules

- **Status codes.** Every fallible function returns a `KoraStatus`. `KORA_STATUS_OK` is 0; other
  values equal the [error codes](errors.md). On failure, `kora_last_error_message()` describes the
  error for the calling thread, valid until that thread's next failing call.
- **Ownership.** Handles are opaque pointers you own. Release each with its `*_destroy` function;
  destroying `NULL` is a no-op. Buffers, kernels, and events stay valid after their runtime is
  destroyed and can still be destroyed safely.
- **Threads.** Every function is thread-safe. A handle may be used from any thread.
- **Panics** never cross the boundary; an internal panic returns `KORA_STATUS_INTERNAL_INVARIANT_VIOLATION`.
- **Versioning.** `KORA_ABI_VERSION` changes only for incompatible declaration changes; check it
  against `kora_abi_version()` at startup. `KoraRuntimeOptions.struct_size` lets later versions
  append fields.

## Functions

| Function | Purpose |
| --- | --- |
| `kora_abi_version()`, `kora_version_string()` | ABI and library versions |
| `kora_status_name(status)`, `kora_last_error_message()` | Error names and the thread's last error |
| `kora_runtime_options_default(&options)` | Fill options with `KORA_DEFAULT` (use the environment) |
| `kora_runtime_create(options_or_NULL, &runtime)`, `kora_runtime_destroy(runtime)` | Create and release a runtime |
| `kora_runtime_device_count(runtime)`, `kora_runtime_device_name(runtime, i, buf, cap, &required)` | Inspect devices |
| `kora_buffer_create(runtime, bytes, &buffer)`, `kora_buffer_destroy(buffer)` | Zero-initialized storage buffers |
| `kora_buffer_size`, `kora_buffer_write(buffer, data, size)`, `kora_buffer_read(buffer, data, size)` | Whole-buffer host access (`size` must equal the buffer size) |
| `kora_kernel_create(runtime, words, count, entry, manifest_json, &kernel)` | Load a kernel from SPIR-V words |
| `kora_kernel_create_from_file(runtime, path, entry, manifest_json, &kernel)`, `kora_kernel_destroy` | Load from a `.spv` file |
| `kora_dispatch_create(runtime, kernel, &dispatch)` | Start a dispatch (global size 1×1×1) |
| `kora_dispatch_set_global(dispatch, x, y, z)` | Global size in invocations |
| `kora_dispatch_bind(dispatch, buffer, KORA_ACCESS_READ / _WRITE / _READ_WRITE)` | Bind the next buffer in manifest order |
| `kora_dispatch_push(dispatch, data, size)` | Push-constant bytes |
| `kora_dispatch_defer_readback(dispatch)` | Keep outputs on the device |
| `kora_dispatch_submit(dispatch, &event)`, `kora_dispatch_destroy(dispatch)` | Submit (once), then release the builder |
| `kora_readback(runtime, buffer, &event)` | Copy device-resident results to the host copy |
| `kora_event_wait(event)`, `kora_event_wait_timeout(event, ns)`, `kora_event_is_done(event)`, `kora_event_destroy(event)` | Completion |

## Options

```c
KoraRuntimeOptions options;
kora_runtime_options_default(&options);
options.device_policy = KORA_DEVICE_POLICY_BY_INDEX;   /* or KORA_DEFAULT */
options.device_index = 1;
options.enable_validation = 1;                        /* 0, 1, or KORA_DEFAULT */
options.enable_startup_calibration = 0;
options.cache_dir = "/tmp/kora-cache";                /* or NULL */
```

## Complete example

```c title="crates/kora-capi/examples/vector_add.c"
--8<-- "crates/kora-capi/examples/vector_add.c"
```
