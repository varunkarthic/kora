# Kernels and manifests

A KORA kernel is a SPIR-V compute module and a JSON **manifest** that declares its interface. Loading
compares the two and compiles the pipeline once; every dispatch is then checked against the manifest.

## Write and compile the shader

Any SPIR-V producer works. With GLSL and `glslangValidator`:

```sh
glslangValidator -V --target-env vulkan1.0 -o vector_add.spv examples/kernels/vector_add.comp
```

The module must contain a `GLCompute` entry point. KORA supports storage-buffer bindings at dispatch
time; push constants and specialization constants are supported.

## The manifest

```json title="examples/kernels/vector_add.json"
--8<-- "examples/kernels/vector_add.json"
```

| Field | Required | Meaning |
| --- | --- | --- |
| `schema_version` | yes | Always `1`. |
| `kernel_id` | yes | A stable name used in errors, traces, and performance records. |
| `bindings` | yes | `set`, `binding`, `name` (must match the SPIR-V variable), `kind` (`storage_buffer`, `uniform_buffer`, `storage_image`, `sampled_image`), and `access` (`read`, `write`, `read_write`). Dispatches bind buffers in this order. |
| `push_constant_size` | yes | Size in bytes of the push-constant block (at most 65,536). |
| `push_constants` | yes | Named fields: `name`, `type` (`u32`, `i32`, `f32`, …), `offset`, `size`. Fields may not overlap. |
| `specialization_constants` | yes | `id`, `name`, `type`, and `default` (bool or number). May be empty. |
| `local_size` | yes | Workgroup size `[x, y, z]`; each dimension non-zero. |
| `cpu_fallback_key` | no | Name of a registered [CPU fallback](cpu.md). |
| `semantics` | no | Free-form tags such as `"elementwise"`. |

Parsing is strict: unknown fields, duplicate members, trailing data, and non-finite numbers are
errors (`InvalidKernel`, with a byte offset or field path). Documents are limited to 1 MiB.

## Load

=== "Rust"

    ```rust
    use kora::{parse_kernel_metadata, Runtime, RuntimeOptions};
    use kora::kernel::spirv::words_from_bytes;

    let runtime = Runtime::new(&RuntimeOptions::default())?;
    let metadata = parse_kernel_metadata(&std::fs::read_to_string("vector_add.json")?)?;

    // From words already in memory, for the runtime's first device:
    let kernel = runtime.load_kernel(&words_from_bytes(&std::fs::read("vector_add.spv")?)?, "main", metadata.clone())?;
    // From a file (at most 16 MiB):
    let kernel = runtime.load_kernel_file("vector_add.spv", "main", metadata.clone())?;
    // For a specific device:
    let kernel = runtime.load_kernel_on(&runtime.devices()[1], &words, "main", metadata)?;
    ```

    `KernelMetadata` can also be built in code; call `validate_kernel_metadata` before using it.

=== "C"

    ```c
    KoraKernel* kernel = NULL;
    KoraStatus status = kora_kernel_create_from_file(runtime, "vector_add.spv", "main", manifest_json, &kernel);
    /* or kora_kernel_create(runtime, words, word_count, "main", manifest_json, &kernel) */
    ```

=== "Python"

    ```python
    metadata = kora.KernelMetadata.from_json(open("vector_add.json").read())
    kernel = runtime.load_kernel("vector_add.spv", "main", metadata)          # path
    kernel = runtime.load_kernel(spirv_bytes, "main", metadata, runtime.devices[0])  # bytes, device
    ```

Loading checks, before any dispatch:

- the module is structurally valid SPIR-V with the named `GLCompute` entry point;
- every manifest binding exists in the module with the same name, set, binding, kind, and a
  compatible access qualifier, and the module uses no binding the manifest omits;
- the push-constant block matches `push_constant_size` and field offsets;
- capabilities the module declares (16- or 64-bit arithmetic, `LocalSizeId`, …) are supported by the
  device;
- the workgroup size fits the device limits.

Pipelines are cached in memory per device, keyed by the module and its interface, and persisted to
the [cache directory](configuration.md#cache-directory) so a second process start skips compilation.

## Push constants

=== "Rust"

    Any `bytemuck::Pod` value whose size equals `push_constant_size`:

    ```rust
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Saxpy { scale: f32, count: u32 }

    runtime.dispatch(&kernel).global(n).read(&x).read_write(&y)
        .push(&Saxpy { scale: 2.0, count: n as u32 })
        .submit()?;
    ```

    `push_bytes(&[u8])` takes raw bytes.

=== "C"

    ```c
    struct { float scale; uint32_t count; } push = {2.0f, n};
    kora_dispatch_push(dispatch, (const uint8_t*)&push, sizeof push);
    ```

=== "Python"

    ```python
    builder.push(struct.pack("<fI", 2.0, n))           # bytes
    builder.push(np.array([n], dtype=np.uint32))       # any C-contiguous buffer
    ```

## Workgroup autotuning

A kernel that declares its workgroup size with `LocalSizeId` (backed by specialization constants)
is tuned per device: KORA times a few candidate sizes with GPU timestamps on early dispatches and
then specializes every dispatch with the fastest. Tuning needs `maintenance4` and timestamp support;
without them, or with `KORA_DISABLE_AUTOTUNE=1`, the declared default is used.
