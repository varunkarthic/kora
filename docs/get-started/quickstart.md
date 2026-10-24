# Your first dispatch

Each example adds two vectors of 2^20 unsigned integers on the first device and checks every result.
They share one kernel, [`examples/kernels/vector_add.comp`](https://github.com/varunkarthic/kora/blob/main/examples/kernels/vector_add.comp),
compiled to `vector_add.spv` and described by the manifest `vector_add.json`.

Run them from a clone of the repository:

=== "Rust"

    ```sh
    cargo run --release -p kora --example vector_add
    ```

    ```rust title="crates/kora/examples/vector_add.rs"
    --8<-- "crates/kora/examples/vector_add.rs"
    ```

=== "C"

    ```sh
    tools/ci/check_capi.sh
    ```

    ```c title="crates/kora-capi/examples/vector_add.c"
    --8<-- "crates/kora-capi/examples/vector_add.c"
    ```

=== "Python"

    ```sh
    python3 -m pip install ".[numpy]"
    python3 examples/vector_add.py
    ```

    ```python title="examples/vector_add.py"
    --8<-- "examples/vector_add.py"
    ```

Expected result: a line ending in `0 mismatches` (the Rust and C examples also name the device).

## What just happened

1. **Runtime creation** loaded the Vulkan loader, picked a device (override with
   [`KORA_DEVICE`](../guides/configuration.md)), and created its queues and memory pools.
2. **Loading the kernel** validated the SPIR-V module against the manifest (binding names, access,
   push-constant layout, workgroup size) and compiled the pipeline once, up front.
3. **Writing buffers** filled their host copies. Nothing touched the GPU yet.
4. **Submitting** checked each binding against the manifest, made the stale inputs resident on the
   device, recorded the commands, and returned an `Event`.
5. **Waiting** returned once the kernel finished and the written buffer was copied back to its host
   copy (pass `defer_readback` to keep it on the device for the next dispatch).

Next: [core concepts](concepts.md), or the [kernel manifest format](../guides/kernels.md) to bring your
own shader.
