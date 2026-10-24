# Debugging

## Turn on validation and logs

```sh
KORA_VALIDATION=1 KORA_LOG_LEVEL=debug cargo run -p kora --example vector_add
```

`KORA_VALIDATION=1` requests the Khronos validation layer; its messages go to the KORA log in the
`device` category. If the layer is advertised but cannot load, KORA warns and continues without it.
On macOS with Homebrew, the layer library is found only with
`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`. `KORA_DEBUG_SAFETY=1` (the default in debug builds)
adds synchronization tracing and extra checks.

Route logs into your own logger with `kora::log::set_sink`, or change thresholds per category with
`kora::log::set_category_level`.

## Read the error

Every error carries a stable code and context:

```rust
match runtime.load_kernel(&words, "main", metadata) {
    Err(error) => eprintln!("{}", error.describe()),
    Ok(kernel) => { /* ... */ }
}
```

`describe()` includes the code, message, device, kernel id, task id, Vulkan result, a suggested
recovery action, and the chain of causes. In C, `kora_last_error_message()` returns the same text; in
Python, `str(error)` does, and `error.code`, `error.vk_result`, and friends hold the fields.

## Common problems

| Symptom | Likely cause |
| --- | --- |
| `NoVulkanImplementation` | No Vulkan loader or driver. Install one, or lavapipe on Linux. |
| `NoEligibleDevice` | `KORA_DEVICE` names a missing device, or no device has a compute queue. |
| `InvalidKernel: SPIR-V: bindings[0].name does not match the module` | The manifest and shader disagree; names must match the shader's variables. |
| `InvalidKernelBinding` at submit | Buffers bound in the wrong order, count, or access. |
| `InvalidArgument: ... newer contents on another device` | A deferred result lives on another device; call `readback` first. |
| `InvalidArgument` while mapped | The same thread used a buffer it still has mapped; drop the mapping first. |
| `DataLoss` on read | The buffer's producing dispatch was on a lost device; rewrite it. |
| `OutOfMemoryBudget` | The heap's soft budget is full; drop buffers or raise `KORA_MEMORY_BUDGET_FRACTION`. |
| Slow first dispatch | Pipeline compilation and startup calibration; both are cached in `KORA_CACHE_DIR`. |

## Inspect devices

`kora-info` shows what the loader reports. For a running `Cluster`, `kora-top` shows devices, workers,
memory, and recent failovers live.
