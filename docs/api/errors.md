# Errors

Every API reports expected failures with the same stable codes. Values never change; new codes may
be added.

| Value | Rust `ErrorCode` | C `KoraStatus` | Python `ErrorCode` | Meaning |
| ---: | --- | --- | --- | --- |
| 0 | — | `KORA_STATUS_OK` | `OK` | Success. |
| 1 | `NoVulkanImplementation` | `KORA_STATUS_NO_VULKAN_IMPLEMENTATION` | `NO_VULKAN_IMPLEMENTATION` | No Vulkan loader or implementation could be loaded. |
| 2 | `NoEligibleDevice` | `KORA_STATUS_NO_ELIGIBLE_DEVICE` | `NO_ELIGIBLE_DEVICE` | No physical device satisfied the selection policy and requirements. |
| 3 | `NoComputeQueue` | `KORA_STATUS_NO_COMPUTE_QUEUE` | `NO_COMPUTE_QUEUE` | The selected device exposes no compute-capable queue family. |
| 4 | `UnsupportedFeature` | `KORA_STATUS_UNSUPPORTED_FEATURE` | `UNSUPPORTED_FEATURE` | A required feature, extension, or capability is not available. |
| 5 | `InvalidKernel` | `KORA_STATUS_INVALID_KERNEL` | `INVALID_KERNEL` | A kernel module or its manifest is malformed or inconsistent. |
| 6 | `InvalidKernelBinding` | `KORA_STATUS_INVALID_KERNEL_BINDING` | `INVALID_KERNEL_BINDING` | Dispatch bindings do not match the kernel interface. |
| 7 | `InvalidDispatch` | `KORA_STATUS_INVALID_DISPATCH` | `INVALID_DISPATCH` | Dispatch dimensions or parameters are outside device limits. |
| 8 | `AllocationFailed` | `KORA_STATUS_ALLOCATION_FAILED` | `ALLOCATION_FAILED` | A host or device allocation failed. |
| 9 | `OutOfMemoryBudget` | `KORA_STATUS_OUT_OF_MEMORY_BUDGET` | `OUT_OF_MEMORY_BUDGET` | The request does not fit the configured memory budget. |
| 10 | `TransferFailed` | `KORA_STATUS_TRANSFER_FAILED` | `TRANSFER_FAILED` | A host/device transfer failed. |
| 11 | `Timeout` | `KORA_STATUS_TIMEOUT` | `TIMEOUT` | A wait expired before the operation completed (the operation continues). |
| 12 | `DeviceLost` | `KORA_STATUS_DEVICE_LOST` | `DEVICE_LOST` | The logical device was lost. |
| 13 | `PipelineCreationFailed` | `KORA_STATUS_PIPELINE_CREATION_FAILED` | `PIPELINE_CREATION_FAILED` | Compute pipeline creation failed. |
| 14 | `SynchronizationError` | `KORA_STATUS_SYNCHRONIZATION_ERROR` | `SYNCHRONIZATION_ERROR` | A synchronization primitive failed or was misused. |
| 15 | `ValidationError` | `KORA_STATUS_VALIDATION_ERROR` | `VALIDATION_ERROR` | Validation (or duplicate verification) detected an error. |
| 16 | `PythonBindingError` | `KORA_STATUS_PYTHON_BINDING_ERROR` | `PYTHON_BINDING_ERROR` | A Python callback failed or returned an invalid value. |
| 17 | `InternalInvariantViolation` | `KORA_STATUS_INTERNAL_INVARIANT_VIOLATION` | `INTERNAL_INVARIANT_VIOLATION` | An internal invariant was violated (a KORA bug; please report it). |
| 18 | `InvalidArgument` | `KORA_STATUS_INVALID_ARGUMENT` | `INVALID_ARGUMENT` | Bad caller input: out-of-range values, mismatched sizes, misuse. |
| 19 | `Overflow` | `KORA_STATUS_OVERFLOW` | `OVERFLOW` | Checked arithmetic overflowed. |
| 20 | `GraphCycle` | `KORA_STATUS_GRAPH_CYCLE` | `GRAPH_CYCLE` | A task graph contains a cycle. |
| 21 | `Cancelled` | `KORA_STATUS_CANCELLED` | `CANCELLED` | Work was cancelled by shutdown or by the caller. |
| 22 | `StaleHandle` | `KORA_STATUS_STALE_HANDLE` | `STALE_HANDLE` | A buffer handle refers to a released or reused slot. |
| 23 | `DataLoss` | `KORA_STATUS_DATA_LOSS` | `DATA_LOSS` | Authoritative data existed only on a lost device. |
| 24 | `IoError` | `KORA_STATUS_IO_ERROR` | `IO_ERROR` | A persistent cache, file, or network operation failed. |

## Context

Beyond the code, an error carries a message and, when known, the device, kernel id, task id, Vulkan
result, a suggested recovery action, and a chain of causes:

- Rust: `error.code()`, `message()`, `device()`, `kernel_id()`, `task_id()`, `vk_result()`,
  `recovery_action()`, `cause()`, and `describe()` for everything on one line.
- C: `kora_last_error_message()` returns `describe()` for the calling thread.
- Python: `KoraError` has `code`, `code_name`, `message`, `device`, `kernel_id`, `task_id`,
  `vk_result`, and `recovery_action`; `str(error)` is `describe()`.

`ErrorCode::from_i32` and `ErrorCode::from_name` convert stored codes back.
