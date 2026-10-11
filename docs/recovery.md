# Admission and out-of-memory recovery

## Device-loss handling

Each Runtime-owned logical device shares a `DeviceFaultState` across its compute and dedicated transfer
queues. A `VK_ERROR_DEVICE_LOST` returned by queue submission or a completion wait atomically marks
that state unhealthy and retains the Vulkan result. The first failing submission receives
`ErrorCode::DeviceLost`; submissions already queued but not submitted are completed with the same
error, and later submissions to any queue sharing that device state are rejected before a Vulkan
queue call. Device loss is never treated as a transient timeout or retried on the same logical device.

An operation that was already submitted when loss is observed has undefined device-side outputs. KORA
marks each writable buffer from that in-flight dispatch as lost; `BufferHandle::read_bytes()` and
read-only or partial mappings then return `ErrorCode::DataLoss` rather than exposing a potentially
stale shadow. A complete `BufferHandle::write_bytes()` replaces the entire host shadow and restores
host readability. A host shadow that remained authoritative for work that was never submitted is
left readable. Errors include the raw `VkResult` and state that device recreation is unavailable;
applications must discard the failed Runtime and create a new one. Runtime-scoped queues, pipelines,
descriptors, synchronization objects, and device replicas are not recreated in place, and automatic
task rescheduling is not attempted because the runtime cannot prove arbitrary tasks are idempotent or
their outputs complete.

The current fault state retains the result code but does not yet query `VK_EXT_device_fault` records or
emit vendor-specific fault addresses. The extension is optional and those diagnostics must remain
best-effort; they cannot be used as a correctness dependency. Timeouts remain poll boundaries and do
not mark a device unhealthy. A future recovery layer may add explicit Runtime recreation and
authoritative-host-buffer restoration without changing this fail-closed behavior.

| API/internal state | Contract |
| --- | --- |
| `DeviceFaultState::healthy()` | False after the first observed `VK_ERROR_DEVICE_LOST`; state is shared across a Runtime device's queues. |
| `DeviceFaultState::last_result()` | Returns the Vulkan result associated with the recorded loss. |
| `QueueContext::submit()` | Rejects work on a lost device before submission; already queued work completes with `DeviceLost`. |
| Device-loss `Error` | Preserves `vk_result` and directs callers to create a new Runtime; no same-device retry is claimed. |

Fault-injection coverage blocks the first submit, injects device loss, queues later work, and verifies
deterministic failure without a completion-thread wait or deadlock. A sibling queue sharing the same
device state is also rejected.

KORA exposes memory admission as an estimate and performs allocator recovery through a finite
sequence. The estimate is advisory: external allocations, heap-budget refreshes, fragmentation, and
concurrent work can change the result before allocation. The allocator remains authoritative and
returns a structured error when recovery is exhausted.

## Admission estimate

`detail::estimate_admission(available, required, likely_eviction, eviction_cost_ms,
fragmentation_risk)` returns an `AdmissionEstimate` containing:

| Field | Meaning |
| --- | --- |
| `required_new_bytes` | Additional allocation size requested by the candidate. |
| `likely_eviction_bytes` | Reclaimable bytes estimated by the caller; this is not a promise that eviction will succeed. |
| `eviction_cost_ms` | Nonnegative estimated cost of preparing those evictions. |
| `budget_after_admission_bytes` | Remaining estimated capacity after admission, or zero when not admitted. |
| `fragmentation_risk` | Caller-provided normalized risk in `[0, 1]`; one means highest risk. |
| `admitted` | Whether available plus estimated reclaimable capacity covers the request. |

Invalid/non-finite cost or risk returns `InvalidArgument`; overflow while combining capacity returns
`Overflow`. The estimate does not mutate budget or evict memory.

## Bounded recovery order

The task-level coordinator invokes each configured action no more than once:

1. Refresh the budget snapshot.
2. Reclaim allocations whose deferred-use timeline has completed.
3. Evict eligible resident buffers (the eviction planner rejects pinned, in-flight, or submitting buffers).
4. Retry the exact requested device-memory allocation once.
5. For a pooled allocation, make one attempt to create a smaller backing block (at least 1 MiB)
   when the request fits; then retry the original suballocation once from that block.
6. At task level, ask the adapter to retry with smaller streaming chunks, if supported.
7. Ask the scheduler to use a registered CPU fallback, if available.
8. Return a structured allocation/budget error with attempted stage names when no action recovers.

Preparation-stage errors do not suppress later actions because a stale budget query or unavailable
reclaim opportunity should not disable the remaining bounded fallbacks. A thrown hook or an invalid
hook result becomes `InternalInvariantViolation`; hooks are expected to report ordinary failures as
`Result<bool>`. `true` means the stage recovered the requested operation, not merely that it ran.
There are no retry loops. The raw device-memory allocator retries only the exact requested size and
returns success only with a real `VkDeviceMemory` handle. Smaller backing-block selection happens in
`MemoryPool::allocate()`, where the original suballocation size is known; the allocation is retried
once only if it fits the smaller block. Chunking and CPU fallback are task-level operations because
neither can satisfy a raw memory-pool allocation request.

Runtime integration must preserve buffer generation, pin counts, queue submission state, and timeline
completion before freeing or reusing device storage. The current runtime connects demand-driven
budget refresh and failure-adaptive pool sizing; deferred reclamation and eviction remain explicit
callbacks until a safe runtime-owned replica release path can satisfy those lifetime checks. The
coordinator and planner do not claim that planning itself frees bytes.

## Function reference

| Function/type | Contract |
| --- | --- |
| `AdmissionEstimate` | Immutable result fields for estimated capacity, eviction cost, fragmentation risk, and admission decision. |
| `estimate_admission()` | Validates cost/risk, uses checked capacity addition, and does not mutate runtime state. |
| `OomRecoveryHooks` | Optional, single-shot stage callbacks; preparation callbacks precede allocation alternatives. |
| `recover_allocation_failure()` | Executes allocator-oriented preparation and one exact-size retry; success requires a successful allocation callback. |
| `recover_task_allocation_failure()` | Executes all configured stages including streaming and CPU fallback, each at most once. |
| `MemoryPool::allocate()` | Uses bounded allocation recovery and may create one smaller block and retry the original suballocation once when it fits. |

Focused tests:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
build/debug/tests/unit/kora_unit_tests --gtest_filter='AdmissionEstimate.*:OomRecovery.*'
```

The expected result is all admission/recovery tests passing. The complete native gate is `ctest
--preset debug --output-on-failure`; Vulkan-unavailable integration cases may be reported as skipped.
