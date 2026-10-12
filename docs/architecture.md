# Architecture

KORA separates public contracts, graph/policy primitives, execution, Vulkan integration, and
operational services. Optional Vulkan capabilities are queried and checked; they are not hidden
correctness requirements.

```{mermaid}
flowchart TD
  CPP["C++ API"] --> RT["Runtime, buffers, kernels, events"]
  PY["Python bindings"] --> RT
  GRAPH["TaskGraph / ExecutionPlan metadata"] -. "caller-controlled" .-> APP["Application orchestration"]
  APP --> RT
  RT -->|"dispatch_adaptive (explicit opt-in)"| SELECT["Calibration-gated device selection"]
  SELECT --> GPU
  RT --> CPU["CPU executor / registered fallbacks"]
  RT --> GPU["Per-device Vulkan dispatch"]
  GPU --> QUEUE["QueueContext and completion"]
  QUEUE --> XFER["Transfers, staging, hazards"]
  QUEUE --> KERNEL["SPIR-V validation and pipeline cache"]
  RT --> MEM["Memory manager and device replicas"]
  RT --> PROFILE["Timestamps, trace export, diagnostics"]
```

The dotted edge is deliberate: a `TaskGraph` and its immutable `ExecutionPlan` describe ordering and
resource access, but replay invokes caller callbacks. They are not implicitly dispatched as Vulkan
operations. Likewise, scheduler cost, policy, transfer-model, and queue-availability helpers are
building blocks. `Runtime::dispatch_adaptive()` is an explicit opt-in for one dispatch; automatic
graph-to-device scheduling is not wired into task-graph replay.

## Ownership and execution

`Runtime` owns its selected Vulkan logical devices and their independent queues, transfer services,
memory managers, pipeline caches, timestamp rings, and device replicas. A `Kernel` is bound to its
owning runtime and selected device. `Runtime::dispatch()` therefore submits to that device; it does
not search for a new device on each call. `Runtime::dispatch_adaptive()` is a separate explicit
alternative that accepts a caller-described workload and returns a builder bound to one compatible
initialized device. It requires complete calibration records for every eligible device and otherwise
retains the kernel's original device. It is deterministic and does not alter default dispatch behavior.

Each `QueueContext` serializes host submission to one Vulkan queue and tracks completion values.
Completion workers resolve events and deferred resource reclamation after GPU completion. Transfers
use per-buffer hazard tracking and staging where required; host-visible direct access is used only
when the selected memory path supports it. Distinct queues do not imply distinct physical engines.

`Buffer` is a logical object with a host shadow and device replicas. Cross-device movement currently
uses explicit host staging. `DevicePolicy::AllEligible` creates independent device state, but task
partitioning, data movement, merging, and result selection remain explicit caller operations; they
are not transparent automatic multi-GPU execution.

## Scheduler and graph boundary

`TaskGraph` infers RAW/WAR/WAW dependencies from generation-aware buffer identities and provides a
deterministic topological plan. `ExecutionPlan` can prepare caller-owned resources and replay callbacks.
The graph layer does not allocate Vulkan resources, submit dispatches, or choose devices.

The scheduler cost evaluator returns a `ScoreBreakdown`; the candidate policy enforces feasibility,
exploration, aging, and hysteresis. Throughput and transfer models expose measurements/estimates.
The opt-in single-dispatch API uses calibration estimates, the EFT scorer, and deterministic candidate
policy, but task-graph replay and online per-kernel performance feedback remain separate. Consequently,
current runtime trace events report completed GPU dispatch timing and queue/device
identity, not scheduler scores. This boundary is important: trace output must not imply a selection
decision KORA did not make.

## Synchronization and failures

Queue submission and completion use timeline values where available, with conservative supported
fallbacks. Buffer RAW/WAR/WAW hazards are lowered to Vulkan barriers. Synchronization2 and
cross-queue ownership transfer are not claimed unless documented by the relevant implementation.

Expected failures use `Result<T>` or `Status`. A device-lost result is retained per logical device,
queued work is failed, and subsequent submissions are rejected. KORA does not recreate a lost logical
device or retry potentially invalid outputs; callers must rebuild from authoritative host data.

## Implementation status map

| Area | Current contract | Detailed reference |
| --- | --- | --- |
| Runtime / Vulkan | Explicit device-bound dispatch by default; opt-in calibration-gated selection among initialized devices | [Runtime API](api/runtime.md), [Vulkan backend](internals/vulkan-backend.md) |
| Task graph | Metadata, hazard inference, deterministic plan and callback replay | [Task graph](task-graph.md) |
| Scheduler | Opt-in deterministic single-dispatch selection; task-graph placement and online per-kernel feedback are not automatic | [Scheduler](scheduler.md), [Multiple devices](multi-gpu.md) |
| Memory | Host shadows, per-device replicas, queried memory-type preferences, budget accounting | [Memory model](memory-model.md) |
| Profiling | Optional completed GPU timestamps and Chrome Trace/Perfetto JSON | [Profiling](profiling.md) |
| Recovery | Device-loss propagation and host-data-loss reporting; no in-place device recreation | [Recovery](recovery.md) |

The [source tree](internals/core.md) and subsystem references describe implementation details. Public
contracts take precedence over implementation notes, and release validation states which checks have
passed for a published version.
