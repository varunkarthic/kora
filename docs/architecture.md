# Architecture

KORA separates public contracts, planning, execution, memory, backend integration, and operational
services so optional Vulkan capabilities do not leak into ordinary application code.

```{mermaid}
flowchart TD
  API["C++ API / Python bindings"] --> TASKS["Tasks, graphs, execution plans"]
  TASKS --> SCHED["Measured scheduler"]
  SCHED --> CPU["CPU executor"]
  SCHED --> GPU["Vulkan queue contexts"]
  GPU --> KERNEL["SPIR-V validation and pipelines"]
  GPU --> SYNC["Hazards and synchronization"]
  GPU --> MEMORY["Pools, staging, residency, eviction"]
  CPU --> EVENTS["Common events and completion"]
  GPU --> EVENTS
  EVENTS --> PROFILE["Profiling, traces, recovery"]
```

## Ownership model

`Runtime` is planned as the top-level owner of devices, queues, schedulers, memory managers, kernel
registry, CPU executor, profiling, persistence, and recovery services. Public handles should retain
safe shared state without exposing Vulkan object ownership.

Each Vulkan device will own queue-context workers. A worker owns its command pool, records and batches
operations, and submits them in queue order. Completion processing resolves events, callbacks,
deferred frees, and staging reclamation away from the submission path.

## Data and execution model

A `Buffer` is a logical object with a host shadow and zero or more device replicas. Residency is
planned from submitted operations rather than inferred only after completion. Task access modes drive
dependency inference and RAW, WAR, and WAW hazard generation.

The scheduler will compare measured execution time, transfer cost, queue delay, data locality, memory
pressure, and downstream movement. Static vendor-specific rules are prohibited.

## Synchronization model

Timeline semaphores and synchronization2 are preferred when available, with conservative Vulkan 1.1
fallbacks. Queue-family ownership transfers must model release and acquire operations correctly for
exclusive resources. Multiple queues never imply independent physical engines.

## Failure model

Expected failures use `Result<T>` and `Status`. Backend errors retain Vulkan results and add device,
task, kernel, recovery, and nested-cause context. Device loss transitions through a defined recovery
state machine; recovery cannot claim success when authoritative data has been lost.

## Implemented versus planned

The directories under `src/` reserve subsystem boundaries, but an empty or scaffolded directory is
not an implementation claim. Consult the current source and closed GitHub issues for implemented
behavior, and use release validation rather than this architecture page alone as evidence.
