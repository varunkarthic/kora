# Task graph data model

`kora::TaskGraph` is a metadata-only builder for operations that later execution and scheduling
layers can consume. It does not execute work, select a device, or retain buffer storage. Each added
task receives a monotonically increasing `TaskId`; dependencies refer to these IDs.

## Declaring work

`TaskMetadata` identifies the operation through `TaskType` (`Upload`, `Download`, `Copy`, `Fill`,
`Dispatch`, `CpuTask`, or `HostCallback`) and carries scheduler hints: kernel or CPU function ID,
problem dimensions, priority, optional deadline, allowed/preferred devices, CPU fallback availability,
and estimated read/write/work-unit counts. These fields are descriptive; the graph builder does not
interpret scheduler policy.

Buffer use is expressed with `BufferAccess{BufferId, AccessMode}`. A task may declare any combination
of `Read`, `Write`, and `ReadWrite` accesses. Repeated declarations for the same generation-checked
buffer ID are folded together; a read plus a write becomes `ReadWrite`. Buffer generations are part
of identity, so a recycled slot is not mistaken for the previous allocation.

```cpp
kora::TaskGraph graph;
kora::TaskMetadata upload;
upload.type = kora::TaskType::Upload;
upload.accesses.push_back({input.id(), kora::AccessMode::Write});
auto upload_id = graph.add_task(std::move(upload));
if (!upload_id) return upload_id.error();

kora::TaskMetadata dispatch;
dispatch.type = kora::TaskType::Dispatch;
dispatch.kernel_id = "vector_add";
dispatch.accesses = {{input.id(), kora::AccessMode::Read},
                     {output.id(), kora::AccessMode::Write}};
auto dispatch_id = graph.add_task(std::move(dispatch));
if (!dispatch_id) return dispatch_id.error();

auto plan = graph.finalize();
if (!plan) return plan.error();
```

## Inferred dependencies

For each buffer generation, the builder tracks the latest writer and the active readers:

- A read depends on the latest writer and joins the active-reader set.
- A write depends on the latest writer and every active reader, clears that set, and becomes the
  latest writer.
- A read-write access applies both rules, then becomes the latest writer.

This creates the required read-after-write (RAW), write-after-read (WAR), and write-after-write
(WAW) edges. Independent reads remain unordered. Dependencies are deduplicated and sorted by task
ID. `add_dependency(prerequisite, dependent)` adds explicit ordering and may refer to either task
once both tasks exist; explicit edges can intentionally create a cycle, which is diagnosed by
`finalize()` before an execution plan is returned.

## Finalization and errors

`finalize()` runs Kahn's topological-sort algorithm. When multiple tasks are ready, the lowest task ID
is emitted first, making the returned `TaskGraphPlan::topological_order` deterministic. The plan also
contains each task's sorted prerequisite IDs. An explicit cycle returns `ErrorCode::GraphCycle` and
sets `Error::task_id` to one task left unresolved by the topological pass. Invalid task/access enums, invalid buffer
generations, and unknown dependency IDs return `ErrorCode::InvalidArgument`; task-ID exhaustion
returns `ErrorCode::Overflow`.

The graph is reusable as metadata, but execution/replay semantics are not part of this API yet. A
successful plan is a dependency order only; it is not proof that the tasks themselves have an
available implementation.
