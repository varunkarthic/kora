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

The graph itself is metadata only. A successful finalized plan is a dependency order, not proof that
the tasks have an available runtime implementation; callback-driven replay is described below.

## Reusable execution plans

`ExecutionPlan::create(graph, preparer, user_data)` finalizes the graph once and stores an immutable
snapshot of its metadata, inferred dependencies, and topological order. When provided, the
`TaskPreparer` runs once per node in topological order so the caller can resolve/cache its own
pipelines, layouts, descriptors, or other task resources in `user_data`. Those resources are
caller-owned and must outlive plan replays that use them. A preparation failure stops plan creation
and receives the current task ID if the error does not already have one.

`ExecutionPlan::replay(executor, user_data)` walks the cached order and invokes the supplied
`TaskExecutor` for each node. It stops at the first failed `Status`, adds that task's ID when the
error does not already identify one, and leaves the plan usable for subsequent replays. The callback
is a plain function pointer plus borrowed user data, so the replay path does not allocate plan or
callback storage.

This layer caches graph analysis and exposes a one-time preparation seam; it does not itself create
Vulkan command buffers, integrate the Runtime pipeline manager, own descriptor resources, or
allocate/alias temporary GPU memory. A caller can prepare and retain such resources in user state and
dispatch them from its executor. Backend resource reuse and temporary-buffer lifetime analysis are
separate follow-up work.
