# Task graphs

`TaskGraph` computes dependencies and a deterministic order for work you describe; `ExecutionPlan`
replays that order by calling your function for each task. KORA does not run graph tasks itself.

```rust
use kora::{AccessMode, BufferAccess, ExecutionPlan, TaskGraph, TaskMetadata, TaskType};

let mut graph = TaskGraph::new();
let produce = graph.add_task(TaskMetadata {
    task_type: TaskType::Dispatch,
    accesses: vec![BufferAccess { buffer: tmp.id(), mode: AccessMode::Write }],
    ..TaskMetadata::default()
})?;
let consume = graph.add_task(TaskMetadata {
    task_type: TaskType::Dispatch,
    accesses: vec![BufferAccess { buffer: tmp.id(), mode: AccessMode::Read }],
    ..TaskMetadata::default()
})?;                                      // depends on `produce` automatically (read after write)
graph.add_dependency(produce, consume)?;  // explicit edges are fine too

let plan = ExecutionPlan::new(&graph)?;   // validates and orders once
plan.replay(|task| {
    // dispatch, copy, or run CPU work for `task.id` here
    Ok(())
})?;
```

- Dependencies are inferred from declared buffer accesses (read-after-write, write-after-read,
  write-after-write) plus explicit edges. A cycle fails `finalize` with `GraphCycle` and a member's
  task id.
- The order is a deterministic topological sort (smallest ready id first).
- `replay` allocates nothing, so a plan can be replayed every frame; it stops at the first error and
  attaches the task id.
- `ExecutionPlan::with_preparation` runs a setup function once per task at creation.

In Python, `plan.replay(fn)` calls `fn(task)`, which must return `None` or `True`; `False`, another
value, or an exception stops replay with `PYTHON_BINDING_ERROR`.

## Temporary-buffer aliasing

`kora::aliasing` plans which temporary buffers can share storage because their lifetimes in a plan
never overlap, using reachability rather than pairwise graph searches.
