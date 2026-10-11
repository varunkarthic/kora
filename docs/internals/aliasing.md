# Temporary-buffer aliasing internals

`plan_temporary_buffer_aliasing()` maps graph-temporary buffers to compatible allocation slots.
It is a planning primitive, not a Vulkan allocator: the caller still creates/binds memory, performs
required synchronization, and retains resources until asynchronous work has completed.

## Lifetime and safety model

The planner collects every task that declares an access to each supplied `BufferId`. Two lifetimes
may share a slot only if every use of one buffer reaches every use of the other through task-graph
dependency edges. This happens-before test is deliberately more conservative than assigning a
single interval from one topological ordering: independent branches may execute concurrently even
if a deterministic topological listing puts one branch first. Same-task uses also overlap for this
purpose and cannot share storage.

Persistent or user-visible resources always receive a private slot. A slot can be reused only when
the exact `usage_class` agrees, its size and alignment satisfy the new requirements, and the
intersection of all Vulkan-style `memory_type_bits` remains non-empty. Placements currently use
offset zero; this API does not suballocate within a slot. The resource-creation layer must provide
requirements in a normalized compatibility class and must not infer alias safety for untracked or
asynchronous uses.

The reported byte totals sum slot sizes and requested sizes, respectively. They estimate peak backing
storage for these requirements; they exclude allocator metadata, external fragmentation, alignment
padding between slots, and resources not supplied to the planner.

## Function reference

| Function or field | Contract |
| --- | --- |
| `plan_temporary_buffer_aliasing(graph, buffers)` | Validates task ordering and requirements, computes DAG lifetimes, and greedily colors compatible non-overlapping lifetimes. |
| `TemporaryBufferRequirements` | Supplies generation-aware identity, byte size, alignment, exact usage compatibility class, memory-type mask, and persistence policy. |
| `TemporaryBufferPlacement` | Associates each input buffer with a slot ID and byte offset (currently always zero). |
| `TemporaryAllocationPlan` | Describes the slot size and aggregate compatibility requirements the allocator must honor. |
| `TemporaryBufferAliasingPlan::planned_savings_bytes()` | Returns the non-negative difference between unaliased requested bytes and planned slot bytes. |

The unit suite checks chain-graph coloring, persistent and incompatible resources, concurrent
branches, invalid requirements, and a simulated callback execution using the planned backing slots.
The simulation validates planner correctness only; device-memory binding and asynchronous retirement
require a later backend integration layer.
