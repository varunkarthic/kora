# Benchmarking

`kora-bench` is a reproducible microbenchmark runner for KORA's CPU-side runtime policies. It emits
one JSON document to standard output, uses `std::chrono::steady_clock`, performs warm-up iterations,
and retains each measured sample. The report contains nearest-rank median and p95 values in
nanoseconds as well as raw samples, so downstream analysis can recompute summaries.

## Build and run

```sh
cmake --preset release
cmake --build --preset release --parallel
build/release/tools/kora-bench --iterations 1000 --warmup 50 > kora-bench.json
```

Expected result: the command exits successfully and writes valid JSON with schema version `1`. The
report is intentionally written to stdout; redirect it to a new or intentionally replaceable path.
`--smoke` runs three measured samples and one warm-up per workload, and is used by CTest without
requiring a Vulkan device:

```sh
build/debug/tools/kora-bench --smoke
ctest --preset debug -R KoraBench.Smoke --output-on-failure
```

The smoke test parses the report with Python's standard JSON parser and verifies stable workload
names, sample counts, and percentile ordering. Invalid options return exit code 2; workload errors
return 1.

## Workloads

| Report name | Measured operation | Notes |
| --- | --- | --- |
| `scheduler_cost` | Evaluate the deterministic additive scheduler cost model. | Synthetic inputs; measures host policy calculation, not a device-selection loop. |
| `task_graph_finalize_64` | Build 64 independent tasks and finalize the graph. | Includes graph construction and deterministic planning. |
| `task_graph_replay_64` | Build a 64-task execution plan and replay callbacks. | Includes plan preparation; callback body only increments a counter. |
| `allocator_allocate_free_256` | Allocate and free 256 aligned suballocations. | Includes allocator creation and invariant validation. |
| `allocator_fragmentation_128` | Allocate 128 varied blocks, free alternating blocks, and query stats. | Confirms nonzero external fragmentation and valid allocator invariants. |
| `host_allocation_4k` | Allocate and initialize a 4 KiB host vector. | Standard-library allocation baseline. |

## Report contract

The root object has `schema_version`, `clock`, `iterations`, `warmup`, and `results`. Each result has
a stable `name`, `unit`, `median`, `p95`, and `samples` array. Percentiles use nearest-rank ordering:
the rank is `ceil(p * N)`. The runner emits workload records in a fixed order and records no paths,
device identifiers, or private configuration. Timing values naturally vary with host load and
compiler, so compare reports only when the machine, build type, compiler, and workload parameters
are recorded alongside them.

## Scope and interpretation

These are CPU-side microbenchmarks, not GPU performance claims. Device dispatch/transfer bandwidth,
direct-versus-staged copies, SAXPY/vector-add/reduction/scan/matmul/stencil kernels, raw-Vulkan
baselines, Python overhead, and CPU/GPU crossover are not currently measured by this executable.
Those require workload-specific shader fixtures and hardware/environment qualification; do not infer
them from the included host measurements. Scheduler results do not include end-to-end Runtime device
selection because the candidate cost model is not yet wired into `Runtime::dispatch()`.
