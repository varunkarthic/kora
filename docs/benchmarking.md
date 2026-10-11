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

`kora-bench` is installed with the other command-line tools under `bin/`. `tools/ci/check_install.sh`
checks the installed CPU smoke report; when shader tooling is available, the optional GPU fixtures are
installed under `share/kora/benchmarks` for `--gpu`.

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

If `glslangValidator` was available when configured, add the optional Vulkan cases:

```sh
build/release/tools/kora-bench --gpu --iterations 1000 --warmup 50 > kora-gpu-bench.json
ctest --preset debug -R KoraBench.GpuSmoke --output-on-failure
```

The GPU run requires a usable Vulkan device and compiled vector-add/SAXPY fixtures. Missing build-time
shader tooling or runtime Vulkan support returns exit code 3; the CTest wrapper reports that as a
skip. The installed executable looks for its shader next to the binary or under
`../share/kora/benchmarks`.

## Workloads

| Report name | Measured operation | Notes |
| --- | --- | --- |
| `scheduler_cost` | Evaluate the deterministic additive scheduler cost model. | Synthetic inputs; measures host policy calculation, not a device-selection loop. |
| `task_graph_finalize_64` | Build 64 independent tasks and finalize the graph. | Includes graph construction and deterministic planning. |
| `task_graph_replay_64` | Build a 64-task execution plan and replay callbacks. | Includes plan preparation; callback body only increments a counter. |
| `allocator_allocate_free_256` | Allocate and free 256 aligned suballocations. | Includes allocator creation and invariant validation. |
| `allocator_fragmentation_128` | Allocate 128 varied blocks, free alternating blocks, and query stats. | Confirms nonzero external fragmentation and valid allocator invariants. |
| `host_allocation_4k` | Allocate and initialize a 4 KiB host vector. | Standard-library allocation baseline. |
| `cpu_vector_add_1024` | Add two 1024-element uint32 arrays. | Arithmetic-only CPU baseline; no allocation in the timed loop. |
| `gpu_tiny_dispatch_1` | Submit one vector-add work item on Vulkan and wait for completion. | End-to-end host latency, including runtime buffer transfers. |
| `gpu_vector_add_1024_end_to_end` | Submit 1024 vector-add work items on Vulkan and wait. | Includes upload, dispatch, and output readback; not kernel-only duration. |
| `gpu_saxpy_1024_end_to_end` | Submit a 1024-element SAXPY on Vulkan and wait. | Resets host input each iteration; includes upload and readback. |

## Report contract

The root object has `schema_version`, `clock`, `iterations`, `warmup`, and `results`. Each result has
a stable `name`, `unit`, `median`, `p95`, and `samples` array. Percentiles use nearest-rank ordering:
the rank is `ceil(p * N)`. The runner emits workload records in a fixed order and records no paths,
device identifiers, or private configuration. Timing values naturally vary with host load and
compiler, so compare reports only when the machine, build type, compiler, and workload parameters
are recorded alongside them.

## Scope and interpretation

CPU-side measurements are always available. Optional GPU cases currently cover tiny dispatch,
vector-add, and SAXPY. Device transfer bandwidth, direct-versus-staged copies, reduction/scan/matmul/
stencil kernels, raw-Vulkan baselines, Python overhead, and size-swept CPU/GPU crossover are not
currently measured. Those require workload-specific fixtures and
hardware/environment qualification; do not infer them from other results. Scheduler results do not
include end-to-end Runtime device selection because the candidate cost model is not yet wired into
`Runtime::dispatch()`.
