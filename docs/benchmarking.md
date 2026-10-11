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

The GPU run requires a usable Vulkan device and compiled vector-add, SAXPY, reduction, scan, tiled
matmul, and stencil fixtures. Missing build-time
shader tooling or runtime Vulkan support returns exit code 3; the CTest wrapper reports that as a
skip. The installed executable looks for its shader next to the binary or under
`../share/kora/benchmarks`.

### Python callback measurement

The Python callback-boundary measurement is an optional companion report and requires the `python`
CMake preset (which builds the native extension):

```sh
cmake --preset python
cmake --build --preset python --parallel
python3 tools/kora-bench/python_overhead.py --iterations 1000 --warmup 50 --tasks 64 \
  > kora-python-bench.json
ctest --test-dir build/python -R KoraBench.PythonSmoke --output-on-failure
```

The script creates one 64-task plan, warms replay, then records the end-to-end duration of each replay
with a Python callback for every task. It emits raw `perf_counter_ns` samples and nearest-rank median
and p95. The timing includes native/Python callback crossings and interpreter scheduling; compare it
only with measurements from the same host, interpreter, and build. It does not represent GPU dispatch
or kernel performance.

## Workloads

| Report name | Measured operation | Notes |
| --- | --- | --- |
| `scheduler_cost` | Evaluate the deterministic additive scheduler cost model. | Synthetic inputs; measures host policy calculation, not a device-selection loop. |
| `task_graph_finalize_64` | Build 64 independent tasks and finalize the graph. | Includes graph construction and deterministic planning. |
| `task_graph_replay_64` | Build a 64-task execution plan and replay callbacks. | Includes plan preparation; callback body only increments a counter. |
| `task_graph_replay_64_reuse_plan` | Replay a prebuilt 64-task plan through a C++ callback. | Measures native replay and callback cost without rebuilding the graph/plan; comparable in scope to the Python companion measurement. |
| `allocator_allocate_free_256` | Allocate and free 256 aligned suballocations. | Includes allocator creation and invariant validation. |
| `allocator_fragmentation_128` | Allocate 128 varied blocks, free alternating blocks, and query stats. | Confirms nonzero external fragmentation and valid allocator invariants. |
| `host_allocation_4k` | Allocate and initialize a 4 KiB host vector. | Standard-library allocation baseline. |
| `cpu_vector_add_N` | Add two uint32 arrays for `N` in 1, 64, 256, 1024, 4096, 65536, 262144, 1048576, and 4194240. | Arithmetic-only CPU baseline; arrays are initialized outside the timed loop. |
| `gpu_tiny_dispatch_1` | Submit one vector-add work item on Vulkan and wait for completion. | Warm steady-state host latency with buffers and pipeline resident. |
| `gpu_tiny_dispatch_batch_10` | Submit 10 one-work-item vector-add dispatches sequentially, waiting for each event. | Measures repeated small-dispatch scheduling and completion latency; output correctness is checked before timing. |
| `gpu_vector_add_N_dispatch_wait` | Dispatch N vector-add work items for N in 64, 256, 1024, 4096, 65536, 262144, 1048576, and 4194240, then wait. | Inputs and pipeline are made resident before timing; output correctness is checked outside the timed loop. |
| `gpu_saxpy_4096_upload_dispatch_wait` | Reset the host input, dispatch 4096 SAXPY elements, and wait. | Includes upload and dispatch wait; correctness is checked before timing. |
| `gpu_reduction_4096_dispatch_wait` | Reduce 4096 float values into 64 workgroup sums and wait. | Inputs are resident; each 64-value tile is validated before timing. |
| `gpu_tiled_scan_4096_dispatch_wait` | Inclusive scan each 64-value workgroup tile and wait. | Inputs are resident; tile prefixes are validated before timing. |
| `gpu_tiled_matmul_64_dispatch_wait` | Multiply two 64×64 matrices using 8×8 shared-memory tiles and wait. | Inputs are resident; every output is checked before timing. |
| `gpu_stencil_4096_dispatch_wait` | Apply a three-point 1D stencil to 4096 values and wait. | Inputs are resident; boundaries and interior are checked before timing. |
| `raw_vulkan_vector_add_4096_dispatch_wait` | Reuse a Vulkan compute pipeline, descriptor set, and command buffer to dispatch 4096 vector-add elements and wait on a fence. | Same vector-add SPIR-V fixture and size as the KORA GPU case; excludes setup, upload, and readback. |
| `command_pool_reuse_steady_state` | Begin, end, and reset a command buffer from a persistent bounded ring. | Host-side Vulkan API overhead; no queue submit or GPU execution. |
| `command_pool_cold_create_record_destroy` | Create a one-slot pool, record/reset one buffer, and destroy it. | Cold setup comparison for the reuse path; no queue submit or GPU execution. |
| `transfer_direct_upload_N` / `transfer_direct_download_N` | Transfer N bytes through a persistently mapped device-local, host-visible Vulkan buffer without a staging ring. | Includes KORA queue submission and completion; the direct-mapped policy is explicitly selected for this controlled comparison. |
| `transfer_staged_upload_N` / `transfer_staged_download_N` | Transfer N bytes through KORA's bounded upload/readback staging rings. | Same device-local, host-visible target buffer and queue as the direct case; correctness is checked before timing. |
| `raw_vulkan_copy_N` | Record `vkCmdCopyBuffer`, submit it with a Vulkan fence, and wait for completion. | Raw API baseline using the same device-local source/destination buffers and transfer queue. |
| `kora_transfer_copy_N` | Copy between the same buffers using `TransferEngine::copy()` and wait for its event. | Includes KORA queue submission, completion, and transfer bookkeeping. |

## Report contract

The root object has `schema_version`, `clock`, `iterations`, `warmup`, and `results`. Each result has
a stable `name`, `unit`, `median`, `p95`, and `samples` array. Transfer results also report
`bytes_transferred` and `throughput_bytes_per_second`, derived from transferred bytes and median
latency. Percentiles use nearest-rank ordering:
the rank is `ceil(p * N)`. The runner emits workload records in a fixed order and records no paths,
device identifiers, or private configuration. Timing values naturally vary with host load and
compiler, so compare reports only when the machine, build type, compiler, and workload parameters
are recorded alongside them.

## Scope and interpretation

CPU-side measurements are always available. Optional GPU cases cover tiny dispatch and size-swept
vector-add/SAXPY, reduction, tiled scan, tiled matmul, stencil, transfer sizes, and raw Vulkan
copy/vector-add dispatch baselines. Transfer results measure end-to-end API latency for the configured
memory path; compute results include KORA runtime scheduling and should not be interpreted as
kernel-only times. Python callback overhead is measured separately as documented above. Scheduler
results do not include end-to-end Runtime device selection because the candidate cost model is not yet
wired into `Runtime::dispatch()`.

## Reference run and results

The following reference was collected on one local machine; it is included to make results
interpretable, not to predict another system. The GPU run explicitly used enumeration index 0 for
KORA (`KORA_DEVICE=0`), matching the raw Vulkan helper's first-enumerated physical device. The
benchmark runner does not yet enforce or emit this identity for arbitrary configurations; until
[issue #170](https://github.com/varunkarthic/kora/issues/170) is resolved, pin the device and record
the environment when comparing KORA against raw Vulkan.

| Environment field | Value |
| --- | --- |
| Operating system | macOS 27.0 (build 26A428), arm64 |
| CPU/device | Apple M4 |
| KORA GPU selection | `KORA_DEVICE=0`, Apple M4 via KosmicKrisp, Vulkan 1.3.354 |
| Compiler | AppleClang 21.0.0 |
| Build | CMake `release` preset, 64-bit |
| CMake / Ninja | CMake 4.4.2; repository preset uses Ninja |
| Python (callback run) | CPython 3.14.7 |
| Shader compiler | glslang 16.6.0 |
| Samples | 1,000 measured iterations; 50 warm-ups unless a separate command states otherwise |

Reproduce CPU/GPU results with:

```sh
cmake --preset release
cmake --build --preset release --parallel
KORA_DEVICE=0 build/release/tools/kora-bench --gpu --iterations 1000 --warmup 50 > kora-bench.json
```

The prepared-plan Python result uses the separate command in the Python callback measurement section above.
The numbers below are medians and nearest-rank p95 values. GPU timings include host-side runtime,
submission, and completion-wait costs; they are not kernel-only duration.

| Workload | Median | p95 | Interpretation |
| --- | ---: | ---: | --- |
| C++ replay, prepared 64-task plan | 0.292 µs | 0.292 µs | Native replay and counter callback |
| Python replay, same plan/callback | 24.333 µs | 24.750 µs | Includes interpreter and Python callback crossings |
| CPU vector add, 4,096 elements | 0.375 µs | 0.542 µs | Host arithmetic only; arrays initialized outside timing |
| KORA vector add, 4,096 elements | 4,783.917 µs | 8,184.458 µs | Warm resident buffers; dispatch and Event completion |
| Raw Vulkan vector add, 4,096 elements | 199.208 µs | 275.334 µs | Same selected device; reused pipeline, descriptors, and buffers |

This local sample has no CPU/GPU crossover within the documented vector-add sizes. The large KORA
versus raw dispatch gap is evidence to investigate this device/driver path, not a portable performance
ratio or product guarantee. Re-run before making design decisions, and preserve the JSON report.

| Vector elements | CPU median | CPU p95 | KORA median | KORA p95 |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0.375 µs | 0.542 µs | 4,783.917 µs | 8,184.458 µs |
| 1,048,576 | 206.750 µs | 234.209 µs | 5,701.500 µs | 9,517.708 µs |
| 4,194,240 | 1,119.875 µs | 4,144.625 µs | 7,253.500 µs | 11,967.208 µs |

### Local command-pool sample

The reference Release run (`KORA_DEVICE=0`, `--gpu --iterations 1000 --warmup 50`) recorded 1,000
host-side samples per command-pool case: persistent-ring reuse had a nearest-rank median of 750 ns
and p95 of 875 ns; creating and destroying a one-slot pool per operation had a median of 1,542 ns
and p95 of 1,583 ns.
This is a single-machine observation of host API overhead, without queue submission or GPU execution;
it is not a portable performance claim and does not establish transfer thresholds or other fast-path
benefits. Re-run the command above on the target system before using these figures for a decision.

The same local Release run compared the transfer paths on enumeration index 0. Values below are
median/p95 microseconds for 1,000 samples after 50 warm-ups; the upload/download cases use a
persistently mapped device-local, host-visible buffer, and the copy cases use a raw Vulkan fence
baseline versus `TransferEngine::copy()`:

| Size | Direct upload | Staged upload | Direct download | Staged download | Raw Vulkan copy | KORA copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 42.125 / 49.375 | 185.292 / 285.084 | 44.792 / 68.167 | 188.125 / 277.583 | 183.875 / 291.709 | 191.208 / 284.708 |
| 64 KiB | 41.292 / 45.209 | 192.792 / 237.083 | 41.625 / 46.625 | 206.625 / 237.917 | 186.500 / 196.500 | 197.375 / 217.625 |
| 1 MiB | 63.709 / 83.458 | 340.708 / 470.625 | 64.875 / 70.375 | 404.750 / 584.584 | 261.583 / 330.625 | 321.208 / 400.625 |

These are end-to-end host-observed latency measurements on one local system, not device-memory
bandwidth claims. The direct path is explicitly selected as a controlled comparison and should not be
inferred to be faster on other memory topologies. Devices without device-local host-visible memory
cannot run this direct/staged comparison, so the optional GPU suite reports unavailable there. The raw
comparison covers buffer copies and one vector-add dispatch, not all KORA compute workloads. Re-run on
each target; do not use these single-host values to set a universal staging threshold.

### Optimization decisions and limits

The evidence above supports keeping the existing bounded command-buffer pool reuse: it reduced
host-side record/reset overhead in this local comparison, and the pool is already the normal path.
It does not justify changing pool sizing or claiming a workload-level speedup because the benchmark
does not submit the recorded buffers to a queue.

No descriptor-update fast path or buffer-device-address (BDA) dispatch path is enabled by these
results. The raw vector-add case reuses a pre-created descriptor set and pipeline, so it isolates
steady-state dispatch overhead but does not compare descriptor allocation/update strategies. The BDA
capability and allocation support likewise do not constitute an implemented or benchmarked BDA
kernel path. Those changes need correctness-equivalent workloads and setup-inclusive as well as
steady-state comparisons before they can be evaluated.

No adaptive dedicated-transfer-queue threshold is inferred. The transfer measurements compare direct
mapping and staging on one host-visible/device-local path; they do not compare dedicated and compute
queues across integrated, discrete, and software devices, nor measure overlap with concurrent compute.
An evidence-backed threshold therefore requires repeatable upload/download size sweeps on those
topologies, queue-family and synchronization validation, concurrent-work measurements, and end-to-end
latency/throughput results. Until then, KORA's conservative capability-based queue selection remains
unchanged. These measurements are diagnostic inputs, not runtime policy or guarantees.
