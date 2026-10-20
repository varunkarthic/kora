# Benchmarking

`kora-bench` measures CPU-side runtime policies and, optionally, CPU arithmetic, KORA-on-Vulkan,
and raw Vulkan paths. It emits one JSON document to standard output, uses
`std::chrono::steady_clock` for host-observed measurements, performs warm-ups, and retains every
measured sample. The report contains nearest-rank median and p95 values in nanoseconds plus the OS,
architecture, compiler, build type, logical CPU count, and (for GPU runs) Vulkan device class and
driver/API identity.

## Build and run

```sh
cmake --preset release
cmake --build --preset release --parallel
build/release/tools/kora-bench --iterations 1000 --warmup 50 > kora-bench.json
```

`kora-bench` is installed with the other command-line tools under `bin/`. `tools/ci/check_install.sh`
checks the installed CPU smoke report; when shader tooling is available, the optional GPU fixtures are
installed under `share/kora/benchmarks` for `--gpu`.

Expected result: the command exits successfully and writes valid JSON with schema version `3`. The
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

GPU reports also include `staging_ring_reserve_{dense,fragmented}_N` host-side measurements for
0, 1, 8, 32, 128, and 512 live ranges. These time only `StagingRing::reserve()`; ring creation,
live-range setup, correctness checks, cancellation, Vulkan submissions, and timeline waits are outside
the interval. Dense cases keep adjacent reservations live. Fragmented cases interleave live ranges
with cancelled ranges and leave a tail gap large enough for the measured request. This isolates
reservation search cost; it is not a transfer-throughput or dispatch-latency measurement.

### Matched raw Vulkan compute workloads

GPU reports include same-device raw Vulkan comparisons for SAXPY, reduction, tiled scan, tiled
matrix multiplication, and stencil at three size tiers: 4,096, 65,536, and 1,048,576 elements, and
16×16, 64×64, and 128×128 matrices. Every tier is a multiple of the fixtures' 64-lane workgroup and
8×8 matmul tile, so each tier runs the identical shader without partial-tile special cases. KORA and raw
Vulkan use the same checked-in SPIR-V fixtures, input values, dispatch dimensions, and output
expectations. Pipelines, descriptors, buffers, and command resources are created before timing and
reused. Each warm-up and measured output is checked for correctness, but the check is outside the
timed interval. The host boundary covers operation setup that is part of the workload, queue submit,
and completion wait; raw Vulkan also performs required mapped-memory flushes for the SAXPY input reset.
GPU timestamp rows measure the dispatch interval on the same queue and selected physical device.

The raw Vulkan buffers are persistent host-visible allocations, while KORA uses its normal buffer and
dispatch path. The report therefore keeps three host boundaries separate: KORA's default completion
(including writable-output readback), KORA with deferred device-resident output, and raw Vulkan using
persistent host-visible buffers. Deferred KORA is the closest dispatch/completion comparison, but it
still differs in allocation and residency policy; none of these host rows is a pure abstraction-only
measurement. Every measured output is explicitly read back and validated after the timer stops,
including deferred outputs. Compute timestamp availability is reported per result name and never
inferred from vector-add timer support. Missing device timestamps are reported as unavailable, never
interpolated. For another device or OS, rerun the same command and retain the raw JSON report:

```sh
KORA_DEVICE=2 build/release/tools/kora-bench --gpu --iterations 1000 --warmup 50 \
  > matched-compute.json
python3 tools/kora-bench/render_comparison.py matched-compute.json --sizes 4096 65536 1048576 4194240
python3 tools/kora-bench/render_charts.py matched-compute.json --output-dir charts
```

The comparison table prints every available workload/size row and its deferred-KORA-to-raw host
median ratio. `render_charts.py` also writes `compute-host-latency.svg`, which groups the 15 matched
cases by workload with p50 dots and p95 whiskers on a logarithmic axis.

One local Release sample on macOS arm64 / Apple M4, AppleClang 21.0.0, Vulkan 1.3.357 / MoltenVK,
used 1,000 measured samples and 50 warm-ups for every case. This is a same-device comparison, not a
cross-platform claim; this machine provides no CUDA result. The complete raw report, including
per-sample values and environment metadata, is
[retained here](../benchmarks/results/macos-arm64-apple-m4-moltenvk-compute-sweep-1000x50-f43b1ef.json)
(source revision f43b1ef91e2cbd118233992b7d2e2e7c49c6c195). The earlier single-size report remains
[available](../benchmarks/results/macos-arm64-apple-m4-moltenvk-compute-1000x50-eb70921.json).
That report is the pre-#227 baseline. The side-by-side tables in
[the generated comparison page](benchmark-compute-results.md) and the chart below are rendered from
the post-#227 report at 8ca0de7 by [`render_comparison.py`](../tools/kora-bench/render_comparison.py)
and [`render_charts.py`](../tools/kora-bench/render_charts.py). Regenerate them after a new run with:

```sh
R=benchmarks/results/macos-arm64-apple-m4-moltenvk-compute-sweep-1000x50-8ca0de7.json
python3 tools/kora-bench/render_comparison.py "$R" \
  --sizes 4096 65536 1048576 4194240 > docs/benchmark-compute-results.md
python3 tools/kora-bench/render_charts.py "$R"
```

![Matched compute host latency across sizes, Apple M4/MoltenVK](../benchmarks/charts/compute-host-latency.svg)

In the pre-#227 baseline, across the 15 matched cases, deferred KORA ranged from 0.82× (64K scan) to
6.38× (1M SAXPY) the raw Vulkan host median and was faster in three cases. Default-to-deferred host medians improved 17–58%.
The gap grows with size for workloads whose inputs are rewritten or whose outputs are large: SAXPY
rewrites its 4 MiB input before every 1M sample, and KORA stages that upload through a separate
transfer submission and waits for it before recording the dispatch, whereas raw Vulkan writes its
persistent mapped buffer directly. Available KORA device timestamps at 1M were 0.96–4.1× raw Vulkan; one
difference is that the runtime selected MoltenVK's host-invisible memory type while raw Vulkan uses
host-visible memory.
KORA timestamps were unavailable for several cases in this run and are shown as gaps. These are
measured boundaries with distinct residency policies, not a pure abstraction overhead delta, and do
not establish CUDA parity or cross-platform performance. The upload and placement causes were
addressed by [#227](#issue-227-unified-memory-direct-access); compare that section's interleaved
before/after pairs rather than these two single runs, because single runs vary several-fold here.

### Staging-ring reservation scaling

The reservation path previously rebuilt and sorted a temporary range list for each gap-search attempt.
The range vector is now maintained in offset order and searched directly, so steady-state reservation
does not allocate scratch memory or sort ranges. A single Release sample set on macOS arm64 / Apple M4,
AppleClang 21.0.0, Vulkan 1.3.357 / MoltenVK used 1,000 measured samples and 50 warm-ups per case.
Both runs used the same physical device and 256-byte requests; each result is host CPU time in
nanoseconds. The full raw-sample reports are retained below.

| Layout / live ranges | Before p50 / p95 (ns) | After p50 / p95 (ns) | p50 change |
| --- | ---: | ---: | ---: |
| Dense / 0 | 42 / 83 | 42 / 83 | 0% |
| Dense / 8 | 83 / 125 | 42 / 84 | -49.4% |
| Dense / 32 | 250 / 250 | 125 / 125 | -50.0% |
| Dense / 128 | 1,000 / 1,042 | 333 / 334 | -66.7% |
| Dense / 512 | 3,250 / 3,958 | 917 / 958 | -71.8% |
| Fragmented / 128 | 1,000 / 1,041 | 334 / 375 | -66.6% |
| Fragmented / 512 | 3,958 / 4,083 | 917 / 958 | -76.8% |

At 0–8 live ranges the observed p50 is flat or lower (quantized timer samples make small differences
coarse); at 128+ ranges p50 falls by 66–77%. The comparison is local evidence from one device and
one 1,000-sample run per revision, not a cross-platform guarantee or repeated-run confidence interval.
It establishes reservation-search improvement, not end-to-end transfer improvement; the remaining
scan and ordered-vector insertion are linear in the number of live ranges. Re-run the GPU suite on
target devices before extrapolating these values.

- Before: [issue #215 baseline report at `a94bef0`](../benchmarks/results/issue215-before-a94bef0.json)
- After: [issue #215 optimized report](../benchmarks/results/issue215-after-b10bdc1.json)

## Optional external CUDA reference

KORA has no CUDA runtime dependency and does not build a CUDA runner. A separate benchmark producer
may emit the version-1 JSON contract accepted by `render_comparison.py`; this keeps CUDA compilation,
execution, and hardware requirements outside the KORA library, default presets, correctness tests,
CI, and release artifacts. The producer must check numerical correctness before timing, retain raw
samples, and verify that the CUDA and Vulkan device UUIDs identify the same physical GPU. It records
that verification method without publishing either raw identifier.

Minimal contract example (four illustrative samples):

```json
{
  "schema_version": 1,
  "backend": "cuda",
  "environment": {
    "os": "linux",
    "architecture": "x86_64",
    "compiler": "nvcc 13.0",
    "build_type": "Release",
    "source_revision": "0123456789abcdef0123456789abcdef01234567"
  },
  "device": {
    "name": "NVIDIA GPU",
    "driver_version": "600.1",
    "runtime_version": "13.0"
  },
  "device_match": {
    "same_physical_device_as_kora_vulkan": true,
    "verification_method": "cuda_uuid_equals_vk_device_uuid"
  },
  "results": [{
    "workload_id": "vector_add",
    "size": 4096,
    "precision": "fp32",
    "semantic_contract": "kora-bench.vector-add.f32.v1",
    "comparison_class": "algorithm_matched",
    "implementation": "one-thread-per-element",
    "measurement_boundary": "kernel_dispatch",
    "kora_result_name": "gpu_vector_add_4096_gpu_execution",
    "raw_vulkan_result_name": "raw_vulkan_vector_add_4096_gpu_execution",
    "correctness_method": "elementwise comparison before timed iterations",
    "correctness_checked": true,
    "correctness_passed": true,
    "timing_domain": "gpu_execution_event",
    "unit": "ns",
    "warmup": 50,
    "iterations": 4,
    "samples": [100, 200, 300, 400],
    "median": 200,
    "p95": 400
  }]
}
```

Required fields are validated, including a same-physical-device UUID-match assertion, timing boundary,
correctness status, commit hash, sample count, and nearest-rank median/p95 recomputed from raw samples.
CUDA event time is kept separate from host-submit/completion latency. Algorithm-matched kernels and
best-in-class CUDA libraries use distinct `comparison_class` values and must not be conflated. Workload
semantics, size, precision, and result names must match the KORA and raw Vulkan references; the external
producer owns the implementation review behind `semantic_contract`.

Render the normal KORA/raw table plus the optional CUDA table with:

```sh
python3 tools/kora-bench/render_comparison.py kora-gpu-bench.json \
  --cuda-report cuda-reference.json \
  --kora-revision "$(git rev-parse HEAD)"
```

`--kora-revision` is required because schema-3 KORA reports predate the source-revision field; the
rendered comparison prints both exact commits, and the CUDA report carries its producer commit. The
CUDA table is emitted only after validation. If a matching KORA or raw Vulkan timestamp is absent,
it prints `unavailable` and does not derive a ratio. The checked-in Apple M4 reports have no CUDA
reference and are not CUDA evidence. A real NVIDIA run is still required to assess the 10% parity
objective and to close the hardware-validation portion of [issue #197](https://github.com/varunkarthic/kora/issues/197).

For a diagnostic phase profile on a host with valid Vulkan device timestamps, run:

```sh
KORA_DEVICE=2 build/release/tools/kora-bench --profile-phases --iterations 1000 --warmup 50 \
  > kora-dispatch-profile.json
```

`--profile-phases` enables `TelemetryLevel::Profile` and sets `diagnostic_profile: true` in the
report. It emits raw samples and p50/p95 for builder setup, the caller's submit call and event wait,
per-dispatch validation, binding preflight, pipeline lookup, Runtime submission preparation, buffer
residency/transfers, submission assembly, queue enqueue, command-buffer acquisition, descriptor allocation/update, command recording,
and queue completion. Device timestamp measurements are included when
available. Host phases can still be reported when the selected device cannot provide GPU timestamps;
the corresponding device-time result is omitted. This mode adds profiling work; do not use its
ordinary benchmark rows as the Release performance baseline or feed the report to the comparison
table or chart generators.

The completion diagnostics further split completion-worker wake delay, the Vulkan timeline/fence
wait, timestamp-query collection, and Runtime completion-callback work. These phases explain where
the caller-visible Event latency is spent; they are measured only in diagnostic mode and are not
themselves an optimization target or a substitute for the ordinary KORA/raw Vulkan comparison.

Phase profiling identified repeated SPIR-V reflection as avoidable work on ordinary pipeline-cache
hits. Static-local-size dispatches now perform an exact cache-key lookup first and only run reflection
when a pipeline is not already cached. `LocalSizeId` dispatches retain the full validation path because
their resolved workgroup size is part of the pipeline key. This internal optimization does not change
the public dispatch contract or relax validation for cache misses.

## Comparing CPU, KORA GPU, and raw GPU paths across operating systems

KORA's supported build/test environments are Linux/GCC, Windows/MSVC, and macOS/AppleClang; the
benchmark report records the OS, architecture, compiler, build configuration, and logical CPU count
so the execution context is visible. Run a Release build with the same workload, warm-up count, sample
count, precision, and input sizes on each target. Compare CPU results only against CPU results, and
compare the same timing domain across reports. CI smoke tests validate that reports are produced and
parsed on supported platforms; GitHub-hosted runner measurements are not retained performance
references and must not be treated as controlled hardware comparisons.

| Path in report | What ran | How to interpret it |
| --- | --- | --- |
| `host_cpu_execution` | Single-thread CPU arithmetic baseline. | CPU workload time; not directly comparable to GPU timestamps or multi-thread throughput. |
| `host_submit_completion` / `host_transfer_completion` | KORA API through completion, including host orchestration and waits. | End-to-end host latency, not kernel-only execution. |
| `gpu_execution_timestamp` | Device timestamp interval around a KORA dispatch, if the selected device/driver supports valid timestamps for that size. | Best available device-side interval here, but not shader active-cycle accounting; unavailable means no value, not zero. |
| `gpu_execution_timestamp` on `raw_vulkan_*` | Equivalent raw Vulkan dispatch interval on the same selected physical device and shader. | Lower-level dispatch reference; it omits KORA orchestration. |

The `device_class` field distinguishes `cpu_vulkan` (software Vulkan, commonly executing on host
CPU cores) from integrated/discrete hardware classes. A `cpu_vulkan` result is useful for correctness
and driver-path overhead, but it is not a GPU-acceleration result and must not be labeled as hardware
GPU throughput. Conversely, the standalone CPU arithmetic case is not a substitute for running the
same shader on a device. CPU model, core topology, power policy, driver, Vulkan implementation, and
clock state can all differ between OS machines; report them alongside any retained results before
calculating ratios.

Current retained performance data is a local Apple M4/MoltenVK run (plus a KosmicKrisp availability
fixture) described below. It does not provide Linux/Windows hardware-GPU results or CUDA evidence.
Those comparisons require machines with the relevant hardware and driver stack; do not infer them
from CI smoke coverage or combine different device models into a single OS ranking.

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
| `cpu_vector_add_N` | Add two uint32 arrays for `N` in 1, 64, 256, 1024, 4096, 65536, 262144, 1048576, and 4194240. | Single-thread C++ host baseline; fixed input arrays are reused. |
| `cpu_saxpy_N`, `cpu_reduction_N`, `cpu_stencil_N`, `cpu_tiled_matmul_N` | Run each operation for three small/medium/large sizes; matrix dimensions are 16, 64, and 128, and vector workloads use 4096, 65536, and 1048576 elements. | Single-thread C++ implementations; reduction emits one sum per 64-value tile to match the Vulkan fixture, and matrix multiplication uses 32×32 cache tiles. Deterministic inputs are initialized before timing, and every output is checked against its reference after each timed sample. |
| `cpu_<workload>_parallel_N` | Four-worker CPU-executor variants of vector add, SAXPY, reduction, and stencil at 4096, 65536, and 1048576 elements; `cpu_tiled_matmul_parallel_N` uses dimensions 16, 64, and 128. | Reports `thread_count: 4`. Worker creation is outside timing; submission, scheduling, execution, and completion wait are inside the host timing interval. All outputs are validated outside the interval. |
| `gpu_tiny_dispatch_1` | Submit one vector-add work item on Vulkan and wait for completion. | Warm steady-state host latency with buffers and pipeline resident. |
| `gpu_tiny_dispatch_batch_10` | Submit 10 one-work-item vector-add dispatches sequentially, waiting for each event. | Measures repeated small-dispatch scheduling and completion latency; output correctness is checked before timing. |
| `gpu_vector_add_N_dispatch_wait` | Dispatch N vector-add work items for N in 64, 256, 1024, 4096, 65536, 262144, 1048576, and 4194240, then wait. | Inputs and pipeline are made resident before timing; output correctness is checked outside the timed loop. |
| `gpu_vector_add_N_deferred_readback_dispatch_wait` | Run the same vector-add workload with `DispatchBuilder::defer_readback()`. | Matched more closely to raw Vulkan's device-resident output; output contents are validated before timing. |
| `gpu_vector_add_N_gpu_execution` / `raw_vulkan_vector_add_N_gpu_execution` | Read a device timestamp interval around the matched dispatch. | Emitted only for element counts with valid timestamp results; exact sizes are listed in `gpu_measurements`. |
| `gpu_saxpy_4096_upload_dispatch_wait` | Reset the host input, dispatch 4096 SAXPY elements, and wait. | Includes upload and dispatch wait; correctness is checked before timing. |
| `gpu_reduction_4096_dispatch_wait` | Reduce 4096 float values into 64 workgroup sums and wait. | Inputs are resident; each 64-value tile is validated before timing. |
| `gpu_tiled_scan_4096_dispatch_wait` | Inclusive scan each 64-value workgroup tile and wait. | Inputs are resident; tile prefixes are validated before timing. |
| `gpu_tiled_matmul_64_dispatch_wait` | Multiply two 64×64 matrices using 8×8 shared-memory tiles and wait. | Inputs are resident; every output is checked before timing. |
| `gpu_stencil_4096_dispatch_wait` | Apply a three-point 1D stencil to 4096 values and wait. | Inputs are resident; boundaries and interior are checked before timing. |
| `raw_vulkan_vector_add_N_dispatch_wait` | Reuse a Vulkan compute pipeline, descriptor set, and command buffer to dispatch N vector-add elements and wait on a fence for N in 64, 256, 1024, 4096, 65536, 262144, 1048576, and 4194240. | Same shader, operation, and element count as the corresponding KORA GPU case; excludes setup, upload, and readback. |
| `command_pool_reuse_steady_state` | Begin, end, and reset a command buffer from a persistent bounded ring. | Host-side Vulkan API overhead; no queue submit or GPU execution. |
| `command_pool_cold_create_record_destroy` | Create a one-slot pool, record/reset one buffer, and destroy it. | Cold setup comparison for the reuse path; no queue submit or GPU execution. |
| `transfer_direct_upload_N` / `transfer_direct_download_N` | Transfer N bytes through a persistently mapped device-local, host-visible Vulkan buffer without a staging ring. | Includes KORA queue submission and completion; the direct-mapped policy is explicitly selected for this controlled comparison. |
| `transfer_staged_upload_N` / `transfer_staged_download_N` | Transfer N bytes through KORA's bounded upload/readback staging rings. | Same device-local, host-visible target buffer and queue as the direct case; correctness is checked before timing. |
| `raw_vulkan_copy_N` | Record `vkCmdCopyBuffer`, submit it with a Vulkan fence, and wait for completion. | Raw API baseline using the same device-local source/destination buffers and transfer queue. |
| `kora_transfer_copy_N` | Copy between the same buffers using `TransferEngine::copy()` and wait for its event. | Includes KORA queue submission, completion, and transfer bookkeeping. |

## CPU compute reference results

The following Release baseline was collected on an Apple M4 (macOS arm64, AppleClang 21.0.0,
`-O3 -DNDEBUG`) with 10 warm-ups and 100 timed samples per workload. Five independent process runs
are retained as `cpu-compute-baseline-macos-arm64-apple-m4-release-run-{1..5}.json` under
`benchmarks/results/`. Values below are the median of the five per-run medians; the range shows the
minimum and maximum of those five medians. These portable C++ references are not a claim of
best-in-class CPU performance, and are not directly rankable against GPU device timestamps.
Single-thread values time the compute loop. Four-thread values include KORA CPU-executor task
submission, scheduling, execution, and wait; the worker pool is created before timing. No
before/after speedup is claimed for adding benchmark coverage.

| Workload | Size | Median of run medians | Run-median range |
| --- | ---: | ---: | ---: |
| Vector add (`uint32`) | 4,096 elements | 2.084 µs | 2.083–2.084 µs |
| Vector add (`uint32`) | 1,048,576 elements | 563.458 µs | 562.000–566.584 µs |
| Vector add (`uint32`) | 4,194,240 elements | 2.253 ms | 2.242–2.293 ms |
| SAXPY (`float32`) | 1,048,576 elements | 204.500 µs | 202.083–321.125 µs |
| Reduction (`float32`, 64-value tile outputs) | 1,048,576 elements | 223.333 µs | 223.250–227.458 µs |
| Three-point stencil (`float32`) | 1,048,576 elements | 659.042 µs | 656.958–661.167 µs |
| Tiled matrix multiply (`float32`) | 128×128 | 439.500 µs | 439.000–441.334 µs |
| Vector add (`uint32`, four threads) | 1,048,576 elements | 78.708 µs | 68.333–321.375 µs |
| SAXPY (`float32`, four threads) | 1,048,576 elements | 72.833 µs | 68.750–75.750 µs |
| Reduction (`float32`, four threads) | 1,048,576 elements | 80.833 µs | 74.291–134.292 µs |
| Three-point stencil (`float32`, four threads) | 1,048,576 elements | 216.834 µs | 209.542–317.875 µs |
| Tiled matrix multiply (`float32`, four threads) | 128×128 | 128.959 µs | 128.000–147.916 µs |

Reproduce with `cmake --preset release`, `cmake --build --preset release --parallel`, then run
`build/release/tools/kora-bench --iterations 100 --warmup 10` five times, saving each stdout JSON
report. Reports include the CPU model, OS, architecture, compiler, build type, exact CMake flags,
source revision, and logical CPU count.
The reports retain every sample; the table is only a compact summary. Small-workload pool overhead is
visible (the four-worker 4096-element cases are slower than their single-thread counterparts), and
this host comparison does not include vendor math libraries or matched raw Vulkan implementations
for SAXPY/reduction/stencil/matmul; those limitations must be resolved before broad CPU-vs-GPU conclusions.

## Report contract

The root object has `schema_version: 3`, `clock`, `iterations`, `warmup`, `environment`, and `results`.
`environment` records OS, architecture, CPU model, compiler, CMake build configuration and flags,
source revision, and logical CPU count. If Git metadata or build flags are unavailable, the report
records `unknown` or an empty flag string instead of implying a value.
Each result has a stable `name`, `unit`, `timing_domain`, `median`, `p95`, and `samples` array. Parallel CPU results also record `thread_count`. Transfer results also report
`bytes_transferred` and `throughput_bytes_per_second`, derived from transferred bytes and median
latency. GPU reports add a `gpu_measurements` object with backend, selected enumeration index,
sanitized device name, Vulkan API version, driver name, broad device class (`cpu_vulkan`, integrated,
discrete, virtual, or other), compute queue-family index, matched-device assertion, compared paths,
and the exact vector sizes with valid KORA/raw Vulkan timestamp samples. Timestamp support is per
workload; unsupported or unreadable samples are omitted, never substituted with host timing. The
report lists timestamp availability and supported element counts in `gpu_measurements`. Every
GPU result identifies its path, timing domain, and selected device index. No
UUID, vendor ID, or device ID is emitted. Percentiles use nearest-rank ordering: the rank is
`ceil(p * N)`. Timing values naturally vary with host load and compiler, so compare reports only when
the machine, build type, compiler, and workload parameters are recorded alongside them.

## Scope and interpretation

CPU-side measurements are always available. CPU compute cases use deterministic, initialized data,
warm-up iterations, and the same host clock/sample policy as other benchmark cases. Correctness
validation runs after (and outside) each timed sample, and scans the full output rather than relying
on one element. CPU implementations are portable C++ and may benefit from compiler auto-vectorization.
Single-thread rows measure the compute loop; parallel rows additionally include KORA CPU-executor
submission, scheduling, and wait overhead for four workers. Neither claims to be a fastest-available
vendor math library. Optional GPU cases cover tiny dispatch and size-swept vector-add/SAXPY, reduction, tiled
scan, tiled matmul, stencil, transfer sizes, and raw Vulkan
copy/vector-add dispatch baselines. `host_cpu_execution` is single-thread host arithmetic;
`host_submit_completion` includes command preparation, submit, and completion wait;
`host_transfer_completion` includes the transfer API and wait; `host_api_only` excludes queue submit
and GPU execution; `gpu_execution_timestamp` is the device timestamp interval between command-stream
timestamps around the dispatch. KORA and raw Vulkan vector-add report these intervals when supported;
the report lists exact workload sizes with samples and omits unavailable samples rather than
substituting host time. This interval is not a promise of shader-only active cycles. `cpu_vulkan`
identifies a software Vulkan implementation (for example, lavapipe) so it cannot be mistaken for a
hardware GPU result. CPU arithmetic and GPU dispatch use different timing domains and must not be
ranked as interchangeable implementations. Transfer results measure end-to-end API latency for the
configured memory path. Python callback overhead is measured separately as documented above. Scheduler
results do not include end-to-end Runtime device selection because the candidate cost model is not yet
wired into `Runtime::dispatch()`.

## Reference run and results

The following reference was collected on one local machine; it is included to make results
interpretable, not to predict another system. KORA and raw Vulkan use the physical device selected by
`Runtime::devices()[0]`. The benchmark resolves that stable enumeration index against the Vulkan
instance's physical-device list and fails the GPU comparison if it cannot find the exact device. Raw
dispatch and transfer helpers create their Vulkan device from that same physical-device record. Set
`KORA_DEVICE` to pin selection when reproducing results. On PowerShell, use `$env:KORA_DEVICE = "2"`
and `build\release\tools\kora-bench.exe`. The JSON report records the sanitized device
name, API and driver names, selected index, compute queue family, and measurement path for each GPU
result; UUIDs and vendor/device IDs are omitted. Runtime and raw helpers share the selected physical
device, while their host orchestration, resource setup, and timing boundaries remain intentionally
different.

| Environment field | Value |
| --- | --- |
| Operating system / architecture | macOS, arm64 |
| CPU / selected Vulkan device | Apple M4 / Apple M4 integrated GPU via MoltenVK |
| Vulkan API / driver | 1.3.357 / MoltenVK |
| Compiler / build | AppleClang 21.0.0 / Release |
| Logical CPUs | 10 |
| Samples | 100 measured iterations; 10 warm-ups |

Reproduce CPU/GPU results with:

```sh
cmake --preset release
cmake --build --preset release --parallel
KORA_DEVICE=2 build/release/tools/kora-bench --gpu --iterations 100 --warmup 10 > kora-bench.json
```

The prepared-plan Python result uses the separate command in the Python callback measurement section above.
The rows below are medians and nearest-rank p95 values from a fresh Apple M4 Release run. Host
submit-to-completion and GPU timestamp intervals are separate. The timestamp interval is measured
around dispatch on the device, not a guarantee of shader-only active cycles. Raw Vulkan deliberately
avoids KORA's runtime orchestration and is a lower-level baseline, not an equivalent API surface.

| Workload | Median | p95 | Interpretation |
| --- | ---: | ---: | --- |
| CPU vector add, 4,096 elements | 0.375 µs | 0.375 µs | Single-thread host arithmetic |
| KORA default readback, 4,096 elements | 2.259 ms | 3.668 ms | Submit, completion, and default writable-output readback |
| KORA deferred readback, 4,096 elements | 595.208 µs | 652.208 µs | Submit/completion with output device-resident |
| Raw Vulkan, 4,096 elements | 404.667 µs | 449.417 µs | Same device/shader; reused pipeline and buffers |
| KORA GPU timestamp, 4,096 elements | 14.041 µs | 18.458 µs | Device timestamp interval around dispatch |
| Raw Vulkan GPU timestamp, 4,096 elements | 21.334 µs | 25.208 µs | Same queue/device; timestamp interval around dispatch |
| CPU vector add, 1,048,576 elements | 224.375 µs | 249.791 µs | Single-thread host arithmetic |
| KORA default readback, 1,048,576 elements | 2.517 ms | 5.543 ms | Submit, completion, and default writable-output readback |
| KORA deferred readback, 1,048,576 elements | 799.750 µs | 871.083 µs | Submit/completion with output device-resident |
| Raw Vulkan, 1,048,576 elements | 996 µs | 1.251 ms | Same device/shader; reused pipeline and buffers |
| KORA GPU timestamp, 1,048,576 elements | unavailable | unavailable | Timer sample unavailable at this size |
| Raw Vulkan GPU timestamp, 1,048,576 elements | 359.125 µs | 404.583 µs | Same queue/device; timestamp interval around dispatch |
| CPU vector add, 4,194,240 elements | 984.125 µs | 1.215 ms | Single-thread host arithmetic |
| KORA default readback, 4,194,240 elements | 2.942 ms | 3.799 ms | Submit, completion, and default writable-output readback |
| KORA deferred readback, 4,194,240 elements | 2.349 ms | 2.981 ms | Submit/completion with output device-resident |
| Raw Vulkan, 4,194,240 elements | 1.964 ms | 2.112 ms | Same device/shader; reused pipeline and buffers |
| KORA GPU timestamp, 4,194,240 elements | unavailable | unavailable | Timer sample unavailable at this size |
| Raw Vulkan GPU timestamp, 4,194,240 elements | 1.315 ms | 1.398 ms | Same queue/device; timestamp interval around dispatch |

This local sample is evidence about one macOS/MoltenVK-backed configuration only. It separates KORA's
default output readback from the deferred-readback host path; raw Vulkan is closest to the latter.
Timestamp support is reported per workload size. The end-to-end KORA/raw difference is being
investigated in [#196](https://github.com/varunkarthic/kora/issues/196). This Apple
device cannot establish CUDA parity. Re-run on every target OS/device before making design decisions.
The full 100-sample report is [retained as JSON](../benchmarks/results/macos-arm64-apple-m4-moltenvk-100x10.json).
An alternate M4/KosmicKrisp report is [also retained](../benchmarks/results/macos-arm64-apple-m4-kosmic-100x10.json);
that Vulkan path reports timestamp measurements unavailable, illustrating why availability is
recorded per selected device/driver rather than assumed from the OS or GPU model.

### Dispatch overhead case study

The phase profile showed the dominant steady-state host cost was not descriptor setup, transfer, or
command recording, but repeated SPIR-V reflection on an exact pipeline-cache hit. For static-local-size
kernels, KORA now checks the fully serialized pipeline key first and uses the already validated
pipeline when the key matches. Cache misses still perform full reflection and validation; `LocalSizeId`
specializations retain the full path because their resolved workgroup size participates in the key.

| Measurement | Before (`8dd2978`) | After, run 1 (`67bec0f`) | After, run 2 (`67bec0f`) |
| --- | ---: | ---: | ---: |
| 4K KORA deferred-readback host completion, median | 676.500 µs | 523.958 µs | 522.208 µs |
| 4K KORA deferred-readback host completion, p95 | 819.958 µs | 674.041 µs | 597.500 µs |
| 4K raw Vulkan host completion, median | 346.875 µs | 406.708 µs | 465.542 µs |
| KORA/raw 4K host median ratio | 1.95× | 1.29× | 1.12× |
| Pipeline lookup phase, median (diagnostic mode) | 181.667 µs | 6.709 µs | — |

The larger-size medians are included to make possible throughput regressions visible:

| Elements | Path | Before (`8dd2978`) | After, run 1 | After, run 2 |
| ---: | --- | ---: | ---: | ---: |
| 1,048,576 | KORA deferred host completion | 1.383 ms | 1.470 ms | 1.274 ms |
| 1,048,576 | Raw Vulkan host completion | 0.958 ms | 1.191 ms | 0.582 ms |
| 4,194,240 | KORA deferred host completion | 2.876 ms | 2.236 ms | 2.222 ms |
| 4,194,240 | Raw Vulkan host completion | 1.493 ms | 1.714 ms | 2.002 ms |

The ordinary runs used `KORA_DEVICE=2 build/release/tools/kora-bench --gpu --iterations 1000
--warmup 50` on macOS arm64 / Apple M4, AppleClang 21.0.0 Release, Vulkan 1.3.357 / MoltenVK,
with KORA and raw Vulkan pinned to the same enumerated physical device. The pipeline phase uses
`--profile-phases --iterations 200 --warmup 20`; it enables telemetry and must not be compared as a
normal performance result. Its pre-fast-path profile was collected on the profiling branch worktree
before the optimization commit, while the post-fast-path phase sample is on `67bec0f`.

The ordinary 1,000-sample reports preserve every sample and the environment fields:
[before `8dd2978`](../benchmarks/results/dispatch-overhead-before-8dd2978-macos-arm64-m4-moltenvk-1000x50.json)
and two reruns after `67bec0f` ([run 1](../benchmarks/results/dispatch-overhead-after-67bec0f-macos-arm64-m4-moltenvk-1000x50.json),
[run 2](../benchmarks/results/dispatch-overhead-after-67bec0f-repeat-macos-arm64-m4-moltenvk-1000x50.json)).
The KORA 4K/4.19M deferred medians were stable across the two post-change runs, but raw Vulkan and
1M results varied materially with the sequential host/driver state. Treat the ratios as local,
promising evidence rather than a controlled cross-device result or CUDA parity claim. Full raw samples
for 1M and 4.19M vector-add are retained in each report so those variations and possible regressions
remain inspectable.

### Visual comparison and methodology

The following plots and the table below are generated from the retained 1,000-sample, 50-warm-up
report taken after #227 (source 8ca0de7); they are not hand-transcribed. The 100-sample table in the
previous section is the original reference run and is kept for history. The vertical latency axis is logarithmic, with units chosen for the range shown in
each panel. Lines and dots show the median (p50); whiskers extend to the nearest-rank p95. The raw
samples remain in the linked JSON report. A missing KORA timestamp is a gap because that timer result
is unavailable for the size; it is not zero, interpolated, or estimated from host time.

CPU arithmetic is isolated from GPU API latency. Host submit/completion is separate from GPU device
timestamps, so the plots do not merge incompatible timing domains. The host chart deliberately labels
KORA's default-readback path separately from deferred KORA and raw Vulkan, whose output stays on the
device. Those latency paths therefore have different synchronization and transfer costs; the raw
Vulkan result is a lower-level baseline, not an equivalent API surface.

![CPU arithmetic timings, Apple M4/MoltenVK](../benchmarks/charts/cpu-arithmetic.svg)

![Host-observed dispatch completion timings, Apple M4/MoltenVK](../benchmarks/charts/host-latency.svg)

![GPU timestamp dispatch intervals, Apple M4/MoltenVK](../benchmarks/charts/gpu-timestamps.svg)

Regenerate the visualizations from the checked-in report with:

```sh
python3 tools/kora-bench/render_charts.py \
  benchmarks/results/macos-arm64-apple-m4-moltenvk-compute-sweep-1000x50-8ca0de7.json \
  --output-dir benchmarks/charts
```

The generator requires schema 3, retained finite samples, timing-domain labels, and an explicit
same-physical-device assertion for KORA/raw Vulkan. It uses only Python's standard library. The plots
show results for one Apple M4/MoltenVK Release configuration; they cannot establish CUDA parity,
hardware results on Linux or Windows, or a general cross-device ranking. The methodology still
requires matching device, operation, input and precision, setup/readback boundaries, warm-up and
sample count before comparing implementations.

| Elements | CPU median | CPU p95 | KORA default median | KORA default p95 | KORA deferred median | KORA deferred p95 | Raw host median | Raw host p95 | KORA GPU median | KORA GPU p95 | Raw GPU median | Raw GPU p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 2.083 µs | 2.125 µs | 1.264 ms | 1.619 ms | 268.416 µs | 303.750 µs | 260.459 µs | 301.958 µs | 12.834 µs | 15.667 µs | 12.417 µs | 13.291 µs |
| 1048576 | 563.166 µs | 573.667 µs | 2.046 ms | 2.333 ms | 1.129 ms | 1.282 ms | 1.247 ms | 1.385 ms | 416.417 µs | 632.042 µs | 582.916 µs | 645.708 µs |
| 4194240 | 2.243 ms | 2.328 ms | 2.670 ms | 3.341 ms | 1.790 ms | 2.092 ms | 1.847 ms | 2.109 ms | 1.240 ms | 1.414 ms | 1.239 ms | 1.406 ms |

Regenerate the comparison rows from the retained raw report with:

```sh
python3 tools/kora-bench/render_comparison.py \
  benchmarks/results/macos-arm64-apple-m4-moltenvk-compute-sweep-1000x50-8ca0de7.json \
  --sizes 4096 1048576 4194240
```

### Local command-pool sample

The earlier KosmicKrisp Release run (`KORA_DEVICE=0`, `--gpu --iterations 1000 --warmup 50`) recorded 1,000
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

### Issue #214: eager submission coalescing and completion latency

The completion-phase diagnostic isolated the dominant avoidable delay before the Vulkan queue submit:
Runtime compute queues held a lone, otherwise-ready dispatch for a 100 µs coalescing collection window.
The completion worker's Vulkan timeline wait was still substantial, but timestamp-query collection and
Runtime callback work each measured only a few microseconds. Runtime coalescing now batches compatible
submissions that are already queued together, without waiting for additional future submissions.

The before/after gate used five interleaved Release runs per variant, each with 50 warm-ups and 1,000
timed samples, on the same Apple M4/MoltenVK device (device index 2, Vulkan 1.3.357). Both variants
ran the same shader and input sizes; output correctness checks occur before timed intervals. KORA and
raw Vulkan were explicitly confirmed to use the same physical device. The before binary was built
from `f854b911f9d447488e305c717af67d119427b6e6` and the after binary from
`32c0cb133fc47e5508a3b907b1591cc60c68206f`; both used AppleClang 21.0.0, Release `-O3 -DNDEBUG`.
Each table entry is the median of five run medians; p95 is the median of the five per-run p95s.
These are host submit/completion measurements, not GPU kernel times. KORA device timestamps were
unavailable on this MoltenVK queue, so no KORA-vs-raw GPU execution ratio is inferred.

| Elements | KORA before p50 / p95 | KORA after p50 / p95 | p50 change | Matched raw Vulkan p50 before / after |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 547.875 / 612.000 µs | 380.958 / 444.083 µs | −30.5% | 409.500 / 408.042 µs |
| 65,536 | 537.583 / 618.125 µs | 391.583 / 448.792 µs | −27.2% | — |
| 262,144 | 610.750 / 678.834 µs | 432.875 / 541.791 µs | −29.1% | — |
| 1,048,576 | 1.331 / 1.554 ms | 1.093 / 1.314 ms | −17.9% | — |
| 4,194,240 | 2.124 / 2.348 ms | 1.917 / 2.291 ms | −9.7% | — |

At 4K, the KORA/raw median host-latency ratio moved from 1.34× to 0.93× in these five paired
runs. The five-run median of per-run p95s improved by 27.4%; the largest workload showed no
regression in either p50 or p95. The improvement is consistent with removing the idle collection
delay; residual latency is dominated by host/driver queue-completion behavior (diagnostic Vulkan wait
was roughly 316–345 µs), not buffer residency or KORA's completion callback. This M4 result is a
local host-latency comparison only; it does not establish CUDA parity or Linux/Windows hardware-GPU
performance.

All raw samples and environment/device assertions are retained in the five interleaved pairs:
[pair 1 before](../benchmarks/results/issue214-before-f854b91-pair-1.json) / [after](../benchmarks/results/issue214-after-32c0cb1-pair-1.json),
[pair 2 before](../benchmarks/results/issue214-before-f854b91-pair-2.json) / [after](../benchmarks/results/issue214-after-32c0cb1-pair-2.json),
[pair 3 before](../benchmarks/results/issue214-before-f854b91-pair-3.json) / [after](../benchmarks/results/issue214-after-32c0cb1-pair-3.json),
[pair 4 before](../benchmarks/results/issue214-before-f854b91-pair-4.json) / [after](../benchmarks/results/issue214-after-32c0cb1-pair-4.json), and
[pair 5 before](../benchmarks/results/issue214-before-f854b91-pair-5.json) / [after](../benchmarks/results/issue214-after-32c0cb1-pair-5.json).

(issue-227-unified-memory-direct-access)=
### Issue #227: unified-memory direct access

Before #227, every dirty input upload and every default-mode output readback went through the
staging ring: KORA copied the host shadow into a temporary vector, staged it, submitted a transfer,
and blocked on that transfer before recording the dispatch; default readback added a second
transfer and copy. MoltenVK on Apple M4 was also classified `DiscreteWithBar`, so runtime buffers
used its host-invisible memory type. #227 classifies single-heap devices as unified memory and, on
unified devices, writes and reads host-visible runtime buffers through their mapping whenever no
queued GPU work can access them (see [the memory model](#unified-memory-direct-access)).
It also fixes dropped KORA device timestamps: results that MoltenVK publishes shortly after the
queue signal are now harvested by `Runtime::profiling_stats()`, so the 1,000-sample report has KORA
device timestamps for all 15 matched compute cases (previously 9) and all eight vector-add sizes
(previously none).

Single runs on this machine are not comparable: the same binary's per-run median moved by up to 5×
between runs (for example raw Vulkan 4M vector add ranged 0.84–2.44 ms), apparently with GPU clock
state. The comparison therefore uses 15 interleaved pairs of the `main` build before the change
(63d5f1b) and the change (8ca0de7), alternating which build ran first, with 200–300 measured samples
and 30 warm-ups per run on macOS arm64 / Apple M4, AppleClang 21.0.0 Release, Vulkan 1.3.357 /
MoltenVK. Change is the median of each pair's after/before p50 ratio, followed by how many of the 15
pairs the new build won:

Before: `63d5f1b6a1617fb5324cf306e68fd0f10393aec1` (15 runs); after: `8ca0de7e6b4e87acd1a432fb93293123bf71d22a` (15 runs). Latencies are the median across runs of the per-run p50. Change is the median of per-pair after/before p50 ratios, with the number of pairs the after build won.

| Workload | Size | KORA default before | after | change | KORA deferred before | after | change | Raw Vulkan (after runs) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAXPY | 4K | 1.085 ms | 434.791 µs | -59% (15/15) | 735.417 µs | 412.917 µs | -44% (15/15) | 323.541 µs |
| SAXPY | 64K | 1.129 ms | 441.333 µs | -60% (15/15) | 825.375 µs | 444.833 µs | -46% (14/15) | 370.833 µs |
| SAXPY | 1M | 2.752 ms | 893.375 µs | -60% (13/15) | 1.633 ms | 758.750 µs | -37% (10/15) | 537.458 µs |
| Reduction | 4K | 713.500 µs | 427.167 µs | -40% (14/15) | 392.917 µs | 419.167 µs | +3% (7/15) | 321.417 µs |
| Reduction | 64K | 714.375 µs | 428.541 µs | -42% (14/15) | 405.333 µs | 412.000 µs | +3% (5/15) | 371.792 µs |
| Reduction | 1M | 1.079 ms | 753.000 µs | -29% (15/15) | 719.334 µs | 738.042 µs | +2% (5/15) | 311.667 µs |
| Scan | 4K | 732.375 µs | 434.083 µs | -44% (15/15) | 415.333 µs | 430.584 µs | +4% (6/15) | 317.833 µs |
| Scan | 64K | 771.208 µs | 432.833 µs | -41% (14/15) | 424.417 µs | 426.833 µs | +0% (7/15) | 398.875 µs |
| Scan | 1M | 2.309 ms | 738.209 µs | -46% (13/15) | 460.708 µs | 460.958 µs | +6% (5/15) | 406.041 µs |
| Stencil | 4K | 690.875 µs | 424.666 µs | -41% (15/15) | 401.292 µs | 424.292 µs | +6% (6/15) | 325.208 µs |
| Stencil | 64K | 805.291 µs | 414.500 µs | -45% (15/15) | 387.083 µs | 417.167 µs | -0% (8/15) | 332.166 µs |
| Stencil | 1M | 2.299 ms | 810.416 µs | -49% (12/15) | 727.000 µs | 458.916 µs | +2% (5/15) | 378.958 µs |
| Tiled matmul | 16² | 712.000 µs | 423.125 µs | -40% (15/15) | 410.542 µs | 420.125 µs | +3% (6/15) | 223.500 µs |
| Tiled matmul | 64² | 676.125 µs | 443.083 µs | -37% (14/15) | 410.583 µs | 413.375 µs | +1% (5/15) | 326.500 µs |
| Tiled matmul | 128² | 781.958 µs | 477.375 µs | -40% (15/15) | 431.417 µs | 474.417 µs | +1% (5/15) | 429.167 µs |
| Vector add | 4K | 2.006 ms | 1.374 ms | -32% (13/15) | 316.875 µs | 402.041 µs | +26% (3/15) | 406.041 µs |
| Vector add | 64K | 2.046 ms | 1.409 ms | -32% (14/15) | 333.000 µs | 358.209 µs | +13% (6/15) | 407.375 µs |
| Vector add | 1M | 4.297 ms | 1.764 ms | -56% (14/15) | 738.958 µs | 1.124 ms | +22% (5/15) | 1.203 ms |
| Vector add | 4M | 3.333 ms | 2.696 ms | -15% (12/15) | 1.659 ms | 1.864 ms | +10% (3/15) | 1.988 ms |

![KORA default readback before and after #227, Apple M4/MoltenVK](../benchmarks/charts/uma-direct-default.svg)

![KORA deferred readback before and after #227, Apple M4/MoltenVK](../benchmarks/charts/uma-direct-deferred.svg)

Default readback, KORA's default dispatch mode, improved in every case: 15–60% lower median latency,
winning at least 12 of 15 pairs everywhere. Deferred SAXPY, which rewrites its input every sample,
improved 37–46%. Deferred compute workloads that upload nothing changed by −0% to +6%, within run
noise. **Deferred vector add regressed** by 10–26% (the new build won only 3–6 of 15 pairs): its
16 MiB buffers now live in MoltenVK's host-visible memory type, which costs more per dispatch than
the host-invisible type it used before. Direct mapping needs host-visible memory, so this is a
trade-off of the change rather than noise. KORA kernels at 1M elements still execute about 3× longer
than raw Vulkan on MoltenVK by device timestamp (but not on KosmicKrisp); #229 tracks that separately.
These are host-boundary measurements on one device, not kernel-only or CUDA comparisons.

Reproduce with two builds and alternate them; then render the table and charts:

```sh
for i in 01 02 03; do
  KORA_DEVICE=2 before/tools/kora-bench --gpu --iterations 200 --warmup 30 > before-$i.json
  KORA_DEVICE=2 after/tools/kora-bench --gpu --iterations 200 --warmup 30 > after-$i.json
done
python3 tools/kora-bench/render_ab.py --before before-*.json --after after-*.json \
  --output-dir benchmarks/charts
```

All 30 retained reports are in
[`benchmarks/results/issue227-uma-direct/`](https://github.com/varunkarthic/kora/tree/main/benchmarks/results/issue227-uma-direct). The
regenerated snapshot charts and [comparison tables](benchmark-compute-results.md) come from a separate
1,000-sample/50-warm-up report at 8ca0de7,
[retained here](../benchmarks/results/macos-arm64-apple-m4-moltenvk-compute-sweep-1000x50-8ca0de7.json).

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
