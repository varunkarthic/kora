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
| `cpu_vector_add_N` | Add two uint32 arrays for `N` in 1, 64, 256, 1024, 4096, 65536, 262144, 1048576, and 4194240. | Arithmetic-only single-thread host baseline; arrays are initialized outside the timed loop. |
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

## Report contract

The root object has `schema_version: 3`, `clock`, `iterations`, `warmup`, `environment`, and `results`.
`environment` records OS, architecture, compiler, CMake build configuration, and logical CPU count.
Each result has a stable `name`, `unit`, `timing_domain`, `median`, `p95`, and `samples` array. Transfer results also report
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

CPU-side measurements are always available. Optional GPU cases cover tiny dispatch and size-swept
vector-add/SAXPY, reduction, tiled scan, tiled matmul, stencil, transfer sizes, and raw Vulkan
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

| Measurement | Before (`8dd2978`) | After (`67bec0f`) | Change |
| --- | ---: | ---: | ---: |
| 4K KORA deferred-readback host completion, median | 676.500 µs | 523.958 µs | 22.5% lower |
| 4K KORA deferred-readback host completion, p95 | 819.958 µs | 674.041 µs | 17.8% lower |
| 4K raw Vulkan host completion, median | 346.875 µs | 406.708 µs | 17.3% higher in later run |
| KORA/raw host median ratio | 1.95× | 1.29× | 33.9% lower |
| Pipeline lookup phase, median (diagnostic mode) | 181.667 µs | 6.709 µs | 96.3% lower |

The ordinary runs used `KORA_DEVICE=2 build/release/tools/kora-bench --gpu --iterations 1000
--warmup 50` on macOS arm64 / Apple M4, AppleClang 21.0.0 Release, Vulkan 1.3.357 / MoltenVK,
with KORA and raw Vulkan pinned to the same enumerated physical device. The pipeline phase uses
`--profile-phases --iterations 200 --warmup 20`; it enables telemetry and must not be compared as a
normal performance result. Its pre-fast-path profile was collected on the profiling branch worktree
before the optimization commit, while the post-fast-path phase sample is on `67bec0f`.

Both ordinary 1,000-sample reports preserve every sample and the environment fields:
[before `8dd2978`](../benchmarks/results/dispatch-overhead-before-8dd2978-macos-arm64-m4-moltenvk-1000x50.json)
and [after `67bec0f`](../benchmarks/results/dispatch-overhead-after-67bec0f-macos-arm64-m4-moltenvk-1000x50.json).
The two runs were sequential and raw Vulkan varied by about 17%; treat the ratio improvement as
promising local evidence, not a controlled cross-device result or a CUDA parity claim. Larger 1M and
4.19M vector-add workloads are retained in each raw report to make throughput regressions inspectable.

### Visual comparison and methodology

The following plots are generated from the same retained JSON file as the table below; they are not
hand-transcribed. The vertical latency axis is logarithmic, with units chosen for the range shown in
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

Regenerate the visualizations from the checked-in 100-sample report (10 warm-ups) with:

```sh
python3 tools/kora-bench/render_charts.py \
  benchmarks/results/macos-arm64-apple-m4-moltenvk-100x10.json \
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
| 4096 | 375 ns | 375 ns | 2.259 ms | 3.668 ms | 595.208 µs | 652.208 µs | 404.667 µs | 449.417 µs | 14.041 µs | 18.458 µs | 21.334 µs | 25.208 µs |
| 1048576 | 224.375 µs | 249.791 µs | 2.517 ms | 5.543 ms | 799.750 µs | 871.083 µs | 996.000 µs | 1.251 ms | unavailable | unavailable | 359.125 µs | 404.583 µs |
| 4194240 | 984.125 µs | 1.215 ms | 2.942 ms | 3.799 ms | 2.349 ms | 2.981 ms | 1.964 ms | 2.112 ms | unavailable | unavailable | 1.315 ms | 1.398 ms |

Regenerate the comparison rows from the retained raw report with:

```sh
python3 tools/kora-bench/render_comparison.py benchmarks/results/macos-arm64-apple-m4-moltenvk-100x10.json
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
