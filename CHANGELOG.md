# Changelog

All notable changes to KORA are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.41] - 2026-10-11

- Add a fixed-size CPU executor with priority lanes, work stealing, graceful draining, and
  exception-aware Events.

## [Unreleased]

### Added
- Start the KORA 1.0 Rust migration: a Cargo workspace (`crates/kora`) builds alongside the C++
  tree, CI checks Rust formatting, Clippy, tests, and rustdoc, and the vendor-neutrality check now
  scans Rust sources.
- Port the core types to Rust: structured `Error`/`ErrorCode` with cause chains, checked size
  arithmetic, categorized logging (`kora_log!`), `RuntimeOptions` resolution with every `KORA_*`
  variable, the bounded strict JSON parser and writer, kernel-metadata parsing/validation/
  serialization, and verification comparators (now a type-safe `VerificationPolicy` enum).
- Port host-side planning to Rust: `Event`/`EventPromise` (an abandoned promise now cancels its
  event instead of hanging waiters), the work-stealing `CpuExecutor` (no global lock on the
  submit/run path), `TaskGraph`/`ExecutionPlan` with allocation-free replay, temporary-buffer
  aliasing (reachability bitsets replace per-pair graph searches), `DynamicPartitioner`, scheduler
  cost/policy models, transfer models, and the persistent performance database.
- Port the Vulkan bootstrap to Rust on `ash`: runtime loader discovery (including macOS SDK and
  Homebrew locations), instance negotiation with portability enumeration, owned physical-device
  snapshots with no escaping `pNext` chains, logical devices that enable only queried features,
  `DeviceInfo` capability tiers and memory topology, device selection, and deterministic fault
  injection. An advertised validation layer that fails to load now falls back to an unvalidated
  instance with a warning instead of failing creation.
- Port the memory subsystem to Rust: the TLSF-style suballocator (statistics are now O(1)
  running counters instead of a full region walk per allocation), device-memory pools that try
  the allocation before running recovery and release surplus empty blocks, the budget ledger and
  memory-type selection, staging rings whose slices keep their mapping alive and reclaim through a
  timeline heap, dirty-range flush/invalidate, residency tracking, LRU-2 eviction planning, and
  the bounded OOM recovery coordinator. Fault injection can now fail a window of consecutive calls.
- Port queue execution and transfers to Rust: `QueueContext` (submission and completion threads,
  timeline or pooled-fence completion, coalescing, device-loss propagation; Vulkan objects are
  released only after both workers exit), the command-buffer ring, hazard tracking with precise
  barriers, queue-family ownership barriers, the GPU timestamp ring, and the `TransferEngine`.
  Staged transfers no longer serialize on the previous staged transfer: producers wait for pending
  reservations to be submitted, and transfers larger than the staging ring are split into pipelined
  chunks instead of failing. Downloads deliver bytes to a completion closure, and transfers keep
  their buffers alive until completion.
- Port the kernel layer to Rust: structural SPIR-V validation and compute-interface reflection,
  the per-device `PipelineManager` (LRU pipeline cache, shared descriptor and pipeline layouts,
  descriptor pools, and the identity-bound persistent pipeline cache in the existing file format),
  and the `LocalSizeId` workgroup autotuner. Cache hits no longer re-validate the module or copy it
  into a key, pipeline compilation runs outside the manager lock, and pipelines, layouts, and
  descriptor-set leases are reference counted so they stay valid after eviction or manager drop.
  Descriptor pools grow by counting their sets instead of relying on driver out-of-pool errors.
  SPIR-V test fixtures are committed, so `cargo test` needs no shader compiler.
- Port logical buffers to Rust: `BufferHandle` and `Pod`-typed `Buffer<T>` with per-buffer locks
  (no global registry mutex), exclusive mapping guards that cannot race host writes, and versions
  that are unique across buffers. Device results that arrive while a buffer is mapped are parked
  and applied on the next access instead of blocking the completion thread. A thread that re-enters
  a buffer it has mapped gets an error instead of a deadlock. A full-range write mapping now
  clears a data-loss mark, and per-device replicas are released with the last handle.
- Port the runtime core to Rust: `Runtime::new` with device selection and `AllEligible`
  quarantine, kernels validated and pre-compiled at load (`load_kernel`, `load_kernel_on`,
  `load_kernel_file`), the `DispatchBuilder` (`read`/`write`/`read_write`, `Pod` push constants,
  priority, deferred readback), `readback`, CPU fallbacks with typed `CpuTaskContext` views,
  memory, transfer, and timestamp statistics, and `VK_EXT_device_fault` reports. Staged input
  uploads no longer block the submitting thread until the copy finishes, and device replicas are
  freed as soon as the last handle and in-flight submission release them.
- Complete the Rust runtime: bounded startup calibration (submission, transfer, and compute
  probes cached per device and driver; corrupt caches are rebuilt, and `Runtime::calibrate`
  forces a refresh), the calibration cost model, `dispatch_adaptive` placement that blends
  calibration with confident online observations and returns to the device that holds
  device-resident inputs, duplicate verification across devices with quarantine on mismatch,
  `recover` and `rebuild_devices` that keep buffers and CPU fallbacks, per-dispatch host phase
  profiling, and Chrome trace export with scheduler scores. Profiling stats report whether the
  device has GPU timestamps.
- Port the Cluster to Rust: `Cluster` with hot-swappable remote workers, failover, split
  dispatches, local device-loss recovery, the device-rescan monitor, and the `kora-top` status
  stream; `WorkerServer` and the `kora-worker` binary (new `kora-tools` crate). The worker protocol
  is now version 2 (peers of different versions reject each other at the first frame) and uses the
  RustCrypto HMAC/SHA-256 and `crc32c` crates instead of hand-written implementations. Released
  buffers are dropped from worker caches within a quarter second, and a malformed worker reply that
  names the same output twice now drops the link instead of being committed.

- Port the command-line tools to Rust in `kora-tools`: `kora-info` (same `--json` schema 2),
  `kora-top` (live view and `--demo`), and `kora-bench`. The Rust `kora-bench` measures the same
  CPU, KORA Runtime, raw Vulkan, transfer, staging-ring, command-pool, and Cluster workloads under the
  same result names, and writes report schema 4: schema 3 plus `implementation: "rust"` and new
  CPU-executor submit and fan-out, allocator-churn, and logical-buffer creation families. CI checks
  both tools' JSON contracts against lavapipe. `MappedPtr::from_raw` lets externally mapped memory
  be used as a transfer buffer.
- Add the stable C ABI (`kora-capi`, built as `libkora` with a cbindgen-generated `kora.h`): opaque
  runtime, buffer, kernel, dispatch, and event handles; `KoraStatus` codes equal to the KORA error
  codes; a per-thread last-error message; and panics contained at the boundary.
  `tools/ci/check_capi.sh` builds and installs the library, compiles the C vector-add example
  against the installed header, and runs it (skipping cleanly without a Vulkan device).
- Replace the pybind11 extension with PyO3/maturin bindings (`kora-runtime` wheel, abi3 for
  Python 3.11+). Names match the 0.x module; new: `Runtime.create_buffer_from` and
  `Buffer.to_numpy` for NumPy, SPIR-V bytes accepted by `Runtime.load_kernel`, `Event.done`,
  `Buffer.id`, `KernelMetadata.from_json`/`to_json`, and `RuntimeOptions.enable_validation` and
  `enable_startup_calibration`. Behavior changes: Python 3.9 and 3.10 are no longer supported;
  `RuntimeOptions` fields default to `None` (environment and built-in defaults apply); a
  `MappedView` belongs to the thread that created it, and while it lives, other access to that
  buffer from the same thread raises `KoraError` (`INVALID_ARGUMENT`) instead of deadlocking;
  `MappedView.release()` ends a mapping early; a `DispatchBuilder` can be submitted once.
- Measure the Rust runtime against 0.3 C++ in ten ABBA-interleaved pairs per device on Apple M4
  (MoltenVK and KosmicKrisp); reports and summaries are retained in
  `benchmarks/results/rust-vs-cpp-apple-m4`. `tools/kora-bench/run_ab.py` runs interleaved pairs and
  `render_ab.py --families` summarizes every benchmark family (schema 3 and 4 reports).
- Task-graph finalization is about 35% faster (no per-task tree sets, compact successor lists), and
  staging-ring reservations no longer slow down with the number of live ranges (cancelling the
  newest reservation returns its space; reservation IDs use a cheap hash): reserve plus cancel with
  512 live ranges went from 1.4 µs to 0.16 µs.
### Fixed
- After a device loss, results that only the lost device held (deferred outputs that were never
  read back) are marked lost on recovery instead of silently reverting to older host contents.
- Dispatches that read or overwrite a buffer written by an earlier dispatch now record a memory
  barrier; independent dispatches still record none. The C++ runtime recorded no barrier between
  dependent dispatches.
- Once a `LocalSizeId` kernel finishes autotuning, every dispatch is specialized with the tuned
  workgroup size. The C++ runtime computed group counts from the tuned size but compiled the
  default size, which could leave elements unprocessed.
- A dispatch whose input has newer, unread results on another device now fails with a clear error
  instead of silently uploading stale host bytes.
- A graceful shutdown now drains follow-up work that completion callbacks enqueue (such as a
  dispatch's output readback) instead of rejecting it.
- Logical devices now enable `shaderInt16`, `shaderInt64`, and `shaderFloat64` when supported,
  so modules declaring those capabilities (which reflection already accepted) are valid, and
  enable `maintenance4` when available; `LocalSizeId` kernels are rejected with a clear error on
  devices without it instead of creating an invalid pipeline.
- A positive integer `default` for an `i32`/`i64` specialization constant, or an integer default
  for an `f32`/`f64` constant, is now accepted and converted to the declared type instead of being
  rejected as a type mismatch.

## [0.3.0] - 2026-10-20

### Changed
- Start the 0.3 release line. Releases are `v0.3.x` and remain Beta until the v0.3 release-validation
  issues close.

## [0.2.28] - 2026-10-20

### Added
- Add `kora-top`, a live terminal view of a `kora::Cluster`. It shows each local device and remote
  worker, which of them received work in the last second, device memory against the driver budget,
  worker cache, and connect, disconnect, failover, rescan, and routing events. `kora-top --demo`
  runs a self-contained Cluster whose loopback workers are unplugged and replugged while every
  result is verified.
- Add `ClusterOptions::status_listen` (or `KORA_STATUS_LISTEN`), an opt-in, loopback-only stream of
  JSON status snapshots, together with `Cluster::events()` and `Cluster::status_port()`.
  `LocalDeviceStatus` gains allocated and budget bytes, and `WorkerStatus` gains in-flight and
  cached bytes.
- Add `Cluster::load_kernel()` for in-memory SPIR-V words.

### Changed
- Redesign the documentation landing page with feature cards and a measured comparison chart, and
  group the sidebar into Getting started, Guides, Performance, API reference, Internals, and
  Project.

### Fixed
- Fix the Windows build of `kora-top`. Make `kora-top --demo` exit with CTest's skip code instead
  of failing when no Vulkan device is available.

## [0.2.27] - 2026-10-20

### Added
- `kora::Cluster` now uses every local GPU as well as remote workers. Each dispatch goes to the target
  with the lowest measured expected completion time. `ClusterDispatch::split()` divides a large
  elementwise dispatch across devices for kernels with a `u32` `base` push constant. The results are
  committed to the host buffer atomically.
- `kora::Cluster` rescans physical devices every `device_rescan_interval` (default 5 s). It rebuilds
  its local devices when a GPU appears or disappears, without losing buffers.
- Add `Cluster::local_devices()` with per-device dispatch, slice, and in-flight counts, and add split,
  slice, and rescan counters to `ClusterStats`.
- Add `Runtime::rebuild_devices()` to recreate every logical device and re-enumerate physical
  devices while keeping buffers.

### Fixed
- Fix a race in which a concurrent staging reservation could reclaim a completed readback slice
  before its completion callback copied the bytes to the host. Concurrent downloads then failed with
  "staging readback slice is unknown or not submitted".

## [0.2.26] - 2026-10-20

### Added
- Add the optional macOS `competitor-bench` tool and `render_competitors.py`, and publish a same-device
  comparison of KORA with Metal, OpenCL 1.2, OpenGL 4.1, Accelerate, and raw Vulkan, including a
  KosmicKrisp-versus-MoltenVK driver comparison.

## [0.2.25] - 2026-10-20

### Fixed
- Fix the Windows build of the remote-worker socket layer, where `windows.h` `min`/`max` macros broke
  `std::min`/`std::max`.

### Added
- Add `kora::Cluster`, `kora::WorkerServer`, and the `kora-worker` tool: offload dispatches to KORA
  workers on other machines or devices you own, with workers connecting, disconnecting, or
  reconnecting at any time. Host buffers stay authoritative; a remote result is committed atomically
  or the dispatch re-runs locally, so unplugging a worker never loses or corrupts work. Clients
  authenticate with an HMAC-SHA256 challenge; without a token, workers accept only loopback clients.
- Add `Runtime::load_kernel()` for in-memory SPIR-V words.
- Add `kora-bench --cluster` to measure `Cluster` dispatches locally and on a loopback worker.

## [0.2.24] - 2026-10-20

### Changed
- Record per-dispatch GPU timestamps only as often as telemetry needs them: none at
  `TelemetryLevel::Off`, one dispatch in 16 at the default `Basic`, and every dispatch at `Profile` and
  `Trace`. Autotuning trials and adaptive observations are always timestamped. Across 10 interleaved
  pairs on Apple M4/MoltenVK, every matched benchmark case got faster at the median (2–27%), with no
  regressions.
- `kora-bench` now measures at the default `Basic` telemetry level, and `render_ab.py` reads
  compressed `.json.gz` reports and takes `--change`/`--prefix` for chart titles and names.

## [0.2.23] - 2026-10-20

### Fixed
- Free each device's copy of a runtime buffer once its last handle is released and no queued GPU
  work can still use it. Previously device allocations accumulated until the `Runtime` was
  destroyed (48 create/dispatch/drop cycles grew live allocations from 5 to 152).

## [0.2.22] - 2026-10-20

### Changed
- On unified-memory devices, write dirty runtime-buffer inputs and read default-readback outputs
  through the buffer's mapping instead of staging each transfer and waiting for it, whenever no
  queued GPU work can access that buffer. On Apple M4/MoltenVK, default-readback dispatches were
  15–60% faster across 19 matched cases and deferred SAXPY 37–46% faster; deferred vector add with
  16 MiB buffers was 10–26% slower because direct mapping requires host-visible memory.
- Classify devices whose memory heaps are all device-local as `Uma` even when they also expose
  host-invisible memory types, and allocate runtime buffers with the `Auto` memory policy.
- Read the host shadow through a pinned map when uploading instead of copying it into a temporary
  vector.

### Added
- Add `render_ab.py` to summarize interleaved before/after benchmark runs as paired-ratio tables and
  SVG charts.

### Fixed
- Report KORA device timestamps that some drivers publish shortly after the queue signal instead of
  intermittently dropping them; `Runtime::profiling_stats()` now waits up to 2 ms for them.

## [0.2.21] - 2026-10-20

### Added
- Sweep matched KORA/raw Vulkan SAXPY, reduction, scan, and stencil benchmarks over 4K, 64K, and
  1M elements and tiled matmul over 16², 64², and 128², with matching CPU tiled-scan baselines, a
  per-size comparison table with deferred-to-raw ratios, and a grouped compute host-latency chart.

### Fixed
- Keep the benchmark chart timing-domain label inside its badge.

## [0.2.20] - 2026-10-12

### Added
- Add separately validated default-readback and deferred-readback host measurements for matched
  KORA compute workloads, and report compute timestamp availability per workload instead of
  inferring it from vector-add timer support.

### Changed
- Render KORA default, KORA deferred, raw Vulkan host, and device-timestamp domains as distinct
  benchmark columns; validate deferred outputs with explicit readback after each timed sample.
  Keep KORA timestamp probes on the default completion path because the deferred path does not
  currently publish Runtime profiling durations; host timing remains separately deferred.

## [0.2.19] - 2026-10-12

### Fixed
- Keep CPU-only `kora-bench` builds compiling when optional GLSL shader tooling is unavailable.

### Added
- Add same-device raw Vulkan baselines for SAXPY, reduction, scan, tiled matrix multiplication, and stencil workloads, including host-submit and device-timestamp comparisons.

### Changed
- Keep live staging-ring ranges ordered by buffer offset, removing per-reservation scratch allocation and sorting; add dense and fragmented reserve-only occupancy benchmarks.
- Exclude per-sample correctness scans from CPU and raw Vulkan benchmark timing while still validating every measured output.

## [0.2.18] - 2026-10-12

### Changed
- Keep live staging-ring ranges ordered by buffer offset, removing per-reservation scratch allocation and sorting; add dense and fragmented reserve-only occupancy benchmarks.

## [0.2.17] - 2026-10-12

### Added
- Split diagnostic GPU completion profiling into completion-worker wake delay, Vulkan wait, timestamp collection, and Runtime callback phases.

### Changed
- Avoid an idle coalescing collection delay for isolated GPU dispatches while retaining eager batching for compatible submissions already queued together.

## [0.2.16] - 2026-10-12

### Added
- Expand `kora-bench` CPU compute coverage with single-thread and four-worker vector-add, SAXPY, reduction, stencil, and tiled matrix multiplication cases; validate full outputs outside timed intervals and record CPU model, build flags, source revision, and worker count.

## [0.2.15] - 2026-10-12

### Added
- Add an opt-in external CUDA reference report contract and same-device comparison table without adding a CUDA build/runtime dependency.

## [0.2.14] - 2026-10-12

### Added
- Expose opt-in per-dispatch host phase timings and add a diagnostic `kora-bench --profile-phases` mode for validation, pipeline lookup, buffer residency, submit, queue, descriptor, command-recording, completion, and device-timestamp measurements.

### Fixed
- Skip repeated SPIR-V reflection on exact pipeline-cache hits for statically sized kernels while preserving full validation for cache misses and `LocalSizeId` specialization.

## [0.2.13] - 2026-10-12

### Changed
- Redesign the docs landing and release guide around user tasks, installation, verification, and release maturity; improve generated benchmark SVG hierarchy while preserving timing-domain separation.

## [0.2.12] - 2026-10-12

### Fixed
- Preserve indented continuation lines when generating release highlights from wrapped changelog
  bullets.

## [0.2.11] - 2026-10-12

### Changed
- Organize generated GitHub release notes into a version overview, platform downloads, highlights,
  install steps, and grouped release-validation gates.

## [0.2.10] - 2026-10-12

### Added
- Generate deterministic, timing-domain-separated benchmark SVG charts from retained schema-3
  reports and document their methodology and limitations.

## [0.2.9] - 2026-10-12

### Changed
- Cache the immutable duplicate-verification device set per Runtime instead of allocating and
  scanning it for every ordinary dispatch; verification still checks current quarantine state at
  submission time.

## [0.2.8] - 2026-10-12

### Changed
- Expand `kora-bench` JSON to schema 3 with platform/compiler/build metadata and explicit timing
  domains; add matched size-swept raw Vulkan vector-add results beside CPU and KORA results, including
  separately labeled default/deferred readback and GPU timestamp measurements when supported.
- Clarify that software Vulkan devices are not hardware-GPU measurements and document the cross-OS
  benchmark smoke coverage.

## [0.2.7] - 2026-10-12

### Fixed
- Correct the benchmark documentation to identify the current `kora-bench` JSON schema as version 3.
- Reject pull requests that omit a new categorized `[Unreleased]` changelog entry before they can merge.

## [0.2.6] - 2026-10-12

### Fixed
- Synchronize CPU executor task-drain counter updates with the worker wait mutex so graceful shutdown
  cannot lose the final wakeup and strand idle workers.
- Reject pull requests that modify published changelog sections, keeping new fixes attached to the tag
  that contains them.

## [0.2.5] - 2026-10-12

### Fixed
- Bound release CTest cases and package jobs so a stalled platform cannot indefinitely hold the
  release queue or publish an incomplete package set.

### Added
- Embed a vendor-neutral compute calibration shader and record validated tiny-dispatch, streaming
  read/write, and integer ALU measurements when supported within the configured best-effort budget.
- Add an explicit adaptive-dispatch API that uses identity- and completeness-gated calibration costs;
  ordinary dispatch remains device-bound and incomplete records retain the current device.
- Learn fresh per-kernel GPU execution costs from completed adaptive-dispatch timestamp samples, with
  identity-safe keys, outlier rejection, expiry, bounded persistence, and conservative calibration blending.
- Add the selected adaptive-placement score breakdown to its completed GPU trace event under Trace
  telemetry; explicit dispatch and score-free cold-start fallback remain unannotated.

### Changed
- Bump the internal calibration-record schema to 2 so older records are recalibrated.

## [0.2.4] - 2026-10-12

### Fixed
- Reject stale issue branches in PR CI and refuse releases with an empty `Unreleased` changelog section.
- Align the v0.2.2 changelog section with the benchmark device-matching change included in that tag.

### Changed
- Document branch synchronization and release changelog requirements.

## [0.2.3] - 2026-10-12

### Fixed
- Paginate release milestone and issue queries, rejecting incomplete or malformed API results before calculating Beta/Stable status.

## [0.2.2] - 2026-10-12

### Fixed
- Ensure KORA and raw Vulkan benchmark measurements use the same Runtime-selected physical device and identify the sanitized device and path in JSON.

### Changed
- Bump the `kora-bench` JSON schema to version 2 with GPU device and measurement-path metadata.

## [0.2.1] - 2026-10-12

### Fixed
- Make `kora-info` report that runtime pipeline-cache state was not queried, rather than implying the cache is absent or uninitialized.

### Changed
- Bump the `kora-info` JSON schema to version 2 and move pipeline-cache reporting to the root-level `pipeline_cache_state` field.
## [0.2.0] - 2026-10-12

### Changed

- Redesign the README and documentation landing page around verified capabilities, setup workflows,
  architecture/dispatch diagrams, support boundaries, and accessible subsystem guides.
- Refresh benchmark snapshots with a pinned reference device, complete environment metadata,
  reproducible commands, and explicit single-host interpretation limits.
- Start the `0.2.x` release line; patch versions remain automatically assigned from Git tags.

## [0.1.53] - 2026-10-11

- Add isolated, budget-bounded H2D, D2H, and device-copy startup calibration probes with byte validation,
  repairable atomic cache replacement, and an explicit `Runtime::calibrate()` refresh operation.

## [0.1.52] - 2026-10-11

- Honor `RuntimeOptions::shutdown_mode` during teardown: graceful shutdown drains accepted work,
  while cancel-pending shutdown completes queued CPU/GPU work with cancellation status and still
  waits for work already executing or submitted. Runtime destruction is explicitly `noexcept`.

## [0.1.51] - 2026-10-11

## [0.1.50] - 2026-10-11

## [0.1.49] - 2026-10-11

## [0.1.48] - 2026-10-11

## [0.1.47] - 2026-10-11

## [0.1.46] - 2026-10-11

## [0.1.45] - 2026-10-11

### Added

- Add `kora-bench` CPU and optional Vulkan JSON microbenchmarks for scheduler cost, task-graph
  planning/replay, allocator throughput/fragmentation, vector addition, SAXPY, reduction, scan, tiled
  matmul, stencil, repeated tiny dispatches, size-swept direct/staged transfers, and CPU/GPU crossover
  measurements, with correctness checks, raw samples, median/p95, and effective transfer throughput.
- Add raw Vulkan buffer-copy and vector-add dispatch baselines beside equivalent KORA workloads, plus
  a Python callback-replay benchmark compared with a reusable-plan C++ callback.
- Add debug safety diagnostics with Release opt-in, Debug host-shadow canaries, Vulkan validation
  requests, and detailed synchronization logs.
- Add bounded Chrome Trace Event JSON export for completed GPU timestamp samples at Trace telemetry
  level, including device/submission correlation and calibrated host clock metadata when available.

- Add one-second Vulkan memory-budget refresh, 80% configurable heap soft limits, backing-allocation
  accounting, and failure-adaptive admission limits.
- Add generation-aware LRU-2 buffer eviction planning with KeepValue scoring, protected-allocation
  filtering, and explicit dirty-copy writeback decisions.
- Add memory admission estimates and finite OOM recovery stages with one retry, smaller-block fallback,
  task-level chunking/CPU fallback hooks, and structured exhaustion diagnostics.
- Add nonblocking GPU dispatch timestamp profiling with valid-bit wrap handling, bounded query rings,
  calibrated host correlation when supported, and runtime profiling statistics.
- Add capability-filtered LocalSizeId workgroup tuning with bounded warmup/measurement trials,
  timestamp-driven selection, driver-version-scoped persistence, and an opt-out control.
- Add an opt-in pybind11 extension and scikit-build-core Python package exposing linked runtime
  version and persistence schema identity.
- Expose Python Runtime, Buffer, Kernel, DispatchBuilder, and awaitable Event wrappers with mapped
  memoryviews, structured `KoraError` codes, GIL release around blocking/native operations, and
  retained owners for in-flight work.
- Expose Python TaskGraph metadata construction, deterministic finalization, and reusable
  ExecutionPlan replay through a thin callback adapter.
- Add explicit device-targeted kernel loading, dispatch routing, and source-device readback for
  cross-device host staging. Keep per-device queue timeline ownership and replicas isolated; no
  peer-memory fast path is enabled without measured evidence.
- Add a measured-throughput task partitioner with per-device memory caps, proportional contiguous
  ranges, minimum/granularity constraints, profitability fallback, and caller-owned range execution.
- Mark logical devices unhealthy after observed Vulkan device loss, stop new submissions across their
  queues, fail queued work deterministically, and preserve raw Vulkan failure diagnostics.
- Return `DataLoss` instead of stale host bytes for outputs of in-flight work lost with a Vulkan
  device; restore host readability only after a complete host rewrite.
- Add transactional `Runtime::recover()` to rebuild logical devices after observed loss while
  retaining host-backed buffer handles and CPU fallback registrations; loaded kernels must be reloaded.
- Drain queued work to `DeviceLost` immediately when the shared logical-device fault state trips, avoiding
  shutdown-order-dependent cancellation of pending submissions.
- Capture optional, bounded `VK_EXT_device_fault` address and vendor records after device loss and expose
  them through `Runtime::device_fault_report()` without writing sensitive details to logs or files.
- Distinguish queue-wait timeouts from device loss and isolate failed device initialization under
  `DevicePolicy::AllEligible`, exposing skipped devices through `Runtime::quarantined_devices()`.
- Add opt-in duplicate dispatch with exact, absolute, relative, ULP, and custom result comparison;
  mismatch marks outputs unavailable and quarantines every participating device.

## [0.1.44] - 2026-10-11

### Added

- Add logarithmic size buckets and measured latency/bandwidth models for five transfer paths.
- Add an inspectable scheduler cost breakdown, memory-pressure penalty, capped locality benefit, and
  queue-availability correction from real completion samples.
- Add a bounded graph-lookahead movement penalty to scheduler cost estimates (default beta 0.25).
- Add constraint-filtered scheduler candidate selection with confidence-decaying exploration,
  hysteresis, priority aging, and deterministic tie-breaking.
- Coalesce compatible low/normal-priority dispatch submissions in a bounded queue window while
  preserving per-dispatch Events and excluding resource conflicts.

## [0.1.42] - 2026-10-11

### Added

- Added a thread-safe scheduler performance database with EWMA/variance, robust bounded-window
  quantiles, MAD outlier rejection, aging, and schema-versioned atomic persistence.

- Added keyed native CPU fallback registration and asynchronous `Runtime::run_cpu_fallback()` with
  access-checked host mappings and GPU readback/upload visibility across mixed dispatch chains.

## [0.1.40] - 2026-10-11

- Reuse version-current device buffers across dispatches and support deferred, explicitly requested
  readback for device-resident chains.

## [0.1.39] - 2026-10-11

- Add conservative task-DAG temporary-buffer alias planning with compatibility checks and persistent
  resource isolation.

## [0.1.38] - 2026-10-11

- Add immutable ExecutionPlan snapshots and callback-driven replay without repeating graph analysis
  or allocating plan storage per replay.

## [0.1.37] - 2026-10-11

- Add public Task/TaskGraph metadata, automatic RAW/WAR/WAW dependencies, explicit ordering, and
  deterministic topological cycle validation.

## [0.1.36] - 2026-10-11

- Add cached queue-family policy, concurrent runtime-buffer sharing, and timeline semaphore ordering
  between dedicated transfer and compute queues when their families differ.

## [0.1.35] - 2026-10-11

- Add capability-selected `vkCmdPipelineBarrier2`/`vkQueueSubmit2` execution with legacy lowering and
  a CI integration pass using Vulkan synchronization validation.

## [0.1.34] - 2026-10-11

- Add per-buffer RAW/WAR/WAW hazard planning with precise host/transfer/compute barrier scopes and
  replace broad transfer barriers with tracker-generated plans.

## [0.1.33] - 2026-10-11

- Add reusable-fence completion fallback for devices without timeline semaphores and wire the
  force-legacy validation option through device creation and Runtime submissions.

## [0.1.32] - 2026-10-11

- Document the per-queue timeline completion contract, resource-lifetime integrations, and add a real
  Vulkan test for dependent timeline submissions and value-based waits.

- Add versioned, device-identity-bound startup calibration records with bounded best-effort probing,
  corruption-safe cache loading, and explicit configuration. Transfer and compute measurements remain
  marked unavailable until safe scratch-buffer probes are implemented.

## [0.1.30] - 2026-10-11

- Add a fluent storage-buffer dispatch API with checked workgroup limits, typed host-shadow buffer
  copies, compiled vector-add/SAXPY integration fixtures, and event completion covering output readback.

## [0.1.29] - 2026-10-11

## [0.1.28] - 2026-10-11

## [0.1.27] - 2026-10-11

- Add strict, bounded `.kora.json` parsing and a programmatic kernel metadata model with duplicate,
  range, type, UTF-8, and device-independent local-size validation.
- Add structural SPIR-V validation and compute-interface reflection for entry points, descriptors,
  push constants, specialization constants, and metadata consistency checks.
- Add per-device compute pipeline and descriptor-layout caches, bounded LRU eviction, recycled
  descriptor sets, specialization-data mapping, and identity-validated persistent Vulkan pipeline caches.

## [0.1.26] - 2026-10-11

## [0.1.25] - 2026-10-11

## [0.1.24] - 2026-10-11

## [0.1.23] - 2026-10-11

- Add asynchronous Vulkan buffer upload, download, copy, and fill tasks with queue-ordered completion
  events, topology-selected staging/direct paths, and non-coherent memory maintenance.
- Extend `Event` with completion callbacks, status continuations, and aggregate `when_all` waits.
- Bound completion-worker timeline waits with polling intervals while retaining submitted resource
  lifetimes until completion; cover repeated real-Vulkan context create/submit/shutdown cycles.
- Add persistent, bounded upload/readback staging rings with timeline-tagged reservations, safe
  reclamation, growth/wait exhaustion handling, and topology-aware transfer-path selection.

## [0.1.22] - 2026-10-11

- Add an MPSC Vulkan queue service with serialized submission, timeline-based GPU completion callbacks,
  Drain/CancelPending shutdown, and concurrent producer stress coverage.

## [0.1.21] - 2026-10-11

- Add a thread-owned Vulkan command-pool set with preallocated reusable command buffers, per-slot
  completion tracking, one-time recording, and mock plus 10,000-submission integration coverage.

## [0.1.20] - 2026-10-11

- Extend memory-pool fragmentation snapshots, add checked cross-pool telemetry aggregation, and add a
  deterministic lavapipe stress test for repeated small allocation/free cycles. Add a move-only public
  Runtime that owns selected Vulkan devices and exposes per-device memory and heap-budget snapshots.

## [0.1.19] - 2026-10-11

## [0.1.18] - 2026-10-11

### Added

- Coalesced dirty-range tracking, non-coherent atom alignment, and dispatch-backed Vulkan flush and
  invalidate helpers with coherent-memory no-ops.
- Versioned host/device residency tracking, stale-copy-aware transfer planning, and invariant-safe
  per-device eviction.

## [0.1.17] - 2026-10-11

### Added

- Reference-counted logical buffer handles with typed host-shadow mappings, generation-checked
  identities, pin tracking, and timeline-gated deferred reclamation.

## [0.1.16] - 2026-10-11

### Added

- Allocation-time memory-policy selection that filters policy rankings by Vulkan resource
  compatibility and required properties, with per-heap reservation accounting from
  `VK_EXT_memory_budget` or a conservative heap-size fallback.

## [0.1.15] - 2026-10-11

## [0.1.14] - 2026-10-10

## [0.1.13] - 2026-10-10

## [0.1.12] - 2026-10-10

## [0.1.11] - 2026-10-10

## [0.1.10] - 2026-10-10

### Added

- Vulkan memory blocks and pools with bounded small-buffer suballocation, persistent host mapping,
  dedicated-allocation bypass, adaptive block sizing, and allocation-call telemetry.
- Checked, alignment-aware two-level segregated-fit suballocation with immediate coalescing,
  fragmentation telemetry, Debug invariant validation, and deterministic randomized tests.
- Detailed human and schema-versioned JSON `kora-info` diagnostics for device queues, memory,
  budgets, topology, subgroup/features, portability, MoltenVK, and cache state.
- Vendor-neutral memory topology classification, benchmark-needed signaling for discrete BAR-like
  memory, and ordered per-policy memory-type preferences.
- Capability-gated logical-device creation with compute and optional dedicated-transfer queues,
  portability-subset support, RAII ownership, and a device-level dispatch table.
- Deterministic, vendor-neutral device eligibility and selection for automatic, device-kind
  preference, explicit index/UUID, and all-eligible policies.

## [0.1.9] - 2026-10-10

## [0.1.8] - 2026-10-10

### Added

- Immutable, vendor-neutral device capability snapshots with limits, features, queue families,
  memory topology, subgroup support, and additive A-D convenience tiers.
- Dedicated-compute and dedicated-transfer queue-family classification with preserved queue
  properties and vendor-neutral rules.

## [0.1.7] - 2026-10-10

## [0.1.6] - 2026-10-10

### Added

- `kora-info`, a command-line report for KORA version, Vulkan loader/instance versions, and devices.
- Raw physical-device discovery with versioned core and extension property/feature queries.

## [0.1.5] - 2026-10-10

### Added

- Vulkan instance creation with API-version clamping, portability-extension negotiation, optional
  validation diagnostics, and deterministic lifecycle tests.

## [0.1.4] - 2026-10-10

### Added

- Initial runtime-loaded Vulkan device dispatch table and deterministic fault injection for memory
  allocation and queue submission.

## [0.1.3] - 2026-10-10

### Fixed

- Windows release archives now use the documented MSVC ABI and fail packaging if CMake selects a
  different compiler.

## [0.1.2] - 2026-10-10

### Added

- Completed checked arithmetic with compiler-assisted overflow detection, byte-count and volume
  helpers, bounded range validation, Result-returning alignment, expanded boundary tests, and an
  internal design reference.

## [0.1.1] - 2026-10-10

### Added

- Completed structured logging with automatic `KORA_LOG_LEVEL` initialization, portable
  zero-dependency fallback formatting, defensive category handling, expanded tests, and a debugging
  guide.

## [0.1.0] - 2026-10-10

### Added

- C++20 CMake project with Debug, Release, sanitizer, test, benchmark, and Python presets.
- Installable `kora::kora` CMake package and standalone package-consumer validation.
- Structured `Error`, `Result<T>`, `Status`, checked arithmetic, runtime options, and logging
  foundations.
- Categorized GoogleTest/CTest framework with software Vulkan ICD and capability-skip support.
- Initial build, testing, API, governance, security, and agent documentation.
- Cross-platform CI, automatic semantic patch releases, platform package archives, and
  milestone-driven Beta/Stable release notes.
- Read the Docs-ready Sphinx/MyST documentation covering setup, architecture, configuration, APIs,
  releases, security, and maintenance.
- Complete environment resolution for deterministic mode, shutdown policy, legacy synchronization,
  and trace telemetry.

### Fixed

- Hosted Windows CI now initializes and verifies MSVC explicitly, while the fetched header-only
  Vulkan dependency no longer builds its unused experimental C++ module.
- Release-status refreshes now exit successfully before the repository has its first release.

[Unreleased]: https://github.com/varunkarthic/kora/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/varunkarthic/kora/releases/tag/v0.3.0
[0.2.28]: https://github.com/varunkarthic/kora/compare/v0.2.27...v0.2.28
[0.2.27]: https://github.com/varunkarthic/kora/compare/v0.2.26...v0.2.27
[0.2.26]: https://github.com/varunkarthic/kora/compare/v0.2.25...v0.2.26
[0.2.25]: https://github.com/varunkarthic/kora/compare/v0.2.24...v0.2.25
[0.2.24]: https://github.com/varunkarthic/kora/compare/v0.2.23...v0.2.24
[0.2.23]: https://github.com/varunkarthic/kora/compare/v0.2.22...v0.2.23
[0.2.22]: https://github.com/varunkarthic/kora/compare/v0.2.21...v0.2.22
[0.2.21]: https://github.com/varunkarthic/kora/compare/v0.2.20...v0.2.21
[0.2.20]: https://github.com/varunkarthic/kora/compare/v0.2.19...v0.2.20
[0.2.19]: https://github.com/varunkarthic/kora/compare/v0.2.18...v0.2.19
[0.2.18]: https://github.com/varunkarthic/kora/compare/v0.2.17...v0.2.18
[0.2.17]: https://github.com/varunkarthic/kora/compare/v0.2.16...v0.2.17
[0.2.16]: https://github.com/varunkarthic/kora/compare/v0.2.15...v0.2.16
[0.2.15]: https://github.com/varunkarthic/kora/compare/v0.2.14...v0.2.15
[0.2.14]: https://github.com/varunkarthic/kora/compare/v0.2.13...v0.2.14
[0.2.13]: https://github.com/varunkarthic/kora/compare/v0.2.12...v0.2.13
[0.2.12]: https://github.com/varunkarthic/kora/compare/v0.2.11...v0.2.12
[0.2.11]: https://github.com/varunkarthic/kora/compare/v0.2.10...v0.2.11
[0.2.10]: https://github.com/varunkarthic/kora/compare/v0.2.9...v0.2.10
[0.2.9]: https://github.com/varunkarthic/kora/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/varunkarthic/kora/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/varunkarthic/kora/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/varunkarthic/kora/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/varunkarthic/kora/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/varunkarthic/kora/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/varunkarthic/kora/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/varunkarthic/kora/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/varunkarthic/kora/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/varunkarthic/kora/releases/tag/v0.2.0
[0.1.53]: https://github.com/varunkarthic/kora/compare/v0.1.52...v0.1.53
[0.1.52]: https://github.com/varunkarthic/kora/compare/v0.1.51...v0.1.52
[0.1.51]: https://github.com/varunkarthic/kora/compare/v0.1.50...v0.1.51
[0.1.50]: https://github.com/varunkarthic/kora/compare/v0.1.49...v0.1.50
[0.1.49]: https://github.com/varunkarthic/kora/compare/v0.1.48...v0.1.49
[0.1.48]: https://github.com/varunkarthic/kora/compare/v0.1.47...v0.1.48
[0.1.47]: https://github.com/varunkarthic/kora/compare/v0.1.46...v0.1.47
[0.1.46]: https://github.com/varunkarthic/kora/compare/v0.1.45...v0.1.46
[0.1.45]: https://github.com/varunkarthic/kora/compare/v0.1.44...v0.1.45
[0.1.44]: https://github.com/varunkarthic/kora/compare/v0.1.43...v0.1.44
[0.1.42]: https://github.com/varunkarthic/kora/compare/v0.1.41...v0.1.42
[0.1.41]: https://github.com/varunkarthic/kora/compare/v0.1.40...v0.1.41
[0.1.40]: https://github.com/varunkarthic/kora/compare/v0.1.39...v0.1.40
[0.1.39]: https://github.com/varunkarthic/kora/compare/v0.1.38...v0.1.39
[0.1.38]: https://github.com/varunkarthic/kora/compare/v0.1.37...v0.1.38
[0.1.37]: https://github.com/varunkarthic/kora/compare/v0.1.36...v0.1.37
[0.1.36]: https://github.com/varunkarthic/kora/compare/v0.1.35...v0.1.36
[0.1.35]: https://github.com/varunkarthic/kora/compare/v0.1.34...v0.1.35
[0.1.34]: https://github.com/varunkarthic/kora/compare/v0.1.33...v0.1.34
[0.1.33]: https://github.com/varunkarthic/kora/compare/v0.1.32...v0.1.33
[0.1.32]: https://github.com/varunkarthic/kora/compare/v0.1.31...v0.1.32
[0.1.30]: https://github.com/varunkarthic/kora/compare/v0.1.29...v0.1.30
[0.1.29]: https://github.com/varunkarthic/kora/compare/v0.1.28...v0.1.29
[0.1.28]: https://github.com/varunkarthic/kora/compare/v0.1.27...v0.1.28
[0.1.27]: https://github.com/varunkarthic/kora/compare/v0.1.26...v0.1.27
[0.1.26]: https://github.com/varunkarthic/kora/compare/v0.1.25...v0.1.26
[0.1.25]: https://github.com/varunkarthic/kora/compare/v0.1.24...v0.1.25
[0.1.24]: https://github.com/varunkarthic/kora/compare/v0.1.23...v0.1.24
[0.1.23]: https://github.com/varunkarthic/kora/compare/v0.1.22...v0.1.23
[0.1.22]: https://github.com/varunkarthic/kora/compare/v0.1.21...v0.1.22
[0.1.21]: https://github.com/varunkarthic/kora/compare/v0.1.20...v0.1.21
[0.1.20]: https://github.com/varunkarthic/kora/compare/v0.1.19...v0.1.20
[0.1.19]: https://github.com/varunkarthic/kora/compare/v0.1.18...v0.1.19
[0.1.18]: https://github.com/varunkarthic/kora/compare/v0.1.17...v0.1.18
[0.1.17]: https://github.com/varunkarthic/kora/compare/v0.1.16...v0.1.17
[0.1.16]: https://github.com/varunkarthic/kora/compare/v0.1.15...v0.1.16
[0.1.15]: https://github.com/varunkarthic/kora/compare/v0.1.14...v0.1.15
[0.1.14]: https://github.com/varunkarthic/kora/compare/v0.1.13...v0.1.14
[0.1.13]: https://github.com/varunkarthic/kora/compare/v0.1.12...v0.1.13
[0.1.12]: https://github.com/varunkarthic/kora/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/varunkarthic/kora/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/varunkarthic/kora/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/varunkarthic/kora/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/varunkarthic/kora/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/varunkarthic/kora/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/varunkarthic/kora/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/varunkarthic/kora/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/varunkarthic/kora/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/varunkarthic/kora/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/varunkarthic/kora/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/varunkarthic/kora/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/varunkarthic/kora/releases/tag/v0.1.0
