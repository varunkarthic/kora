# Changelog

All notable changes to KORA are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.41] - 2026-10-11

- Add a fixed-size CPU executor with priority lanes, work stealing, graceful draining, and
  exception-aware Events.

## [Unreleased]

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

[Unreleased]: https://github.com/varunkarthic/kora/compare/v0.1.53...HEAD
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
