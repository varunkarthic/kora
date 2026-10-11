# Changelog

All notable changes to KORA are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/varunkarthic/kora/compare/v0.1.26...HEAD
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
