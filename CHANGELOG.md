# Changelog

All notable changes to KORA are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/varunkarthic/kora/compare/v0.1.9...HEAD
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
