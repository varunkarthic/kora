# KORA

**K**ernel **O**ffload **R**untime **A**rchitecture is a C++20 runtime for explicit CPU and Vulkan
compute workloads. It gives applications building blocks for device-bound kernels, buffers,
transfers, events, task graphs, and diagnostics while keeping Vulkan capabilities—and their limits—
visible to the caller.

[![CI](https://github.com/varunkarthic/kora/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/varunkarthic/kora/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/varunkarthic/kora?include_prereleases&label=latest%20release)](https://github.com/varunkarthic/kora/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-Read%20the%20Docs-2f6f9f)](https://kora.readthedocs.io/en/latest/)

> **Maturity:** KORA is in active development. Releases are evaluation builds, not a production
> suitability guarantee. The v0.2.x line is tracked by the [`v0.2` milestone](https://github.com/varunkarthic/kora/milestone/20);
> GitHub release automation publishes patch versions and keeps the line Beta until its validation
> issues are closed.

## What KORA does

- Loads and validates prebuilt SPIR-V compute kernels and caches Vulkan pipelines.
- Manages host shadows, per-device buffer replicas, transfers, queue work, and completion events.
- Provides task-graph dependency analysis and reusable callback plans for caller-controlled
  orchestration.
- Offers CPU execution/fallback, multi-device primitives, profiling, trace export, recovery
  diagnostics, and a thin optional Python API.
- Uses Vulkan as a vendor-neutral GPU interface; it does not require CUDA, ROCm/HIP, oneAPI, or
  OpenCL.

KORA is explicit by default: a loaded kernel is bound to a Runtime device, and task graphs replay
caller-owned callbacks. Applications may opt into deterministic, calibration-gated placement for an
individual dispatch with `Runtime::dispatch_adaptive()`; it requires an explicit workload description
and `DevicePolicy::AllEligible` to consider more than one device. It does not automatically schedule
arbitrary graph work across CPUs and GPUs. See [what is implemented](docs/overview.md) and the
[platform support matrix](docs/platform-support.md) before choosing it for a workload.

## Architecture at a glance

```mermaid
flowchart LR
  APP["Your application"] --> API["C++ Runtime API"]
  PY["Optional Python API"] --> API
  APP -. "owns graph order and placement" .-> GRAPH["TaskGraph / ExecutionPlan"]
  API --> CPU["CPU executor / explicit fallback"]
  API --> DISPATCH["Device-bound dispatch"]
  API -->|"explicit opt-in"| ADAPT["Calibration-gated device selection"]
  ADAPT --> DISPATCH
  DISPATCH --> PIPE["SPIR-V validation / pipeline"]
  DISPATCH --> QUEUE["Queue / Event completion"]
  QUEUE --> GPU["Vulkan compute device"]
  API --> MEMORY["Buffers / transfers / replicas"]
  API --> OBS["Profiling / trace / diagnostics"]
```

### A dispatch lifecycle

```mermaid
sequenceDiagram
  participant App as Application
  participant K as KORA Runtime
  participant Q as Device queue
  participant D as Vulkan device
  App->>K: load validated SPIR-V + metadata
  App->>K: bind buffers, dimensions, push data
  App->>K: submit()
  K->>K: validate access and prepare residency/transfers
  K->>Q: enqueue command and return Event
  Q->>D: submit dispatch
  D-->>Q: complete timeline value
  Q-->>K: resolve Event and eligible readback
  K-->>App: Event becomes complete
```

The Event covers dispatch completion and, by default, writable-output readback. Deferred readback
keeps results device-resident until the caller requests them. See the [Runtime API guide](docs/api/runtime.md)
for lifetime, buffer, and error contracts.

## Quick start

Requirements: CMake 3.21+, Ninja, a C++20 compiler, and Git. A Vulkan loader/compute implementation
is needed only for GPU execution; KORA resolves the loader at runtime, so a CPU-only machine can
still build the library and run applicable tests.

```sh
git clone https://github.com/varunkarthic/kora.git
cd kora
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug --output-on-failure
```

For an optimized build and installable CMake package:

```sh
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release --output-on-failure
cmake --install build/release --prefix "$PWD/install"
```

Link an application to the exported target:

```cmake
find_package(kora CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE kora::kora)
```

The package and first-dispatch walkthrough are in [Getting started](docs/getting-started.md); all
options and build presets are in [Building](docs/building.md).

## Performance snapshot

These are captured local Release measurements, not portable promises. The comparison separates
single-thread CPU arithmetic from KORA-on-Vulkan and raw Vulkan host-submit/completion times. These
are not kernel-only GPU times; rerun on each target OS, compiler, driver, device, and workload. The
[benchmark guide](docs/benchmarking.md) includes the methodology and limitations.

These generated charts use separate panels for CPU arithmetic, host completion, and device
timestamps. Lines and dots show p50; whiskers extend to p95. Each panel uses a logarithmic latency
axis with units selected for its range. Missing device timestamps remain visibly unavailable rather
than being estimated. This single Apple M4/MoltenVK sample is not CUDA or cross-platform performance
evidence.

![CPU arithmetic benchmark chart](benchmarks/charts/cpu-arithmetic.svg)

![KORA and raw Vulkan host-latency chart](benchmarks/charts/host-latency.svg)

![KORA and raw Vulkan device-timestamp chart](benchmarks/charts/gpu-timestamps.svg)

| Elements | CPU median | KORA default | KORA deferred | Raw Vulkan | KORA GPU | Raw GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 2.084 µs | 4.404 ms | 526.291 µs | 402.500 µs | unavailable | 20.375 µs |
| 1048576 | 562.500 µs | 5.008 ms | 1.343 ms | 920.250 µs | unavailable | 388.042 µs |
| 4194240 | 2.312 ms | 5.627 ms | 1.031 ms | 1.859 ms | unavailable | 1.260 ms |

Benchmark sample: 100 measured iterations after 10 warm-ups on macOS arm64 / Apple M4, AppleClang 21
Release; GPU paths were pinned to device 2 (MoltenVK, Vulkan 1.3.357). KORA default includes output
readback, while deferred KORA and raw Vulkan keep outputs device-resident. CPU arithmetic, host
submit/completion, and GPU timestamp columns are different timing domains; KORA device timestamps
were unavailable on this run. The benchmarking guide also reports separate single-thread and
four-worker CPU baselines for SAXPY, reduction, stencil, and matrix multiplication. No CUDA comparison
is available from this non-CUDA device. Full workload definitions and caveats are in
[Benchmarking](docs/benchmarking.md).

The same device also runs matched raw-Vulkan compute fixtures for SAXPY, reduction, scan, tiled
matrix multiplication, and stencil. At 4,096 elements, device-time medians were within 3.5% of raw
Vulkan for the four KORA timestamp workloads that completed; host completion was about 2.1–3.3× the
raw path. This separates near-parity in shader execution from remaining host-side overhead. One KORA
stencil timestamp was unavailable, and a single M4/MoltenVK run is not cross-platform or CUDA evidence.
See the [matched compute methodology](docs/benchmarking.md#matched-raw-vulkan-compute-workloads),
[generated comparison tables](docs/benchmark-compute-results.md),
and [raw sample report](benchmarks/results/macos-arm64-apple-m4-moltenvk-compute-1000x50-ec51125.json).

The retained table is currently one macOS/MoltenVK machine sample. Linux, Windows, and macOS are
build/test targets, but CI smoke coverage is not a cross-OS performance study; no Linux/Windows
hardware-GPU or CUDA result is claimed. The benchmark guide explains how to distinguish CPU
arithmetic, software Vulkan, hardware-GPU timestamps, and KORA host overhead.

The subsequent #196 dispatch-path optimization was measured separately against raw Vulkan on the
same Apple M4/MoltenVK device. In 1,000-sample Release runs, 4K deferred-readback KORA changed from
676.5 µs to 524.0 µs median (raw Vulkan: 346.9 µs to 406.7 µs). The KORA/raw host-latency ratio
changed from 1.95× to 1.29×. These sequential local runs show host-load variation and do not establish
CUDA parity; a second post-change sample measured KORA at 522.2 µs and raw Vulkan at 465.5 µs.
Raw JSON, exact revisions, and the profiling breakdown are in
[the dispatch-overhead study](docs/benchmarking.md#dispatch-overhead-case-study).
An optional external CUDA report can be validated and rendered separately; no CUDA result is included
until a same-device NVIDIA run is available. The report contract and invocation are in the
[benchmark guide](docs/benchmarking.md#optional-external-cuda-reference).

The follow-up #214 optimization removed the idle coalescing collection delay while preserving
batching of compatible submissions already queued together. Across five interleaved 1,000-sample
Release pairs on the same M4/MoltenVK device, 4K deferred KORA host completion improved from a
547.9 µs median to 381.0 µs (−30.5%); median per-run p95 improved from 612.0 µs to 444.1 µs. The
matched raw Vulkan 4K median was 409.5 µs before and 408.0 µs after, putting post-change KORA at
0.93× raw host latency on this device. Deferred KORA medians also improved at 64K, 256K, 1M, and
4M elements; the 4M p50 improved 9.7% with no p95 regression. These are host-side timings, not
kernel-only GPU timings, and do not establish CUDA parity. The [full paired study](docs/benchmarking.md#issue-214-eager-submission-coalescing-and-completion-latency)
includes the methodology, per-size p50/p95 table, remaining bottlenecks, and all ten raw reports.

## Documentation

Start with [Overview](docs/overview.md), then choose a path:

| If you want to… | Start here |
| --- | --- |
| Build, install, and run tests | [Getting started](docs/getting-started.md) · [Building](docs/building.md) · [Testing](docs/testing.md) |
| Understand the boundaries and ownership model | [Architecture](docs/architecture.md) · [Invariants](docs/internals/invariants.md) |
| Select devices and manage buffers/transfers | [Device selection](docs/device-selection.md) · [Memory model](docs/memory-model.md) |
| Load kernels and dispatch work | [Kernel format](docs/kernel-format.md) · [Runtime API](docs/api/runtime.md) |
| Profile or compare performance | [Profiling](docs/profiling.md) · [Benchmarking](docs/benchmarking.md) |
| Diagnose failures or device loss | [Debugging](docs/debugging.md) · [Recovery](docs/recovery.md) |
| Use Python | [Python API](docs/python-api.md) |
| Check release status and installation | [Releases](docs/releases.md) · [Platform support](docs/platform-support.md) |

The full, searchable handbook is published on [Read the Docs](https://kora.readthedocs.io/en/latest/).

## Releases and compatibility

[`VERSION`](VERSION) contains `MAJOR.MINOR`; successful main-branch CI triggers release automation,
which increments the patch from existing tags (`0.2.0`, `0.2.1`, …). A release is Beta while any
issue in its matching `vMAJOR.MINOR` milestone is open, and Stable only after every issue in that
release line is closed. Stable describes completion of repository validation gates, not a guarantee
for every application or GPU. See [release policy](docs/releases.md) and [CHANGELOG](CHANGELOG.md).

## Project and security policy

KORA is a solo-maintained project and does not accept external feature/support issues or pull
requests. Please use published releases and documentation as provided. Report vulnerabilities
privately as described in [SECURITY.md](SECURITY.md).

## License

Apache-2.0; see [LICENSE](LICENSE).
