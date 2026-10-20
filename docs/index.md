# KORA documentation

<div align="center">

**A C++20 runtime for explicit CPU and Vulkan compute.**

Build, integrate, measure, and diagnose KORA with behavior-focused guides and API references.

[Get started](getting-started.md) · [Platform support](platform-support.md) · [API reference](api/runtime.md) · [Releases](releases.md)

</div>

```{mermaid}
flowchart LR
    A[Choose a goal] --> B{What do you need?}
    B -->|Try KORA| C[Install or build]
    B -->|Integrate| D[Select device and load kernel]
    B -->|Understand| E[Architecture and invariants]
    B -->|Measure| F[Profile and benchmark]
    C --> G[Run tests and inspect devices]
    D --> H[Manage buffers and dispatch]
    E --> I[Trace, debug, recover]
    F --> J[Compare timing domains]
```

## Start with your task

| Your goal | Recommended path | What you will get |
| --- | --- | --- |
| Build KORA and run a first check | [Getting started](getting-started.md) → [Building](building.md) → [Testing](testing.md) | A local build, test results, and a capability report from `kora-info` |
| Add KORA to a CMake application | [Getting started](getting-started.md) → [Runtime API](api/runtime.md) | `find_package(kora CONFIG)` integration and explicit dispatch contracts |
| Choose CPU or a Vulkan device | [Platform support](platform-support.md) → [Device selection](device-selection.md) → [CPU executor](cpu-executor.md) | Capability-aware choices and platform limitations |
| Load a kernel and move data | [Kernel format](kernel-format.md) → [Memory model](memory-model.md) → [Runtime API](api/runtime.md) | SPIR-V package requirements, buffer ownership, and execution flow |
| Understand system behavior | [Architecture](architecture.md) → [Invariants](internals/invariants.md) | Component boundaries, synchronization, and lifetime rules |
| Explain a performance result | [Profiling](profiling.md) → [Benchmarking](benchmarking.md) | Reproducible commands and clearly separated timing domains |
| Diagnose a failure or device loss | [Debugging](debugging.md) → [Recovery](recovery.md) | Structured errors, diagnostic evidence, and recovery boundaries |
| Check release maturity or install an archive | [Releases and packages](releases.md) | Beta/Stable meaning, platform assets, setup, and verification |

## Quick start

KORA can be built without a GPU. A working Vulkan implementation is needed only for Vulkan device
initialization and GPU integration checks. The first build downloads pinned development dependencies
when they are not already installed.

```sh
git clone https://github.com/varunkarthic/kora.git
cd kora
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug --output-on-failure
./build/debug/tools/kora-info
```

Expected outcome: CTest reports passing native tests (and may explicitly skip tests requiring an
unavailable Vulkan capability); `kora-info` reports detected Vulkan devices or a structured
availability error. A successful build alone does not establish GPU execution support. See
[Getting started](getting-started.md) for interpretation and installation steps.

:::{note}
KORA is pre-1.0. The documented API surface may evolve between minor versions. Pin a release tag in
production projects and review its release notes before upgrading.
:::

## Browse the handbook

```{toctree}
:maxdepth: 1
:caption: Build and try

overview
platform-support
getting-started
building
testing
```

```{toctree}
:maxdepth: 1
:caption: Integrate and use

configuration
device-selection
memory-model
task-graph
cpu-executor
multi-gpu
cluster
kernel-format
python-api
api/runtime
api/options
api/buffer
api/event
api/kernel
api/errors
api/verification
api/logging
api/version
api/checked
```

```{toctree}
:maxdepth: 1
:caption: Design, diagnostics, and performance

architecture
internals/invariants
synchronization
profiling
calibration
autotuning
scheduler
benchmarking
benchmark-compute-results
debugging
recovery
```

```{toctree}
:maxdepth: 1
:caption: Internals and operations

internals/core
internals/checked-arithmetic
internals/vulkan-backend
internals/instance
internals/physical-device
internals/capabilities
internals/queues
internals/logical-device
internals/suballocator
internals/memory-pool
internals/eviction
internals/command-pools
internals/queue-context
internals/completion
internals/staging-ring
internals/transfers
internals/dirty-ranges
internals/spirv
internals/pipelines
internals/aliasing
internals/performance-db
ci-cd
releases
security
maintenance
tools/kora-info
```

## How to read performance results

Benchmark observations are specific to a named device, driver, OS, compiler, build, and workload.
CPU arithmetic, host submit/completion, and device timestamp measurements are separate domains;
they are not interchangeable measures of “GPU speed.” Software Vulkan is useful for correctness
coverage but is not a hardware-GPU result. The [benchmark guide](benchmarking.md) documents the
methodology and limitations; retained charts and tables are generated from machine-readable reports.

---

This handbook is published on [Read the Docs](https://kora.readthedocs.io/en/latest/) and maintained
alongside the [KORA source repository](https://github.com/varunkarthic/kora).
