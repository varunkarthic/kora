# KORA

**A vendor-neutral C++20 runtime for Vulkan compute, from one GPU to several devices you can plug
in and out.**

KORA turns explicit Vulkan compute into buffers, kernels, and events. It keeps a host copy of every
buffer authoritative, so a dispatch can run on any local GPU or a remote `kora-worker`. A device can
disappear mid-stream without losing or corrupting results.

[Get started](getting-started.md) · [Platform support](platform-support.md) ·
[API reference](api/runtime.md) · [Releases](releases.md) ·
[GitHub](https://github.com/varunkarthic/kora)

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Get started
:link: getting-started
:link-type: doc

Build without a GPU, run the tests, and list your devices with `kora-info`.
:::

:::{grid-item-card} Multi-device and hot-swap
:link: cluster
:link-type: doc

Route dispatches across every local GPU and remote workers, split large ones, and survive unplugs.
:::

:::{grid-item-card} Watch it live
:link: tools/kora-top
:link-type: doc

`kora-top` shows detected devices, where work is going, memory, and failovers as they happen.
:::

:::{grid-item-card} Memory model
:link: memory-model
:link-type: doc

Host shadows, device replicas, staging, deferred readback, and when bytes actually move.
:::

:::{grid-item-card} Benchmarks
:link: benchmarking
:link-type: doc

Interleaved, reproducible measurements against raw Vulkan, Metal, OpenCL, and OpenGL.
:::

:::{grid-item-card} API reference
:link: api/runtime
:link-type: doc

Runtime, buffers, events, kernels, errors, and the throwing `checked` layer.
:::
::::

## Measured, not claimed

Host-observed latency of the same compute kernels on one Apple M4 through KORA, raw Vulkan, Metal,
OpenCL, and OpenGL. The chart is generated from the retained JSON reports by
`tools/kora-bench/render_competitors.py`. See [the comparison](benchmark-competitors.md) for the
method, the per-size tables, and where KORA is slower.

![KORA, raw Vulkan, Metal, OpenCL, and OpenGL host latency on Apple M4](../benchmarks/charts/competitors-host.svg)

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

## How to read performance results

Benchmark observations are specific to a named device, driver, OS, compiler, build, and workload.
CPU arithmetic, host submit/completion, and device timestamp measurements are separate domains;
they are not interchangeable measures of “GPU speed.” Software Vulkan is useful for correctness
coverage but is not a hardware-GPU result. The [benchmark guide](benchmarking.md) documents the
methodology and limitations; retained charts and tables are generated from machine-readable reports.

---

This handbook is published on [Read the Docs](https://kora.readthedocs.io/en/latest/) and maintained
alongside the [KORA source repository](https://github.com/varunkarthic/kora).

```{toctree}
:hidden:
:maxdepth: 1
:caption: Getting started

overview
platform-support
getting-started
building
testing
configuration
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Guides

device-selection
memory-model
task-graph
cpu-executor
multi-gpu
cluster
tools/kora-top
tools/kora-info
kernel-format
python-api
debugging
recovery
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Performance

benchmarking
benchmark-competitors
benchmark-compute-results
profiling
calibration
autotuning
scheduler
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: API reference

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
:hidden:
:maxdepth: 1
:caption: Internals

architecture
internals/invariants
synchronization
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
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project

releases
ci-cd
security
maintenance
```
