# KORA documentation

KORA is a C++20 runtime for explicit CPU and Vulkan compute. This handbook documents the current
contracts, practical setup, operating limits, and internal design. It distinguishes working features
from planned integrations: in particular, graph replay is caller-controlled and scheduler scoring is
not yet wired into automatic Runtime placement.

## Choose a path

```{mermaid}
flowchart TD
  START["What are you doing?"] --> BUILD["Trying KORA"]
  START --> USE["Using the Runtime"]
  START --> DEV["Developing or diagnosing"]
  START --> PERF["Measuring performance"]
  BUILD --> QUICK["Getting started → Building → Testing"]
  USE --> API["Runtime API → Kernel format → Configuration"]
  DEV --> DESIGN["Architecture → Invariants → Subsystem internals"]
  DEV --> DEBUG["Debugging → Recovery → Security"]
  PERF --> BENCH["Profiling → Benchmarking → Autotuning"]
```

## Get running

```{toctree}
:maxdepth: 1
:caption: Build and try

overview
platform-support
getting-started
building
testing
```

## Use KORA

```{toctree}
:maxdepth: 1
:caption: User guides

configuration
device-selection
memory-model
task-graph
cpu-executor
multi-gpu
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

## Design, diagnostics, and performance

```{toctree}
:maxdepth: 1
:caption: Design and operations

architecture
internals/invariants
synchronization
profiling
calibration
autotuning
scheduler
benchmarking
debugging
recovery
```

```{toctree}
:maxdepth: 1
:caption: Internal references

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

## Install, release, and maintain

```{toctree}
:maxdepth: 1
:caption: Project operations

ci-cd
releases
security
maintenance
```

## Tools and API reference

```{toctree}
:maxdepth: 1
:caption: Tools

tools/kora-info
```

The API reference pages above describe stable behavior only to the extent stated by each page; KORA
is pre-1.0 and its public API may evolve. Commands use the checked-in presets unless explicitly
labelled platform-specific. Benchmark values are observations from named local configurations, not
hardware-independent guarantees. See [release maturity](releases.md) and [support boundaries](platform-support.md).
