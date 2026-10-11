# Overview

## What KORA provides

Raw Vulkan exposes devices, queues, memory types, command buffers, pipelines, and synchronization.
KORA provides a C++20 runtime and an optional thin Python binding for compute applications. The
implemented services include:

- logical buffers with host and device residency;
- validated SPIR-V kernel packages and pipeline caching;
- asynchronous events and dependency-aware task-graph primitives;
- buffer hazard tracking and supported Vulkan barrier lowering;
- pooled allocation, staging, eviction, and memory-budget handling;
- CPU task execution and explicit multi-device operations, with scheduler/partitioning primitives;
- profiling, traces, recovery, and diagnostic tools;
- a C++ API and an optional Python binding for its documented subset.

Automatic graph execution and scheduler-driven device placement are not implied by those interfaces:
task graphs currently build dependency plans for caller callbacks, and kernels are bound to a device
when created. Consult [Architecture](architecture.md) and [Multiple devices](multi-gpu.md) for those
boundaries. This documentation distinguishes available runtime behavior from optional capabilities
and unimplemented integration work.

## Portability model

The GPU backend targets Vulkan 1.1 as its broad baseline and prefers newer core features or extension
equivalents when available. Optional features may improve performance but cannot become hidden
correctness requirements.

Linux and Windows use the installed Vulkan loader and driver. macOS uses MoltenVK over Metal, so its
capability set may be a portability subset. A machine without a usable Vulkan compute implementation
can still build KORA and use non-GPU components; GPU initialization reports a structured error.
See [Platform and feature support](platform-support.md) for the capability tiers, portability limits,
and numerical reproducibility expectations.

## Non-goals

KORA does not compile a general programming language into SPIR-V, emulate CUDA APIs, or hide data
dependencies from the caller. It does not guarantee that multiple Vulkan queues map to independent
hardware engines. It does not use vendor IDs as scheduling policy.

## Maturity and release channels

Beta releases have one or more open validation issues and are intended for evaluation. Stable means
only that all release-line validation gates are closed; applications must still evaluate hardware,
driver, workload, and security suitability for their own environment.
