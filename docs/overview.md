# Overview

## What KORA provides

Raw Vulkan exposes devices, queues, memory types, command buffers, pipelines, and synchronization.
KORA is intended to add the runtime services normally needed by compute applications:

- logical buffers with host and device residency;
- validated SPIR-V kernel packages and pipeline caching;
- asynchronous events and dependency-aware task graphs;
- hazard tracking and portable synchronization lowering;
- pooled allocation, staging, eviction, and memory-budget handling;
- measured CPU/GPU scheduling and multi-device placement;
- profiling, traces, recovery, and diagnostic tools;
- equivalent C++ and thin Python interfaces.

The current repository implements the build/package foundation, structured errors, checked
arithmetic, configuration resolution, structured logging, and categorized tests. Later subsystems
remain tracked as GitHub issues and must not be assumed available yet.

## Portability model

The GPU backend targets Vulkan 1.1 as its broad baseline and prefers newer core features or extension
equivalents when available. Optional features may improve performance but cannot become hidden
correctness requirements.

Linux and Windows use the installed Vulkan loader and driver. macOS uses MoltenVK over Metal, so its
capability set may be a portability subset. A machine without a usable Vulkan compute implementation
can still build KORA and use non-GPU components; GPU initialization reports a structured error.

## Non-goals

KORA does not compile a general programming language into SPIR-V, emulate CUDA APIs, or hide data
dependencies from the caller. It does not guarantee that multiple Vulkan queues map to independent
hardware engines. It does not use vendor IDs as scheduling policy.

## Maturity and release channels

Beta releases have one or more open validation issues and are intended for evaluation. Stable means
only that all release-line validation gates are closed; applications must still evaluate hardware,
driver, workload, and security suitability for their own environment.
