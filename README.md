# KORA

KORA (Kernel Offload Runtime Architecture) is a vendor-neutral heterogeneous compute runtime for
explicit CPU and GPU workloads. Its GPU execution layer uses Vulkan and prebuilt SPIR-V compute
modules rather than CUDA, ROCm/HIP, oneAPI, OpenCL, or another vendor compute stack.

> **Project status:** early development. The build system and foundational C++ contracts are in
> place; Vulkan discovery, execution, scheduling, memory management, Python bindings, and production
> packages are being implemented incrementally. Beta releases are suitable for evaluation, not
> production workloads.

## Goals

- One C++20 runtime across Linux, Windows, and macOS through Vulkan/MoltenVK.
- Explicit buffers, kernels, events, task graphs, and reusable execution plans.
- Capability- and measurement-driven scheduling across CPU, integrated GPU, discrete GPU, and
  multiple GPUs without vendor-ID heuristics.
- Memory pooling, residency tracking, transfer overlap, hazard management, recovery, profiling, and
  persistent performance models above raw Vulkan.
- Thin Python bindings over the same C++ implementation.

## Current architecture

KORA is organized into focused runtime layers:

```text
Public C++ / Python APIs
          |
Task graph, scheduler, and execution plans
          |
CPU executor | Vulkan kernel and queue execution
          |
Memory, synchronization, profiling, and recovery
          |
Vulkan loader and platform implementation
```

Vulkan entry points are loaded at runtime. Building KORA therefore requires Vulkan headers but does
not require a Vulkan loader or GPU to be present. GPU operations will report a structured capability
error when no usable implementation is available.

## Requirements

- CMake 3.21 or newer
- Ninja
- A C++20 compiler
- Git and network access when CMake must download a pinned build dependency
- A Vulkan 1.1-capable implementation for GPU execution; MoltenVK is used on macOS

## Build and test

```sh
git clone https://github.com/varunkarthic/kora.git
cd kora
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug
```

Release and sanitizer configurations are also available:

```sh
cmake --preset release
cmake --build --preset release --parallel

cmake --preset asan
cmake --build --preset asan --parallel
ctest --preset asan
```

See [building](docs/building.md) and [testing](docs/testing.md) for every preset, option, dependency,
software Vulkan ICD configuration, installation, and package-consumer instructions.

## Install and consume from CMake

```sh
cmake --install build/release --prefix /path/to/kora-install
```

Then use the exported target:

```cmake
find_package(kora CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE kora::kora)
```

The public API is still evolving before the first stable release. The currently documented error
contract is available in [docs/api/errors.md](docs/api/errors.md).

## Releases

Every push to `main` is validated and produces a `MAJOR.MINOR.PATCH` release. `MAJOR.MINOR` comes
from [`VERSION`](VERSION); automation increments `PATCH` from the latest matching tag. Releases are
marked **Beta** while any release-validation issue in the matching milestone remains open and become
**Stable** when all tracked issues are closed. Release notes include installation and usage guidance,
the changelog, downloadable source/package artifacts, and the current validation table.

See [CHANGELOG.md](CHANGELOG.md) for user-visible changes once release automation is installed.

## Documentation

The documentation source lives in [`docs/`](docs/) and is structured for future Read the Docs
hosting. It covers setup, API contracts, architecture, testing, releases, security, and maintenance.

## Project policy

KORA is a solo-maintained personal open-source project. The repository does not accept external pull
requests or feature/support issues. Please use the published source and documentation as provided.
Security vulnerabilities are the exception: report them privately as described in
[SECURITY.md](SECURITY.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
