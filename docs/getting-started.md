# Getting started

```{mermaid}
flowchart LR
  CLONE["Clone source"] --> BUILD["Configure and build"]
  BUILD --> TEST["Run CTest"]
  TEST --> INFO["Inspect Vulkan devices"]
  INFO --> PACKAGE["Install / find_package"]
  PACKAGE --> APP["Load SPIR-V and dispatch"]
  INFO -. "no Vulkan?" .-> CPU["Build still works; GPU calls return a structured error"]
```

KORA's native build does not require a GPU. GPU examples and integration tests require a working
Vulkan implementation; use the platform notes to understand MoltenVK and software Vulkan limits.

## Clone and build

```sh
git clone https://github.com/varunkarthic/kora.git
cd kora
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug --output-on-failure
```

A successful test run currently reports the native unit tests as passed and may report the deliberate
no-Vulkan capability test as skipped. A skip means the environment lacks a required capability; it
does not prove GPU execution works.

After building, inspect what the local loader exposes:

```sh
./build/debug/tools/kora-info
./build/debug/tools/kora-info --json | python3 -m json.tool
```

`kora-info` exits nonzero when Vulkan loader/device enumeration is unavailable. Its report is a
capability snapshot, not a GPU execution test. `tools/ci/run_device_matrix.py` and GPU integration
tests provide stronger execution evidence when the corresponding devices exist.

## Build an optimized installation

```sh
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release --output-on-failure
cmake --install build/release --prefix "$PWD/install"
```

The installation contains the static library, public headers, generated version header, and CMake
package files.

## Consume KORA

```cmake
cmake_minimum_required(VERSION 3.21)
project(example LANGUAGES CXX)

find_package(kora CONFIG REQUIRED)
add_executable(example main.cpp)
target_link_libraries(example PRIVATE kora::kora)
```

Configure the consumer with the installation prefix:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/absolute/path/to/kora/install
cmake --build build
```

## Verify the package contract

KORA includes a standalone downstream consumer check:

```sh
tools/ci/check_install.sh
```

It builds Release, installs to a temporary prefix, discovers KORA with `find_package`, links an
independent executable, runs it, and removes the temporary files.

## Next steps

- Read [Building KORA](building.md) for presets and dependency behavior.
- Read [Configuration](configuration.md) before setting environment variables.
- Read [Errors and results](api/errors.md) before calling fallible APIs.
- Read [Testing KORA](testing.md) before interpreting device-dependent results.
- Read [Kernel format](kernel-format.md) and [Runtime API](api/runtime.md) before loading and dispatching
  a kernel; metadata, buffer ownership, and device binding are explicit.
