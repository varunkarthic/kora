# Building KORA

KORA uses CMake 3.21 or newer and C++20. Ninja is the default generator for all checked-in
presets so local builds and CI exercise the same build graph.

## Prerequisites

- A C++20 compiler (AppleClang, Clang, GCC, or MSVC)
- CMake 3.21 or newer
- Ninja
- Git and network access when a build dependency is not installed locally

The Vulkan loader is deliberately not a link-time dependency. KORA loads it at runtime, which
allows the library and CPU functionality to work on machines without a Vulkan implementation.
Vulkan headers are still required to compile the GPU backend. CMake first looks for installed
headers and otherwise downloads the pinned Khronos Vulkan-Headers release.

SPIRV-Headers are required for the parser's Khronos instruction and enum definitions; CMake uses an
installed package or downloads the pinned Vulkan-SDK-aligned release. SPIRV-Tools is optional; when
found, it supplements KORA's built-in structural SPIR-V checks. GoogleTest is required only when
`KORA_BUILD_TESTS` is enabled. CMake uses an installed package when available and otherwise
downloads the pinned release.

## Quick start

```sh
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug
```

Build trees are written to `build/<preset>`. Generated files and downloaded dependencies stay
inside the selected build tree.

## Presets

| Preset | Build type | Purpose |
| --- | --- | --- |
| `debug` | Debug | Ordinary development with assertions and debug information |
| `release` | Release | Optimized build |
| `asan` | Debug | AddressSanitizer instrumentation |
| `ubsan` | Debug | UndefinedBehaviorSanitizer instrumentation |
| `tsan` | Debug | ThreadSanitizer instrumentation |
| `tests` | Debug | Explicit test-enabled build |
| `benchmarks` | Release | Enables benchmark targets as they are added |
| `python` | Release | Enables Python binding targets as they are added |

Sanitizer presets require compiler and platform support. Each sanitizer is isolated in its own
build tree because AddressSanitizer and ThreadSanitizer cannot be combined safely.

To configure every preset, or configure and build every preset, run:

```sh
tools/ci/check_presets.sh
tools/ci/check_presets.sh --build
tools/ci/check_install.sh
```

`check_install.sh` additionally installs the Release build into a temporary prefix, configures a
standalone consumer through `find_package(kora CONFIG)`, links it, and runs it.

## Build options

| Option | Default | Meaning |
| --- | --- | --- |
| `KORA_BUILD_TESTS` | `ON` | Build unit and integration tests |
| `KORA_BUILD_TOOLS` | `ON` | Build command-line tools |
| `KORA_BUILD_BENCHMARKS` | `OFF` | Build benchmarks |
| `KORA_BUILD_PYTHON` | `OFF` | Build Python bindings |
| `KORA_ENABLE_SPIRV_TOOLS` | `ON` | Use SPIRV-Tools validation when the package is available |
| `KORA_SANITIZE_ADDRESS` | `OFF` | Enable AddressSanitizer |
| `KORA_SANITIZE_UNDEFINED` | `OFF` | Enable UndefinedBehaviorSanitizer |
| `KORA_SANITIZE_THREAD` | `OFF` | Enable ThreadSanitizer |

Override an option at configure time when a preset is not sufficient:

```sh
cmake --preset release -DKORA_BUILD_TESTS=OFF -DKORA_BUILD_TOOLS=OFF
cmake --build --preset release --parallel
```

## CMake targets and helper functions

`kora` is the implementation target. In both the build tree and installed packages, consumers
should link the namespaced `kora::kora` target:

```cmake
find_package(kora CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE kora::kora)
```

`kora_configure_target(target)` is defined in `cmake/KoraWarnings.cmake`. It applies the project's
warning policy to KORA-owned targets and adds the sanitizer compile and link flags selected by the
active preset. Third-party targets are intentionally excluded from this warning policy.

`kora_add_tests(executable label sources...)` is defined in `tests/CMakeLists.txt`. It creates a
GoogleTest executable, links it to `kora::kora`, applies KORA's warning policy, and registers its
individual tests with CTest under the supplied label.

## Build invariants

- All KORA targets compile as C++20 with compiler extensions disabled.
- KORA never links a vendor compute runtime such as CUDA, HIP, oneAPI, or OpenCL.
- Vulkan entry points are resolved at runtime; a missing loader is a runtime error, not a process
  load failure.
- Optional dependencies may improve validation or tooling but may not become correctness
  requirements for the core runtime.
- KORA-owned code receives the same warning policy in library, tool, and test targets.
- Presets use independent build directories so instrumentation and generated files cannot leak
  between configurations.
- `VERSION` contains only `MAJOR.MINOR`; release automation supplies `KORA_VERSION_PATCH`.

## Installation

Install from a configured build tree with an explicit prefix:

```sh
cmake --install build/release --prefix /path/to/kora-install
```

This installs the library, public headers, generated version header, and CMake package files. A
consumer can then set `CMAKE_PREFIX_PATH=/path/to/kora-install` and use `find_package(kora CONFIG)`.
