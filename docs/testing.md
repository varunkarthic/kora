# Testing KORA

KORA uses GoogleTest for native tests and CTest as the common runner. Tests are grouped by runtime
cost and purpose so developers and CI can select the smallest useful suite.

## Running tests

Configure, build, and run the default Debug suite with:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug
```

Run only one category with its CTest label:

```sh
ctest --preset debug -L unit
ctest --preset debug -L integration
ctest --preset debug -L stress
ctest --preset debug -L fault_injection
```

The `asan`, `ubsan`, and `tsan` presets instrument both KORA and its tests. Build a sanitizer preset
before invoking its matching CTest preset.

## Test categories

| Directory | Label | Purpose |
| --- | --- | --- |
| `tests/unit` | `unit` | Deterministic tests of isolated components |
| `tests/integration` | `integration` | Multiple subsystems, Vulkan loader, and device behavior |
| `tests/stress` | `stress` | Repetition, memory pressure, and concurrency endurance |
| `tests/fault_injection` | `fault_injection` | Injected backend failures and recovery paths |

Tests must not silently move between categories. A test belongs in the narrowest category that
matches its dependencies and expected runtime.

## Software Vulkan ICD

Set `KORA_TEST_VULKAN_ICD` to an ICD JSON file when Vulkan tests should use a software
implementation such as Mesa lavapipe:

```sh
cmake --preset debug -DKORA_TEST_VULKAN_ICD=/path/to/lvp_icd.json
cmake --build --preset debug --parallel
ctest --preset debug -L integration
```

The `kora_add_test` helper applies that value as `VK_ICD_FILENAMES` only to targets declared with
`USES_VULKAN`. Native unit tests and unrelated tools do not inherit it. An unset value preserves
the process environment and lets the Vulkan loader perform normal discovery.

## Tests on systems without Vulkan

A missing Vulkan loader or eligible compute device is an expected environment condition. Tests
that specifically require Vulkan must call `KORA_REQUIRE_VULKAN(condition)` after discovery. A
false condition reports a GoogleTest skip and returns from the test body; it must not be converted
into a pass, crash, or assertion failure.

The integration meta-test deliberately passes `false` to this macro. CTest must report the test as
passed overall while GoogleTest records it as skipped, proving that Vulkan-dependent suites remain
portable to CPU-only build hosts.

## CMake test API

`kora_add_test(executable label [USES_VULKAN] SOURCES source...)` creates one GoogleTest executable,
links `kora::kora` and `GTest::gtest_main`, applies KORA's warnings and sanitizer settings, and uses
`gtest_discover_tests` to register each test case with CTest. `SOURCES` is required; unknown or empty
arguments fail during configuration so misspelled test declarations cannot disappear silently.

`KORA_REQUIRE_VULKAN(condition)` is a test-only macro in `tests/support/vulkan_test.hpp`. Call it
from the test body, not from a helper function, because `GTEST_SKIP` returns from the function in
which it expands.

## Invariants

- Every test is discoverable through CTest and carries exactly one category label.
- Unit tests do not require a GPU, Vulkan loader, network access, or mutable global machine state.
- Vulkan-dependent tests skip only for missing environmental capability, never for product defects.
- A configured software ICD is scoped to tests marked `USES_VULKAN`.
- Tests are repeatable and leave no persistent cache or device state behind.
- Third-party GoogleTest targets do not inherit KORA's warning-as-policy settings.
