# Install

## Requirements

| | Needed for | Notes |
| --- | --- | --- |
| Rust 1.88 or newer | building anything | The repository pins a toolchain in `rust-toolchain.toml`. |
| A Vulkan loader and compute driver | running work on a device | Not needed to build. Linux: your GPU driver or Mesa (`lavapipe` for CPU-only machines). Windows: your GPU driver. macOS: [MoltenVK or KosmicKrisp](../project/platform-support.md#macos). |
| A C11 compiler | the C API | Any compiler that can link a shared library. |
| Python 3.11 or newer | the Python API | One abi3 wheel covers every CPython ≥ 3.11. NumPy is optional. |

KORA loads the Vulkan loader at run time. Without one, creating a runtime fails with
`NoVulkanImplementation`; everything else (task graphs, the CPU executor, metadata parsing) works.

## Rust

```sh
cargo add kora --git https://github.com/varunkarthic/kora
```

The crate has no build-time native dependencies: `ash` loads Vulkan dynamically.

## C

Build `libkora` and use the committed header:

```sh
git clone https://github.com/varunkarthic/kora.git
cd kora
cargo build --release -p kora-capi
```

This produces `target/release/libkora.so` (Linux), `libkora.dylib` (macOS), or `kora.dll` (Windows),
plus a static `libkora.a`. The header is `crates/kora-capi/include/kora.h`. Link with `-lkora`; see
[the C API](../api/c.md). `tools/ci/check_capi.sh` performs the full build, install, compile, and run
cycle and is the quickest way to confirm a toolchain:

```sh
tools/ci/check_capi.sh
```

Expected result: `vector add of 4099 elements: 0 mismatches`, or `skipped: ...` without a Vulkan
device.

## Python

From a clone, build and install the wheel (the build backend is
[maturin](https://www.maturin.rs/)):

```sh
python3 -m pip install ".[numpy]"
python3 -c "import kora; print(kora.version())"
```

## Command-line tools

```sh
cargo install --git https://github.com/varunkarthic/kora kora-tools
```

This installs [`kora-info`](../tools/kora-info.md), [`kora-top`](../tools/kora-top.md),
[`kora-bench`](../tools/kora-bench.md), and [`kora-worker`](../tools/kora-worker.md).

## Check your devices

```sh
kora-info
```

Expected result: the KORA version, the loader and instance API versions, and one block per physical
device with its queues, memory heaps, and memory types. On a machine without Vulkan it exits non-zero
with `NoVulkanImplementation`. `kora-info --json` prints the same data as JSON (schema version 2).

!!! tip "No GPU? Use lavapipe"
    On Linux, `sudo apt-get install mesa-vulkan-drivers` installs Mesa's software Vulkan driver
    (`lavapipe`). It is the device KORA's CI tests against: good for correctness, not for performance
    numbers.
