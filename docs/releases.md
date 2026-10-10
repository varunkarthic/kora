# Releases and packages

KORA releases use `MAJOR.MINOR.PATCH`. The repository `VERSION` file selects the release line and
automation increments the patch for every successfully validated push to `main`.

## Choosing a release

Use Stable releases when all published validation gates are closed. Use Beta releases only when you
can evaluate the remaining open gates against your workload. Neither channel replaces application-
specific benchmarking, driver qualification, or security review.

## Package contents

Platform archives contain an installed development prefix: static library, public headers, generated
version header, and CMake package files. They do not bundle a Vulkan driver, MoltenVK, Python runtime,
or system C/C++ runtime.

## Source build

```sh
git clone --branch v0.1.0 https://github.com/varunkarthic/kora.git
cd kora
cmake --preset release -DKORA_VERSION_PATCH=0
cmake --build --preset release --parallel
ctest --preset release --output-on-failure
```

Replace the example tag and patch with the selected release. See [CI and releases](ci-cd.md) for the
automation and failure-recovery model.
