# Releases

## Versions

`VERSION` holds `MAJOR.MINOR`. Every push to `main` that passes CI publishes
`vMAJOR.MINOR.PATCH`, where the patch is one more than the latest tag in that line.

| Status | Meaning |
| --- | --- |
| **Stable** | Every release-validation issue in the line's milestone (`v1.0` for 1.0.x) is closed. |
| **Beta** | At least one validation issue is open. Published as a pre-release. |

Stable means the project's validation checklist is closed, not a warranty for every GPU, driver, or
workload. Qualify the hardware and workload you ship. A workflow re-renders the status table on each
release page when a tracked issue changes.

The release workflow starts when CI succeeds on `main`. GitHub runs `workflow_run` workflows from
the repository's **default branch**, so the default branch must carry the current
`.github/workflows/release.yml`. A copy that differs from `main` packages with stale steps.

## What a release contains

| Asset | Contents |
| --- | --- |
| `kora-<version>-linux-x86_64.tar.gz` | `include/kora.h`, `lib/libkora.so` and `libkora.a`, `bin/` tools |
| `kora-<version>-macos-arm64.tar.gz` | `include/kora.h`, `lib/libkora.dylib` and `libkora.a`, `bin/` tools |
| `kora-<version>-windows-x86_64.zip` | `include/kora.h`, `lib/kora.dll`, `kora.dll.lib`, and `kora.lib`, `bin/` tools |
| `kora_runtime-<version>-cp311-abi3-*.whl` | The Python package for each platform |

The release notes include usage instructions, the changelog section, and the validation table.
Archives do not bundle a Vulkan loader or driver.

## Build a release from source

```sh
git clone --branch v1.0.0 https://github.com/varunkarthic/kora.git
cd kora
KORA_VERSION_PATCH=0 cargo build --release --workspace
```

`KORA_VERSION_PATCH` sets the patch component that `kora::version::version_string()` reports.

See the [changelog](https://github.com/varunkarthic/kora/blob/main/CHANGELOG.md) for every change.
