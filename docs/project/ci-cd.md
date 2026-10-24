# CI/CD

| Trigger | Jobs |
| --- | --- |
| Pull request | Linux (Ubuntu 24.04, lavapipe): changelog and branch-freshness checks, rustfmt, Clippy (`-D warnings`), all tests (the core crate again under synchronization validation), rustdoc, vendor-neutrality scan, `kora-info`/`kora-bench` JSON contracts, the C ABI install check, the Python wheel and tests, and `mkdocs build --strict` |
| Push to `main` | The same on Linux, plus build and test on macOS and Windows (hosted runners have no Vulkan driver, so device tests skip there) |
| Successful `main` CI | Release workflow: version and tag, per-platform archives (C ABI and tools) and abi3 wheels, release notes, Stable/Beta status |
| Issue changes in a release milestone | Re-render the validation table and status of affected releases |

Every pull request must add an entry under **Unreleased** in `CHANGELOG.md`; published sections are
immutable.

Read the Docs builds this site with `mkdocs build --strict` and publishes the rustdoc reference
alongside it.
