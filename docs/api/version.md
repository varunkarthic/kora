# Version API

Header: generated `kora/version.hpp`

`KORA_VERSION_MAJOR`, `KORA_VERSION_MINOR`, and `KORA_VERSION_PATCH` are compile-time integer macros.
`kora::version_string()` returns the same version as `major.minor.patch`.

The repository [`VERSION`](../../VERSION) file stores only `MAJOR.MINOR`. Local builds default patch
to zero unless `-DKORA_VERSION_PATCH=N` is provided. Release automation derives and injects the next
patch number from Git tags.

`kora::kSchemaVersion` versions persisted calibration, performance, tuning, and pipeline-cache data.
It is independent of the product version and changes only when a persisted format becomes
incompatible.

## Invariants

- Public packages use exactly three numeric semantic-version components.
- A published tag and compiled version must match.
- Moving or reusing a published tag is prohibited.
- Stale schema data is discarded safely rather than parsed as the current format.
