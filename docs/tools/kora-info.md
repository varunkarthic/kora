# `kora-info`

`kora-info` reports the KORA build version, Vulkan loader and negotiated instance versions, and a
detailed capability report for every enumerated physical device.

```sh
cmake --preset debug
cmake --build --preset debug --parallel
./build/debug/tools/kora-info
./build/debug/tools/kora-info --json | python3 -m json.tool
```

The command uses KORA's runtime-loaded Vulkan path and does not link the Vulkan loader directly.
Human and JSON reports include queue-family classifications, memory heaps and types, optional heap
budget/usage, memory topology and benchmark-needed state, subgroup properties, synchronization and
buffer-address capabilities, descriptor indexing, portability-subset and MoltenVK detection.
`kora-info` does not initialize a KORA `Runtime`, so it cannot observe a runtime pipeline cache. Human
output states `Pipeline cache state: not queried`; JSON reports the same fact at the root as
`pipeline_cache_state: "not_queried"`. This is not evidence that caching is absent or uninitialized.

Reports deliberately omit device UUIDs, vendor IDs, device IDs, and other serial identifiers.

## JSON contract

The root `schema_version` is `2`. Root fields contain KORA, loader, and instance versions, plus
`pipeline_cache_state`, which is always `not_queried` because this device-only command does not create
a Runtime. Schema version 2 removes the misleading per-device `cache_state` field from version 1.
`devices`
contains one object per physical device with:

- public name, Vulkan API version, and driver name;
- `features` and `subgroup` capability objects;
- `queue_families`, including counts, flags, timestamp bits, and dedicated classifications;
- `memory.heaps` and `memory.types`, with budget fields only when `VK_EXT_memory_budget` exists;
- memory topology, portability, MoltenVK, and benchmark-needed fields.

The dependency-free writer escapes JSON strings and emits native JSON booleans and integers. Field
additions remain compatible within a schema version; incompatible changes require incrementing the
schema version.

## Data sources and invariants

| KORA API | Reported information |
| --- | --- |
| `kora::version_string()` | KORA semantic version |
| `Loader::supported_api_version()` | Loader-advertised maximum API version |
| `Instance::api_version()` | Negotiated instance API version |
| `Instance::physical_devices()` | Device, driver, queue, memory-budget, and feature queries |
| `DeviceCaps::from_vulkan()` | Vendor-neutral topology and capability classifications |

A loader, instance, or enumeration failure is printed to standard error and returns a nonzero exit.
Unknown arguments also fail and print `Usage: kora-info [--json]`. A missing Vulkan implementation
is never represented as a successful empty report.

CTest verifies the human report, missing-ICD failure, and JSON output. The JSON test invokes the
binary, parses standard output with Python `json.loads`, and validates required nested types and
collections. Vulkan-dependent checks skip only when no implementation exists; Linux CI installs
lavapipe and requires the report to execute.
