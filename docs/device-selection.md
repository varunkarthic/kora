# Device selection

KORA separates eligibility from preference. `select_devices` is a pure function over resolved
runtime options and synthetic `DeviceCandidate` records, so policy is deterministic and can be
tested without a Vulkan driver.

## Eligibility and identity

A candidate is eligible when it has at least one compute queue and satisfies the workload
requirements supplied by discovery. `DeviceId` records the current enumeration index and a
32-hex-character UUID. Indices are convenient but are not assumed stable across runs; persistent
configuration should use UUIDs.

Explicit index and UUID policies return `NoEligibleDevice` when the requested device is absent or
ineligible. A malformed UUID returns `InvalidArgument`.

## Policies

| Policy | Result |
| --- | --- |
| `Auto` | Highest capability score, without a device-type bonus |
| `PreferDiscrete` | Eligible discrete devices first, then the capability score; falls back when none exist |
| `PreferIntegrated` | Eligible integrated devices first, then the capability score; falls back when none exist |
| `ByIndex` | Exactly the eligible candidate with the requested current enumeration index |
| `ByUuid` | Exactly the eligible candidate with the requested case-insensitive UUID |
| `AllEligible` | Every eligible candidate in deterministic capability order |

The initial capability score orders capability tier, device-local memory, and compute-queue count.
Device kind is deliberately absent from `Auto`, so a discrete GPU is not selected merely because
it is discrete. Issue #20 will add measured startup calibration ahead of these static tie-breakers.
Equal scores use ascending enumeration index and then lowercase UUID.

## API reference

`select_devices(const ResolvedOptions&, const std::vector<DeviceCandidate>&)` returns selected
`DeviceId` values or a structured `Error`. `DeviceCandidate::satisfies_requirements` is the caller's
summary of kernel-specific SPIR-V, descriptor, push-constant, storage-buffer, and memory constraints;
selection never silently weakens those requirements.

Run the focused tests with:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug --output-on-failure -R DeviceSelection
```
