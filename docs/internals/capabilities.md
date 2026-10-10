# Device capabilities

`kora::DeviceInfo` is an immutable, vendor-neutral snapshot built from the raw
`kora::vk::PhysicalDeviceInfo` query. The raw Vulkan record stays in the backend; the public record
uses standard C++ scalar values, arrays, booleans, and vectors so consumers do not need Vulkan
headers to inspect capabilities.

## Snapshot contents

The snapshot includes the public device name and identifiers, the device API version, compute
workgroup and storage/push-constant limits, timestamp period in nanoseconds, subgroup size and
operation flags, queue-family flags/counts, memory heaps and memory-type properties, and a focused
set of Vulkan feature booleans. All accessors are const-only and references returned for nested
records and vectors are read-only.

`DeviceCaps::from_vulkan()` performs the conversion. Promoted timeline-semaphore, buffer-device
address, descriptor-indexing, and synchronization2 features are read from the core 1.2/1.3 query
structures when those structures were queried; otherwise the corresponding extension query
structures are used. Unqueried capabilities remain false. Queue transfer support is reported for
graphics and compute queues as well as queues advertising the explicit transfer bit, following
Vulkan's implicit transfer-queue rules.

## Convenience tiers

Tiers are additive convenience profiles, not device eligibility decisions. A queried device always
receives at least Tier A; the runtime must still check each exact feature and limit needed by an
operation.

| Tier | Required capabilities |
| --- | --- |
| A | Any successfully queried Vulkan physical device; no optional feature is required |
| B | Tier A, timeline semaphores, and basic subgroup operations |
| C | Tier B and buffer device address |
| D | Tier C, runtime descriptor arrays, and synchronization2 |

Missing a capability for a higher tier only leaves the device in its best lower tier. No tier rule
uses `vendorID`, device name, or other vendor identity to classify hardware.

## Invariants and tests

- A `DeviceInfo` cannot be default-constructed or assigned after construction.
- `DeviceCaps` maps raw Vulkan structures without exposing Vulkan types in the public header.
- Tier requirements are monotonic and never reject a physical device for missing a higher-tier
  capability.
- Synthetic Vulkan property/feature records cover each tier and extension fallback.
- CI scans `src/` to reject conditional branches or direct comparisons on `vendorID`.

The raw query path and its Vulkan entry-point references are documented in
[physical-device discovery](./physical-device.md).
