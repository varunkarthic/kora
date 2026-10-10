# Physical-device discovery

`Instance::physical_devices()` enumerates every `VkPhysicalDevice` associated with a live
`kora::vk::Instance` and returns a `PhysicalDeviceInfo` snapshot for each one. The returned handles
are borrowed: keep the owning `Instance` alive while using them.

## Queried data

Each snapshot contains core properties, features, and memory properties; the complete device
extension-property list; and the versioned Vulkan 1.1, 1.2, and 1.3 property/feature structures
supported by both the physical device and the instance API version.

Promoted feature/property structures are queried through the Vulkan 1.1, 1.2, and 1.3 aggregate
structures when the effective API version supports them. Below those versions, KORA queries the
corresponding extension structures only when the device advertises the extension. Other optional
queries include subgroup-size control, memory budget and priority, calibrated timestamp domains,
device fault, portability subset, and robustness2. Calibrated timestamp domains are enumerated
through the KHR entry point or its EXT alias.

On Vulkan 1.0 instances, KORA resolves `vkGetPhysicalDeviceProperties2KHR`,
`vkGetPhysicalDeviceFeatures2KHR`, and `vkGetPhysicalDeviceMemoryProperties2KHR` when the instance
extension was enabled during creation. If properties2 is unavailable, it still returns the Vulkan
1.0 properties/features/memory data through the original entry points. Queried `pNext` links are
cleared before a record is returned, so no returned structure retains a pointer to temporary query
storage. A structure with zero `sType` was not queried because its core version or extension was
unavailable. KORA uses the Vulkan beta registry declarations for the portability-subset query.

The records are raw capability data, not an eligibility decision and not device-creation requests.
Later selection and logical-device setup can apply policy using the exact queried values.

| Entry point | Stage | Purpose |
| --- | --- | --- |
| `vkEnumeratePhysicalDevices` | Instance | Discover all physical devices |
| `vkEnumerateDeviceExtensionProperties` | Physical device | Determine which extension queries are valid |
| `vkGetPhysicalDeviceProperties2` / `KHR` | Physical device | Query core and extension properties via `pNext` |
| `vkGetPhysicalDeviceFeatures2` / `KHR` | Physical device | Query core and extension features via `pNext` |
| `vkGetPhysicalDeviceMemoryProperties2` / `KHR` | Physical device | Query memory heaps and optional budget data |
| `vkGetPhysicalDeviceCalibrateableTimeDomainsKHR` / `EXT` | Physical device | Enumerate calibrated timestamp domains |

The integration test requests validation when available, queries and prints each device's public
name, API version, and extension count, and fails if the validation callback reports an error.
Neither serial identifiers nor UUIDs are printed.

Run it with:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug --output-on-failure -R VulkanPhysicalDevice
```
