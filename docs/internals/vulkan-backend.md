# Vulkan backend dispatch

KORA loads the Vulkan loader at runtime instead of linking to a system loader or ICD. This keeps
library loading independent of GPU-driver installation; a missing or unusable loader is reported as
`ErrorCode::NoVulkanImplementation`.

## Dispatch ownership

`kora::vk::Loader` owns the dynamically loaded library and resolves
`vkGetInstanceProcAddr`. `Loader::device_api` uses the supplied live instance to resolve
`vkGetDeviceProcAddr`, then loads device entry points and returns an `Api` that retains the loader
and dispatch state for as long as the table is in use. The `Api` is an internal interface in
`src/vulkan/api.hpp`, not a public Vulkan-ABI guarantee.

`Loader::supported_api_version()` returns the maximum Vulkan API version reported by the loader,
or Vulkan 1.0 when `vkEnumerateInstanceVersion` is unavailable. Instance creation clamps its
requested version against that value. On macOS, loader discovery tries the standard loader name,
then `VULKAN_SDK` locations and common Homebrew library prefixes so an installed executable does
not depend on a build-tree runtime path.

The implemented table currently contains:

| Entry point | Vulkan signature | Loader used |
| --- | --- | --- |
| `allocate_memory` | `vkAllocateMemory` | `vkGetDeviceProcAddr` |
| `queue_submit` | `vkQueueSubmit` | `vkGetDeviceProcAddr` |

Callers must keep the associated `VkInstance` and `VkDevice` alive while using its table. A missing
required entry point returns `ErrorCode::UnsupportedFeature`; invalid/null handles are rejected
before resolution. Vulkan return values are passed through unchanged so callers can retain the
original `VkResult` in their structured error.

## Fault injection

`kora::vk::FaultInjector` decorates an `Api` table without changing production loader behavior.
`fail_on_nth` configures one one-shot failure for a supported function and call number. Calls before
and after the selected call are forwarded to the underlying table. The decorator and returned table
share their state, so the table remains valid if it outlives the `FaultInjector` object.

Configuration is intended to finish before calls begin; changing failure rules concurrently with
dispatch is not supported. The per-call counters are atomic so independently submitted calls can
advance the same rule safely.

## Invariants and scope

- Vulkan loader symbols are resolved at runtime; production code does not link against an ICD.
- Function pointers are resolved once per device and reused.
- The `Api` owns the storage required by its callbacks, but does not own the Vulkan device.
- Fault injection is opt-in and never changes the normal API result path.
- The table currently covers only memory allocation and queue submission. Other Vulkan calls must
  be added to this table before backend code uses them; this is not yet the full Vulkan backend.

Build and run the focused tests with:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug --output-on-failure -R VulkanFaultInjection
```
