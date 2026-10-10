# Logical-device creation

`kora::vk::Instance::create_device` creates an owned Vulkan logical device from a previously queried
`PhysicalDeviceInfo`. The returned `Device` keeps its instance and loader alive, destroys the
`VkDevice` through the loaded dispatch path, and exposes the compute and transfer queues selected at
creation time.

## Queue policy

KORA requires one compute-capable queue family. It selects the first reported family with a nonzero
queue count and `VK_QUEUE_COMPUTE_BIT`. When `DeviceOptions::enable_dedicated_transfer_queue` is
true, a transfer-only family is selected when one exists; otherwise the compute queue is also the
transfer queue. Only one queue is requested from each selected family.

Selection is based on queried queue flags and counts, never a vendor or device identifier. A device
without a compute queue returns `ErrorCode::NoComputeQueue`.

## Feature and extension negotiation

KORA enables only features that were reported by the physical-device query and that the current
backend can use:

- timeline semaphores;
- buffer device address;
- descriptor indexing, including the queried runtime-array and binding modes;
- synchronization2; and
- robust buffer access from the core feature set.

Promoted features use the Vulkan 1.2 or 1.3 feature structures. On older API versions, the matching
KHR or EXT structure and advertised extension name are used. `VK_KHR_portability_subset` is enabled
whenever it is advertised, which is required by portability implementations such as MoltenVK. An
advertised flag is not assumed to be available unless it was present in the query snapshot.

## Lifecycle and API reference

| API | Behavior |
| --- | --- |
| `Instance::create_device(info, options)` | Validates queue availability, negotiates supported features/extensions, creates queues, and returns `Result<Device>` |
| `Device::handle()` | Borrowed `VkDevice`; ownership remains with the wrapper |
| `Device::compute_queue()` | Borrowed queue from `compute_queue_family()` |
| `Device::transfer_queue()` | Dedicated transfer queue when available and requested, otherwise the compute queue |
| `Device::enabled_features()` | Exact optional backend features requested at device creation |
| `Device::enabled_extensions()` | Exact advertised extension names enabled at device creation |
| `Device::api()` | Device-level allocation and submission dispatch table |

The device is move-only. Its queues and handle become invalid when it is destroyed. The instance is
retained internally so Vulkan's required instance-before-device lifetime ordering is preserved.

## Verification

The integration test creates and destroys a logical device with validation requested on every local
compute-capable physical device:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug --output-on-failure -R VulkanPhysicalDevice.CreatesLogicalDevice
```
