# Queue-family discovery and classification

Each raw physical-device query includes the ordered Vulkan queue-family properties. Internal
`DeviceCaps::classify_queue_families()` converts those properties into the vendor-neutral
`DeviceInfo::QueueFamily` records consumed by later scheduling and device-creation work.

## Classification rules

- `compute` is true when `VK_QUEUE_COMPUTE_BIT` is present.
- `dedicated_compute` is true when the family supports compute but not graphics.
- `transfer` is true when the transfer bit is explicit or when graphics or compute support implies
  transfer operations under Vulkan queue-family rules.
- `dedicated_transfer` is true when transfer is supported and neither graphics nor compute is
  supported.
- Queue count, timestamp-valid bits, sparse-binding support, and minimum image-transfer granularity
  are copied from the raw family properties.

The classifier preserves the input family order and does not discard or reject a family. These are
capability facts, not vendor-specific labels or a queue-selection decision. In particular, buffer
transfer suitability must not be inferred from `min_image_transfer_granularity`: that limit applies
to image transfers. For buffer transfers, use transfer capability and available queue count.

## Function reference

| Function | Purpose |
| --- | --- |
| `DeviceCaps::classify_queue_families()` | Purely maps a synthetic or queried array of `VkQueueFamilyProperties` to public queue-family records |
| `DeviceCaps::from_vulkan()` | Includes classified queue families in the immutable `DeviceInfo` snapshot |

Unit tests exercise separate compute/transfer families, shared graphics/compute families, and a
single-family layout. All use only queue flags and reported properties; no vendor identifiers are
consulted.
