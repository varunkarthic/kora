# Vulkan instance lifecycle

`kora::vk::Loader` dynamically loads the Vulkan loader and creates an owned `kora::vk::Instance`.
The instance wrapper keeps the loader loaded, destroys its optional debug messenger before the
`VkInstance`, and does not expose the handle's ownership to callers.

## Version and extension negotiation

The default requested API version is Vulkan 1.3. The loader queries
`vkEnumerateInstanceVersion` when available (older loaders are treated as Vulkan 1.0) and requests
the minimum of the caller's requested version, the loader's supported version, and Vulkan 1.3.
This records the supported version without requiring newer APIs merely to create the instance.

Instance extensions are enumerated before creation. When present, KORA enables
`VK_KHR_portability_enumeration` together with its required instance-create flag, so portability
implementations such as MoltenVK can enumerate devices. If the selected API version is Vulkan 1.0,
`VK_KHR_get_physical_device_properties2` is enabled when available for the Vulkan 1.1-style
physical-device query path.

| Entry point | Stage | Purpose |
| --- | --- | --- |
| `vkGetInstanceProcAddr` | Loader/global | Resolve all Vulkan entry points without a static loader link |
| `vkEnumerateInstanceVersion` | Loader/global | Determine the maximum instance API version (optional on 1.0 loaders) |
| `vkEnumerateInstanceExtensionProperties` | Loader/global | Detect portability, properties2, and debug-utils extensions |
| `vkEnumerateInstanceLayerProperties` | Loader/global | Detect optional validation-layer availability |
| `vkCreateInstance` | Loader/global | Create an instance with negotiated version, layers, extensions, and flags |
| `vkEnumeratePhysicalDevices` | Instance | Reject implementations that expose no physical devices |
| `vkCreateDebugUtilsMessengerEXT` | Instance | Install optional validation diagnostics |
| `vkDestroyDebugUtilsMessengerEXT` | Instance | Release the optional validation callback |
| `vkDestroyInstance` | Instance | Destroy the owned instance |

Every enumeration retries when Vulkan returns `VK_INCOMPLETE`, rather than trusting a stale count.
Loader, instance, extension enumeration, and zero-device failures return
`ErrorCode::NoVulkanImplementation`, retaining the `VkResult` when available.

## Validation diagnostics

Set `Loader::InstanceOptions::enable_validation` to request
`VK_LAYER_KHRONOS_validation`. Instance creation enables that layer only when it is installed; if
absent, it logs a warning and continues without validation. When both the layer and
`VK_EXT_debug_utils` are available, KORA installs a debug messenger and routes warning/error
messages to the `Device` log category. Validation is diagnostic support, not a security boundary.

## Lifecycle checks

The integration suite exercises 100 create/destroy cycles, version clamping, and the missing-ICD
failure path. Run it with:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug --output-on-failure -R VulkanInstance
```
