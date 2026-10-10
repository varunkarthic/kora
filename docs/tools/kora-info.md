# `kora-info`

`kora-info` reports the KORA build version, the maximum Vulkan API version supported by the
dynamically loaded Vulkan loader, the instance API version negotiated by KORA, and every physical
device returned by the Vulkan implementation.

Build and run it with:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
./build/debug/tools/kora-info
```

The command uses KORA's loader, instance, and physical-device query interfaces. It does not link
against the Vulkan loader or call Vulkan entry points directly. Device output includes the public
device name and Vulkan API version, and includes the Vulkan driver name when the driver-properties
query is available. It does not print device UUIDs or other serial identifiers.

## Data sources

| KORA API | Reported information |
| --- | --- |
| `kora::version_string()` | KORA semantic version |
| `kora::vk::Loader::load()` | Runtime Vulkan loader availability |
| `kora::vk::Loader::supported_api_version()` | Loader-advertised maximum API version |
| `kora::vk::Loader::create_instance()` and `Instance::api_version()` | Negotiated instance API version |
| `kora::vk::Instance::physical_devices()` | Device names, API versions, and available driver names |

If the Vulkan loader cannot be loaded, instance creation fails, or physical-device enumeration
fails, the command prints a diagnostic to standard error and exits non-zero. This makes it suitable
for scripting and health checks without treating a missing Vulkan implementation as an empty
successful result.

The CTest integration checks verify the successful output contains the device count and that a
missing ICD produces a non-zero exit with a Vulkan diagnostic.
