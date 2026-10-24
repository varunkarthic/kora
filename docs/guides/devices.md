# Choosing devices

## Which devices KORA sees

`kora-info` lists every physical device the Vulkan loader reports. A device is **eligible** when it
has a compute queue. Each selected device is identified by a `DeviceId`: its enumeration index plus
its 32-hex-digit UUID. Indices can change between runs; store UUIDs.

## Selection policies

| Policy | `KORA_DEVICE` | Selects |
| --- | --- | --- |
| `Auto` (default) | `auto` | The highest capability score: capability tier, then device-local memory, then compute queues. Device type earns no bonus. |
| `PreferDiscrete` | `discrete` | Discrete GPUs first, then the score; falls back when there are none. |
| `PreferIntegrated` | `integrated` | Integrated GPUs first, then the score. |
| `ByIndex` | `0`, `1`, … | Exactly that enumeration index. |
| `ByUuid` | 32 hex digits | Exactly that device. |
| `AllEligible` | `all` | Every eligible device. Devices that fail to initialize are quarantined, not fatal. |

Ties break by enumeration index, then UUID. An explicit index or UUID that is absent returns
`NoEligibleDevice`; a malformed UUID returns `InvalidArgument`.

=== "Rust"

    ```rust
    use kora::{DevicePolicy, Runtime, RuntimeOptions};

    let runtime = Runtime::new(&RuntimeOptions {
        device_policy: Some(DevicePolicy::AllEligible),
        ..RuntimeOptions::default()
    })?;
    for device in runtime.devices() {
        let info = runtime.device_info(&device)?;
        println!("{} {:?} {:?}", info.name, info.kind, info.memory_topology);
    }
    ```

=== "C"

    ```c
    KoraRuntimeOptions options;
    kora_runtime_options_default(&options);
    options.device_policy = KORA_DEVICE_POLICY_ALL_ELIGIBLE;
    kora_runtime_create(&options, &runtime);
    ```

=== "Python"

    ```python
    options = kora.RuntimeOptions()
    options.device_policy = kora.DevicePolicy.ALL_ELIGIBLE
    runtime = kora.Runtime.create(options)
    print(runtime.devices)
    ```

Explicit options win over `KORA_DEVICE`, which wins over the default.

## Capabilities, not vendors

`runtime.device_info(&id)` returns a `DeviceInfo`: name, kind, API version, driver, limits, features
(timeline semaphores, synchronization2, buffer device address, descriptor indexing, portability
subset, …), subgroup support, queue families, memory heaps and types, memory topology, and a
capability tier. KORA's own decisions use these fields and measurements; vendor and device IDs are
never consulted.

Optional features are enabled only when the device reports them. A kernel that needs a capability
the device lacks fails to load with `UnsupportedFeature` instead of failing at dispatch time.

## Using several devices

With `AllEligible`, kernels are still bound to one device: load a kernel per device with
`load_kernel_on`, or let KORA place work:

- [`dispatch_adaptive`](adaptive.md) chooses one device per dispatch from calibration and observed
  timings.
- [`Cluster`](cluster.md) routes, splits, and fails over across every local device and remote
  workers.

A buffer is shared by all devices of a runtime; each device that uses it gets its own replica.
