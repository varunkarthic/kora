# Platform and feature support

KORA uses Vulkan as a vendor-neutral compute interface. Support is determined from the loader,
physical-device properties, queue families, memory types, extensions, and feature bits reported at
runtime. A platform name or vendor identity is not a substitute for those capability queries.

## Capability tiers

| Tier | What it means | Examples |
| --- | --- | --- |
| Build-time | Available when the relevant project option and dependencies are enabled. | C++ library and tools; optional Python bindings; optional SPIR-V validation tooling. |
| Runtime baseline | Required by the configured KORA Vulkan backend and checked during initialization. | Vulkan compute device, usable queue, supported kernel interface, and required synchronization support. |
| Queried optional | Enabled only when the selected physical device reports the corresponding feature/extension. | Calibrated timestamps, memory budget, device-fault diagnostics, subgroup operations, buffer device address. |
| Workload-dependent | Requires kernel metadata, resource compatibility, and per-operation validation. | Shader scalar widths, descriptor/resource limits, dispatch dimensions, memory type masks. |

An optional feature may improve diagnostics or unlock a workload, but KORA does not silently require
it for unrelated kernels. Unsupported optional behavior returns a structured error or is omitted as
documented for that API. `kora-info` reports queried device information; see [Device selection](device-selection.md),
[Kernel format](kernel-format.md), and [Tool reference](tools/kora-info.md).

## Operating-system and portability notes

| Environment | Vulkan path | Practical limit |
| --- | --- | --- |
| Linux | System Vulkan loader and installed driver; CI also exercises Mesa lavapipe. | Driver/device features and queue topology vary. A software device is useful for correctness tests, not representative performance. |
| Windows | System Vulkan loader and installed driver. | The selected SDK/toolchain affects building; runtime capabilities still come from the driver. |
| macOS | Vulkan through MoltenVK over Metal. | The portability subset may expose fewer extensions, queues, memory types, timestamp domains, or shader capabilities than a native Vulkan implementation. |

MoltenVK is not a separate KORA backend. KORA follows the Vulkan capabilities it reports, including
portability-subset requirements. The presence of MoltenVK does not promise support for a specific
extension or performance behavior. Optional capabilities may be absent; code should query
`DeviceInfo` and handle the reported `Result` rather than infer support from the operating system.

KORA does not bundle a Vulkan driver or promise that multiple queues map to independent hardware
engines. Missing Vulkan compute support does not prevent building the library or using non-GPU
components, but GPU runtime creation will fail with a structured diagnostic. See [Building](building.md)
and [CI/CD](ci-cd.md) for build and verification requirements.

## Multiple devices

`DevicePolicy::AllEligible` can initialize more than one eligible device. Each has independent
queues, memory state, pipelines, and buffer replicas. Kernel dispatch remains bound to the device
selected when the kernel was created. Cross-device transfer uses explicit host staging; partitioning
and output merging require caller-defined task semantics. KORA does not claim transparent automatic
multi-GPU scheduling. See [Multiple devices](multi-gpu.md) for limits and examples.

## Numerical reproducibility

KORA does not promise bitwise-identical floating-point output across devices, drivers, or shader
implementations. Arithmetic contraction, instruction selection, supported precision, and operation
ordering can differ; applications requiring reproducibility should specify their own kernel
constraints and test the actual targets. Integer operations are not made numerically equivalent by
the runtime either: overflow and representation semantics belong to the kernel and API contract.

For duplicate execution, `VerificationPolicy` provides exact numeric equality, absolute/relative
tolerance, ULP distance for IEEE-754 float/double, or a caller comparator. A tolerance is an
application-selected acceptance rule, not an accuracy guarantee supplied by KORA. A mismatch
returns `ValidationError` and quarantines all participating devices because the runtime cannot know
which result is authoritative. See [Duplicate-result comparators](api/verification.md).

## Related references

- [Configuration and environment variables](configuration.md)
- [Memory model and budget](memory-model.md)
- [Synchronization](synchronization.md)
- [Profiling and trace limitations](profiling.md)
- [Recovery and data-loss semantics](recovery.md)
- [Benchmark methodology and interpretation](benchmarking.md)
