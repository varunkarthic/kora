# Kernels and pipelines

## SPIR-V validation and reflection

`kernel::spirv` parses the module without external tools: header and instruction bounds, the
`GLCompute` entry point, execution modes (`LocalSize`, `LocalSizeId`), decorations, and the types of
every resource variable. It then checks the module against the manifest (names, sets, bindings,
kinds, access, push-constant layout, specialization constants) and against the device's capability
snapshot. A mutation fuzz test feeds it corrupted modules.

## PipelineManager

One per device. A pipeline is keyed by the module hash, its interface, and specialization values:

- the cache-hit path hashes the module and compares it byte-for-byte, without re-validating or
  copying it;
- compilation runs outside the manager's lock, and concurrent requests for one variant share a single
  pipeline;
- pipelines, layouts, and descriptor-set leases are reference counted, so they stay valid after LRU
  eviction or after the manager is dropped;
- descriptor pools grow by counting their sets rather than waiting for driver out-of-pool errors;
- the persistent `VkPipelineCache` is stored in the cache directory, bound to the device and driver
  identity, and rebuilt if corrupt.

## Autotuning

Kernels whose workgroup size comes from `LocalSizeId` specialization constants are tuned per device.
The autotuner proposes candidates within device limits, times them with GPU timestamps on real
dispatches, and then specializes every dispatch with the winner. It requires `maintenance4` and
timestamps; `KORA_DISABLE_AUTOTUNE=1` turns it off.
