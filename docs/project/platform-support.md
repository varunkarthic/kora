# Platform support

KORA uses Vulkan as a vendor-neutral compute interface. What works on a machine is decided by what its
loader and driver report at run time (queue families, memory types, features, extensions, and limits),
never by the operating system or vendor name.

## Operating systems

| OS | Vulkan path | Tested |
| --- | --- | --- |
| Linux | System loader + GPU driver; Mesa lavapipe for CPU-only machines | CI on every pull request and push (lavapipe) |
| Windows | System loader + GPU driver | CI on every push to `main` (build and tests; skips without a driver) |
| macOS | MoltenVK or KosmicKrisp (Mesa) over Metal | CI on every push to `main`; benchmarks on Apple M4 |

### macOS

Two Vulkan drivers can expose the same Apple GPU:

| Driver | Notes |
| --- | --- |
| **MoltenVK** | Portability subset. Has GPU timestamps. Install with the Vulkan SDK or `brew install molten-vk vulkan-loader`. |
| **KosmicKrisp** | Mesa's Vulkan-on-Metal driver. Lower submission latency for small dispatches in our measurements; no GPU timestamps. |

Both appear in `kora-info`; choose one with `KORA_DEVICE=<index>`. KORA follows the portability subset
when a driver reports it.

## Capability tiers

| Tier | Meaning | Examples |
| --- | --- | --- |
| Baseline | Required for a device to run work | Vulkan 1.0+, a compute queue, storage buffers |
| Queried optional | Enabled only when reported | Timeline semaphores, synchronization2, buffer device address, descriptor indexing, memory budget, calibrated timestamps, device-fault diagnostics, maintenance4 |
| Workload-dependent | Checked per kernel and dispatch | Shader integer/float widths, workgroup size and count limits, storage buffer range |

A missing optional capability never breaks unrelated work: KORA falls back (fences instead of
timeline semaphores, legacy barriers instead of synchronization2, no autotuning without
maintenance4 or timestamps), or rejects only the kernel that needs it.

## Multiple devices

`DevicePolicy::AllEligible` initializes every eligible device and quarantines any that fail. Multiple
queues are not assumed to map to independent hardware engines. Transfers currently share the compute
queue.

## Not supported

KORA does not use or require CUDA, ROCm/HIP, oneAPI, or OpenCL, and does not bundle a Vulkan driver.
