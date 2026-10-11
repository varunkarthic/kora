# KORA documentation

KORA is a vendor-neutral C++20 runtime for explicit heterogeneous CPU and Vulkan GPU workloads. It
turns prebuilt SPIR-V compute kernels, buffers, dependencies, and runtime policies into portable
execution without requiring CUDA, ROCm/HIP, oneAPI, or OpenCL.

KORA is under active development. The documentation distinguishes implemented behavior from planned
architecture. Interfaces described as planned are not yet available in the current source tree.

```{toctree}
:maxdepth: 2
:caption: Start here

overview
getting-started
building
testing
configuration
```

```{toctree}
:maxdepth: 2
:caption: Design and operation

architecture
debugging
internals/core
internals/checked-arithmetic
internals/vulkan-backend
internals/instance
internals/physical-device
internals/capabilities
internals/queues
internals/logical-device
internals/suballocator
internals/memory-pool
internals/command-pools
internals/queue-context
internals/completion
internals/staging-ring
internals/transfers
internals/dirty-ranges
internals/spirv
internals/pipelines
device-selection
memory-model
ci-cd
releases
security
maintenance
```

```{toctree}
:maxdepth: 2
:caption: Tools

tools/kora-info
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/errors
api/options
api/runtime
kernel-format
api/logging
api/version
api/checked
api/buffer
api/event
```

## Documentation guarantees

- Commands are written for the checked-in CMake presets.
- Required and optional dependencies are identified separately.
- Security, platform, and Vulkan limitations are stated explicitly.
- A source-level design is not presented as runtime-verified behavior.
- Release status reflects milestone gates: open gates mean Beta; all closed means Stable.
