# Guides

Task-oriented guides. Each shows Rust first, with C and Python where the API differs.

| If you want to… | Read |
| --- | --- |
| Bring your own shader | [Kernels and manifests](kernels.md) |
| Understand when bytes move and what a buffer holds | [Buffers and memory](memory.md) |
| Chain dispatches, wait, and keep results on the device | [Dispatch and synchronization](dispatch.md) |
| Pick a GPU, or use all of them | [Choosing devices](devices.md) |
| Spread work over local GPUs and other machines, and survive unplugs | [Multiple devices and remote workers](cluster.md) |
| Let KORA choose the device for one dispatch | [Adaptive placement and calibration](adaptive.md) |
| Order work with dependencies you declare | [Task graphs](task-graphs.md) |
| Run CPU implementations next to GPU ones | [CPU executor and fallbacks](cpu.md) |
| Handle a lost device | [Recovery and device loss](recovery.md) |
| Measure where time goes | [Profiling and tracing](profiling.md) |
| Diagnose a failure | [Debugging](debugging.md) |
| Configure KORA with environment variables | [Configuration](configuration.md) |
