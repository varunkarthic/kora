# Internals

How the runtime is put together, for contributors to the codebase and for anyone debugging below the
public API. The generated [rustdoc](https://kora.readthedocs.io/en/latest/rustdoc/kora/) documents
every module, including the lower-level `vulkan`, `memory`, and `kernel` building blocks.

- [Architecture](architecture.md): layers, threads, and the life of a dispatch.
- [Invariants](invariants.md): the cross-layer correctness rules and the tests that enforce them.
- [Memory subsystem](memory.md): pools, the suballocator, budgets, staging, and replicas.
- [Queues and transfers](queues.md): submission, completion, barriers, and copies.
- [Kernels and pipelines](pipelines.md): SPIR-V reflection, pipeline caching, and autotuning.
