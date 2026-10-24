# Tools

All four ship in the `kora-tools` crate:

```sh
cargo install --git https://github.com/varunkarthic/kora kora-tools
```

| Tool | What it does |
| --- | --- |
| [`kora-info`](kora-info.md) | Lists Vulkan devices and their capabilities, as text or JSON. |
| [`kora-top`](kora-top.md) | Live view of a running `Cluster`: devices, workers, memory, failovers. |
| [`kora-bench`](kora-bench.md) | CPU, KORA, raw Vulkan, transfer, and Cluster microbenchmarks as JSON. |
| [`kora-worker`](kora-worker.md) | Runs dispatches for a remote `Cluster`. |
