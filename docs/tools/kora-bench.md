# kora-bench

Measures CPU, KORA runtime, raw Vulkan, transfer, and Cluster workloads and prints one JSON report on
stdout.

```sh
kora-bench --smoke                     # 3 samples per workload, CPU only (seconds)
kora-bench --gpu > report.json         # + Vulkan workloads on the default device
kora-bench --gpu --cluster             # + Cluster local and loopback-worker dispatches
kora-bench --profile-phases            # + per-phase host timings of a 4096-element dispatch
kora-bench --iterations 1000 --warmup 50 --gpu
```

Exit status: 0 on success, 1 if a workload failed or produced a wrong result, 2 for invalid arguments,
3 when the GPU path is unavailable. Every workload's output is checked during warm-up and after every
timed sample.

## What it measures

| Family | Examples | Timing domain |
| --- | --- | --- |
| Host operations | `scheduler_cost`, `task_graph_finalize_64`, `task_graph_replay_64`, `allocator_allocate_free_256`, `allocator_churn_4096`, `buffer_create_drop_4096`, `cpu_executor_submit_wait`, `cpu_executor_fanout_1024` | host |
| CPU baselines | `cpu_{vector_add,saxpy,reduction,stencil,tiled_scan}_{size}`, `cpu_tiled_matmul_{dim}`, and `_parallel_` variants on 4 workers | host CPU |
| KORA runtime | `gpu_tiny_dispatch_1`, `gpu_vector_add_{size}_dispatch_wait`, `..._deferred_readback_dispatch_wait`, the same for SAXPY, reduction, scan, stencil, and tiled matmul | host submit to completion |
| Device time | `..._gpu_execution` | GPU timestamps (when available) |
| Raw Vulkan reference | `raw_vulkan_*`, the same kernels with one command buffer and a fence on the same physical device | host submit to completion |
| Transfers | `transfer_{direct,staged}_{upload,download}_{size}`, `kora_transfer_copy_*`, `raw_vulkan_copy_*`, `staging_ring_reserve_*`, `command_pool_*` | host |
| Cluster | `cluster_{local,remote_loopback}_saxpy_{size}_dispatch_wait` | host submit to completion |

## Report format

The report has schema version `4`. (The 0.x C++ `kora-bench` wrote schema version `3`, with the same
result names minus the four host families added in 1.0.) The root has `schema_version`, `implementation` (`"rust"`),
`clock`, `iterations`, `warmup`, `diagnostic_profile`, `environment` (OS, architecture, compiler,
CPU model, build type and flags, source revision, logical CPU count), `gpu_measurements` with `--gpu`
(device, driver, queue family, timestamp availability), and `results[]`. Each result has `name`,
`unit` (`ns`), `median`, `p95`, `timing_domain`, every raw sample, and where relevant `path`,
`device_index`, `bytes_transferred`, `throughput_bytes_per_second`, and `thread_count`.

`tests/check_kora_bench_json.py` validates the contract and runs in CI.

## Comparing builds

Never compare single runs: GPU clock state can move one run's numbers several-fold. Interleave:

```sh
python3 tools/kora-bench/run_ab.py --a old/kora-bench --b new/kora-bench --out ab --pairs 10 --device 1
python3 tools/kora-bench/render_ab.py --families --before ab/a-*.json.gz --after ab/b-*.json.gz
```

See [Performance methodology](../performance/methodology.md).
