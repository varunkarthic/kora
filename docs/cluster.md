# Multi-device, remote workers, and hot-swap

`kora::Cluster` lets one application use the GPUs of other machines or devices it owns. A
`kora-worker` process runs KORA on the other device; the host offloads dispatches to it over TCP and
keeps working when the worker connects, disconnects, or reconnects at any time. The idea comes from
[Backburner](https://github.com/StayLameBro/backburner), which splits LLM inference between a Mac and
an iPhone over USB-C. Backburner itself is Apple- and LLM-specific (a llama.cpp fork and a
Metal/Neural Engine iOS app), so KORA does not include its code; it implements the same architecture
vendor-neutrally for any KORA kernel: a worker on the attached device, a link that only resends data
the worker does not already hold, and placement chosen from measured cost.

## Quick start

On the device that lends its GPU:

```sh
export KORA_WORKER_TOKEN="a long random secret"
build/release/tools/kora-worker --listen 0.0.0.0:7400
```

Without a token the worker accepts only clients on the same machine. A USB-C network link,
Thunderbolt bridge, or ordinary LAN all work; the protocol does not encrypt traffic, so use a trusted
link or an SSH/VPN tunnel across untrusted networks.

On the host, use a `Cluster` where you would use a `Runtime`:

```cpp
#include "kora/cluster.hpp"

kora::ClusterOptions options;
options.workers = {"192.168.2.5:7400"};   // or KORA_WORKERS=host:port,host:port
auto cluster = kora::Cluster::create(options).value();
auto kernel = cluster.load_kernel("saxpy.spv", "main", metadata).value();
auto x = cluster.create_buffer<float>(n).value();
auto y = cluster.create_buffer<float>(n).value();
// ... write x and y ...
auto done = cluster.dispatch(kernel).global(n).read(x).read_write(y).push(params).submit().value();
done.wait();            // y now holds the result, wherever it ran
```

Workers listed in `ClusterOptions::workers` or `KORA_WORKERS` may be started before or after the
host; absent ones are retried every `reconnect_interval`. `attach()` and `detach()` add or remove
workers while dispatches are running. `workers()` and `stats()` report connection state, how many
dispatches each target ran, and how many were failed over.

## What happens when a worker is unplugged

Buffers always live on the host. A remote dispatch sends the inputs the worker does not already have
(tracked per buffer and host version), runs the kernel there, and returns its outputs. Outputs are
committed to the host buffers only after the complete reply has arrived and been validated, and only
if the host buffer has not changed since the dispatch started. Any failure before that point (a
closed connection, a timeout, a frame whose CRC32C does not match, a worker error) leaves every host
buffer untouched and re-runs the same dispatch on the local Runtime. A dispatch is therefore applied
exactly once, wherever it ends up running.

```{mermaid}
sequenceDiagram
    participant App
    participant Cluster
    participant Worker
    participant Local as Local Runtime
    App->>Cluster: submit(y += x)
    Cluster->>Worker: Dispatch (only data the worker lacks)
    Worker--xCluster: link drops before the reply
    Note over Cluster: no host buffer was changed
    Cluster->>Local: re-run the same dispatch
    Local-->>Cluster: done
    Cluster-->>App: Event completes, y correct
    Cluster->>Worker: reconnect in the background
```

Dispatches that touch the same buffer keep their order (read-after-write, write-after-read,
write-after-write) whichever device runs each one. The dispatcher never blocks on a dependency, so a
job that fails over can always run even while later jobs wait for it.

The local GPU is protected the same way: if the local device is lost, the Cluster recreates the
local Runtime, reloads its kernels, and re-runs the failed dispatch once. Outputs of work that was
in flight on the lost device follow the normal [recovery rules](recovery.md).

## Every local GPU is a target

When `ClusterOptions::local.device_policy` is unset (and `KORA_DEVICE` is not set), the Cluster opens
every eligible local Vulkan device and routes each dispatch among those devices and the connected
workers. Host buffers remain the single source of truth, so a dispatch can run on any device and the
next one can run somewhere else. `Cluster::local_devices()` reports dispatches, slices, and
in-flight work for each local device.

On an Apple M4 with three Vulkan drivers, 4,096-element SAXPY through the Cluster went from 335 µs
to 78 µs p50 after the Cluster measured all three drivers and picked the fastest. Previously it
always used the default device.

### Hot-plugged GPUs

Every `device_rescan_interval` (default 5 s; zero disables it) a monitor thread compares the physical
device list with the one in use. When a GPU appears or disappears, the Cluster waits until no local
work is in flight, then calls `Runtime::rebuild_devices()`. Buffers survive, kernels reload on the
new devices, and queued dispatches continue. `ClusterStats::device_rescans` counts rebuilds. Rescan
is off when you pin a device policy, because a pinned policy is a request for specific devices.

### Splitting one dispatch across devices

```cpp
auto event = cluster.dispatch(kernel)
                 .global(count)
                 .read(x)
                 .read_write(y)
                 .push(Params{2.0F, count, 0})
                 .split(sizeof(float))  // Bytes of each output element per invocation.
                 .submit();
```

`split()` lets the Cluster divide a large one-dimensional elementwise dispatch among its devices.
The kernel must declare a `u32` push constant named `base` and handle element `base + gl_GlobalInvocationID.x`;
invocation *i* of every written buffer must touch only element *i*. Slices are multiples of the
workgroup size. Each device's range is read back, merged into one snapshot, and committed with a
single version-checked write, so a failed slice (an unplugged worker, a lost device) re-runs the
whole dispatch and never leaves a partial result.

The planner splits only when doing so should beat the best single device, using measured
per-device throughput and `split_overhead` (default 150 µs) for each extra slice. Dispatches below
`split_min_elements` (default 65,536) are never split. The throughput model assumes the devices run
independently. That is false for two drivers of one GPU, or for a CPU renderer that shares memory
bandwidth. So the planner also records how long finished splits actually take and stops splitting a
kernel and size once splits measure no faster than one device, re-trying every 16th time.

Measured on an Apple M4 whose three "devices" are one GPU behind two drivers plus llvmpipe (Release,
20 interleaved ABBA pairs after warm-up, median of per-pair ratios):

| SAXPY elements | One device p50 | `split()` p50 | Speedup | Splits chosen |
| ---: | ---: | ---: | ---: | ---: |
| 65,536 | 127 µs | 129 µs | 1.01× | 0 (below `split_min_elements`) |
| 1,048,576 | 679 µs | 680 µs | 1.00× | 3 of 46 |
| 2,097,152 | 1.32 ms | 1.27 ms | 1.00× | 3 of 46 |

Before the planner checked its predictions against measured splits, the same 1M-element dispatch
split nearly every time and took 1.19 ms, 0.70× the single-device speed. On this machine the
right result is not to split, and the planner now reaches it on its own. Machines with
independent GPUs are where splitting can pay off; this repository has not measured one.

## Watching it run

[`kora-top`](tools/kora-top.md) shows local devices and workers, where work is going, device
memory, and hot-swap events live. Set `KORA_STATUS_LISTEN=127.0.0.1:7979` for your application and
run `kora-top`, or run `kora-top --demo`.

## Placement

| `OffloadPolicy` | Behavior |
| --- | --- |
| `Auto` (default) | Runs each kernel on each local device first to measure it, tries each connected worker occasionally, then picks the target with the lowest expected completion time for that kernel and data size, including work already queued on each target. |
| `PreferRemote` | Runs on the least-busy connected worker that can load the kernel; local otherwise. |
| `LocalOnly` | Never offloads. |

A worker that cannot load a kernel (for example, a missing device feature) is not used for that
kernel again; other kernels can still go there.

## Security model

- The handshake is an HMAC-SHA256 challenge-response over a fresh 32-byte nonce, using the shared
  token. Wrong tokens are rejected before the worker does any work, and an unauthenticated peer
  cannot displace an active client.
- With no token configured, the worker accepts only loopback clients.
- Workers validate every size, count, and index in each request, cap frame size
  (`max_message_bytes`) and cached bytes (`max_cache_bytes`, least-recently-used eviction), and load
  kernels through the same SPIR-V and metadata validation as `Runtime::load_kernel`.
- Traffic is authenticated at connection time but not encrypted or integrity-protected against an
  active attacker afterwards; CRC32C only detects accidental corruption.

## Protocol

Each frame is a 32-byte little-endian header (magic, version, message type, request ID, payload
size, CRC32C) followed by the payload. Message types are `Challenge`, `Auth`, `Welcome`, `Reject`,
`LoadKernel`, `KernelLoaded`, `Failure`, `Dispatch`, `Outputs`, `NeedData`, `Release`, `Ping`, and
`Pong`; see `src/remote/protocol.hpp`. A worker serves one client at a time; a newly authenticated
client replaces a vanished one, so a host that reconnects after a cable pull is not locked out.

## Measured cost of offloading

`kora-bench --gpu --cluster` runs chained SAXPY (`y += 2x`) through a `Cluster` with `LocalOnly`,
then with `PreferRemote` against an in-process worker on `127.0.0.1` that uses the same GPU. Every
remote run is checked to have executed every dispatch remotely with no failovers. Five runs of 300
samples and 30 warm-ups on Apple M4 / MoltenVK, Release, source c48eb6b (median of per-run values):

| SAXPY elements | Local p50 | Local p95 | Remote (loopback) p50 | Remote p95 | Added by going remote |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 335 µs | 430 µs | 431 µs | 518 µs | +96 µs (1.29×) |
| 65,536 | 417 µs | 472 µs | 650 µs | 737 µs | +234 µs (1.56×) |
| 1,048,576 | 664 µs | 1.59 ms | 4.76 ms | 5.23 ms | +4.1 ms (7.2×) |

Small and medium dispatches cost one loopback round trip more. At 1M elements the 4 MiB output
returning to the host dominates, so offload pays only when the worker's GPU is much faster or the
local one is busy; `OffloadPolicy::Auto` measures exactly this per kernel and size. Before the link
was tuned (hardware CRC32C, 4 MiB socket buffers, outputs sent straight from the pinned host copy)
the same 1M remote dispatch took about 37 ms. These are same-machine numbers: a real USB-C or LAN link
adds its own latency and bandwidth limits. The raw reports are in
[`benchmarks/results/cluster-loopback/`](https://github.com/varunkarthic/kora/tree/main/benchmarks/results/cluster-loopback).

## Limits

- One request is in flight per worker; transfers and kernel execution on a worker are not pipelined.
- Outputs always return to the host, so chains of remote dispatches pay a round trip each; the
  worker keeps returned outputs cached, so the next dispatch does not resend them.
- A 1D dispatch is still limited by each device's `maxComputeWorkGroupCount[0]` (65,535 groups on
  the M4, about 4.19M elements at a workgroup size of 64), including with `split()`.
- Measurements in this repository use a loopback worker on the same Apple M4; they demonstrate
  correctness and hot-swap behavior, not cross-machine speed.

## Tests

`tests/integration/test_cluster.cpp` checks remote results against local ones, a worker that starts
after the Cluster, unplug/replug and detach mid-stream, wrong tokens, and a randomized test that
stops and restarts the worker every 5–60 ms while 300 dependent dispatches run; every result must be
exact. Further tests cover routing across every local device, split results against a CPU
reference with an element count that is not a multiple of the workgroup, splits while a worker is
toggled every 5–40 ms, and rejection of `split()` for kernels without a `base` field. `tests/unit/test_remote_protocol.cpp` covers SHA-256/HMAC test vectors, CRC32C (hardware and table paths), framing, and
corrupted, oversized, and truncated frames.
