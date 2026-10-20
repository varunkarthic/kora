# Remote workers and hot-swap

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
closed connection, a timeout, a frame whose CRC32 does not match, a worker error) leaves every host
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

## Placement

| `OffloadPolicy` | Behavior |
| --- | --- |
| `Auto` (default) | Runs each kernel locally first to measure it, tries each connected worker occasionally, then picks the target with the lowest expected completion time for that kernel and data size, including work already queued on each target. |
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
  active attacker afterwards; CRC32 only detects accidental corruption.

## Protocol

Each frame is a 32-byte little-endian header (magic, version, message type, request ID, payload
size, CRC32) followed by the payload. Message types are `Challenge`, `Auth`, `Welcome`, `Reject`,
`LoadKernel`, `KernelLoaded`, `Failure`, `Dispatch`, `Outputs`, `NeedData`, `Release`, `Ping`, and
`Pong`; see `src/remote/protocol.hpp`. A worker serves one client at a time; a newly authenticated
client replaces a vanished one, so a host that reconnects after a cable pull is not locked out.

## Limits

- One request is in flight per worker; transfers and kernel execution on a worker are not pipelined.
- Outputs always return to the host, so chains of remote dispatches pay a round trip each; the
  worker keeps returned outputs cached, so the next dispatch does not resend them.
- Measurements in this repository use a loopback worker on the same Apple M4; they demonstrate
  correctness and hot-swap behavior, not cross-machine speed.

## Tests

`tests/integration/test_cluster.cpp` checks remote results against local ones, a worker that starts
after the Cluster, unplug/replug and detach mid-stream, wrong tokens, and a randomized test that
stops and restarts the worker every 5–60 ms while 300 dependent dispatches run; every result must be
exact. `tests/unit/test_remote_protocol.cpp` covers SHA-256/HMAC test vectors, CRC32, framing, and
corrupted, oversized, and truncated frames.
