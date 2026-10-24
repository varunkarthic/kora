# kora-top

`kora-top` shows what a running `Cluster` is doing: every local GPU and remote worker,
which of them received work in the last second, device memory, worker cache, and a log of
connects, disconnects, failovers, device rescans, and routing changes.

## Watch the demo

```sh
kora-top --demo
```

The demo needs no setup and exits with status 77 when no Vulkan device is available. It creates a
Cluster with every local device and two loopback workers,
A and B, and runs a 4K-element dispatch and a 1M-element `split()` dispatch in a loop. A script
plugs worker B in at 3 s, unplugs A at 7 s, plugs A back in at 11 s, and unplugs B at 15 s, then
repeats. Every eighth round the demo reads both result buffers and compares every element with the
expected value. The footer counts verified results and mismatches, and `kora-top --demo` exits
non-zero if any result was wrong.

This frame was captured from `kora-top --demo --no-color` on an Apple M4 (KosmicKrisp and MoltenVK
drivers for the same GPU, plus two loopback workers), 9.8 s in. Worker A was unplugged at 7.0 s: its
in-flight dispatch re-ran locally, work moved to worker B, and all 250 checked results were exact.

```text
kora-top  127.0.0.1:50243  up 9.8 s  policy prefer-remote
dispatches: local 3, remote 1,414, split 573 (1,157 slices)   failovers 1   device rescans 0   recoveries 0

  TARGET                                              WORK NOW               TOTAL     RUN  MEMORY
* local 1   Apple M4                                  █████░░░░░░░░░░░   33%  575       0    ░░░░░░░░ 16.0 MiB of 14.2 GiB
* local 0   Apple M4                                  █████░░░░░░░░░░░   33%  574       0    16.0 MiB
o worker    127.0.0.1:50241  unplugged                ░░░░░░░░░░░░░░░░    0%  922       0    1 connects, 1 failovers
* worker    127.0.0.1:50242  Apple M4                 █████░░░░░░░░░░░   33%  503       0    cached 8.0 MiB

  EVENTS
     3.0s  demo.saxpy now runs on worker 127.0.0.1:50241, split(local 0 + local 1), split(local 1 + local 0)
     3.1s  worker 127.0.0.1:50242 connected (Apple M4)
     4.0s  demo.saxpy now runs on worker 127.0.0.1:50241, worker 127.0.0.1:50242, split(local 1 + local 0), ~
     5.0s  demo.saxpy now runs on worker 127.0.0.1:50241, split(local 1 + local 0)
     6.0s  demo.saxpy now runs on worker 127.0.0.1:50241, worker 127.0.0.1:50242, split(local 1 + local 0), ~
     7.0s  worker 127.0.0.1:50241 failed a dispatch; re-running it locally
     7.0s  worker 127.0.0.1:50241 disconnected
     7.0s  demo.saxpy now runs on worker 127.0.0.1:50241, worker 127.0.0.1:50242, local 0, split(local 0 + l~
     8.0s  demo.saxpy now runs on worker 127.0.0.1:50242, split(local 1 + local 0), split(local 1 + local 0 ~
     9.1s  demo.saxpy now runs on worker 127.0.0.1:50242, split(local 1 + local 0)

demo  results verified 250, 0 mismatches   next: plug in worker A (127.0.0.1:50241) in 2 s   (Ctrl-C to quit)
```

Each line shows a target: local devices first, then workers. **WORK NOW** is the target's share of the
work completed in roughly the last second; **TOTAL** counts whole dispatches plus split slices; **RUN**
is work in flight. **MEMORY** is what KORA has allocated on a device, against the driver's budget when
it reports one; for a worker, `cached` is how much of the host's buffer data it holds at the current
version, which the next dispatch does not resend.

## Attach to your application

Turn on the status endpoint, either without code changes:

```sh
KORA_STATUS_LISTEN=127.0.0.1:7979 ./my-app
```

or in code:

```rust
let cluster = kora::Cluster::new(kora::ClusterOptions {
    status_listen: "127.0.0.1:7979".into(), // port 0 picks a free port; see status_port()
    ..kora::ClusterOptions::default()
})?;
```

Then run `kora-top` in another terminal. `127.0.0.1:7979` is its default; pass `host:port` for any
other endpoint. `kora-top` waits until the application is up, and it reconnects if the application
restarts.

| Option | Effect |
| --- | --- |
| `host:port` | Endpoint to attach to (default `127.0.0.1:7979`). |
| `--demo` | Run the built-in demo described above. |
| `--frames N` | Draw `N` snapshots as plain frames, then exit. Useful in scripts and CI. |
| `--no-color` | No ANSI colors. `NO_COLOR` has the same effect. |

## Status stream

The endpoint sends one JSON object per line every `status_interval` (default 250 ms) to each
connected viewer, with at most eight viewers. A viewer that stops reading is disconnected rather
than allowed to slow the Cluster. Fields:

| Field | Contents |
| --- | --- |
| `version` | Stream format version, currently 1. |
| `uptime_ms`, `policy` | Time since the Cluster was created, and its `OffloadPolicy`. |
| `devices[]` | `index`, `name`, `dispatches`, `slices`, `inflight`, `allocated_bytes`, `budget_bytes` for each local device. |
| `workers[]` | `endpoint`, `connected`, `name`, `device`, `inflight`, `dispatches`, `failovers`, `connects`, `disconnects`, `cached_bytes` for each worker. |
| `stats` | The `ClusterStats` counters. |
| `events[]` | Up to 64 recent events: `at_ms`, `message`, and `repeats` (identical consecutive events are merged). |

The same data is available in-process through `Cluster::local_devices()`, `workers()`, `stats()`,
and `events()`.

## Security

Snapshots name devices and worker endpoints, so `status_listen` accepts only loopback addresses
(`127.0.0.1`, `::1`, `localhost`). Accepted connections that do not come from loopback are closed.
The stream is read-only: viewers cannot send commands to the Cluster.
