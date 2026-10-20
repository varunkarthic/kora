# kora-top

`kora-top` shows what a running `kora::Cluster` is doing: every local GPU and remote worker,
which of them received work in the last second, device memory, worker cache, and a log of
connects, disconnects, failovers, device rescans, and routing changes.

## Watch the demo

```sh
./build/release/tools/kora-top --demo
```

The demo needs no setup. It creates a Cluster with every local device and two loopback workers,
A and B, and runs a 4K-element dispatch and a 1M-element `split()` dispatch in a loop. A script
plugs worker B in at 3 s, unplugs A at 7 s, plugs A back in at 11 s, and unplugs B at 15 s, then
repeats. Every eighth round the demo reads both result buffers and compares every element with the
expected value. The footer counts verified results and mismatches, and `kora-top --demo` exits
non-zero if any result was wrong.

This frame was captured from `kora-top --demo --no-color` on an Apple M4, 10 s in, after worker A
had been unplugged:

```text
kora-top  127.0.0.1:54876  up 9.9 s  policy prefer-remote
dispatches: local 1, remote 1,217, split 1,214 (3,645 slices)   failovers 1   device rescans 0   recoveries 0

  TARGET                                    WORK NOW               TOTAL     RUN  MEMORY
* local 2   Apple M4                        ████░░░░░░░░░░░░   25%  1,215     0    ░░░░░░░░ 16.0 MiB of 14.2 GiB
* local 1   llvmpipe (LLVM 22.1.8, 128 bit~ ████░░░░░░░░░░░░   25%  1,214     0    ░░░░░░░░ 16.0 MiB of 4.5 GiB
* local 0   Apple M4                        ████░░░░░░░░░░░░   25%  1,214     0    16.0 MiB
o worker    127.0.0.1:54874  unplugged      ░░░░░░░░░░░░░░░░    0%  871       0    1 connects, 1 failovers
* worker    127.0.0.1:54875  Apple M4       ████░░░░░░░░░░░░   25%  349       0    cached 8.0 MiB

  EVENTS
     0.0s  worker 127.0.0.1:54874 connected (Apple M4)
     1.0s  demo.saxpy now runs on worker 127.0.0.1:54874, split(local 1 + worker 127.0.0.1:54874 +~
     2.0s  demo.saxpy now runs on worker 127.0.0.1:54874, split(local 1 + local 0 + local 2), spli~
     3.1s  worker 127.0.0.1:54875 connected (Apple M4)
     4.0s  demo.saxpy now runs on worker 127.0.0.1:54874, split(local 1 + worker 127.0.0.1:54875 +~
     5.0s  demo.saxpy now runs on worker 127.0.0.1:54874, split(local 1 + local 0 + local 2), spli~
     7.0s  worker 127.0.0.1:54874 failed a dispatch; re-running it locally
     7.0s  worker 127.0.0.1:54874 disconnected
     8.0s  demo.saxpy now runs on worker 127.0.0.1:54874, worker 127.0.0.1:54875, local 2, split(l~
     9.0s  demo.saxpy now runs on worker 127.0.0.1:54875, split(local 1 + local 0 + local 2), spli~

demo  results verified 304, 0 mismatches   next: plug in worker A (127.0.0.1:54874) in 2 s   (Ctrl-C to quit)
```

The frame shows:

- The three local devices (two Vulkan drivers for the M4 GPU, plus llvmpipe) and worker B each ran
  a quarter of the last second's work, mostly split slices.
- Worker A is gone. At 7.0 s it failed one in-flight dispatch, that dispatch re-ran locally, and
  work moved to B.
- 304 results were checked across the unplug with no mismatches.
- Memory shows what KORA has allocated on each device, against the budget the driver reports
  when it supports `VK_EXT_memory_budget`. `cached` is how much of the host's buffer data the worker
  holds at the current version, which the next dispatch does not resend.

## Attach to your application

Turn on the status endpoint, either without code changes:

```sh
KORA_STATUS_LISTEN=127.0.0.1:7979 ./my-app
```

or in code:

```cpp
kora::ClusterOptions options;
options.status_listen = "127.0.0.1:7979";  // Port 0 picks a free port; see status_port().
auto cluster = kora::Cluster::create(options);
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

The same data is available in-process through `Cluster::local_devices()`, `Cluster::workers()`,
`Cluster::stats()`, and `Cluster::events()`.

## Security

Snapshots name devices and worker endpoints, so `status_listen` accepts only loopback addresses
(`127.0.0.1`, `::1`, `localhost`). Accepted connections that do not come from loopback are closed.
The stream is read-only: viewers cannot send commands to the Cluster.
