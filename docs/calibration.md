# Startup calibration

Startup calibration is an optional, best-effort facility that stores versioned measurements beside
the other runtime caches. Records are keyed by the selected physical device UUID and Vulkan driver
version and include the KORA calibration schema. Any missing, corrupt, oversized, incompatible, or
unreadable record is treated as a cache miss; it never prevents runtime creation.

## Configuration

Calibration is enabled by default. Disable it with `RuntimeOptions::enable_startup_calibration = false`
or `KORA_STARTUP_CALIBRATION=false`. Set `RuntimeOptions::calibration_budget_ms` or
`KORA_CALIBRATION_BUDGET_MS` to bound the per-device calibration budget (1–60,000 ms; default
250 ms). API options take precedence over environment variables.

Cache files are written below `<cache_dir>/calibration/<device-uuid>.json` using a temporary file,
flush, and rename. Never share a writable cache directory across mutually untrusted users.

## Record schema

Schema version 2 stores identity (`device_uuid`, `driver_version`, `kora_schema`) plus nanoseconds for
submit overhead and bytes/operations per second for tiny dispatch, H2D, D2H, device copy, streaming
kernel bandwidth, and ALU throughput. Zero means that measurement is not available; it must not be
interpreted as infinite performance or zero latency.

On a cache miss, Runtime measures the median wall time of up to three empty compute-queue submissions
(enqueue through queue completion) and tries isolated transfer probes using temporary Runtime-owned
Vulkan buffers. It samples 4 KiB, 64 KiB, and 256 KiB host-to-device, device-to-host, and device-copy
operations, validates the copied bytes, and stores the best observed throughput for each supported
path. A built-in SPIR-V compute module is embedded in KORA; runtime use does not require a shader
compiler. After structural/reflection validation for each device, the probe measures the smallest
dispatch, a storage-buffer read/write pass, and a deterministic integer ALU loop. Every output element
is checked against a host-computed expected result before its rate is accepted. The read/write rate
counts input and output bytes; ALU throughput uses four integer operations per loop iteration as a
work estimate, not a hardware instruction count. These are host-observed end-to-end times, including
KORA submission/completion and output visibility costs; they are not isolated GPU timestamp queries
or hardware performance counters.

The configured per-device budget is best-effort and split between queue/transfer and compute work,
with a small reserve for atomically publishing the record. Pipeline creation and driver calls can
themselves exceed the remaining budget; KORA checks the deadline before probes and uses a bounded
completion wait, but cannot preempt a driver call already in progress. A timeout, allocation failure,
unsupported path, or failed correctness check leaves that metric at zero. Calibration uses private
buffers only and never changes application buffers. Schema 2 invalidates records from older versions
whose compute fields could only have been zero. Zero still marks an unsupported or unmeasured path.

Call `Runtime::calibrate()` to bypass the cache and repeat best-effort calibration for every selected
device, including when startup calibration was disabled. The call uses the Runtime's configured
per-device budget. It succeeds when the request was accepted; individual unsupported probes remain
zero and cache I/O failures are non-fatal. Calibration records are currently diagnostic inputs only:
they do not change device selection or automatic scheduling.

## Invariants

```{mermaid}
flowchart TD
    A[Select device and inspect schema 2 cache] -->|valid, complete| Z[Reuse measured values]
    A -->|miss or missing compute metric| B[Bounded queue and private transfer probes]
    B --> C[Validate embedded SPIR-V for device]
    C -->|unsupported or invalid| F[Keep unavailable metric at zero]
    C -->|valid| D[Dispatch tiny, streaming, and ALU modes]
    D --> E{Host output matches expected result?}
    E -->|no or timed out| F
    E -->|yes| G[Record only validated metric]
    F --> H[Atomically publish best-effort record]
    G --> H
```

- Calibration is performed only after selected devices are initialized and never changes device
  selection.
- Cache parsing has a strict size cap and requires exact schema, UUID, driver, and KORA schema match.
- Calibration errors are non-fatal. Failed writes leave the runtime usable and the next startup retries.
- Probe work is bounded by the configured per-device budget and does not run for devices without a
  stable Vulkan UUID. Pending transfer completions retain their private scratch buffers until the
  queue completion callback releases them.

The record codec and atomic file helper are internal implementation details, not a stable public API.
