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

Schema version 1 stores identity (`device_uuid`, `driver_version`, `kora_schema`) plus nanoseconds for
submit overhead and bytes/operations per second for tiny dispatch, H2D, D2H, device copy, streaming
kernel bandwidth, and ALU throughput. Zero means that measurement is not available; it must not be
interpreted as infinite performance or zero latency.

On a cache miss, Runtime measures the median wall time of up to three empty compute-queue submissions
(enqueue through queue completion) and tries isolated
transfer probes using temporary Runtime-owned Vulkan buffers. It samples 4 KiB, 64 KiB, and 256 KiB
host-to-device, device-to-host, and device-copy operations, validates the copied bytes, and stores the
best observed throughput for each supported path. Probes stop before the configured measurement
deadline to leave a small reserve for atomically publishing the record. A timeout, allocation failure,
unsupported path, or failed correctness check leaves that metric at zero; calibration never consumes
or changes application buffers. `tiny_dispatch_ns`, `streaming_bytes_per_second`, and
`alu_operations_per_second` remain zero until validated built-in compute probes are available.

Call `Runtime::calibrate()` to bypass the cache and repeat best-effort calibration for every selected
device, including when startup calibration was disabled. The call uses the Runtime's configured
per-device budget. It succeeds when the request was accepted; individual unsupported probes remain
zero and cache I/O failures are non-fatal. Calibration records are currently diagnostic inputs only:
they do not change device selection or automatic scheduling.

## Invariants

- Calibration is performed only after selected devices are initialized and never changes device
  selection.
- Cache parsing has a strict size cap and requires exact schema, UUID, driver, and KORA schema match.
- Calibration errors are non-fatal. Failed writes leave the runtime usable and the next startup retries.
- Probe work is bounded by the configured per-device budget and does not run for devices without a
  stable Vulkan UUID. Pending transfer completions retain their private scratch buffers until the
  queue completion callback releases them.

The record codec and atomic file helper are internal implementation details, not a stable public API.
