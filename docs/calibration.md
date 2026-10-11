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

The current runtime performs only a bounded host-side submit-overhead probe. The Vulkan dispatch and
transfer probes remain zero until scratch-buffer benchmarking can be isolated from application work
and measured reliably across supported devices. Consequently, current records are schema/storage
infrastructure and are not yet used to change device selection or scheduling decisions.

## Invariants

- Calibration is performed only after selected devices are initialized and never changes device
  selection.
- Cache parsing has a strict size cap and requires exact schema, UUID, driver, and KORA schema match.
- Calibration errors are non-fatal. Failed writes leave the runtime usable and the next startup retries.
- Measurement work is bounded by the configured per-device budget and does not run for devices without
  a stable Vulkan UUID.

The record codec and atomic file helper are internal implementation details, not a stable public API.
