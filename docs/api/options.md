# Runtime options API

Header: `include/kora/options.hpp`

## Types

`DevicePolicy`
: Selects automatic placement, preference for discrete or integrated devices, an explicit index or
  UUID, or all eligible devices. Preference is not a guarantee that a requested device exists.

`TelemetryLevel`
: Controls off, basic, profile, or trace data collection.

`ShutdownMode`
: Chooses graceful completion or cancellation of pending work during runtime shutdown.

`enable_startup_calibration`, `calibration_budget_ms`
: Enable/disable best-effort per-device startup calibration and configure its budget (1–60,000 ms,
  default 250 ms). Corresponding environment variables are documented in `configuration.md`.
  `Runtime::calibrate()` can explicitly rerun probes while bypassing the persistent cache.

`disable_autotune`
: Disable runtime workgroup tuning. When unset, `KORA_DISABLE_AUTOTUNE` controls this setting and
  defaults to tuning enabled where supported. Tuning applies only to SPIR-V kernels that declare
  `LocalSizeId` and only when GPU timestamp profiling is available. See
  [workgroup autotuning](../autotuning.md).

`RuntimeOptions`
: User-facing optional overrides. An unset field permits environment/default resolution.

`ResolvedOptions`
: Fully concrete values plus non-fatal parsing warnings. Runtime construction should consume this
  type after calling `resolve`.

## Functions

`system_env()`
: Returns the production environment lookup. Tests inject an `EnvLookup` instead of mutating the
  process environment.

`resolve(options, env)`
: Merges explicit values, environment variables, and defaults. Bad environment input never throws;
  it adds a warning and preserves a safe value.

## Invariants

- Explicit API values always beat environment variables.
- `RuntimeOptions::debug_safety` / `KORA_DEBUG_SAFETY` requests Vulkan validation when available and
  trace-level synchronization logging; it defaults on in non-Release builds and off in Release.
- Memory-budget fractions remain in `(0,1]`.
- UUID selection uses exactly 32 hexadecimal characters.
- Selection policy is capability based; a vendor ID is not a policy input.
- Test-only fallback switches must not silently become production defaults.
