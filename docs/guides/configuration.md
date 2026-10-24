# Configuration

Settings resolve in this order: an explicit option in code, then the `KORA_*` environment variable,
then the built-in default. Invalid environment values produce a warning and fall back to the default.

## Runtime

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `KORA_DEVICE` | `auto`, `discrete`, `integrated`, `all`, an index, or a 32-hex UUID | `auto` | [Device selection](devices.md) |
| `KORA_VALIDATION` | boolean | `false` | Request the Vulkan validation layer |
| `KORA_DEBUG_SAFETY` | boolean | on in debug builds | Extra checks and synchronization tracing (also requests validation) |
| `KORA_PROFILE` | `off`, `basic`, `profile`, `trace` | `basic` | [Telemetry level](profiling.md) |
| `KORA_TRACE` | boolean | `false` | Force `trace` telemetry |
| `KORA_LOG_LEVEL` | `trace`, `debug`, `info`, `warn`, `error`, `off` | `warn` | Default log threshold |
| `KORA_MEMORY_BUDGET_FRACTION` | number in (0, 1] | `0.80` | Share of each heap's free memory KORA may use |
| `KORA_CACHE_DIR` | path | platform cache directory | Pipeline cache, calibration, and performance database |
| `KORA_STARTUP_CALIBRATION` | boolean | `true` | [Startup calibration](adaptive.md) |
| `KORA_CALIBRATION_BUDGET_MS` | 1–60000 | `250` | Calibration time per device |
| `KORA_DISABLE_AUTOTUNE` | boolean | `false` | Disable `LocalSizeId` workgroup tuning |
| `KORA_DETERMINISTIC` | boolean | `false` | Disable exploration in scheduler policy models |
| `KORA_SHUTDOWN_MODE` | `graceful`, `cancel_pending` | `graceful` | Drain or cancel queued work when a runtime is dropped |
| `KORA_FORCE_LEGACY_SYNC` | boolean | `false` | Use fences instead of timeline semaphores (testing) |
| `KORA_DISABLE_MULTI_QUEUE`, `KORA_DISABLE_CPU_BACKEND` | boolean | `false` | Accepted for compatibility; no effect in 1.0 |

Booleans accept `1/true/on/yes` and `0/false/off/no`, case-insensitively.

## Cluster and workers

| Variable | Effect |
| --- | --- |
| `KORA_WORKERS` | Comma-separated `host:port` endpoints added to every `Cluster` |
| `KORA_WORKER_TOKEN` | Shared secret for the worker handshake (Cluster and `kora-worker`) |
| `KORA_STATUS_LISTEN` | Loopback `host:port` for the [`kora-top`](../tools/kora-top.md) status stream |

## In code

=== "Rust"

    ```rust
    let options = kora::RuntimeOptions {
        enable_validation: Some(true),
        device_policy: Some(kora::DevicePolicy::PreferIntegrated),
        memory_budget_fraction: Some(0.7),
        ..kora::RuntimeOptions::default()
    };
    let resolved = options.resolve();   // what the runtime will use, after the environment
    ```

=== "C"

    ```c
    KoraRuntimeOptions options;
    kora_runtime_options_default(&options);   /* every field KORA_DEFAULT */
    options.enable_validation = 1;
    options.device_policy = KORA_DEVICE_POLICY_PREFER_INTEGRATED;
    ```

=== "Python"

    ```python
    options = kora.RuntimeOptions()
    options.enable_validation = True
    options.device_policy = kora.DevicePolicy.PREFER_INTEGRATED
    ```

## Cache directory

Without `KORA_CACHE_DIR`, KORA uses the platform's user cache directory. The cache holds performance
data only (pipeline caches bound to the device and driver identity, calibration records, and observed
timings) and is rebuilt if corrupt. Do not share a writable cache across security boundaries.
