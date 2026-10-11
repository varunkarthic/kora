# Configuration

`kora::RuntimeOptions` stores optional API overrides. Resolution follows this precedence:

1. an explicitly set C++ option;
2. the corresponding `KORA_*` environment variable;
3. the built-in default.

Invalid environment values add a warning and fall back safely. Invalid explicit memory-budget values
also reset to the default with a warning.

## Environment variables

| Variable | Values | Default | Purpose |
| --- | --- | --- | --- |
| `KORA_VALIDATION` | Boolean | `false` | Request Vulkan validation layers |
| `KORA_DEVICE` | `auto`, `discrete`, `integrated`, `all`, index, or 32-hex UUID | `auto` | Device selection policy |
| `KORA_MEMORY_BUDGET_FRACTION` | Number in `(0,1]` | `0.80` | Fraction of usable memory budget |
| `KORA_PROFILE` | `off`, `basic`, `profile`, `trace` | `basic` | Telemetry detail |
| `KORA_LOG_LEVEL` | `trace`, `debug`, `info`, `warn`, `error`, `off` | `warn` | Default log threshold |
| `KORA_CACHE_DIR` | Filesystem path | Platform cache path | Persistent cache location |
| `KORA_STARTUP_CALIBRATION` | Boolean | `true` | Enable best-effort startup calibration |
| `KORA_CALIBRATION_BUDGET_MS` | Integer, 1–60000 | `250` | Per-device startup calibration time budget |
| `KORA_DETERMINISTIC` | Boolean | `false` | Deterministic scheduling policy |
| `KORA_SHUTDOWN_MODE` | `graceful`, `cancel_pending` | `graceful` | Runtime shutdown behavior |
| `KORA_DISABLE_AUTOTUNE` | Boolean | `false` | Disable online tuning |
| `KORA_DISABLE_MULTI_QUEUE` | Boolean | `false` | Restrict device queue use |
| `KORA_DISABLE_CPU_BACKEND` | Boolean | `false` | Exclude CPU execution |
| `KORA_FORCE_LEGACY_SYNC` | Boolean | `false` | Test hook for legacy synchronization |
| `KORA_TRACE` | Boolean | `false` | Force trace-level telemetry |

Boolean parsing accepts `1/true/on/yes` and `0/false/off/no`, case-insensitively.

## Cache directory

If `KORA_CACHE_DIR` is unset, KORA derives a user cache location from the platform environment. Cache
files are performance data, not trusted configuration. Do not share writable caches across security
boundaries or place secrets in paths and cache keys.

## API example

```cpp
kora::RuntimeOptions options;
options.enable_validation = true;
options.device_policy = kora::DevicePolicy::PreferIntegrated;
options.memory_budget_fraction = 0.70;

kora::ResolvedOptions resolved = kora::resolve(options);
```

Explicit fields above take precedence over environment values; unset fields remain eligible for the
environment overlay.
