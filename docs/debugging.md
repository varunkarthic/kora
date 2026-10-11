# Debugging and diagnostics

KORA's diagnostic design keeps observability separate from runtime correctness. Logging can explain
device choice, scheduling, synchronization, and recovery decisions, but changing a log threshold
must never change those decisions. The public API is documented in {doc}`api/logging`.

## Debug safety mode

`RuntimeOptions::debug_safety` or `KORA_DEBUG_SAFETY` enables extra diagnostics. It defaults on for
non-Release builds and off for Release builds; an explicit API value takes precedence over the
environment. When enabled, KORA requests `VK_LAYER_KHRONOS_validation` when installed and emits
trace-level queue submission details in the `Sync` category. If the layer is unavailable, startup
continues with a warning. Vulkan robust buffer access is enabled whenever the selected device reports
support, independently of this option.

Several checks are unconditional safety contracts rather than mode switches: SPIR-V is validated and
reflected before pipeline creation, buffer slot generations are checked on handle operations, and the
suballocator and queue ownership invariants are asserted in Debug builds. Debug builds also place
canaries around each host-shadow buffer and report an internal-invariant error when a later operation
detects an overwritten guard. These guards are absent from Release allocations. Vulkan validation and
robust buffer access are diagnostic/hardening aids, not a substitute for validating untrusted input or
bounds in application kernels.

Internal assertion semantics and the source/test map for the audited invariants are listed in
[Internal invariant audit](internals/invariants.md). `KORA_ASSERT` does not replace recoverable input
checks; its expression is side-effect-free and omitted from evaluation in Release builds.

For example, enable safety mode explicitly in a Release build with:

```sh
KORA_DEBUG_SAFETY=1 ./your-application
```

Expected outcome: if the Vulkan validation layer is installed, it is enabled and synchronization
submission details appear on standard error. Otherwise startup continues and reports that validation
could not be enabled.

## Enable useful logs

Set the process-wide threshold before starting the application:

```sh
KORA_LOG_LEVEL=info ./your-application
```

Accepted values, from most to least verbose, are `trace`, `debug`, `info`, `warn`, `error`, and
`off`. Matching is case-insensitive. An unset or invalid value falls back to `warn`.

For a narrow investigation, keep the global threshold conservative and lower one category in code:

```cpp
kora::log::set_level(kora::log::Level::Warn);
kora::log::set_category_level(kora::log::Category::Sync,
                              kora::log::Level::Trace);
```

This is preferable to global trace logging when investigating one subsystem because it reduces
noise and the chance of recording sensitive application context.

## Capture diagnostics in an application

The default sink writes one serialized line to standard error as `[CATEGORY] message`. Applications
can replace it with a callback:

```cpp
kora::log::set_sink([](kora::log::Level severity,
                       kora::log::Category category,
                       std::string_view message) {
    application_log(kora::log::to_string(severity),
                    kora::log::to_string(category), message);
});
```

The callback runs while KORA holds the sink mutex. It must return promptly, must not call KORA
logging recursively, and must copy `message` if it needs the text after the callback returns. Restore
the built-in sink with `set_sink(nullptr)`.

## Diagnose common failures

### No log output

1. Confirm `KORA_LOG_LEVEL` was set before the first logging API call.
2. Check the relevant category threshold with `kora::log::level(category)`.
3. Confirm the threshold is not `Off` and the statement level is at least the threshold.
4. If a custom sink is installed, verify that it does not discard the event.

### Unexpectedly verbose output

Call `set_level` once to reset all category overrides, then add only the category overrides needed
for the investigation. Remember that `Trace` and `Debug` can be high-volume in queue and scheduler
paths.

### A custom sink hangs

Check for recursive logging and lock inversion. Sink calls are serialized to protect callbacks that
are not thread-safe. Forward events to an application queue if processing is slow; do not perform
blocking I/O while holding unrelated application locks.

### Formatting differs across compilers

KORA uses `std::format` when the standard library provides it. The portable fallback supports only
`{}`, `{{`, and `}}`. Keep library log statements within that common subset unless all supported
toolchains provide `<format>`.

## Collect a useful report

When reporting a defect, include:

- KORA version from `kora::version_string()`;
- operating system, architecture, compiler, and build type;
- the selected device and relevant category logs;
- the smallest reproducible workload and whether deterministic mode changes the result;
- whether the failure reproduces with a software Vulkan implementation.

Remove credentials, personal paths, buffer contents, proprietary kernels, and private device data.
Prefer a short log around the failure over an unbounded trace.

## Concurrency invariants

- Level storage is atomic, so thresholds may be changed while worker threads are active.
- The enabled path serializes sink calls; the disabled path does not acquire the sink mutex.
- Macro arguments are evaluated only after the threshold check succeeds.
- Log order reflects entry into the sink critical section, not a global causal ordering of runtime
  operations.
- Logging must not be used as a synchronization primitive.
