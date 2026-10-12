# Telemetry and GPU timestamp profiling

`RuntimeOptions::telemetry` and `KORA_PROFILE` select `Off`, `Basic`, `Profile`, or `Trace` in
increasing detail. The default is `Basic`; `KORA_TRACE=true` is a compatibility shorthand for
`Trace`. `Runtime::profiling_stats()` provides completed timestamp counters and the latest sample.
Trace export is explicitly opt-in: create the runtime with `TelemetryLevel::Trace`, then call
`Runtime::export_trace_json()` after dispatch work. The returned document uses Chrome Trace Event
JSON (`traceEvents`, complete `X` events) and can be opened directly in Perfetto. It includes GPU
dispatch duration, submission ID, queue timeline value, device index, and calibrated monotonic host
timestamp/deviation when supported. When timestamp calibration is unavailable, timestamps are zero
and `host_clock_calibrated` is false; GPU duration remains useful but cannot be accurately placed on
the host timeline.

```cpp
kora::RuntimeOptions options;
options.telemetry = kora::TelemetryLevel::Trace;
auto runtime = kora::Runtime::create(options);
// Submit and wait for work, then:
auto trace = runtime.value().export_trace_json();
if (trace) {
    std::ofstream("kora-trace.json") << trace.value();
}
```

`export_trace_json()` polls only completed queue work and never waits for pending GPU submissions.
The retained history is bounded to the newest 4096 completed samples per device. Exported traces
contain operational timing metadata only; KORA does not persist them. Adaptive Runtime dispatch records
include the actual selected candidate's `ScoreBreakdown`, associated with the GPU event by submission
ID. Score components are in nanoseconds and include the predicted finish estimate. Ordinary explicitly
bound dispatches have no scheduler score, and cold-start fallback that lacks a complete calibration
record also omits the score rather than inventing one. See [scheduler cost models](scheduler.md) for
the meaning of the terms.

## Per-function reference

| API | Behavior |
| --- | --- |
| `resolve(options, env)` | Applies API-over-environment-over-default precedence for telemetry level. |
| `Runtime::profiling_stats(device)` | Polls and returns cumulative counts and latest completed GPU timestamp data; no waiting. |
| `Runtime::export_trace_json()` | At Trace level, polls all owned devices and returns bounded Chrome Trace Event JSON. |
| `TelemetryLevel::Off/Basic/Profile/Trace` | Selects requested telemetry detail; trace-file export requires `Trace`. |

KORA records optional compute-dispatch durations with Vulkan timestamp queries. Profiling is
best-effort: devices or queue families without timestamp support continue to execute normally, and
`Runtime::profiling_stats()` reports `UnsupportedFeature` when no query ring could be created.

## Collection model

Each runtime compute queue owns a fixed ring of 128 dispatch samples. A sample records timestamps at
the beginning and end of the dispatch command buffer, associates them with that queue submission's
completion value, and is readable only after the queue timeline reaches that value. Query results
are polled without `VK_QUERY_RESULT_WAIT_BIT`; pending results remain in the ring and polling never
waits for GPU work. If all ring slots are occupied, new samples are dropped and counted rather than
stalling submission.

Timestamp ticks are converted using `DeviceInfo::timestamp_period_nanoseconds()`. Delta arithmetic
uses the selected compute queue family's `timestamp_valid_bits`, including wraparound for widths
below 64 bits. Invalid periods and overflow return structured errors. Dispatch success is independent
of sample collection.

Completed measurements from opt-in adaptive dispatch are persisted for later placement decisions when
timestamp queries are available. Their identity includes an opaque device UUID, driver/schema versions,
SPIR-V and entry-point fingerprint, explicit workload class, work-size bucket, and timing domain. The local
bounded database applies outlier rejection and age expiry; no kernel source or device name is persisted.
GPU execution duration is kept separate from CPU-side dispatch preparation, blocking input-upload time,
and queue-enqueue-to-completion overhead (with GPU execution subtracted). Queue completion overhead also
includes host completion service and is not a pure hardware queue-wait measurement. Placement currently
uses fresh GPU execution observations; other recorded domains remain separate from the calibration model.
Adaptive database snapshots are coalesced to the first sample, the five-sample confidence threshold, and
then every eighth sample to limit completion-thread filesystem work.

## Host correlation

When the calibrated-timestamps extension is enabled and both device and monotonic host domains were
reported, KORA samples a correlated host timestamp at command recording time. The sample also
preserves Vulkan's maximum deviation. If either domain or the extension entry point is unavailable,
host timestamp and deviation are zero; GPU duration profiling remains available.

## API

After dispatch events complete, poll the runtime for completed measurements:

```cpp
auto stats = runtime.profiling_stats(device_id);
if (stats) {
    // stats.value().samples_collected is cumulative for this runtime/device.
    // last_duration_nanoseconds is the most recently collected dispatch duration.
    // dropped_samples counts samples omitted because the bounded ring was full.
}
```

`RuntimeProfilingStats` reports cumulative collected and dropped counts, the latest GPU-timestamped
submission ID and duration, calibrated host time and deviation when supported, and the separate
`last_profile_submission_id` plus host-side phase durations for the latest completed profile dispatch.
The phase fields separate per-dispatch metadata, argument, and limit validation; binding preflight;
pipeline lookup; Runtime submit preparation; buffer-residency and transfer checks; submission assembly; queue enqueue; command-buffer
acquisition; descriptor allocation/update; command encoding; and queue completion. They are diagnostic measurements
collected only when telemetry is set to `Profile` or `Trace`; they are not zero-overhead timers and
must not be mixed with ordinary Release benchmark runs. Runtime submit preparation includes residency
checks, transfers, and host-side submission assembly. It excludes synchronous upload wait time, which
has its own transfer observation. SPIR-V parsing/reflection in `load_kernel()` is outside the
steady-state dispatch and is not included. `submission_preparation` is an inclusive host phase; the
preflight, pipeline, residency, and assembly fields help explain it and should not be added to it.
On an exact pipeline-cache hit for a kernel with a static local size, the Runtime can use the
previously validated pipeline directly. A cache miss still runs SPIR-V reflection and full validation;
`LocalSizeId` specialization also retains full validation because specialization affects the resolved
pipeline key.
Queue completion is measured from the pre-enqueue point until the completion callback (so it includes
queue enqueue) and excludes default output readback performed after GPU completion. Additional
completion diagnostics separate completion-worker wake delay, the Vulkan timeline/fence wait,
timestamp-query collection, and Runtime callback work. Builder
construction can be timed by the caller around `Runtime::dispatch()` and builder configuration.
Calling the API may collect ready samples but does not wait for pending
submissions. When phase profiling is enabled but the device cannot provide GPU timestamps, the method
still returns host phases with zero GPU sample counts and duration. Without phase profiling and without
a timestamp ring, it returns `UnsupportedFeature`. A device not owned by the runtime returns
`InvalidArgument`.

## Invariants and limitations

- Query indices are reused only after their previous slot has been collected.
- Only the queue owner records query commands; host collection is mutex-protected.
- Timeline completion is checked before result polling, and query reads omit the wait flag.
- Coalesced submissions retain independent sample IDs while sharing their queue completion value.
- The measurement encloses the GPU command-buffer dispatch region, not host queue latency or readback.
- Zero timestamp-valid bits, missing query entry points, or query-pool creation failure disable
  profiling rather than failing runtime startup.

## Verification

```sh
cmake --preset debug
cmake --build --preset debug --parallel
build/debug/tests/unit/kora_unit_tests --gtest_filter='GpuTimestamp*'
ctest --preset debug --output-on-failure
```

The focused tests validate valid-bit wraparound, tick conversion/overflow, completed-only polling,
and no-wait query flags. The runtime integration dispatch checks for a positive plausible duration on
devices that support timestamp queries and accepts the documented unsupported-feature fallback.
