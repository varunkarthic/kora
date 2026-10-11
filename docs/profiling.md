# GPU timestamp profiling

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

`RuntimeProfilingStats` reports cumulative collected and dropped counts, the latest submission ID and
duration, plus calibrated host time and deviation when the relevant domains are supported. Calling
the API may collect ready samples but does not wait for pending submissions. A device not owned by
the runtime returns `InvalidArgument`.

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
