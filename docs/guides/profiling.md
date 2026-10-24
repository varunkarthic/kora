# Profiling and tracing

## Telemetry levels

| `KORA_PROFILE` / `telemetry` | GPU timestamps | Host phase breakdown | Trace export |
| --- | --- | --- | --- |
| `off` | never | no | no |
| `basic` (default) | 1 dispatch in 16 | no | no |
| `profile` | every dispatch | yes | no |
| `trace` | every dispatch | yes | yes |

Timestamps need a queue with `timestampValidBits`; `profiling_stats().gpu_timestamps` reports whether
the device has them (for example, KosmicKrisp on Apple silicon does not).

## Per-device statistics

```rust
let stats = runtime.profiling_stats(&device)?;
println!("last kernel took {} ns on the GPU", stats.last_duration_nanoseconds);
```

With `profile` or `trace`, the stats also break the last dispatch into host phases: validation,
binding preflight, pipeline lookup, residency transfers, submission assembly, queue enqueue,
command-buffer acquisition, descriptor setup, recording, completion-thread delay, device wait,
timestamp collection, and the completion callback. `kora-bench --profile-phases` prints these as
`profile_kora_4096_*` rows.

`runtime.transfer_stats()` counts uploads and downloads; `memory_stats(&device)` reports pool usage.

## Chrome trace

```rust
let runtime = kora::Runtime::new(&kora::RuntimeOptions {
    telemetry: Some(kora::TelemetryLevel::Trace),
    ..Default::default()
})?;
// ... run work ...
std::fs::write("kora-trace.json", runtime.export_trace_json()?)?;
```

Open the file in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing`. Each completed dispatch is
a slice on its device's track (the last 4096 per device), and adaptive dispatches carry the scheduler
score that chose their device.

## Benchmarks

For comparisons, use [`kora-bench`](../tools/kora-bench.md) and the interleaving method in
[Performance methodology](../performance/methodology.md); single runs on one machine can vary by
several times with GPU clock state.
