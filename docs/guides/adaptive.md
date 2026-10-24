# Adaptive placement and calibration

By default you decide where each dispatch runs. `dispatch_adaptive` lets KORA choose among the
runtime's devices for one dispatch at a time.

## Calibration

At startup each device runs a short, bounded set of probes (submission latency, host-to-device and
device-to-host bandwidth, and compute throughput), 250 ms per device by default. Results are cached
per device and driver version in the cache directory, so later starts skip the probes; a corrupt
cache is rebuilt. `runtime.calibrate()` forces a refresh and `runtime.calibration(&device)` returns
the record.

| Setting | Variable | Default |
| --- | --- | --- |
| Run startup calibration | `KORA_STARTUP_CALIBRATION` | `true` |
| Budget per device | `KORA_CALIBRATION_BUDGET_MS` | `250` |

## Adaptive dispatch

```rust
use kora::{AdaptiveDispatchWorkload, AdaptiveKernelWorkKind, DevicePolicy, RuntimeOptions};

let runtime = kora::Runtime::new(&RuntimeOptions {
    device_policy: Some(DevicePolicy::AllEligible),
    ..RuntimeOptions::default()
})?;
let workload = AdaptiveDispatchWorkload {
    input_bytes: 8 * n,
    output_bytes: 4 * n,
    kernel_work: n,
    kernel_kind: AdaptiveKernelWorkKind::StreamingBytes,
    ..AdaptiveDispatchWorkload::default()
};
runtime.dispatch_adaptive(&kernel, &workload)?
    .global(n).read(&a).read(&b).write(&out).push(&(n as u32))
    .submit()?
    .wait()?;
```

KORA scores every eligible device that supports the kernel: predicted transfer time from calibrated
bandwidth, execution time from calibrated throughput (or from confident online observations of this
kernel and size class), memory pressure, and locality. A dispatch whose inputs already live on a
device prefers that device rather than moving them. The choice is deterministic for a given state.
With trace telemetry, each adaptive dispatch records its score in the [trace](profiling.md).

The kernel can be loaded on any device; KORA loads it on the chosen device as needed. Without
`AllEligible`, adaptive dispatch has a single candidate.
