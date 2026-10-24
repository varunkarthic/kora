# Recovery and device loss

## What happens when a device is lost

The first `VK_ERROR_DEVICE_LOST` from a submission or completion wait marks that device unhealthy.
From then on:

- the failing submission and everything queued behind it complete with `DeviceLost` (the error keeps
  the raw `VkResult`);
- later submissions to that device are rejected before any Vulkan call;
- each buffer written by an in-flight dispatch is marked **lost**: reading it returns `DataLoss`
  instead of stale bytes, until a complete `write` replaces its contents;
- buffers whose host copy was authoritative stay readable.

A `Timeout` from `wait_timeout` is only your deadline: it does not cancel work or mark the device
unhealthy.

## Recover

```rust
let status = runtime.dispatch(&kernel).global(n).read(&x).write(&y).submit()?.wait();
if let Err(error) = status && error.code() == kora::ErrorCode::DeviceLost {
    let report = runtime.device_fault_report(&runtime.devices()[0])?;   // query before recovering
    runtime.recover()?;                       // needs &mut Runtime
    let kernel = runtime.load_kernel(&words, "main", metadata.clone())?;  // kernels must be reloaded
    // resubmit work whose inputs are still authoritative
}
```

`recover` recreates every device from the original options. It is transactional: if creation fails,
the runtime is unchanged. Buffers and CPU fallbacks survive; their contents are re-uploaded lazily.
Results that only the lost device held (outputs of deferred dispatches never read back) are marked
lost rather than reverting silently to older host contents. KORA does not resubmit work for you,
because it cannot know which of your tasks are safe to repeat.

`rebuild_devices` performs the same rebuild without a loss, picking up devices that appeared or
disappeared. Call it only when no work is in flight. A `Cluster` does both for you (see
[failover](cluster.md#failover-and-hot-swap)).

## Device-fault reports

When the device supports `VK_EXT_device_fault`, KORA captures its description and bounded address and
vendor records at the first loss. `device_fault_report` returns them (`supported == false` otherwise).
Reports are never logged or persisted, and vendor binaries are not captured.

## Partial initialization

With `DevicePolicy::AllEligible`, a device that fails to initialize is quarantined and the others are
used; `quarantined_devices()` lists them. If every device fails, creation returns the first error.

## Testing your recovery path

`runtime.simulate_device_loss(&device)` marks a device lost exactly as a real loss would, and
`kora::vulkan::FaultInjector` can fail specific Vulkan calls deterministically in tests.
