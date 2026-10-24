# Dispatch and synchronization

## Describe, submit, wait

```rust
let event = runtime
    .dispatch(&kernel)        // a DispatchBuilder for the kernel's device
    .global(n)                // or global_3d(x, y, z), in invocations (not workgroups)
    .read(&a)                 // bind buffers in manifest order, with matching access
    .read(&b)
    .write(&out)
    .push(&params)            // exactly push_constant_size bytes
    .priority(kora::DispatchPriority::High)   // optional scheduling hint
    .submit()?;               // returns as soon as the work is queued
event.wait()?;
```

`submit` validates everything that can be checked on the host (binding count and access, push size,
dispatch dimensions against device limits) and returns an error without queuing anything. Device
failures surface through the event.

## Ordering between dispatches

KORA tracks how each submission uses each buffer and records only the barriers that are needed:

| Earlier dispatch | Later dispatch | Result |
| --- | --- | --- |
| writes `x` | reads `x` | memory barrier; later waits for the write (RAW) |
| reads `x` | writes `x` | execution barrier (WAR) |
| writes `x` | writes `x` | memory barrier (WAW) |
| touches `x` | touches only `y` | no barrier; both may overlap |

So a chain of dependent dispatches on one device needs no waits in between; submit them all and wait
on the last event.

## Events

An `Event` is a cheap, cloneable handle to a completion.

| Rust | Python | C |
| --- | --- | --- |
| `wait()` | `wait()` (releases the GIL) | `kora_event_wait` |
| `wait_timeout(d)` → `Timeout` | `wait_for(seconds)` | `kora_event_wait_timeout` |
| `is_done()` | `done` | `kora_event_is_done` |
| `on_complete(f)` | `await event` | — |
| `Event::when_all(&[...])` | `asyncio.gather(*events)` | — |
| `then(f)` | — | — |

A callback registered on a completed event runs immediately on the calling thread; otherwise it runs
on a KORA completion thread, so keep it short. In Python, `await event` resolves through
`loop.call_soon_threadsafe` without blocking a thread per await.

## Deferred readback

`defer_readback()` keeps written buffers on the device; `runtime.readback(&buffer)` copies them back
later. See [Keeping results on the device](memory.md#keeping-results-on-the-device).

## Duplicate verification

For results you want checked across devices, `verify_with` runs the dispatch on two devices and
compares their outputs before committing:

```rust
use kora::{VerificationPolicy, VerificationDataType};

runtime.dispatch(&kernel).global(n).read(&x).write(&y)
    .verify_with(VerificationPolicy::Ulp(4), VerificationDataType::Float32)
    .submit()?
    .wait()?;   // ValidationError if the devices disagree
```

Policies are `Exact`, `AbsoluteTolerance(t)`, `RelativeTolerance(t)`, `Ulp(n)`, and `Custom(f)`.
A mismatch quarantines both devices for later adaptive placement and marks the outputs lost. It needs
a runtime created with `DevicePolicy::AllEligible` and at least two devices that support the kernel.

## Threads

`Runtime`, `Kernel`, `Buffer`, and `Event` are `Send + Sync`. Submitting from several threads is
safe; each device serializes recording internally and keeps a dedicated submission thread and
completion thread, so `submit` does not wait for the GPU.
