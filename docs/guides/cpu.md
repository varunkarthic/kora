# CPU executor and fallbacks

## The executor

`CpuExecutor` is a fixed-size work-stealing thread pool with three priority lanes. Tasks return a
`Status`, and each submission returns an `Event`.

```rust
use kora::{CpuExecutor, CpuExecutorOptions, CpuTaskPriority};

let executor = CpuExecutor::new(CpuExecutorOptions { worker_count: 4, ..Default::default() })?;
let event = executor.submit(CpuTaskPriority::High, || {
    // work
    Ok(())
})?;
executor.wait(&event)?;
```

A panicking task completes its event with `InternalInvariantViolation` instead of tearing down the
pool. Shutdown drains queued work (or cancels it with `ShutdownMode::CancelPending`); tasks already
running may still submit follow-up work while draining.

## CPU fallbacks

Register a CPU implementation of a kernel, and run it on the runtime's executor with the same
bindings and push constants as a dispatch:

```rust
use kora::KernelAccess;

runtime.register_cpu_fallback("vector.add", |ctx| {
    let n = ctx.push_as::<u32>()? as usize;
    let a = ctx.read_as::<u32>(0)?.to_vec();
    let b = ctx.read_as::<u32>(1)?.to_vec();
    let out: &mut [u32] = kora::bytemuck::cast_slice_mut(ctx.write(2)?);
    for i in 0..n {
        out[i] = a[i] + b[i];
    }
    Ok(())
})?;
runtime
    .run_cpu_fallback("vector.add", &[(a.handle(), KernelAccess::Read), (b.handle(), KernelAccess::Read),
        (sums.handle(), KernelAccess::Write)], kora::bytemuck::bytes_of(&(n as u32)))?
    .wait()?;
```

Readable buffers are read back from any device first, so a fallback always sees current data. A
kernel manifest can name its fallback in `cpu_fallback_key`.
