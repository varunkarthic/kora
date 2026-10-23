//! Bounded startup calibration: submission overhead, transfer bandwidths, and compute probes.
//!
//! Probes use private scratch buffers, never application data, and stop at the time budget.
//! Calibration is best effort: a failed or skipped probe leaves its measurement at zero and never
//! fails runtime creation.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::context::{DeviceContext, lock};
use super::dispatch::DispatchBuilder;
use super::kernel;
use super::replica::Replica;
use crate::buffer::{Buffer, BufferManager, BufferUsage};
use crate::calibration::{CalibrationRecord, PROBE_SHADER};
use crate::event::{Event, EventPromise};
use crate::kernel::spirv::words_from_bytes;
use crate::metadata::{
    KernelAccess, KernelBindingKind, KernelBindingMetadata, KernelMetadata, PushConstantFieldMetadata,
};
use crate::vulkan::Submission;

const PROBE_ELEMENTS: u32 = 16 * 1024;
const PROBE_SEED: u32 = 0x9e37_79b9;
const ALU_ITERATIONS: u32 = 256;

fn waited(event: &Event, deadline: Instant) -> Option<u64> {
    let remaining = deadline.checked_duration_since(Instant::now())?;
    event.wait_timeout(remaining).ok()?;
    Some(0)
}

/// Runs `operation`, waits for its event within the deadline, and returns the elapsed time.
fn timed(deadline: Instant, operation: impl FnOnce() -> Option<Event>) -> Option<u64> {
    if Instant::now() >= deadline {
        return None;
    }
    let started = Instant::now();
    let event = operation()?;
    waited(&event, deadline)?;
    Some(
        u64::try_from(started.elapsed().as_nanos())
            .unwrap_or(u64::MAX)
            .max(1),
    )
}

fn rate(bytes: u64, nanoseconds: u64) -> u64 {
    let value = bytes as f64 * 1e9 / nanoseconds as f64;
    if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value as u64
    }
}

fn median(samples: &mut [u64]) -> u64 {
    samples.sort_unstable();
    samples.get(samples.len() / 2).copied().unwrap_or(0)
}

fn submission_overhead(ctx: &DeviceContext, deadline: Instant) -> u64 {
    let mut samples = Vec::with_capacity(3);
    for _ in 0..3 {
        let sample = timed(deadline, || {
            let promise = EventPromise::new();
            let event = promise.event();
            ctx.queue
                .submit(Submission {
                    on_complete: Some(Box::new(move |_, status| promise.complete(status))),
                    ..Submission::default()
                })
                .ok()?;
            Some(event)
        });
        match sample {
            Some(sample) => samples.push(sample),
            None => break,
        }
    }
    median(&mut samples)
}

fn transfer_probes(
    buffers: &BufferManager,
    ctx: &DeviceContext,
    deadline: Instant,
    record: &mut CalibrationRecord,
) {
    let size = 256 * 1024;
    let (Ok(first), Ok(second)) = (
        buffers.create(size, BufferUsage::Storage),
        buffers.create(size, BufferUsage::Storage),
    ) else {
        return;
    };
    let (Ok(a), Ok(b)) = (ctx.replica(&first), ctx.replica(&second)) else {
        return;
    };
    let input: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
    for bytes in [4096_u64, 64 * 1024, size] {
        let len = bytes as usize;
        let Some(upload) = timed(deadline, || {
            ctx.transfers.upload(&a.transfer_view(), 0, &input[..len]).ok()
        }) else {
            return;
        };
        record.h2d_bytes_per_second = record.h2d_bytes_per_second.max(rate(bytes, upload));
        let readback = |source: &Arc<Replica>| -> Option<(u64, bool)> {
            let output = Arc::new(Mutex::new(Vec::new()));
            let sink = {
                let output = Arc::clone(&output);
                move |data: &[u8]| {
                    *lock(&output) = data.to_vec();
                    Ok(())
                }
            };
            let elapsed = timed(deadline, || {
                ctx.transfers
                    .download(&source.transfer_view(), 0, bytes, None, sink)
                    .ok()
            })?;
            let matches = *lock(&output) == input[..len];
            Some((elapsed, matches))
        };
        let Some((download, true)) = readback(&a) else {
            return;
        };
        record.d2h_bytes_per_second = record.d2h_bytes_per_second.max(rate(bytes, download));
        let Some(copy) = timed(deadline, || {
            ctx.transfers
                .copy(&a.transfer_view(), 0, &b.transfer_view(), 0, bytes)
                .ok()
        }) else {
            return;
        };
        let Some((_, true)) = readback(&b) else {
            return;
        };
        record.device_copy_bytes_per_second = record.device_copy_bytes_per_second.max(rate(bytes, copy));
    }
}

fn probe_metadata() -> KernelMetadata {
    let binding = |binding, name: &str, access| KernelBindingMetadata {
        set: 0,
        binding,
        name: name.into(),
        kind: KernelBindingKind::StorageBuffer,
        access,
    };
    let field = |name: &str, offset| PushConstantFieldMetadata {
        name: name.into(),
        type_name: "u32".into(),
        offset,
        size: 4,
    };
    KernelMetadata {
        kernel_id: "kora.internal.calibration-probe.v1".into(),
        local_size: [64, 1, 1],
        bindings: vec![
            binding(0, "input_values", KernelAccess::Read),
            binding(1, "output_values", KernelAccess::Write),
        ],
        push_constant_size: 16,
        push_constants: vec![
            field("count", 0),
            field("iterations", 4),
            field("mode", 8),
            field("seed", 12),
        ],
        ..KernelMetadata::default()
    }
}

fn expected(input: u32, iterations: u32, mode: u32) -> u32 {
    match mode {
        0 => input ^ PROBE_SEED,
        2 => (0..iterations).fold(input, |value, _| {
            let value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            value ^ (value >> 13)
        }),
        _ => input,
    }
}

fn compute_probes(buffers: &BufferManager, ctx: &Arc<DeviceContext>, deadline: Instant) {
    let Ok(words) = words_from_bytes(PROBE_SHADER) else {
        return;
    };
    let Ok(probe) = kernel::load(ctx, &words, "main", probe_metadata()) else {
        return;
    };
    let create = |count| buffers.create_typed::<u32>(u64::from(count));
    let (Ok(input), Ok(output), Ok(tiny_input), Ok(tiny_output)) = (
        create(PROBE_ELEMENTS),
        create(PROBE_ELEMENTS),
        create(1),
        create(1),
    ) else {
        return;
    };
    let values: Vec<u32> = (0..PROBE_ELEMENTS)
        .map(|i| i.wrapping_mul(747_796_405).wrapping_add(2_891_336_453))
        .collect();
    if input.write(&values).is_err() || tiny_input.write(&values[..1]).is_err() {
        return;
    }
    let peers: Arc<[Arc<DeviceContext>]> = Arc::from([]);
    let run = |elements: u32, iterations: u32, mode: u32| -> Option<u64> {
        let (source, destination): (&Buffer<u32>, &Buffer<u32>) = if elements == 1 {
            (&tiny_input, &tiny_output)
        } else {
            (&input, &output)
        };
        let elapsed = timed(deadline, || {
            DispatchBuilder::new(probe.clone(), Arc::clone(&peers))
                .global(u64::from(elements))
                .read(source)
                .write(destination)
                .push(&[elements, iterations, mode, PROBE_SEED])
                .submit()
                .ok()
        })?;
        let produced = destination.read().ok()?;
        let correct = (0..elements as usize).all(|i| produced[i] == expected(values[i], iterations, mode));
        correct.then_some(elapsed)
    };

    run(1, 0, 0); // Warms the pipeline and the smallest dispatch path.
    let mut tiny: Vec<u64> = (0..3).map_while(|_| run(1, 0, 0)).collect();
    let mut record = lock(&ctx.calibration);
    if !tiny.is_empty() {
        record.tiny_dispatch_ns = median(&mut tiny);
    }
    drop(record);
    run(PROBE_ELEMENTS, 0, 1);
    if let Some(elapsed) = run(PROBE_ELEMENTS, 0, 1) {
        let bytes = u64::from(PROBE_ELEMENTS) * 4 * 2;
        lock(&ctx.calibration).streaming_bytes_per_second = rate(bytes, elapsed);
    }
    run(PROBE_ELEMENTS, ALU_ITERATIONS, 2);
    if let Some(elapsed) = run(PROBE_ELEMENTS, ALU_ITERATIONS, 2) {
        let operations = u64::from(PROBE_ELEMENTS) * u64::from(ALU_ITERATIONS) * 4;
        lock(&ctx.calibration).alu_operations_per_second = rate(operations, elapsed);
    }
}

/// Calibrates one device within `budget`. A valid cached record is reused unless `force`; compute
/// probes missing from a cached record are measured again.
pub(super) fn calibrate_device(
    buffers: &BufferManager,
    ctx: &Arc<DeviceContext>,
    budget: Duration,
    force: bool,
) {
    let Some(path) = ctx.calibration_path.clone() else {
        return;
    };
    let started = Instant::now();
    let hard_deadline = started + budget;
    let write_reserve = (budget / 5).min(Duration::from_millis(10));
    let compute_reserve = (budget / 2).min(Duration::from_millis(150));
    let compute_deadline = hard_deadline - write_reserve;
    let probe_deadline = compute_deadline - compute_reserve;
    let uuid = ctx.id.uuid.to_string();
    let driver = ctx.info.driver_version;
    let cached = (!force)
        .then(|| CalibrationRecord::read(&path, &uuid, driver))
        .flatten();
    let record = match cached {
        Some(record) if !record.compute_pending() => {
            *lock(&ctx.calibration) = record;
            return;
        }
        Some(record) => record,
        None => {
            let mut record = CalibrationRecord::new(uuid, driver);
            record.submit_overhead_ns = submission_overhead(ctx, probe_deadline);
            transfer_probes(buffers, ctx, probe_deadline, &mut record);
            record
        }
    };
    *lock(&ctx.calibration) = record;
    compute_probes(buffers, ctx, compute_deadline);
    // Persisting is best effort; an unwritable cache directory only costs a recalibration.
    let snapshot = lock(&ctx.calibration).clone();
    let _ = snapshot.write(&path);
}
