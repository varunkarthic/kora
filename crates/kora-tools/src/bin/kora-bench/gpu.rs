//! KORA Runtime GPU benchmarks: dispatch overhead, vector add across sizes, the matched compute
//! workloads, device timestamps, and optional per-phase dispatch profiling.

use std::time::Instant;

use kora::kernel::spirv::words_from_bytes;
use kora::log::Level;
use kora::{
    Buffer, DeviceId, Kernel, KernelAccess, KernelBindingKind, KernelBindingMetadata, KernelMetadata,
    PushConstantFieldMetadata, Runtime, RuntimeOptions, RuntimeProfilingStats, TelemetryLevel,
};

use crate::report::{
    Failure, GpuDevice, Measurement, Options, Outcome, failed, run, run_checked, run_timestamps, unavailable,
};

pub const VECTOR_ADD: &[u8] = include_bytes!("../../../../kora/tests/fixtures/vector_add.spv");
pub const SAXPY: &[u8] = include_bytes!("../../../../kora/tests/fixtures/saxpy.spv");
pub const REDUCTION: &[u8] = include_bytes!("../../../../kora/tests/fixtures/reduction.spv");
pub const SCAN: &[u8] = include_bytes!("../../../../kora/tests/fixtures/scan.spv");
pub const STENCIL: &[u8] = include_bytes!("../../../../kora/tests/fixtures/stencil.spv");
pub const TILED_MATMUL: &[u8] = include_bytes!("../../../../kora/tests/fixtures/tiled_matmul.spv");

pub const COMPUTE_SIZES: [u32; 3] = [4096, 65536, 1_048_576];
pub const VECTOR_SIZES: [u32; 8] = [64, 256, 1024, 4096, 65536, 262_144, 1_048_576, 4_194_240];
pub const MATRIX_DIMENSIONS: [u32; 3] = [16, 64, 128];

fn field(name: &str, type_name: &str, offset: u32) -> PushConstantFieldMetadata {
    PushConstantFieldMetadata {
        name: name.into(),
        type_name: type_name.into(),
        offset,
        size: 4,
    }
}

fn metadata(
    kernel_id: &str,
    local_size: [u32; 3],
    push: Vec<PushConstantFieldMetadata>,
    bindings: &[(&str, KernelAccess)],
) -> KernelMetadata {
    KernelMetadata {
        kernel_id: kernel_id.into(),
        local_size,
        push_constant_size: 4 * push.len() as u32,
        push_constants: push,
        bindings: bindings
            .iter()
            .enumerate()
            .map(|(index, (name, access))| KernelBindingMetadata {
                set: 0,
                binding: index as u32,
                name: (*name).into(),
                kind: KernelBindingKind::StorageBuffer,
                access: *access,
            })
            .collect(),
        ..KernelMetadata::default()
    }
}

pub fn saxpy_metadata() -> KernelMetadata {
    metadata(
        "kora_bench.saxpy",
        [64, 1, 1],
        vec![field("scale", "f32", 0), field("count", "u32", 4)],
        &[
            ("input_x", KernelAccess::Read),
            ("input_y", KernelAccess::ReadWrite),
        ],
    )
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SaxpyParams {
    pub scale: f32,
    pub count: u32,
}

fn load(runtime: &Runtime, words: &[u8], metadata: KernelMetadata) -> Outcome<Kernel> {
    let words = words_from_bytes(words).map_err(unavailable("GPU benchmarks unavailable"))?;
    runtime
        .load_kernel(&words, "main", metadata)
        .map_err(unavailable("GPU benchmarks unavailable"))
}

fn buffer<T: kora::bytemuck::Pod>(runtime: &Runtime, values: &[T]) -> Outcome<Buffer<T>> {
    let buffer = runtime
        .create_buffer::<T>(values.len() as u64)
        .map_err(failed("GPU buffer creation"))?;
    buffer.write(values).map_err(failed("GPU buffer upload"))?;
    Ok(buffer)
}

/// Panics are not used for control flow: dispatch failures are recorded and reported after the
/// timing loop (each timed closure must be infallible).
struct Errors(Option<kora::Error>);

impl Errors {
    fn note(&mut self, status: kora::Status) {
        if let Err(error) = status {
            self.0.get_or_insert(error);
        }
    }

    fn check(&mut self, context: &str) -> Outcome {
        match self.0.take() {
            Some(error) => Err(Failure::Failed(format!("{context}: {error}"))),
            None => Ok(()),
        }
    }
}

/// Basic telemetry timestamps a sample of dispatches, so repeat the operation until a new
/// device-time sample arrives; each sample is still one dispatch's interval.
fn sample_timestamp(runtime: &Runtime, device: &DeviceId, operation: &mut dyn FnMut()) -> Option<u64> {
    let before = runtime.profiling_stats(device).ok()?;
    for _ in 0..32 {
        operation();
        let after = runtime.profiling_stats(device).ok()?;
        if after.last_submission_id != before.last_submission_id {
            return Some(after.last_duration_nanoseconds);
        }
    }
    None
}

/// The matched compute workloads: a dispatch with readback, one with deferred readback (both
/// validated), and device timestamps when the device has them.
#[allow(clippy::too_many_arguments)]
fn add_workload(
    results: &mut Vec<Measurement>,
    options: &Options,
    runtime: &Runtime,
    device: &DeviceId,
    name: &str,
    errors: &mut Errors,
    mut dispatch: impl FnMut(bool) -> kora::Status,
    mut validate: impl FnMut() -> bool,
) -> Outcome {
    errors.note(dispatch(false));
    errors.check(name)?;
    if !validate() {
        return Err(Failure::Failed(format!(
            "benchmark correctness check failed: {name}"
        )));
    }
    let errors = std::cell::RefCell::new(errors);
    let dispatch = std::cell::RefCell::new(&mut dispatch);
    let run_one = |defer: bool| {
        let status = (dispatch.borrow_mut())(defer);
        errors.borrow_mut().note(status);
    };
    results.push(run_checked(
        format!("{name}_dispatch_wait"),
        options,
        || run_one(false),
        &mut validate,
    )?);
    results.push(run_checked(
        format!("{name}_deferred_readback_dispatch_wait"),
        options,
        || run_one(true),
        &mut validate,
    )?);
    if let Some(samples) = run_timestamps(format!("{name}_gpu_execution"), options, || {
        sample_timestamp(runtime, device, &mut || run_one(false))
    }) {
        results.push(samples);
    }
    errors.borrow_mut().check(name)
}

/// Returns the Runtime's first device so raw Vulkan measurements can use the same one.
pub fn add_gpu_benchmarks(
    results: &mut Vec<Measurement>,
    options: &Options,
    report: &mut GpuDevice,
) -> Outcome<(Runtime, DeviceId)> {
    let runtime = Runtime::new(&RuntimeOptions {
        telemetry: Some(if options.profile_phases {
            TelemetryLevel::Profile
        } else {
            TelemetryLevel::Basic
        }),
        enable_startup_calibration: Some(false),
        disable_autotune: Some(true),
        debug_safety: Some(false),
        enable_validation: Some(false),
        log_level: Some(Level::Off),
        ..RuntimeOptions::default()
    })
    .map_err(unavailable("GPU benchmarks unavailable"))?;
    let device = runtime.devices().into_iter().next().ok_or_else(|| {
        Failure::Unavailable("GPU benchmarks unavailable: Runtime selected no devices".into())
    })?;
    let mut errors = Errors(None);

    let vector_add = load(
        &runtime,
        VECTOR_ADD,
        metadata(
            "kora_bench.vector_add",
            [64, 1, 1],
            vec![field("count", "u32", 0)],
            &[
                ("input_a", KernelAccess::Read),
                ("input_b", KernelAccess::Read),
                ("output_data", KernelAccess::Write),
            ],
        ),
    )?;
    let largest = VECTOR_SIZES[VECTOR_SIZES.len() - 1] as usize;
    let a = buffer(&runtime, &vec![3_u32; largest])?;
    let b = buffer(&runtime, &vec![7_u32; largest])?;
    let output = buffer(&runtime, &vec![0_u32; largest])?;
    let dispatch = |count: u32, defer: bool| -> kora::Status {
        let mut builder = runtime
            .dispatch(&vector_add)
            .global(u64::from(count))
            .read(&a)
            .read(&b)
            .write(&output)
            .push(&count);
        if defer {
            builder = builder.defer_readback();
        }
        builder.submit()?.wait()
    };
    let sums_ok = |count: u32| {
        output
            .read()
            .is_ok_and(|v| v[..count as usize].iter().all(|x| *x == 10))
    };
    results.push(run("gpu_tiny_dispatch_1", options, || {
        errors.note(dispatch(1, false))
    }));
    let mut batch = || {
        for _ in 0..10 {
            errors.note(dispatch(1, false));
        }
    };
    batch();
    results.push(run("gpu_tiny_dispatch_batch_10", options, batch));
    errors.check("tiny dispatch")?;
    for count in VECTOR_SIZES {
        errors.note(dispatch(count, false));
        errors.check("vector add")?;
        if !sums_ok(count) {
            return Err(Failure::Failed(format!(
                "gpu_vector_add_{count} produced wrong sums"
            )));
        }
        results.push(run(
            format!("gpu_vector_add_{count}_dispatch_wait"),
            options,
            || {
                errors.note(dispatch(count, false));
            },
        ));
        results.push(run(
            format!("gpu_vector_add_{count}_deferred_readback_dispatch_wait"),
            options,
            || errors.note(dispatch(count, true)),
        ));
        if let Some(samples) =
            run_timestamps(format!("gpu_vector_add_{count}_gpu_execution"), options, || {
                sample_timestamp(&runtime, &device, &mut || errors.note(dispatch(count, true)))
            })
        {
            report.kora_timestamp_sizes.push(count);
            results.push(samples);
        }
        errors.check("vector add")?;
    }

    if options.profile_phases {
        add_phase_profile(results, options, &runtime, &device, &mut || {
            runtime
                .dispatch(&vector_add)
                .global(4096)
                .read(&a)
                .read(&b)
                .write(&output)
                .push(&4096_u32)
                .defer_readback()
        });
    }

    let saxpy = load(&runtime, SAXPY, saxpy_metadata())?;
    for count in COMPUTE_SIZES {
        let n = count as usize;
        let x = buffer(&runtime, &vec![2.0_f32; n])?;
        let y = buffer(&runtime, &vec![1.0_f32; n])?;
        let initial = vec![1.0_f32; n];
        add_workload(
            results,
            options,
            &runtime,
            &device,
            &format!("gpu_saxpy_{count}_upload"),
            &mut errors,
            |defer| {
                y.write(&initial)?;
                let mut builder = runtime
                    .dispatch(&saxpy)
                    .global(u64::from(count))
                    .read(&x)
                    .read_write(&y)
                    .push(&SaxpyParams { scale: 2.0, count });
                if defer {
                    builder = builder.defer_readback();
                }
                builder.submit()?.wait()
            },
            || {
                runtime.readback(&y).is_ok_and(|e| e.wait().is_ok())
                    && y.read().is_ok_and(|v| v.iter().all(|x| *x == 5.0))
            },
        )?;
    }

    let two = |id, words, count_name: &str, output_access| -> Outcome<Kernel> {
        load(
            &runtime,
            words,
            metadata(
                id,
                [64, 1, 1],
                vec![field(count_name, "u32", 0)],
                &[("input_data", KernelAccess::Read), ("output_data", output_access)],
            ),
        )
    };
    let reduction = two("kora_bench.reduction", REDUCTION, "count", KernelAccess::Write)?;
    let scan = two("kora_bench.scan_tile64", SCAN, "count", KernelAccess::Write)?;
    let stencil = two("kora_bench.stencil", STENCIL, "count", KernelAccess::Write)?;
    for count in COMPUTE_SIZES {
        let n = count as usize;
        let simple =
            |kernel: &Kernel, input: &Buffer<f32>, output: &Buffer<f32>, defer: bool| -> kora::Status {
                let mut builder = runtime
                    .dispatch(kernel)
                    .global(u64::from(count))
                    .read(input)
                    .write(output)
                    .push(&count);
                if defer {
                    builder = builder.defer_readback();
                }
                builder.submit()?.wait()
            };
        let ones = buffer(&runtime, &vec![1.0_f32; n])?;
        let reduced = buffer(&runtime, &vec![0.0_f32; n / 64])?;
        add_workload(
            results,
            options,
            &runtime,
            &device,
            &format!("gpu_reduction_{count}"),
            &mut errors,
            |defer| simple(&reduction, &ones, &reduced, defer),
            || {
                runtime.readback(&reduced).is_ok_and(|e| e.wait().is_ok())
                    && reduced.read().is_ok_and(|v| v.iter().all(|x| *x == 64.0))
            },
        )?;

        let scan_input = buffer(&runtime, &vec![1_u32; n])?;
        let scanned = buffer(&runtime, &vec![0_u32; n])?;
        add_workload(
            results,
            options,
            &runtime,
            &device,
            &format!("gpu_tiled_scan_{count}"),
            &mut errors,
            |defer| {
                let mut builder = runtime
                    .dispatch(&scan)
                    .global(u64::from(count))
                    .read(&scan_input)
                    .write(&scanned)
                    .push(&count);
                if defer {
                    builder = builder.defer_readback();
                }
                builder.submit()?.wait()
            },
            || {
                runtime.readback(&scanned).is_ok_and(|e| e.wait().is_ok())
                    && scanned
                        .read()
                        .is_ok_and(|v| v.iter().enumerate().all(|(i, x)| *x == (i as u32 % 64) + 1))
            },
        )?;

        let stencil_output = buffer(&runtime, &vec![0.0_f32; n])?;
        add_workload(
            results,
            options,
            &runtime,
            &device,
            &format!("gpu_stencil_{count}"),
            &mut errors,
            |defer| simple(&stencil, &ones, &stencil_output, defer),
            || {
                runtime.readback(&stencil_output).is_ok_and(|e| e.wait().is_ok())
                    && stencil_output.read().is_ok_and(|v| {
                        v[0] == 2.0 && v[n - 1] == 2.0 && v[1..n - 1].iter().all(|x| *x == 3.0)
                    })
            },
        )?;
    }

    let matmul = load(
        &runtime,
        TILED_MATMUL,
        metadata(
            "kora_bench.tiled_matmul",
            [8, 8, 1],
            vec![field("dimension", "u32", 0)],
            &[
                ("matrix_a", KernelAccess::Read),
                ("matrix_b", KernelAccess::Read),
                ("matrix_c", KernelAccess::Write),
            ],
        ),
    )?;
    for dimension in MATRIX_DIMENSIONS {
        let elements = (dimension * dimension) as usize;
        let ma = buffer(&runtime, &vec![1.0_f32; elements])?;
        let mb = buffer(&runtime, &vec![1.0_f32; elements])?;
        let mc = buffer(&runtime, &vec![0.0_f32; elements])?;
        add_workload(
            results,
            options,
            &runtime,
            &device,
            &format!("gpu_tiled_matmul_{dimension}"),
            &mut errors,
            |defer| {
                let mut builder = runtime
                    .dispatch(&matmul)
                    .global_3d(u64::from(dimension), u64::from(dimension), 1)
                    .read(&ma)
                    .read(&mb)
                    .write(&mc)
                    .push(&dimension);
                if defer {
                    builder = builder.defer_readback();
                }
                builder.submit()?.wait()
            },
            || {
                runtime.readback(&mc).is_ok_and(|e| e.wait().is_ok())
                    && mc.read().is_ok_and(|v| v.iter().all(|x| *x == dimension as f32))
            },
        )?;
    }
    Ok((runtime, device))
}

const PHASES: [&str; 18] = [
    "builder_setup",
    "submit_call",
    "event_wait",
    "dispatch_validation",
    "binding_preflight",
    "pipeline_lookup",
    "submission_preparation",
    "buffer_residency_transfer",
    "submission_assembly",
    "queue_enqueue",
    "command_buffer_acquire",
    "descriptor_setup",
    "command_recording",
    "queue_completion",
    "completion_worker_delay",
    "device_completion_wait",
    "timestamp_collection",
    "completion_callback",
];

fn phase_values(builder: u64, submit: u64, wait: u64, s: &RuntimeProfilingStats) -> [u64; 18] {
    [
        builder,
        submit,
        wait,
        s.last_dispatch_validation_nanoseconds,
        s.last_binding_preflight_nanoseconds,
        s.last_pipeline_lookup_nanoseconds,
        s.last_host_submission_nanoseconds,
        s.last_buffer_residency_nanoseconds,
        s.last_submission_assembly_nanoseconds,
        s.last_queue_enqueue_nanoseconds,
        s.last_command_buffer_acquire_nanoseconds,
        s.last_descriptor_setup_nanoseconds,
        s.last_command_recording_nanoseconds,
        s.last_queue_completion_nanoseconds,
        s.last_completion_worker_delay_nanoseconds,
        s.last_device_completion_wait_nanoseconds,
        s.last_timestamp_collection_nanoseconds,
        s.last_completion_callback_nanoseconds,
    ]
}

/// Telemetry-instrumented 4096-element dispatches split into host phases (diagnostic only).
fn add_phase_profile(
    results: &mut Vec<Measurement>,
    options: &Options,
    runtime: &Runtime,
    device: &DeviceId,
    make: &mut dyn FnMut() -> kora::DispatchBuilder,
) {
    let mut profile = || -> Option<[u64; 18]> {
        let before = runtime.profiling_stats(device).ok()?;
        let started = Instant::now();
        let builder = make();
        let builder_ns = started.elapsed().as_nanos() as u64;
        let started = Instant::now();
        let event = builder.submit().ok()?;
        let submit_ns = started.elapsed().as_nanos() as u64;
        let started = Instant::now();
        event.wait().ok()?;
        let wait_ns = started.elapsed().as_nanos() as u64;
        let stats = runtime.profiling_stats(device).ok()?;
        (stats.last_profile_submission_id != before.last_profile_submission_id)
            .then(|| phase_values(builder_ns, submit_ns, wait_ns, &stats))
    };
    for _ in 0..options.warmup {
        if profile().is_none() {
            eprintln!("KORA dispatch phase profiling unavailable on the selected device");
            return;
        }
    }
    let mut samples: [Vec<u64>; 18] = Default::default();
    for _ in 0..options.iterations {
        let Some(values) = profile() else {
            eprintln!("KORA dispatch phase profiling stopped: timestamp sample unavailable");
            return;
        };
        for (phase, value) in values.into_iter().enumerate() {
            samples[phase].push(value);
        }
    }
    for (phase, values) in samples.into_iter().enumerate() {
        results.push(Measurement::from_samples(
            format!("profile_kora_4096_{}", PHASES[phase]),
            values,
        ));
    }
}
