//! Runtime end to end against real Vulkan devices: kernels, dispatch residency, readback, CPU
//! fallbacks, telemetry, and failure paths. Every test runs under the validation layer when it is
//! installed and fails on any validation error.

mod common;

use std::sync::Barrier;

use kora::kernel::spirv::words_from_bytes;
use kora::{
    DevicePolicy, DispatchPriority, ErrorCode, KernelAccess, KernelBindingKind, KernelBindingMetadata,
    KernelMetadata, PushConstantFieldMetadata, Runtime, RuntimeOptions, ShutdownMode,
    SpecializationConstantMetadata, SpecializationValue, TelemetryLevel,
};

const VECTOR_ADD: &[u8] = include_bytes!("fixtures/vector_add.spv");
const SAXPY: &[u8] = include_bytes!("fixtures/saxpy.spv");
const AUTOTUNE: &[u8] = include_bytes!("fixtures/autotune.spv");

fn cache_dir(name: &str) -> String {
    std::env::temp_dir()
        .join(format!("kora-runtime-test-{}-{name}", std::process::id()))
        .to_string_lossy()
        .into_owned()
}

fn options(name: &str) -> RuntimeOptions {
    RuntimeOptions {
        enable_validation: Some(true),
        cache_dir: Some(cache_dir(name)),
        enable_startup_calibration: Some(false),
        ..RuntimeOptions::default()
    }
}

/// A runtime, or `None` (after printing why) on machines without a usable Vulkan device.
fn runtime(options: &RuntimeOptions) -> Option<Runtime> {
    match Runtime::new(options) {
        Ok(runtime) => Some(runtime),
        Err(error)
            if matches!(
                error.code(),
                ErrorCode::NoVulkanImplementation | ErrorCode::NoEligibleDevice
            ) =>
        {
            eprintln!("skipping: {error}");
            None
        }
        Err(error) => panic!("runtime creation failed: {error}"),
    }
}

/// Runs `test` with validation-error capture; the runtime is dropped before the check.
fn with_runtime(name: &str, test: impl FnOnce(&Runtime)) {
    with_runtime_options(&options(name), test);
}

fn with_runtime_options(options: &RuntimeOptions, test: impl FnOnce(&Runtime)) {
    let (errors, _guard) = common::ValidationErrors::capture();
    let Some(runtime) = runtime(options) else { return };
    test(&runtime);
    drop(runtime);
    errors.assert_none();
}

fn binding(binding: u32, name: &str, access: KernelAccess) -> KernelBindingMetadata {
    KernelBindingMetadata {
        set: 0,
        binding,
        name: name.into(),
        kind: KernelBindingKind::StorageBuffer,
        access,
    }
}

fn field(name: &str, type_name: &str, offset: u32) -> PushConstantFieldMetadata {
    PushConstantFieldMetadata {
        name: name.into(),
        type_name: type_name.into(),
        offset,
        size: 4,
    }
}

fn vector_add_metadata() -> KernelMetadata {
    KernelMetadata {
        kernel_id: "integration.vector_add".into(),
        local_size: [64, 1, 1],
        bindings: vec![
            binding(0, "input_a", KernelAccess::Read),
            binding(1, "input_b", KernelAccess::Read),
            binding(2, "output_data", KernelAccess::Write),
        ],
        push_constant_size: 4,
        push_constants: vec![field("count", "u32", 0)],
        ..KernelMetadata::default()
    }
}

fn saxpy_metadata() -> KernelMetadata {
    KernelMetadata {
        kernel_id: "integration.saxpy".into(),
        local_size: [64, 1, 1],
        bindings: vec![
            binding(0, "input_x", KernelAccess::Read),
            binding(1, "input_y", KernelAccess::ReadWrite),
        ],
        push_constant_size: 8,
        push_constants: vec![field("scale", "f32", 0), field("count", "u32", 4)],
        ..KernelMetadata::default()
    }
}

fn vector_add(runtime: &Runtime) -> kora::Kernel {
    runtime
        .load_kernel(
            &words_from_bytes(VECTOR_ADD).unwrap(),
            "main",
            vector_add_metadata(),
        )
        .unwrap()
}

fn saxpy(runtime: &Runtime) -> kora::Kernel {
    runtime
        .load_kernel(&words_from_bytes(SAXPY).unwrap(), "main", saxpy_metadata())
        .unwrap()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SaxpyParameters {
    scale: f32,
    count: u32,
}

#[test]
fn rapid_create_and_destroy_honors_both_shutdown_modes() {
    for mode in [ShutdownMode::Graceful, ShutdownMode::CancelPending] {
        for _ in 0..5 {
            let options = RuntimeOptions {
                shutdown_mode: Some(mode),
                ..options("lifecycle")
            };
            let Some(runtime) = runtime(&options) else { return };
            assert!(!runtime.devices().is_empty());
        }
    }
}

#[test]
fn owns_selected_devices_and_reports_memory_and_selection() {
    with_runtime("ownership", |runtime| {
        let devices = runtime.devices();
        assert!(!devices.is_empty());
        assert!(runtime.quarantined_devices().is_empty());
        for device in &devices {
            let info = runtime.device_info(device).unwrap();
            assert!(!info.name.is_empty());
            let stats = runtime.memory_stats(device).unwrap();
            assert_eq!(stats.heaps.len(), info.memory_heaps.len());
            assert!(stats.heaps.iter().any(|h| h.budget_bytes > 0));
            let report = runtime.device_fault_report(device).unwrap();
            assert_eq!(report.loss_result, 0, "healthy devices report no loss");
        }
        let foreign = kora::DeviceId::new(999, "not-a-device");
        assert_eq!(
            runtime.memory_stats(&foreign).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            runtime.device_info(&foreign).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
    });
    // Explicit index selection keeps the enumeration index.
    let Some(all) = runtime(&RuntimeOptions {
        device_policy: Some(DevicePolicy::AllEligible),
        ..options("selection")
    }) else {
        return;
    };
    let devices = all.devices();
    drop(all);
    for device in devices {
        let Some(single) = runtime(&RuntimeOptions {
            device_policy: Some(DevicePolicy::ByIndex),
            device_index: Some(device.index),
            ..options("selection")
        }) else {
            return;
        };
        assert_eq!(single.devices(), [device]);
    }
}

#[test]
fn dispatches_vector_add_on_every_device_with_timeline_and_fence_fallback() {
    let Some(all) = runtime(&RuntimeOptions {
        device_policy: Some(DevicePolicy::AllEligible),
        ..options("vector-add")
    }) else {
        return;
    };
    let devices = all.devices();
    drop(all);
    let count = 257_u32;
    let a: Vec<u32> = (0..count).map(|i| i * 3).collect();
    let b: Vec<u32> = (0..count).map(|i| i + 11).collect();
    for device in devices {
        for legacy in [false, true] {
            let options = RuntimeOptions {
                device_policy: Some(DevicePolicy::ByIndex),
                device_index: Some(device.index),
                force_legacy_sync: Some(legacy),
                telemetry: Some(TelemetryLevel::Trace),
                ..options("vector-add")
            };
            with_runtime_options(&options, |runtime| {
                let kernel = vector_add(runtime);
                assert_eq!(kernel.entry_point(), "main");
                assert_eq!(kernel.metadata().kernel_id, "integration.vector_add");
                let input_a = runtime.create_buffer::<u32>(u64::from(count)).unwrap();
                let input_b = runtime.create_buffer::<u32>(u64::from(count)).unwrap();
                let output = runtime.create_buffer::<u32>(u64::from(count)).unwrap();
                input_a.write(&a).unwrap();
                input_b.write(&b).unwrap();
                runtime
                    .dispatch(&kernel)
                    .global(u64::from(count))
                    .read(&input_a)
                    .read(&input_b)
                    .write(&output)
                    .push(&count)
                    .submit()
                    .unwrap()
                    .wait()
                    .unwrap();
                let values = output.read().unwrap();
                for index in 0..count as usize {
                    assert_eq!(
                        values[index],
                        a[index] + b[index],
                        "device {device:?} legacy {legacy} index {index}"
                    );
                }
                match runtime.profiling_stats(&device) {
                    Ok(stats) => {
                        assert!(stats.samples_collected > 0);
                        assert!(stats.last_duration_nanoseconds > 0);
                    }
                    Err(error) => assert_eq!(error.code(), ErrorCode::UnsupportedFeature),
                }

                let code = |builder: kora::DispatchBuilder| builder.submit().unwrap_err().code();
                let base = || runtime.dispatch(&kernel).global(u64::from(count));
                assert_eq!(
                    code(
                        base()
                            .global(0)
                            .read(&input_a)
                            .read(&input_b)
                            .write(&output)
                            .push(&count)
                    ),
                    ErrorCode::InvalidDispatch
                );
                assert_eq!(
                    code(base().write(&input_a).read(&input_b).write(&output).push(&count)),
                    ErrorCode::InvalidKernelBinding
                );
                assert_eq!(
                    code(base().read(&input_a).read(&input_b).push(&count)),
                    ErrorCode::InvalidKernelBinding
                );
                assert_eq!(
                    code(base().read(&input_a).read(&input_b).write(&output).push(&0_u64)),
                    ErrorCode::InvalidArgument
                );
                assert_eq!(
                    code(
                        base()
                            .global_3d(1, 1 << 31, 1)
                            .read(&input_a)
                            .read(&input_b)
                            .write(&output)
                            .push(&count)
                    ),
                    ErrorCode::InvalidDispatch
                );
                let stats = runtime.transfer_stats();
                assert_eq!((stats.uploads, stats.downloads), (2, 1));
            });
        }
    }
}

#[test]
fn rejects_invalid_kernels_at_load_time() {
    with_runtime("invalid-kernels", |runtime| {
        let words = words_from_bytes(VECTOR_ADD).unwrap();
        let mut wrong = vector_add_metadata();
        wrong.local_size = [32, 1, 1];
        assert_eq!(
            runtime.load_kernel(&words, "main", wrong).unwrap_err().code(),
            ErrorCode::InvalidKernel
        );
        assert_eq!(
            runtime
                .load_kernel(&words, "missing", vector_add_metadata())
                .unwrap_err()
                .code(),
            ErrorCode::InvalidKernel
        );
        assert_eq!(
            runtime
                .load_kernel(&[], "main", vector_add_metadata())
                .unwrap_err()
                .code(),
            ErrorCode::InvalidKernel
        );
        let missing = std::path::Path::new("/nonexistent/kernel.spv");
        assert_eq!(
            runtime
                .load_kernel_file(missing, "main", vector_add_metadata())
                .unwrap_err()
                .code(),
            ErrorCode::InvalidKernel
        );
    });
}

#[test]
fn reuses_device_resident_intermediates_across_a_dispatch_chain() {
    with_runtime("chain", |runtime| {
        let kernel = vector_add(runtime);
        let n = 1000_u32;
        let a = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let b = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let middle = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let out = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        a.write(&(0..n).collect::<Vec<_>>()).unwrap();
        b.write(&vec![5; n as usize]).unwrap();
        let first = runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&a)
            .read(&b)
            .write(&middle)
            .push(&n)
            .defer_readback()
            .submit()
            .unwrap();
        let second = runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&middle)
            .read(&b)
            .write(&out)
            .push(&n)
            .submit()
            .unwrap();
        second.wait().unwrap();
        first.wait().unwrap();
        let expected: Vec<u32> = (0..n).map(|i| i + 10).collect();
        assert_eq!(out.read().unwrap(), expected);
        // `middle` stayed on the device: only `a` and `b` were uploaded, only `out` read back.
        let stats = runtime.transfer_stats();
        assert_eq!((stats.uploads, stats.downloads), (2, 1));
        assert_eq!(
            middle.read().unwrap(),
            vec![0; n as usize],
            "host shadow not yet refreshed"
        );
        runtime.readback(&middle).unwrap().wait().unwrap();
        assert_eq!(middle.read().unwrap(), (0..n).map(|i| i + 5).collect::<Vec<_>>());
        // A second readback finds the host current and does nothing.
        runtime.readback(&middle).unwrap().wait().unwrap();
        assert_eq!(runtime.transfer_stats().downloads, 2);
    });
}

#[test]
fn dependent_read_write_dispatches_are_ordered_on_the_device() {
    with_runtime("hazards", |runtime| {
        let kernel = saxpy(runtime);
        let n = 4096_u32;
        let x = runtime.create_buffer::<f32>(u64::from(n)).unwrap();
        let y = runtime.create_buffer::<f32>(u64::from(n)).unwrap();
        x.write(&vec![1.0; n as usize]).unwrap();
        y.write(&vec![0.5; n as usize]).unwrap();
        let rounds = 24;
        let mut events = Vec::new();
        for round in 0..rounds {
            let mut builder = runtime
                .dispatch(&kernel)
                .global(u64::from(n))
                .read(&x)
                .read_write(&y)
                .push(&SaxpyParameters { scale: 2.0, count: n });
            if round + 1 < rounds {
                builder = builder.defer_readback();
            }
            events.push(builder.submit().unwrap());
        }
        for event in events {
            event.wait().unwrap();
        }
        let expected = 0.5 + 2.0 * rounds as f32;
        assert!(y.read().unwrap().iter().all(|&v| v == expected));
        assert_eq!(
            runtime.transfer_stats().uploads,
            2,
            "y stays resident across the chain"
        );
    });
}

#[test]
fn host_rewrites_between_dispatches_are_uploaded_with_default_and_deferred_readback() {
    with_runtime("rewrites", |runtime| {
        let kernel = vector_add(runtime);
        let n = 64_u32;
        let a = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let b = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let out = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        b.write(&vec![1; n as usize]).unwrap();
        for (round, deferred) in [(1_u32, false), (2, true), (3, false), (4, true)] {
            a.write(&vec![round * 100; n as usize]).unwrap();
            let mut builder = runtime
                .dispatch(&kernel)
                .global(u64::from(n))
                .read(&a)
                .read(&b)
                .write(&out)
                .push(&n);
            if deferred {
                builder = builder.defer_readback();
            }
            builder.submit().unwrap().wait().unwrap();
            if deferred {
                runtime.readback(&out).unwrap().wait().unwrap();
            }
            assert!(
                out.read().unwrap().iter().all(|&v| v == round * 100 + 1),
                "round {round}"
            );
        }
        // `a` changed every round; `b` was uploaded once.
        assert_eq!(runtime.transfer_stats().uploads, 5);
    });
}

#[test]
fn in_flight_dispatches_keep_inputs_submitted_before_a_later_host_write() {
    with_runtime("in-flight", |runtime| {
        let kernel = vector_add(runtime);
        let n = 1 << 16;
        let a = runtime.create_buffer::<u32>(n).unwrap();
        let b = runtime.create_buffer::<u32>(n).unwrap();
        let out = runtime.create_buffer::<u32>(n).unwrap();
        a.write(&vec![7; n as usize]).unwrap();
        b.write(&vec![1; n as usize]).unwrap();
        let event = runtime
            .dispatch(&kernel)
            .global(n)
            .read(&a)
            .read(&b)
            .write(&out)
            .push(&(n as u32))
            .submit()
            .unwrap();
        a.write(&vec![1000; n as usize]).unwrap();
        event.wait().unwrap();
        assert!(out.read().unwrap().iter().all(|&v| v == 8));
        runtime
            .dispatch(&kernel)
            .global(n)
            .read(&a)
            .read(&b)
            .write(&out)
            .push(&(n as u32))
            .submit()
            .unwrap()
            .wait()
            .unwrap();
        assert!(out.read().unwrap().iter().all(|&v| v == 1001));
    });
}

#[test]
fn a_host_write_during_a_deferred_dispatch_wins_over_device_results() {
    with_runtime("host-wins", |runtime| {
        let kernel = vector_add(runtime);
        let n = 256_u32;
        let a = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let b = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let out = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&a)
            .read(&b)
            .write(&out)
            .push(&n)
            .defer_readback()
            .submit()
            .unwrap()
            .wait()
            .unwrap();
        out.write(&vec![42; n as usize]).unwrap();
        runtime.readback(&out).unwrap().wait().unwrap();
        assert!(out.read().unwrap().iter().all(|&v| v == 42));
    });
}

#[test]
fn runs_cpu_fallbacks_between_gpu_dispatches() {
    with_runtime("cpu-fallback", |runtime| {
        let kernel = vector_add(runtime);
        let n = 512_u32;
        let a = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let b = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let middle = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let doubled = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let out = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        a.write(&(0..n).collect::<Vec<_>>()).unwrap();
        b.write(&vec![3; n as usize]).unwrap();
        runtime
            .register_cpu_fallback("double", |task| {
                let factor: u32 = task.push_as()?;
                let (input, output) = task.input_output::<u32, u32>(0, 1)?;
                for (o, i) in output.iter_mut().zip(input) {
                    *o = i * factor;
                }
                Ok(())
            })
            .unwrap();
        // GPU result stays on the device; the CPU task reads it back first.
        runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&a)
            .read(&b)
            .write(&middle)
            .push(&n)
            .defer_readback()
            .submit()
            .unwrap();
        runtime
            .run_cpu_fallback(
                "double",
                &[
                    (middle.handle(), KernelAccess::Read),
                    (doubled.handle(), KernelAccess::Write),
                ],
                bytemuck::bytes_of(&2_u32),
            )
            .unwrap()
            .wait()
            .unwrap();
        // The CPU output is uploaded to the next GPU dispatch.
        runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&doubled)
            .read(&b)
            .write(&out)
            .push(&n)
            .submit()
            .unwrap()
            .wait()
            .unwrap();
        let expected: Vec<u32> = (0..n).map(|i| (i + 3) * 2 + 3).collect();
        assert_eq!(out.read().unwrap(), expected);

        let unknown = runtime.run_cpu_fallback("missing", &[], &[]).unwrap_err();
        assert_eq!(unknown.code(), ErrorCode::InvalidKernel);
        let duplicate = runtime
            .run_cpu_fallback(
                "double",
                &[
                    (a.handle(), KernelAccess::Read),
                    (a.handle(), KernelAccess::Write),
                ],
                &[],
            )
            .unwrap_err();
        assert_eq!(duplicate.code(), ErrorCode::InvalidKernelBinding);
        runtime
            .register_cpu_fallback("misdeclared", |task| task.write(0).map(drop))
            .unwrap();
        let misdeclared = runtime
            .run_cpu_fallback("misdeclared", &[(a.handle(), KernelAccess::Read)], &[])
            .unwrap()
            .wait()
            .unwrap_err();
        assert_eq!(misdeclared.code(), ErrorCode::InvalidKernelBinding);
        assert!(runtime.register_cpu_fallback("", |_| Ok(())).is_err());
    });
}

#[test]
fn concurrent_dispatch_threads_share_a_runtime() {
    with_runtime("threads", |runtime| {
        let kernel = vector_add(runtime);
        let threads = 6;
        let barrier = Barrier::new(threads);
        std::thread::scope(|scope| {
            for thread in 0..threads as u32 {
                let (kernel, barrier) = (kernel.clone(), &barrier);
                scope.spawn(move || {
                    let n = 333_u32;
                    let a = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
                    let b = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
                    let out = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
                    b.write(&vec![thread; n as usize]).unwrap();
                    barrier.wait();
                    for round in 0..20_u32 {
                        a.write(&vec![round; n as usize]).unwrap();
                        runtime
                            .dispatch(&kernel)
                            .global(u64::from(n))
                            .read(&a)
                            .read(&b)
                            .write(&out)
                            .push(&n)
                            .priority(if round % 2 == 0 {
                                DispatchPriority::Normal
                            } else {
                                DispatchPriority::High
                            })
                            .submit()
                            .unwrap()
                            .wait()
                            .unwrap();
                        assert!(out.read().unwrap().iter().all(|&v| v == round + thread));
                    }
                });
            }
        });
    });
}

#[test]
fn frees_device_copies_of_released_buffers() {
    with_runtime("release", |runtime| {
        let kernel = vector_add(runtime);
        let device = runtime.devices()[0].clone();
        // Warm the staging rings (persistent allocations) before taking the baseline.
        {
            let warm = runtime.create_buffer::<u32>(1 << 14).unwrap();
            runtime
                .dispatch(&kernel)
                .global(1 << 14)
                .read(&warm)
                .read(&warm)
                .write(&warm)
                .push(&(1_u32 << 14))
                .submit()
                .unwrap()
                .wait()
                .unwrap();
        }
        let baseline = runtime.memory_stats(&device).unwrap().live_allocations;
        {
            let n = 1 << 14;
            let a = runtime.create_buffer::<u32>(n).unwrap();
            let b = runtime.create_buffer::<u32>(n).unwrap();
            let out = runtime.create_buffer::<u32>(n).unwrap();
            let event = runtime
                .dispatch(&kernel)
                .global(n)
                .read(&a)
                .read(&b)
                .write(&out)
                .push(&(n as u32))
                .submit()
                .unwrap();
            assert!(runtime.memory_stats(&device).unwrap().live_allocations >= baseline + 3);
            // Dropping the handles while the dispatch runs is safe: the submission keeps them.
            drop((a, b, out));
            event.wait().unwrap();
        }
        assert_eq!(runtime.memory_stats(&device).unwrap().live_allocations, baseline);
        assert_eq!(runtime.buffer_manager().live_buffers(), 0);
    });
}

#[test]
fn records_dispatch_timestamps_only_as_often_as_telemetry_needs() {
    for (level, expected) in [
        (TelemetryLevel::Off, Some(0)),
        (TelemetryLevel::Basic, Some(2)),
        (TelemetryLevel::Profile, Some(20)),
    ] {
        let options = RuntimeOptions {
            telemetry: Some(level),
            disable_autotune: Some(true),
            ..options("timestamps")
        };
        with_runtime_options(&options, |runtime| {
            let device = runtime.devices()[0].clone();
            if runtime.profiling_stats(&device).is_err() {
                return;
            }
            let kernel = vector_add(runtime);
            let n = 64_u32;
            let a = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
            let out = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
            for _ in 0..20 {
                runtime
                    .dispatch(&kernel)
                    .global(u64::from(n))
                    .read(&a)
                    .read(&a)
                    .write(&out)
                    .push(&n)
                    .submit()
                    .unwrap()
                    .wait()
                    .unwrap();
            }
            let stats = runtime.profiling_stats(&device).unwrap();
            assert_eq!(
                Some(stats.samples_collected + stats.dropped_samples),
                expected,
                "{level:?}"
            );
        });
    }
}

#[test]
fn tunes_local_size_id_dispatches_and_keeps_results_exact() {
    let options = RuntimeOptions {
        telemetry: Some(TelemetryLevel::Profile),
        ..options("autotune")
    };
    with_runtime_options(&options, |runtime| {
        let device = runtime.devices()[0].clone();
        let info = runtime.device_info(&device).unwrap().clone();
        let spec = |id, name: &str| SpecializationConstantMetadata {
            id,
            name: name.into(),
            type_name: "u32".into(),
            default_value: SpecializationValue::U64(if id == 7 { 64 } else { 1 }),
        };
        let metadata = KernelMetadata {
            kernel_id: "integration.autotune".into(),
            local_size: [64, 1, 1],
            bindings: vec![binding(0, "data_buffer", KernelAccess::ReadWrite)],
            push_constant_size: 4,
            push_constants: vec![field("count", "u32", 0)],
            specialization_constants: vec![spec(7, "local_x"), spec(8, "local_y"), spec(9, "local_z")],
            ..KernelMetadata::default()
        };
        let words = words_from_bytes(AUTOTUNE).unwrap();
        let kernel = match runtime.load_kernel(&words, "main", metadata) {
            Ok(kernel) => kernel,
            Err(error) => {
                assert!(!info.features.maintenance4, "{error}");
                assert_eq!(error.code(), ErrorCode::InvalidKernel);
                return;
            }
        };
        let n = 10_000_u32;
        let data = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let rounds = 60_u32;
        for _ in 0..rounds {
            runtime
                .dispatch(&kernel)
                .global(u64::from(n))
                .read_write(&data)
                .push(&n)
                .defer_readback()
                .submit()
                .unwrap()
                .wait()
                .unwrap();
            // Observing timestamps advances the tuner between dispatches.
            let _ = runtime.profiling_stats(&device);
        }
        runtime.readback(&data).unwrap().wait().unwrap();
        assert!(
            data.read().unwrap().iter().all(|&v| v == rounds),
            "every size covers every element"
        );
        if kernel.is_autotuned() {
            let size = kernel.local_size();
            assert!(size[0] >= 1 && size[0] <= info.limits.max_compute_work_group_size[0]);
        }
    });
}

#[test]
fn device_loss_fails_new_work_and_reports_the_loss() {
    with_runtime("device-loss", |runtime| {
        let kernel = vector_add(runtime);
        let device = runtime.devices()[0].clone();
        let n = 64_u32;
        let a = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let out = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        runtime.simulate_device_loss(&device).unwrap();
        let result = runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&a)
            .read(&a)
            .write(&out)
            .push(&n)
            .submit()
            .and_then(|event| event.wait());
        assert_eq!(result.unwrap_err().code(), ErrorCode::DeviceLost);
        let report = runtime.device_fault_report(&device).unwrap();
        assert_eq!(report.loss_result, -4, "VK_ERROR_DEVICE_LOST");
    });
}

#[test]
fn kernels_and_buffers_outlive_their_runtime_safely() {
    let (errors, _guard) = common::ValidationErrors::capture();
    let Some(runtime) = runtime(&options("outlive")) else {
        return;
    };
    let kernel = vector_add(&runtime);
    let buffer = runtime.create_buffer::<u32>(64).unwrap();
    buffer.write(&[2; 64]).unwrap();
    let pending = runtime
        .dispatch(&kernel)
        .global(64)
        .read(&buffer)
        .read(&buffer)
        .write(&buffer)
        .push(&64_u32)
        .submit()
        .unwrap();
    let late = runtime.dispatch(&kernel).global(64);
    // Graceful shutdown drains the dispatch before the runtime is gone.
    drop(runtime);
    pending.wait().unwrap();
    assert_eq!(buffer.read().unwrap(), [4; 64]);
    let error = late
        .read(&buffer)
        .read(&buffer)
        .write(&buffer)
        .push(&64_u32)
        .submit()
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::Cancelled);
    drop((kernel, buffer));
    errors.assert_none();
}
