//! Calibration, adaptive placement, duplicate verification, recovery, and tracing against real
//! devices. Multi-device cases skip on machines with a single Vulkan device.

mod common;

use std::sync::Arc;

use kora::calibration::CalibrationRecord;
use kora::kernel::spirv::words_from_bytes;
use kora::verification::{VerificationDataType, VerificationPolicy};
use kora::{
    AdaptiveDispatchWorkload, AdaptiveKernelWorkKind, DevicePolicy, ErrorCode, KernelAccess,
    KernelBindingKind, KernelBindingMetadata, KernelMetadata, PushConstantFieldMetadata, Runtime,
    RuntimeOptions, TelemetryLevel,
};

const VECTOR_ADD: &[u8] = include_bytes!("fixtures/vector_add.spv");
const SAXPY: &[u8] = include_bytes!("fixtures/saxpy.spv");

fn cache_dir(name: &str) -> String {
    let path = std::env::temp_dir().join(format!("kora-advanced-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path.to_string_lossy().into_owned()
}

fn options(name: &str) -> RuntimeOptions {
    RuntimeOptions {
        enable_validation: Some(true),
        device_policy: Some(DevicePolicy::AllEligible),
        cache_dir: Some(cache_dir(name)),
        enable_startup_calibration: Some(false),
        ..RuntimeOptions::default()
    }
}

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

fn with_runtime(options: &RuntimeOptions, minimum_devices: usize, test: impl FnOnce(&mut Runtime)) {
    let (errors, _guard) = common::ValidationErrors::capture();
    let Some(mut runtime) = runtime(options) else {
        return;
    };
    if runtime.devices().len() < minimum_devices {
        eprintln!("skipping: needs {minimum_devices} devices");
        return;
    }
    test(&mut runtime);
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

fn vector_add(runtime: &Runtime, device: &kora::DeviceId) -> kora::Kernel {
    let metadata = KernelMetadata {
        kernel_id: "advanced.vector_add".into(),
        local_size: [64, 1, 1],
        bindings: vec![
            binding(0, "input_a", KernelAccess::Read),
            binding(1, "input_b", KernelAccess::Read),
            binding(2, "output_data", KernelAccess::Write),
        ],
        push_constant_size: 4,
        push_constants: vec![field("count", "u32", 0)],
        ..KernelMetadata::default()
    };
    runtime
        .load_kernel_on(device, &words_from_bytes(VECTOR_ADD).unwrap(), "main", metadata)
        .unwrap()
}

fn saxpy(runtime: &Runtime, device: &kora::DeviceId) -> kora::Kernel {
    let metadata = KernelMetadata {
        kernel_id: "advanced.saxpy".into(),
        local_size: [64, 1, 1],
        bindings: vec![
            binding(0, "input_x", KernelAccess::Read),
            binding(1, "input_y", KernelAccess::ReadWrite),
        ],
        push_constant_size: 8,
        push_constants: vec![field("scale", "f32", 0), field("count", "u32", 4)],
        ..KernelMetadata::default()
    };
    runtime
        .load_kernel_on(device, &words_from_bytes(SAXPY).unwrap(), "main", metadata)
        .unwrap()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Saxpy {
    scale: f32,
    count: u32,
}

fn complete_record(rate: u64) -> CalibrationRecord {
    CalibrationRecord {
        submit_overhead_ns: 100,
        tiny_dispatch_ns: 100,
        h2d_bytes_per_second: rate,
        d2h_bytes_per_second: rate,
        device_copy_bytes_per_second: rate,
        streaming_bytes_per_second: rate,
        alu_operations_per_second: rate,
        ..CalibrationRecord::default()
    }
}

fn workload(n: u32) -> AdaptiveDispatchWorkload {
    AdaptiveDispatchWorkload {
        input_bytes: u64::from(n) * 8,
        output_bytes: u64::from(n) * 4,
        kernel_work: u64::from(n) * 12,
        kernel_kind: AdaptiveKernelWorkKind::StreamingBytes,
        ..AdaptiveDispatchWorkload::default()
    }
}

#[test]
fn verified_dispatch_publishes_matching_results_and_quarantines_mismatches() {
    with_runtime(&options("verify"), 2, |runtime| {
        let devices = runtime.devices();
        let kernel = saxpy(runtime, &devices[0]);
        let n = 1000_u32;
        let x = runtime.create_buffer::<f32>(u64::from(n)).unwrap();
        let y = runtime.create_buffer::<f32>(u64::from(n)).unwrap();
        x.write(&(0..n).map(|i| i as f32).collect::<Vec<_>>()).unwrap();
        y.write(&vec![1.0; n as usize]).unwrap();
        runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&x)
            .read_write(&y)
            .push(&Saxpy { scale: 0.5, count: n })
            .verify_with(VerificationPolicy::Exact, VerificationDataType::Float32)
            .submit()
            .unwrap()
            .wait()
            .unwrap();
        let expected: Vec<f32> = (0..n).map(|i| 1.0 + 0.5 * i as f32).collect();
        assert_eq!(y.read().unwrap(), expected);
        assert!(runtime.quarantined_devices().is_empty());

        // A disagreeing comparator quarantines every participant and loses the outputs.
        let disagree = VerificationPolicy::Custom(Arc::new(|_: &[u8], _: &[u8]| false));
        let error = runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&x)
            .read_write(&y)
            .push(&Saxpy { scale: 0.5, count: n })
            .verify_with(disagree, VerificationDataType::Float32)
            .submit()
            .unwrap()
            .wait()
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::ValidationError);
        assert_eq!(runtime.quarantined_devices().len(), devices.len());
        assert_eq!(y.read().unwrap_err().code(), ErrorCode::DataLoss);
        let blocked = runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&x)
            .read_write(&y)
            .push(&Saxpy { scale: 0.5, count: n })
            .submit()
            .unwrap_err();
        assert_eq!(blocked.code(), ErrorCode::ValidationError);
    });
}

#[test]
fn verification_needs_two_devices_and_a_writable_output() {
    with_runtime(&options("verify-single"), 1, |runtime| {
        let device = runtime.devices()[0].clone();
        let kernel = vector_add(runtime, &device);
        let n = 64_u32;
        let a = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let out = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let result = runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&a)
            .read(&a)
            .write(&out)
            .push(&n)
            .verify_with(VerificationPolicy::Exact, VerificationDataType::Float32)
            .submit();
        if runtime.devices().len() < 2 {
            assert_eq!(result.unwrap_err().code(), ErrorCode::UnsupportedFeature);
        } else {
            result.unwrap().wait().unwrap();
        }
        let tolerance = VerificationPolicy::AbsoluteTolerance(-1.0);
        let invalid = runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&a)
            .read(&a)
            .write(&out)
            .push(&n)
            .verify_with(tolerance, VerificationDataType::Float32)
            .submit()
            .unwrap_err();
        assert_eq!(invalid.code(), ErrorCode::InvalidArgument);
    });
}

#[test]
fn stages_results_between_every_device_pair_through_host_memory() {
    with_runtime(&options("staging"), 2, |runtime| {
        let devices = runtime.devices();
        let n = 512_u32;
        for producer in &devices {
            for consumer in &devices {
                if producer == consumer {
                    continue;
                }
                let first = vector_add(runtime, producer);
                let second = vector_add(runtime, consumer);
                let a = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
                let middle = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
                let out = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
                a.write(&(0..n).collect::<Vec<_>>()).unwrap();
                runtime
                    .dispatch(&first)
                    .global(u64::from(n))
                    .read(&a)
                    .read(&a)
                    .write(&middle)
                    .push(&n)
                    .defer_readback()
                    .submit()
                    .unwrap()
                    .wait()
                    .unwrap();
                // The consumer refuses the stale host copy until the producer's result is read back.
                let stale = runtime
                    .dispatch(&second)
                    .global(u64::from(n))
                    .read(&middle)
                    .read(&a)
                    .write(&out)
                    .push(&n)
                    .submit()
                    .unwrap_err();
                assert_eq!(stale.code(), ErrorCode::InvalidArgument);
                runtime.readback_from(&middle, producer).unwrap().wait().unwrap();
                runtime
                    .dispatch(&second)
                    .global(u64::from(n))
                    .read(&middle)
                    .read(&a)
                    .write(&out)
                    .push(&n)
                    .submit()
                    .unwrap()
                    .wait()
                    .unwrap();
                assert_eq!(out.read().unwrap(), (0..n).map(|i| i * 3).collect::<Vec<_>>());
            }
        }
    });
}

#[test]
fn adaptive_dispatch_keeps_the_bound_device_without_complete_calibration() {
    with_runtime(&options("adaptive-fallback"), 1, |runtime| {
        let device = runtime.devices()[0].clone();
        let kernel = vector_add(runtime, &device);
        let builder = runtime.dispatch_adaptive(&kernel, &workload(256)).unwrap();
        assert_eq!(builder.device(), &device);
        let n = 256_u32;
        let a = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let out = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        a.write(&vec![4; n as usize]).unwrap();
        builder
            .global(u64::from(n))
            .read(&a)
            .read(&a)
            .write(&out)
            .push(&n)
            .submit()
            .unwrap()
            .wait()
            .unwrap();
        assert!(out.read().unwrap().iter().all(|&v| v == 8));
        let zero = AdaptiveDispatchWorkload {
            kernel_work: 0,
            ..workload(n)
        };
        assert_eq!(
            runtime.dispatch_adaptive(&kernel, &zero).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
    });
}

#[test]
fn adaptive_dispatch_follows_calibration_and_returns_to_the_producer_device() {
    let options = RuntimeOptions {
        telemetry: Some(TelemetryLevel::Trace),
        ..options("adaptive")
    };
    with_runtime(&options, 2, |runtime| {
        let devices = runtime.devices();
        let (bound, fast) = (&devices[0], &devices[1]);
        if bound.uuid.is_empty() || fast.uuid.is_empty() {
            eprintln!("skipping: devices without UUIDs cannot carry calibration records");
            return;
        }
        runtime
            .set_calibration(bound, complete_record(1_000_000))
            .unwrap();
        runtime
            .set_calibration(fast, complete_record(10_000_000_000))
            .unwrap();
        let kernel = vector_add(runtime, bound);
        let n = 1024_u32;
        let a = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let out = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        a.write(&(0..n).collect::<Vec<_>>()).unwrap();
        for _ in 0..6 {
            let builder = runtime.dispatch_adaptive(&kernel, &workload(n)).unwrap();
            assert_eq!(builder.device(), fast, "calibration prefers the faster device");
            builder
                .global(u64::from(n))
                .read(&a)
                .read(&a)
                .write(&out)
                .push(&n)
                .submit()
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(out.read().unwrap(), (0..n).map(|i| i * 2).collect::<Vec<_>>());
        }
        // Completed executions are recorded per device (when the device has GPU timestamps).
        let stats = runtime.profiling_stats(fast).unwrap();
        if stats.gpu_timestamps {
            assert!(!runtime.performance_database().is_empty());
            let trace = runtime.export_trace_json().unwrap();
            assert!(trace.contains("scheduler_score"), "{trace}");
            kora::json::parse(trace.as_bytes()).unwrap();
        }

        // A deferred result on the bound device pulls the next adaptive dispatch back there.
        let middle = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&a)
            .read(&a)
            .write(&middle)
            .push(&n)
            .defer_readback()
            .submit()
            .unwrap()
            .wait()
            .unwrap();
        let builder = runtime.dispatch_adaptive(&kernel, &workload(n)).unwrap();
        builder
            .global(u64::from(n))
            .read(&middle)
            .read(&a)
            .write(&out)
            .push(&n)
            .submit()
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(out.read().unwrap(), (0..n).map(|i| i * 3).collect::<Vec<_>>());
    });
}

#[test]
fn recovery_rebuilds_devices_and_keeps_authoritative_host_data() {
    with_runtime(&options("recover"), 1, |runtime| {
        let device = runtime.devices()[0].clone();
        assert_eq!(
            runtime.recover().unwrap_err().code(),
            ErrorCode::InvalidArgument,
            "nothing was lost"
        );
        let kernel = vector_add(runtime, &device);
        let n = 128_u32;
        let a = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        let deferred = runtime.create_buffer::<u32>(u64::from(n)).unwrap();
        a.write(&vec![21; n as usize]).unwrap();
        runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&a)
            .read(&a)
            .write(&deferred)
            .push(&n)
            .defer_readback()
            .submit()
            .unwrap()
            .wait()
            .unwrap();
        runtime.simulate_device_loss(&device).unwrap();
        runtime.recover().unwrap();
        assert!(runtime.devices().contains(&device));
        // Host-backed data survives; results only the lost device held are gone.
        assert_eq!(a.read().unwrap(), vec![21; n as usize]);
        assert_eq!(deferred.read().unwrap_err().code(), ErrorCode::DataLoss);
        // Old kernels are invalidated; reloaded kernels work on the rebuilt device.
        let stale = runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&a)
            .read(&a)
            .write(&deferred)
            .push(&n)
            .submit()
            .unwrap_err();
        assert_eq!(stale.code(), ErrorCode::Cancelled);
        let fresh = vector_add(runtime, &device);
        runtime
            .dispatch(&fresh)
            .global(u64::from(n))
            .read(&a)
            .read(&a)
            .write(&deferred)
            .push(&n)
            .submit()
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(deferred.read().unwrap(), vec![42; n as usize]);
        runtime.rebuild_devices().unwrap();
        assert_eq!(a.read().unwrap(), vec![21; n as usize]);
    });
}

#[test]
fn startup_calibration_survives_corrupt_caches_and_recalibrates_on_request() {
    let directory = cache_dir("calibration");
    let calibrated = RuntimeOptions {
        enable_startup_calibration: Some(true),
        calibration_budget_ms: Some(2_000),
        cache_dir: Some(directory.clone()),
        ..options("calibration-unused")
    };
    let Some(probe) = runtime(&calibrated) else { return };
    let device = probe.devices()[0].clone();
    drop(probe);
    if device.uuid.is_empty() {
        eprintln!("skipping: device has no UUID for calibration records");
        return;
    }
    let calibration_file = std::path::Path::new(&directory)
        .join("calibration")
        .join(format!("{}.json", device.uuid));
    let performance_file = std::path::Path::new(&directory).join("performance-db.json");
    std::fs::write(&calibration_file, "not json").unwrap();
    std::fs::write(&performance_file, "{corrupt").unwrap();
    with_runtime(&calibrated, 1, |runtime| {
        let record = runtime.calibration(&device).unwrap();
        assert_eq!(record.device_uuid, &*device.uuid);
        assert!(record.submit_overhead_ns > 0, "{record:?}");
        assert!(record.tiny_dispatch_ns > 0, "{record:?}");
        assert!(record.streaming_bytes_per_second > 0, "{record:?}");
        assert!(record.alu_operations_per_second > 0, "{record:?}");
        // The corrupt cache was replaced with a valid record for this device.
        let stored = std::fs::read_to_string(&calibration_file).unwrap();
        CalibrationRecord::decode(&stored, &device.uuid, record.driver_version).unwrap();
        std::fs::write(&calibration_file, "corrupt again").unwrap();
        runtime.calibrate();
        let stored = std::fs::read_to_string(&calibration_file).unwrap();
        CalibrationRecord::decode(&stored, &device.uuid, record.driver_version).unwrap();
    });
}
