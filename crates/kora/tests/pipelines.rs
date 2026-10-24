//! Pipeline manager against real Vulkan devices.

mod common;

use std::path::PathBuf;
use std::sync::{Arc, Barrier};

use kora::kernel::spirv::words_from_bytes;
use kora::kernel::{DeviceCapabilities, PipelineCacheIdentity, PipelineManager, PipelineManagerOptions};
use kora::vulkan::ash::vk;
use kora::vulkan::{Device, DeviceOptions, FaultFunction, InstanceOptions, device_info};
use kora::{
    DeviceInfo, ErrorCode, KernelAccess, KernelBindingKind, KernelBindingMetadata, KernelMetadata,
    PushConstantFieldMetadata, SpecializationConstantMetadata, SpecializationValue,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/spirv_reflection.spv");
const AUTOTUNE: &[u8] = include_bytes!("fixtures/autotune.spv");

struct Setup {
    device: Device,
    info: DeviceInfo,
    identity: PipelineCacheIdentity,
    errors: common::ValidationErrors,
    _log: std::sync::MutexGuard<'static, ()>,
}

impl Setup {
    /// Destroys the device (and with it every child object) and asserts that the validation layer,
    /// when it is installed, reported nothing.
    fn finish(self) {
        let Self { device, errors, .. } = self;
        drop(device);
        errors.assert_none();
    }
}

fn setup() -> Option<Setup> {
    let (errors, log) = common::ValidationErrors::capture();
    let validated = InstanceOptions {
        enable_validation: true,
        ..InstanceOptions::default()
    };
    let (_, instance) = common::vulkan(&validated)?;
    if !instance.validation_enabled() {
        eprintln!("note: validation layer unavailable; validation checks are skipped");
    }
    let physical = instance.physical_devices().ok()?;
    let physical = physical.into_iter().find(|p| {
        p.queue_families
            .iter()
            .any(|f| f.queue_count > 0 && f.queue_flags.contains(vk::QueueFlags::COMPUTE))
    })?;
    let info = device_info(&physical);
    let identity = PipelineCacheIdentity::from_physical(&physical);
    let device = instance
        .create_device(&physical, &DeviceOptions::default())
        .expect("device");
    Some(Setup {
        device,
        info,
        identity,
        errors,
        _log: log,
    })
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("kora-pipelines-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Runs `test` against a validated device; every object the test created is dropped before the
/// device is destroyed and validation messages are checked.
fn with_device(test: impl FnOnce(&Setup)) {
    let Some(setup) = setup() else { return };
    test(&setup);
    setup.finish();
}

fn module() -> Vec<u32> {
    words_from_bytes(FIXTURE).unwrap()
}

fn spec(id: u32, name: &str, value: u64) -> SpecializationConstantMetadata {
    SpecializationConstantMetadata {
        id,
        name: name.into(),
        type_name: "u32".into(),
        default_value: SpecializationValue::U64(value),
    }
}

fn fixture_metadata(scale: u64) -> KernelMetadata {
    let binding = |binding, name: &str| KernelBindingMetadata {
        set: 0,
        binding,
        name: name.into(),
        kind: KernelBindingKind::StorageBuffer,
        access: KernelAccess::ReadWrite,
    };
    KernelMetadata {
        kernel_id: "pipeline-cache-integration".into(),
        local_size: [8, 2, 1],
        bindings: vec![binding(0, "data_buffer"), binding(1, "second_buffer")],
        push_constant_size: 4,
        push_constants: vec![PushConstantFieldMetadata {
            name: "count".into(),
            type_name: "u32".into(),
            offset: 0,
            size: 4,
        }],
        specialization_constants: vec![spec(7, "SCALE", scale)],
        ..KernelMetadata::default()
    }
}

/// The fixture with its `SCALE` default changed from 2 to 3.
fn alternate_default(mut words: Vec<u32>) -> Vec<u32> {
    let mut cursor = 5;
    while cursor < words.len() {
        let count = (words[cursor] >> 16) as usize;
        if words[cursor] & 0xffff == 50 && count == 4 {
            words[cursor + 3] = 3;
            return words;
        }
        cursor += count;
    }
    panic!("fixture has no OpSpecConstant");
}

fn manager(setup: &Setup, cache: Option<PathBuf>, options: PipelineManagerOptions) -> PipelineManager {
    manager_with_identity(setup, setup.identity, cache, options)
}

fn manager_with_identity(
    setup: &Setup,
    identity: PipelineCacheIdentity,
    cache: Option<PathBuf>,
    options: PipelineManagerOptions,
) -> PipelineManager {
    PipelineManager::new(
        setup.device.clone(),
        DeviceCapabilities::from_device(&setup.info),
        identity,
        cache,
        options,
    )
    .unwrap()
}

const TINY: PipelineManagerOptions = PipelineManagerOptions {
    max_cached_pipelines: 1,
    descriptor_sets_per_pool: 1,
    max_descriptor_pools_per_layout: 2,
};

#[test]
fn caches_pipelines_recycles_descriptors_and_recovers_persistent_cache() {
    with_device(|setup| {
        let words = module();
        let temp = TempDir::new("persist");
        let cache_file = temp.0.join("pipelines").join("cache.bin");
        let metadata = fixture_metadata(2);
        {
            let manager = manager(setup, Some(cache_file.clone()), TINY);
            let first = manager.get_or_create(&words, &metadata, &[], "main").unwrap();
            assert_ne!(first.handle(), vk::Pipeline::null());
            assert_eq!(first.local_size(), [8, 2, 1]);
            assert_eq!(first.descriptor_set_count(), 1);
            assert_eq!(first.push_constant_size(), 4);

            let cached = manager.find_cached(&words, &metadata, &[], "main").unwrap();
            assert!(Arc::ptr_eq(&first, &cached));
            let repeated = manager.get_or_create(&words, &metadata, &[], "main").unwrap();
            assert!(Arc::ptr_eq(&first, &repeated));
            let stats = manager.stats();
            assert_eq!((stats.pipeline_cache_hits, stats.pipeline_cache_misses), (2, 1));

            let mut invalid = metadata.specialization_constants.clone();
            invalid[0].type_name = "unsupported".into();
            assert!(manager.find_cached(&words, &metadata, &invalid, "main").is_none());
            assert_eq!(
                manager
                    .get_or_create(&words, &metadata, &invalid, "main")
                    .unwrap_err()
                    .code(),
                ErrorCode::InvalidArgument
            );

            // One set per pool: the second lease grows a pool, a released set is recycled.
            let a = manager.allocate_descriptor_set(&first, 0).unwrap();
            let b = manager.allocate_descriptor_set(&first, 0).unwrap();
            assert_ne!(a.handle(), b.handle());
            assert_eq!(manager.stats().descriptor_pool_growths, 2);
            let recycled_handle = a.handle();
            drop(a);
            let recycled = manager.allocate_descriptor_set(&first, 0).unwrap();
            assert_eq!(recycled.handle(), recycled_handle);
            // Both pools are full and the growth limit is two.
            let exhausted = manager.allocate_descriptor_set(&first, 0).unwrap_err();
            assert_eq!(exhausted.code(), ErrorCode::AllocationFailed);
            assert_eq!(
                manager.allocate_descriptor_set(&first, 1).unwrap_err().code(),
                ErrorCode::InvalidArgument
            );
            drop((b, recycled));
            assert_eq!(manager.stats().descriptor_sets_allocated, 3);

            // A different module evicts the only cache slot; the evicted pipeline stays usable.
            let alternate = alternate_default(words.clone());
            let second = manager
                .get_or_create(&alternate, &fixture_metadata(3), &[], "main")
                .unwrap();
            assert!(!Arc::ptr_eq(&first, &second));
            assert_eq!(manager.stats().pipeline_evictions, 1);
            assert_eq!(manager.cached_pipelines(), 1);
            assert!(manager.find_cached(&words, &metadata, &[], "main").is_none());
            let lease = manager.allocate_descriptor_set(&first, 0).unwrap();
            drop((lease, first));
            // Layouts are shared between pipelines with the same interface.
            assert!(manager.stats().descriptor_layout_cache_hits >= 1);
            assert!(manager.stats().pipeline_layout_cache_hits >= 1);

            manager.save_pipeline_cache().unwrap();
            assert!(cache_file.exists());
        }
        {
            let manager = manager(setup, Some(cache_file.clone()), TINY);
            manager.get_or_create(&words, &metadata, &[], "main").unwrap();
            assert!(manager.stats().persisted_pipeline_cache_loaded);
        }
        {
            let mut other = setup.identity;
            other.device_uuid[0] ^= 0xff;
            let manager = manager_with_identity(setup, other, Some(cache_file.clone()), TINY);
            manager.get_or_create(&words, &metadata, &[], "main").unwrap();
            assert!(!manager.stats().persisted_pipeline_cache_loaded);
        }
        // Dropping the mismatched manager rewrote the file for the other identity; corrupt it outright.
        std::fs::write(&cache_file, b"not a KORA pipeline cache").unwrap();
        {
            let manager = manager(setup, Some(cache_file.clone()), TINY);
            manager.get_or_create(&words, &metadata, &[], "main").unwrap();
            assert!(!manager.stats().persisted_pipeline_cache_loaded);
        }
        // ...and the drop above rewrote a valid cache.
        let manager = manager(setup, Some(cache_file), TINY);
        manager.get_or_create(&words, &metadata, &[], "main").unwrap();
        assert!(manager.stats().persisted_pipeline_cache_loaded);
    });
}

#[test]
fn rejects_invalid_modules_before_reaching_the_driver() {
    with_device(|setup| {
        let manager = manager(setup, None, PipelineManagerOptions::default());
        let words = module();
        let mut wrong = fixture_metadata(2);
        wrong.local_size = [1, 1, 1];
        let error = manager.get_or_create(&words, &wrong, &[], "main").unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidKernel);
        assert!(error.message().contains("local_size"), "{error}");
        assert_eq!(
            manager
                .get_or_create(&words[..4], &fixture_metadata(2), &[], "main")
                .unwrap_err()
                .code(),
            ErrorCode::InvalidKernel
        );
        assert_eq!(
            manager
                .get_or_create(&words, &fixture_metadata(2), &[], "missing")
                .unwrap_err()
                .code(),
            ErrorCode::InvalidKernel
        );
        assert_eq!(manager.cached_pipelines(), 0);
    });
}

#[test]
fn specialization_overrides_create_distinct_cached_variants() {
    with_device(|setup| {
        let manager = manager(setup, None, PipelineManagerOptions::default());
        let words = module();
        let metadata = fixture_metadata(2);
        let base = manager.get_or_create(&words, &metadata, &[], "main").unwrap();
        let five = [spec(7, "SCALE", 5)];
        let overridden = manager.get_or_create(&words, &metadata, &five, "main").unwrap();
        assert!(!Arc::ptr_eq(&base, &overridden));
        assert_ne!(base.digest(), overridden.digest());
        let again = manager.find_cached(&words, &metadata, &five, "main").unwrap();
        assert!(Arc::ptr_eq(&overridden, &again));
        // Overriding with the default value resolves to the base pipeline.
        let same = manager
            .get_or_create(&words, &metadata, &[spec(7, "SCALE", 2)], "main")
            .unwrap();
        assert!(Arc::ptr_eq(&base, &same));
        assert_eq!(manager.cached_pipelines(), 2);
    });
}

#[test]
fn concurrent_requests_for_one_variant_share_a_single_pipeline() {
    with_device(|setup| {
        let manager = Arc::new(manager(setup, None, PipelineManagerOptions::default()));
        let words = Arc::new(module());
        let threads = 8;
        let barrier = Arc::new(Barrier::new(threads));
        let handles: Vec<_> = (0..threads)
            .map(|index| {
                let (manager, words, barrier) =
                    (Arc::clone(&manager), Arc::clone(&words), Arc::clone(&barrier));
                std::thread::spawn(move || {
                    let scale = [spec(7, "SCALE", 10 + (index as u64 % 2))];
                    barrier.wait();
                    manager
                        .get_or_create(&words, &fixture_metadata(2), &scale, "main")
                        .unwrap()
                })
            })
            .collect();
        let pipelines: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        for (index, pipeline) in pipelines.iter().enumerate() {
            assert!(
                Arc::ptr_eq(pipeline, &pipelines[index % 2]),
                "variant {}",
                index % 2
            );
        }
        assert!(!Arc::ptr_eq(&pipelines[0], &pipelines[1]));
        assert_eq!(manager.cached_pipelines(), 2);
        let stats = manager.stats();
        assert_eq!(
            stats.pipeline_cache_hits + stats.pipeline_cache_misses,
            threads as u64
        );
    });
}

#[test]
fn local_size_id_kernels_are_specialized_or_rejected_without_maintenance4() {
    with_device(|setup| {
        let manager = manager(setup, None, PipelineManagerOptions::default());
        let words = words_from_bytes(AUTOTUNE).unwrap();
        let metadata = KernelMetadata {
            kernel_id: "autotune".into(),
            local_size: [1, 1, 1],
            bindings: vec![KernelBindingMetadata {
                set: 0,
                binding: 0,
                name: "data_buffer".into(),
                kind: KernelBindingKind::StorageBuffer,
                access: KernelAccess::ReadWrite,
            }],
            push_constant_size: 4,
            push_constants: vec![PushConstantFieldMetadata {
                name: "count".into(),
                type_name: "u32".into(),
                offset: 0,
                size: 4,
            }],
            specialization_constants: vec![
                spec(7, "local_x", 1),
                spec(8, "local_y", 1),
                spec(9, "local_z", 1),
            ],
            ..KernelMetadata::default()
        };
        let wide = [spec(7, "local_x", 64)];
        let result = manager.get_or_create(&words, &metadata, &wide, "main");
        if !setup.info.features.maintenance4 {
            let error = result.unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidKernel);
            assert!(error.message().contains("maintenance4"), "{error}");
            return;
        }
        let pipeline = result.unwrap();
        assert_eq!(pipeline.local_size(), [64, 1, 1]);
        let zero = manager
            .get_or_create(&words, &metadata, &[spec(7, "local_x", 0)], "main")
            .unwrap_err();
        assert_eq!(zero.code(), ErrorCode::InvalidArgument);
    });
}

#[test]
fn injected_driver_failures_are_reported_and_not_cached() {
    with_device(|setup| {
        let manager = manager(setup, None, PipelineManagerOptions::default());
        let words = module();
        let metadata = fixture_metadata(2);
        let faults = setup.device.faults();
        faults
            .fail_on_nth(
                FaultFunction::CreateComputePipelines,
                1,
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            )
            .unwrap();
        let error = manager.get_or_create(&words, &metadata, &[], "main").unwrap_err();
        assert_eq!(error.code(), ErrorCode::PipelineCreationFailed);
        assert_eq!(error.vk_result(), vk::Result::ERROR_OUT_OF_DEVICE_MEMORY.as_raw());
        assert_eq!(manager.cached_pipelines(), 0);
        let pipeline = manager.get_or_create(&words, &metadata, &[], "main").unwrap();

        faults
            .fail_on_nth(
                FaultFunction::AllocateDescriptorSets,
                1,
                vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            )
            .unwrap();
        let error = manager.allocate_descriptor_set(&pipeline, 0).unwrap_err();
        assert_eq!(error.code(), ErrorCode::PipelineCreationFailed);
        faults.clear();
        manager.allocate_descriptor_set(&pipeline, 0).unwrap();
    });
}

#[test]
fn pipelines_and_leases_outlive_their_manager_and_foreign_pipelines_are_rejected() {
    with_device(|setup| {
        let words = module();
        let first = manager(setup, None, PipelineManagerOptions::default());
        let other = manager(setup, None, PipelineManagerOptions::default());
        let pipeline = first
            .get_or_create(&words, &fixture_metadata(2), &[], "main")
            .unwrap();
        assert_eq!(
            other.allocate_descriptor_set(&pipeline, 0).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        let lease = first.allocate_descriptor_set(&pipeline, 0).unwrap();
        drop(first);
        assert_ne!(pipeline.handle(), vk::Pipeline::null());
        assert_ne!(lease.handle(), vk::DescriptorSet::null());
        drop(pipeline);
        drop(lease);
        setup.device.wait_idle().unwrap();
    });
}
