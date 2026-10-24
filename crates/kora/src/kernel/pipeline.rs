//! Per-device compute pipelines, descriptor layouts and pools, and the persistent pipeline cache.
//!
//! Every Vulkan object here is reference counted and destroyed by its own `Drop`, so a
//! [`PipelineRef`] or [`DescriptorSetLease`] stays valid even after its [`PipelineManager`] is
//! dropped. Pipelines evicted from the LRU cache are destroyed once the last reference goes away.
//! A command buffer that uses a pipeline or descriptor set must keep the reference alive until the
//! GPU has finished with it (the runtime stores them in the submission's keep-alive list).
//!
//! Compared with the C++ implementation, a cache hit no longer re-validates the module and no
//! longer copies it into a key string: the key holds a hash of the module plus the serialized
//! interface, and a hit is confirmed by comparing the module words. Pipeline compilation runs
//! outside the manager lock, so threads warming different kernels compile in parallel.

use std::collections::HashMap;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ash::vk;

use super::spirv::{DeviceCapabilities, validate_and_reflect};
use crate::error::{Error, ErrorCode, Result, Status};
use crate::fsutil;

use crate::metadata::{
    KernelBindingKind, KernelBindingMetadata, KernelMetadata, SpecializationConstantMetadata,
    SpecializationValue,
};
use crate::vulkan::{Device, FaultFunction, PhysicalDeviceInfo};

const CACHE_MAGIC: [u8; 8] = *b"KORAPCH1";
const CACHE_SCHEMA: u32 = 1;
const CACHE_HEADER_SIZE: usize = 72;
const MAXIMUM_CACHE_FILE_SIZE: usize = 64 * 1024 * 1024;
const MAXIMUM_DESCRIPTOR_SET_INDEX: u32 = 31;

fn pipeline_error(code: ErrorCode, message: impl AsRef<str>) -> Error {
    Error::new(code, format!("Pipeline manager: {}", message.as_ref()))
}

fn vk_failure(operation: &str, result: vk::Result) -> Error {
    pipeline_error(
        ErrorCode::PipelineCreationFailed,
        format!("{operation} failed with {result:?}"),
    )
    .with_vk(result.as_raw())
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Word-at-a-time hash of a SPIR-V module. Collisions are harmless: a hit is confirmed by
/// comparing the stored module words.
fn module_hash(words: &[u32]) -> u64 {
    words.iter().fold(words.len() as u64, |hash, &word| {
        (hash.rotate_left(5) ^ u64::from(word)).wrapping_mul(0x517c_c1b7_2722_0a95)
    })
}

/// Identity a persisted pipeline cache is bound to. A cache written for any other identity is
/// ignored (and rebuilt), never handed to the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PipelineCacheIdentity {
    /// `VkPhysicalDeviceProperties::vendorID` (cache identity only, never used for decisions).
    pub vendor_id: u32,
    /// `VkPhysicalDeviceProperties::deviceID`.
    pub device_id: u32,
    /// `VkPhysicalDeviceProperties::driverVersion`.
    pub driver_version: u32,
    /// `VkPhysicalDeviceIDProperties::deviceUUID` (zero when unavailable).
    pub device_uuid: [u8; vk::UUID_SIZE],
    /// `VkPhysicalDeviceIDProperties::driverUUID` (zero when unavailable).
    pub driver_uuid: [u8; vk::UUID_SIZE],
}

impl PipelineCacheIdentity {
    /// Identity of a queried physical device.
    pub fn from_physical(physical: &PhysicalDeviceInfo) -> Self {
        Self {
            vendor_id: physical.properties.vendor_id, // kora:identity-check
            device_id: physical.properties.device_id,
            driver_version: physical.properties.driver_version,
            device_uuid: physical.device_uuid.unwrap_or_default(),
            driver_uuid: physical.driver_uuid.unwrap_or_default(),
        }
    }
}

/// Wraps a driver cache blob in KORA's versioned, identity-bound, checksummed file format.
pub(crate) fn encode_cache_file(identity: &PipelineCacheIdentity, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(CACHE_HEADER_SIZE + payload.len());
    out.extend_from_slice(&CACHE_MAGIC);
    out.extend_from_slice(&CACHE_SCHEMA.to_le_bytes());
    out.extend_from_slice(&identity.vendor_id.to_le_bytes()); // kora:identity-check
    out.extend_from_slice(&identity.device_id.to_le_bytes());
    out.extend_from_slice(&identity.driver_version.to_le_bytes());
    out.extend_from_slice(&identity.device_uuid);
    out.extend_from_slice(&identity.driver_uuid);
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&fnv1a(payload).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Returns the driver payload of a cache file, or `None` when the file is foreign, stale,
/// truncated, or corrupt.
pub(crate) fn decode_cache_file<'a>(bytes: &'a [u8], identity: &PipelineCacheIdentity) -> Option<&'a [u8]> {
    if bytes.len() < CACHE_HEADER_SIZE || bytes.len() > MAXIMUM_CACHE_FILE_SIZE || bytes[..8] != CACHE_MAGIC {
        return None;
    }
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap_or_default());
    let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or_default());
    let payload = &bytes[CACHE_HEADER_SIZE..];
    let same_identity = u32_at(12) == identity.vendor_id // kora:identity-check
        && u32_at(16) == identity.device_id
        && u32_at(20) == identity.driver_version
        && bytes[24..40] == identity.device_uuid
        && bytes[40..56] == identity.driver_uuid;
    let intact = u64_at(56) == payload.len() as u64 && u64_at(64) == fnv1a(payload);
    (u32_at(8) == CACHE_SCHEMA && same_identity && intact && !payload.is_empty()).then_some(payload)
}

/// Exact identity of a pipeline variant: the module (by hash, confirmed by comparison on lookup),
/// the entry point, and every metadata field the module is validated against, with
/// specialization overrides applied.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    module_hash: u64,
    module_words: usize,
    interface: Box<[u8]>,
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_value(out: &mut Vec<u8>, value: SpecializationValue) {
    let (tag, bits) = match value {
        SpecializationValue::Bool(b) => (0, u64::from(b)),
        SpecializationValue::I64(i) => (1, i.cast_unsigned()),
        SpecializationValue::U64(u) => (2, u),
        SpecializationValue::F64(f) => (3, f.to_bits()),
    };
    out.push(tag);
    out.extend_from_slice(&bits.to_le_bytes());
}

/// Checks that every override names a declared constant of the same type.
fn check_overrides(metadata: &KernelMetadata, overrides: &[SpecializationConstantMetadata]) -> Status {
    for o in overrides {
        let declared = metadata
            .specialization_constants
            .iter()
            .any(|s| s.id == o.id && s.type_name == o.type_name);
        if !declared {
            return Err(pipeline_error(
                ErrorCode::InvalidArgument,
                format!(
                    "specialization override {} has an unknown ID or mismatched type",
                    o.id
                ),
            ));
        }
    }
    Ok(())
}

fn resolved_value(
    constant: &SpecializationConstantMetadata,
    overrides: &[SpecializationConstantMetadata],
) -> SpecializationValue {
    overrides
        .iter()
        .rev()
        .find(|o| o.id == constant.id)
        .map_or(constant.default_value, |o| o.default_value)
}

impl PipelineKey {
    /// Builds the key for `module` specialized by `overrides` (which must name declared constants
    /// of the same type).
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for an unknown or mistyped override.
    pub fn new(
        module: &[u32],
        metadata: &KernelMetadata,
        overrides: &[SpecializationConstantMetadata],
        entry_point: &str,
    ) -> Result<Self> {
        check_overrides(metadata, overrides)?;
        let mut out = Vec::with_capacity(128 + metadata.bindings.len() * 24);
        put_str(&mut out, entry_point);
        for dimension in metadata.local_size {
            put_u32(&mut out, dimension);
        }
        let mut bindings: Vec<&KernelBindingMetadata> = metadata.bindings.iter().collect();
        bindings.sort_by_key(|b| (b.set, b.binding));
        put_u32(&mut out, u32::try_from(bindings.len()).unwrap_or(u32::MAX));
        for b in bindings {
            put_u32(&mut out, b.set);
            put_u32(&mut out, b.binding);
            out.push(b.kind as u8);
            out.push(b.access as u8);
            put_str(&mut out, &b.name);
        }
        put_u32(&mut out, metadata.push_constant_size);
        put_u32(
            &mut out,
            u32::try_from(metadata.push_constants.len()).unwrap_or(u32::MAX),
        );
        for field in &metadata.push_constants {
            put_str(&mut out, &field.name);
            put_str(&mut out, &field.type_name);
            put_u32(&mut out, field.offset);
            put_u32(&mut out, field.size);
        }
        let mut constants: Vec<&SpecializationConstantMetadata> =
            metadata.specialization_constants.iter().collect();
        constants.sort_by_key(|s| s.id);
        put_u32(&mut out, u32::try_from(constants.len()).unwrap_or(u32::MAX));
        for constant in constants {
            put_u32(&mut out, constant.id);
            put_str(&mut out, &constant.name);
            put_str(&mut out, &constant.type_name);
            put_value(&mut out, constant.default_value);
            put_value(&mut out, resolved_value(constant, overrides));
        }
        Ok(Self {
            module_hash: module_hash(module),
            module_words: module.len(),
            interface: out.into_boxed_slice(),
        })
    }

    /// Stable 64-bit digest of the key (for logs, traces, and performance records).
    pub fn digest(&self) -> u64 {
        fnv1a(&self.interface) ^ self.module_hash.rotate_left(17) ^ self.module_words as u64
    }
}

/// Encodes specialization values as Vulkan expects them: `VkBool32` for `bool`, native-width
/// little-endian integers and floats otherwise.
fn specialization_data(
    constants: &[SpecializationConstantMetadata],
) -> (Vec<vk::SpecializationMapEntry>, Vec<u8>) {
    let mut entries = Vec::with_capacity(constants.len());
    let mut data = Vec::with_capacity(constants.len() * 8);
    for constant in constants {
        let offset = u32::try_from(data.len()).unwrap_or(u32::MAX);
        let start = data.len();
        // Metadata validation guarantees the value variant matches the declared type and range;
        // the casts below only narrow to the declared width.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "values are range-checked by validation"
        )]
        match (constant.type_name.as_str(), constant.default_value) {
            (_, SpecializationValue::Bool(b)) => data.extend_from_slice(&u32::from(b).to_le_bytes()),
            ("i32", SpecializationValue::I64(i)) => data.extend_from_slice(&(i as i32).to_le_bytes()),
            ("u32", SpecializationValue::U64(u)) => data.extend_from_slice(&(u as u32).to_le_bytes()),
            ("f32", SpecializationValue::F64(f)) => data.extend_from_slice(&(f as f32).to_le_bytes()),
            (_, SpecializationValue::I64(i)) => data.extend_from_slice(&i.to_le_bytes()),
            (_, SpecializationValue::U64(u)) => data.extend_from_slice(&u.to_le_bytes()),
            (_, SpecializationValue::F64(f)) => data.extend_from_slice(&f.to_le_bytes()),
        }
        entries.push(
            vk::SpecializationMapEntry::default()
                .constant_id(constant.id)
                .offset(offset)
                .size(data.len() - start),
        );
    }
    (entries, data)
}

const fn descriptor_type(kind: KernelBindingKind) -> vk::DescriptorType {
    match kind {
        KernelBindingKind::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        KernelBindingKind::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        KernelBindingKind::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        KernelBindingKind::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
    }
}

/// Limits for the pipeline and descriptor caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineManagerOptions {
    /// Pipelines kept in the LRU cache.
    pub max_cached_pipelines: usize,
    /// Descriptor sets per descriptor pool.
    pub descriptor_sets_per_pool: u32,
    /// Descriptor pools per set layout before allocation fails.
    pub max_descriptor_pools_per_layout: u32,
}

impl Default for PipelineManagerOptions {
    fn default() -> Self {
        Self {
            max_cached_pipelines: 128,
            descriptor_sets_per_pool: 32,
            max_descriptor_pools_per_layout: 8,
        }
    }
}

/// Cache and descriptor counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PipelineManagerStats {
    /// Lookups served from the pipeline cache.
    pub pipeline_cache_hits: u64,
    /// Lookups that compiled a pipeline.
    pub pipeline_cache_misses: u64,
    /// Pipelines evicted from the LRU cache.
    pub pipeline_evictions: u64,
    /// Descriptor set layouts reused.
    pub descriptor_layout_cache_hits: u64,
    /// Pipeline layouts reused.
    pub pipeline_layout_cache_hits: u64,
    /// Descriptor pools created.
    pub descriptor_pool_growths: u64,
    /// Descriptor sets handed out (fresh or recycled).
    pub descriptor_sets_allocated: u64,
    /// The persisted driver cache was accepted at startup.
    pub persisted_pipeline_cache_loaded: bool,
}

#[derive(Debug, Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    layout_hits: AtomicU64,
    pipeline_layout_hits: AtomicU64,
    pool_growths: AtomicU64,
    sets_allocated: AtomicU64,
    persisted_loaded: AtomicBool,
}

fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

#[derive(Debug, Default)]
struct DescriptorPools {
    pools: Vec<vk::DescriptorPool>,
    /// Sets allocated from the newest pool; older pools are full.
    newest_used: u32,
    free: Vec<vk::DescriptorSet>,
}

/// A descriptor set layout together with the pools its sets come from.
#[derive(Debug)]
struct SetLayout {
    device: Device,
    handle: vk::DescriptorSetLayout,
    pool_sizes: Vec<vk::DescriptorPoolSize>,
    sets_per_pool: u32,
    max_pools: u32,
    counters: Arc<Counters>,
    pools: Mutex<DescriptorPools>,
}

impl SetLayout {
    fn grow(&self, state: &mut DescriptorPools) -> Result<vk::DescriptorPool> {
        if state.pools.len() >= self.max_pools as usize {
            return Err(pipeline_error(
                ErrorCode::AllocationFailed,
                "descriptor pool growth limit reached",
            ));
        }
        let mut sizes = Vec::with_capacity(self.pool_sizes.len().max(1));
        for size in &self.pool_sizes {
            let count = size
                .descriptor_count
                .checked_mul(self.sets_per_pool)
                .ok_or_else(|| {
                    pipeline_error(ErrorCode::Overflow, "descriptor pool size overflows uint32")
                })?;
            sizes.push(vk::DescriptorPoolSize {
                ty: size.ty,
                descriptor_count: count,
            });
        }
        if sizes.is_empty() {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: self.sets_per_pool,
            });
        }
        // Sets are recycled through the free list and released with their pool, so the pool does
        // not need FREE_DESCRIPTOR_SET (which lets drivers use a simpler linear allocator).
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(self.sets_per_pool)
            .pool_sizes(&sizes);
        // SAFETY: `info` and `sizes` are valid for the call; the device is live.
        let pool = unsafe { self.device.raw().create_descriptor_pool(&info, None) }
            .map_err(|r| vk_failure("vkCreateDescriptorPool", r))?;
        state.pools.push(pool);
        state.newest_used = 0;
        bump(&self.counters.pool_growths);
        Ok(pool)
    }

    fn allocate_from(&self, pool: vk::DescriptorPool) -> std::result::Result<vk::DescriptorSet, vk::Result> {
        if let Some(injected) = self.device.faults().check(FaultFunction::AllocateDescriptorSets) {
            return Err(injected);
        }
        let layouts = [self.handle];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and layout belong to this device and outlive the call; access to the
        // pool is externally synchronized by `self.pools`, which the caller holds.
        unsafe { self.device.raw().allocate_descriptor_sets(&info) }.map(|sets| sets[0])
    }

    fn allocate(self: &Arc<Self>) -> Result<DescriptorSetLease> {
        let mut state = lock(&self.pools);
        let set = if let Some(set) = state.free.pop() {
            set
        } else {
            // Pools are filled in order and never asked for more than `max_sets`, so allocation
            // does not depend on drivers reporting VK_ERROR_OUT_OF_POOL_MEMORY.
            let pool = match state.pools.last() {
                Some(&pool) if state.newest_used < self.sets_per_pool => pool,
                _ => self.grow(&mut state)?,
            };
            let set = self
                .allocate_from(pool)
                .map_err(|r| vk_failure("vkAllocateDescriptorSets", r))?;
            state.newest_used += 1;
            set
        };
        bump(&self.counters.sets_allocated);
        Ok(DescriptorSetLease {
            layout: Arc::clone(self),
            set,
        })
    }
}

impl Drop for SetLayout {
    fn drop(&mut self) {
        let state = self
            .pools
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for &pool in &state.pools {
            // SAFETY: every lease holds an Arc to this layout, so no set from these pools is still
            // in use; destroying a pool frees its sets.
            unsafe { self.device.raw().destroy_descriptor_pool(pool, None) };
        }
        // SAFETY: no pipeline layout or pool referencing this layout remains (they hold Arcs).
        unsafe { self.device.raw().destroy_descriptor_set_layout(self.handle, None) };
    }
}

/// A descriptor set borrowed from a pipeline's pools. Dropping the lease recycles the set; it must
/// not be dropped while a pending command buffer still uses it.
#[derive(Debug)]
pub struct DescriptorSetLease {
    layout: Arc<SetLayout>,
    set: vk::DescriptorSet,
}

impl DescriptorSetLease {
    /// The descriptor set handle.
    pub fn handle(&self) -> vk::DescriptorSet {
        self.set
    }
}

impl Drop for DescriptorSetLease {
    fn drop(&mut self) {
        lock(&self.layout.pools).free.push(self.set);
    }
}

#[derive(Debug)]
struct PipelineLayout {
    device: Device,
    handle: vk::PipelineLayout,
    sets: Vec<Arc<SetLayout>>,
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        // SAFETY: only pipelines reference this layout and each holds an Arc to it.
        unsafe { self.device.raw().destroy_pipeline_layout(self.handle, None) };
    }
}

/// A compiled compute pipeline and the layout it was created with.
#[derive(Debug)]
pub struct ComputePipeline {
    device: Device,
    handle: vk::Pipeline,
    layout: Arc<PipelineLayout>,
    bindings: Vec<KernelBindingMetadata>,
    local_size: [u32; 3],
    push_constant_size: u32,
    digest: u64,
    owner: u64,
}

/// Shared reference to a cached pipeline.
pub type PipelineRef = Arc<ComputePipeline>;

impl ComputePipeline {
    /// Pipeline handle.
    pub fn handle(&self) -> vk::Pipeline {
        self.handle
    }
    /// Pipeline layout handle.
    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout.handle
    }
    /// Number of descriptor sets (`0..=max set index`).
    pub fn descriptor_set_count(&self) -> u32 {
        u32::try_from(self.layout.sets.len()).unwrap_or(u32::MAX)
    }
    /// Bindings in metadata order.
    pub fn bindings(&self) -> &[KernelBindingMetadata] {
        &self.bindings
    }
    /// Effective workgroup size, including `LocalSizeId` specialization.
    pub fn local_size(&self) -> [u32; 3] {
        self.local_size
    }
    /// Push-constant block size in bytes.
    pub fn push_constant_size(&self) -> u32 {
        self.push_constant_size
    }
    /// Digest of the pipeline key.
    pub fn digest(&self) -> u64 {
        self.digest
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        // SAFETY: the last reference is gone, so no command buffer recorded by KORA still uses the
        // pipeline (submissions keep their pipelines alive until completion).
        unsafe { self.device.raw().destroy_pipeline(self.handle, None) };
    }
}

#[derive(Debug)]
struct CacheEntry {
    module: Arc<[u32]>,
    pipeline: PipelineRef,
    last_used: u64,
}

#[derive(Debug, Default)]
struct State {
    driver_cache: Option<vk::PipelineCache>,
    pipelines: HashMap<PipelineKey, CacheEntry>,
    clock: u64,
    set_layouts: HashMap<Vec<u8>, Arc<SetLayout>>,
    pipeline_layouts: HashMap<(Vec<vk::DescriptorSetLayout>, u32), Arc<PipelineLayout>>,
}

#[derive(Debug)]
struct Shared {
    id: u64,
    device: Device,
    capabilities: DeviceCapabilities,
    identity: PipelineCacheIdentity,
    cache_file: Option<PathBuf>,
    options: PipelineManagerOptions,
    counters: Arc<Counters>,
    state: Mutex<State>,
}

/// Per-device owner of compute pipelines, descriptor layouts and pools, and the driver pipeline
/// cache. Thread safe; cheap to share behind an `Arc`.
#[derive(Debug)]
pub struct PipelineManager {
    shared: Shared,
}

impl PipelineManager {
    /// Creates a manager. `cache_file`, when set, is loaded lazily on first compilation and
    /// written back by [`save_pipeline_cache`](Self::save_pipeline_cache) and on drop.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] when a limit is zero.
    pub fn new(
        device: Device,
        capabilities: DeviceCapabilities,
        identity: PipelineCacheIdentity,
        cache_file: Option<PathBuf>,
        options: PipelineManagerOptions,
    ) -> Result<Self> {
        if options.max_cached_pipelines == 0
            || options.descriptor_sets_per_pool == 0
            || options.max_descriptor_pools_per_layout == 0
        {
            return Err(pipeline_error(
                ErrorCode::InvalidArgument,
                "cache and descriptor-pool limits must be greater than zero",
            ));
        }
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Ok(Self {
            shared: Shared {
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                device,
                capabilities,
                identity,
                cache_file,
                options,
                counters: Arc::default(),
                state: Mutex::default(),
            },
        })
    }

    /// The device pipelines are created on.
    pub fn device(&self) -> &Device {
        &self.shared.device
    }

    /// Current counters.
    pub fn stats(&self) -> PipelineManagerStats {
        let c = &self.shared.counters;
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        PipelineManagerStats {
            pipeline_cache_hits: load(&c.hits),
            pipeline_cache_misses: load(&c.misses),
            pipeline_evictions: load(&c.evictions),
            descriptor_layout_cache_hits: load(&c.layout_hits),
            pipeline_layout_cache_hits: load(&c.pipeline_layout_hits),
            descriptor_pool_growths: load(&c.pool_growths),
            descriptor_sets_allocated: load(&c.sets_allocated),
            persisted_pipeline_cache_loaded: c.persisted_loaded.load(Ordering::Relaxed),
        }
    }

    /// Number of pipelines currently in the LRU cache.
    pub fn cached_pipelines(&self) -> usize {
        lock(&self.shared.state).pipelines.len()
    }

    fn lookup(&self, key: &PipelineKey, module: &[u32]) -> Option<PipelineRef> {
        let mut state = lock(&self.shared.state);
        state.clock += 1;
        let now = state.clock;
        let entry = state.pipelines.get_mut(key)?;
        if *entry.module != *module {
            return None;
        }
        entry.last_used = now;
        bump(&self.shared.counters.hits);
        Some(Arc::clone(&entry.pipeline))
    }

    /// Returns a cached pipeline without compiling or validating anything. `None` also covers an
    /// invalid override.
    pub fn find_cached(
        &self,
        module: &[u32],
        metadata: &KernelMetadata,
        overrides: &[SpecializationConstantMetadata],
        entry_point: &str,
    ) -> Option<PipelineRef> {
        let key = PipelineKey::new(module, metadata, overrides, entry_point).ok()?;
        self.lookup(&key, module)
    }

    /// Returns the pipeline for `module` with `overrides` applied to its specialization constants,
    /// validating and compiling it on a cache miss.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidKernel`] when the module or metadata is invalid,
    /// [`ErrorCode::InvalidArgument`] for bad overrides, and
    /// [`ErrorCode::PipelineCreationFailed`] when the driver rejects the pipeline.
    pub fn get_or_create(
        &self,
        module: &[u32],
        metadata: &KernelMetadata,
        overrides: &[SpecializationConstantMetadata],
        entry_point: &str,
    ) -> Result<PipelineRef> {
        let key = PipelineKey::new(module, metadata, overrides, entry_point)?;
        if let Some(pipeline) = self.lookup(&key, module) {
            return Ok(pipeline);
        }
        bump(&self.shared.counters.misses);
        let reflection = validate_and_reflect(module, metadata, &self.shared.capabilities, entry_point)?;
        let mut resolved = metadata.clone();
        for constant in &mut resolved.specialization_constants {
            constant.default_value = resolved_value(constant, overrides);
        }
        if reflection.local_size_id_based {
            for dimension in 0..3 {
                let spec_id = reflection.local_size_spec_ids[dimension];
                let value = resolved
                    .specialization_constants
                    .iter()
                    .find(|s| s.id == spec_id)
                    .and_then(|s| match (s.type_name.as_str(), s.default_value) {
                        ("u32", SpecializationValue::U64(v)) => u32::try_from(v).ok().filter(|&v| v != 0),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        pipeline_error(
                            ErrorCode::InvalidArgument,
                            "LocalSizeId override must be a nonzero u32 specialization value",
                        )
                    })?;
                resolved.local_size[dimension] = value;
            }
        }
        let entry = CString::new(entry_point)
            .map_err(|_| pipeline_error(ErrorCode::InvalidArgument, "entry point contains a NUL byte"))?;

        let (layout, driver_cache) = {
            let mut state = lock(&self.shared.state);
            let layout = self.pipeline_layout(&mut state, &resolved)?;
            (layout, self.driver_cache(&mut state)?)
        };
        let handle = self.compile(module, &resolved, &entry, layout.handle, driver_cache)?;
        let pipeline = Arc::new(ComputePipeline {
            device: self.shared.device.clone(),
            handle,
            layout,
            bindings: resolved.bindings,
            local_size: resolved.local_size,
            push_constant_size: resolved.push_constant_size,
            digest: key.digest(),
            owner: self.shared.id,
        });

        let mut state = lock(&self.shared.state);
        // Another thread may have compiled the same variant while the lock was released.
        if let Some(existing) = state.pipelines.get(&key)
            && *existing.module == *module
        {
            return Ok(Arc::clone(&existing.pipeline));
        }
        state.clock += 1;
        let entry = CacheEntry {
            module: Arc::from(module),
            pipeline: Arc::clone(&pipeline),
            last_used: state.clock,
        };
        state.pipelines.insert(key, entry);
        while state.pipelines.len() > self.shared.options.max_cached_pipelines {
            let Some(victim) = state
                .pipelines
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            state.pipelines.remove(&victim);
            bump(&self.shared.counters.evictions);
        }
        Ok(pipeline)
    }

    fn compile(
        &self,
        module: &[u32],
        metadata: &KernelMetadata,
        entry: &CString,
        layout: vk::PipelineLayout,
        driver_cache: vk::PipelineCache,
    ) -> Result<vk::Pipeline> {
        let device = self.shared.device.raw();
        let shader_info = vk::ShaderModuleCreateInfo::default().code(module);
        // SAFETY: `module` was validated structurally and outlives the call.
        let shader = unsafe { device.create_shader_module(&shader_info, None) }
            .map_err(|r| vk_failure("vkCreateShaderModule", r))?;
        let (entries, data) = specialization_data(&metadata.specialization_constants);
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&data);
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(entry);
        if !entries.is_empty() {
            stage = stage.specialization_info(&specialization);
        }
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let created = match self
            .shared
            .device
            .faults()
            .check(FaultFunction::CreateComputePipelines)
        {
            Some(injected) => Err(injected),
            // SAFETY: every handle and borrowed array in `info` is live for the call; the driver
            // cache is internally synchronized.
            None => unsafe { device.create_compute_pipelines(driver_cache, &[info], None) }
                .map(|pipelines| pipelines[0])
                .map_err(|(partial, result)| {
                    for pipeline in partial.into_iter().filter(|p| *p != vk::Pipeline::null()) {
                        // SAFETY: the failed call returned this handle and nothing else uses it.
                        unsafe { device.destroy_pipeline(pipeline, None) };
                    }
                    result
                }),
        };
        // SAFETY: the shader module is not needed once pipeline creation has returned.
        unsafe { device.destroy_shader_module(shader, None) };
        created.map_err(|r| vk_failure("vkCreateComputePipelines", r))
    }

    fn set_layout(&self, state: &mut State, bindings: &[&KernelBindingMetadata]) -> Result<Arc<SetLayout>> {
        let mut key = Vec::with_capacity(bindings.len() * 8);
        for binding in bindings {
            put_u32(&mut key, binding.binding);
            put_u32(&mut key, descriptor_type(binding.kind).as_raw().cast_unsigned());
        }
        if let Some(layout) = state.set_layouts.get(&key) {
            bump(&self.shared.counters.layout_hits);
            return Ok(Arc::clone(layout));
        }
        let vk_bindings: Vec<_> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(descriptor_type(b.kind))
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let mut pool_sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for binding in bindings {
            let ty = descriptor_type(binding.kind);
            match pool_sizes.iter_mut().find(|s| s.ty == ty) {
                Some(size) => size.descriptor_count += 1,
                None => pool_sizes.push(vk::DescriptorPoolSize {
                    ty,
                    descriptor_count: 1,
                }),
            }
        }
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);
        // SAFETY: `info` borrows `vk_bindings`, which outlives the call; the device is live.
        let handle = unsafe { self.shared.device.raw().create_descriptor_set_layout(&info, None) }
            .map_err(|r| vk_failure("vkCreateDescriptorSetLayout", r))?;
        let layout = Arc::new(SetLayout {
            device: self.shared.device.clone(),
            handle,
            pool_sizes,
            sets_per_pool: self.shared.options.descriptor_sets_per_pool,
            max_pools: self.shared.options.max_descriptor_pools_per_layout,
            counters: Arc::clone(&self.shared.counters),
            pools: Mutex::default(),
        });
        state.set_layouts.insert(key, Arc::clone(&layout));
        Ok(layout)
    }

    fn pipeline_layout(&self, state: &mut State, metadata: &KernelMetadata) -> Result<Arc<PipelineLayout>> {
        let mut sets = Vec::new();
        if let Some(max_set) = metadata.bindings.iter().map(|b| b.set).max() {
            if max_set > MAXIMUM_DESCRIPTOR_SET_INDEX {
                return Err(pipeline_error(
                    ErrorCode::InvalidKernel,
                    "descriptor set index exceeds the supported limit of 31",
                ));
            }
            for set in 0..=max_set {
                let mut bindings: Vec<_> = metadata.bindings.iter().filter(|b| b.set == set).collect();
                bindings.sort_by_key(|b| b.binding);
                sets.push(self.set_layout(state, &bindings)?);
            }
        }
        let key = (
            sets.iter().map(|s| s.handle).collect::<Vec<_>>(),
            metadata.push_constant_size,
        );
        if let Some(layout) = state.pipeline_layouts.get(&key) {
            bump(&self.shared.counters.pipeline_layout_hits);
            return Ok(Arc::clone(layout));
        }
        let ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: metadata.push_constant_size,
        }];
        let ranges: &[_] = if metadata.push_constant_size == 0 {
            &[]
        } else {
            &ranges
        };
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&key.0)
            .push_constant_ranges(ranges);
        // SAFETY: the set layouts are live (held by `sets`) and the arrays outlive the call.
        let handle = unsafe { self.shared.device.raw().create_pipeline_layout(&info, None) }
            .map_err(|r| vk_failure("vkCreatePipelineLayout", r))?;
        let layout = Arc::new(PipelineLayout {
            device: self.shared.device.clone(),
            handle,
            sets,
        });
        state.pipeline_layouts.insert(key, Arc::clone(&layout));
        Ok(layout)
    }

    fn driver_cache(&self, state: &mut State) -> Result<vk::PipelineCache> {
        if let Some(cache) = state.driver_cache {
            return Ok(cache);
        }
        let file = self
            .shared
            .cache_file
            .as_deref()
            .and_then(|path| fsutil::read_bounded(path, MAXIMUM_CACHE_FILE_SIZE as u64).ok());
        let payload = file
            .as_deref()
            .and_then(|bytes| decode_cache_file(bytes, &self.shared.identity));
        let create = |data: &[u8]| {
            let info = vk::PipelineCacheCreateInfo::default().initial_data(data);
            // SAFETY: `data` outlives the call; drivers validate the blob header themselves and
            // KORA only passes blobs written for this exact device and driver identity.
            unsafe { self.shared.device.raw().create_pipeline_cache(&info, None) }
        };
        let cache = match payload.map(create) {
            Some(Ok(cache)) => {
                self.shared
                    .counters
                    .persisted_loaded
                    .store(true, Ordering::Relaxed);
                cache
            }
            // A rejected or absent blob falls back to an empty cache.
            Some(Err(_)) | None => create(&[]).map_err(|r| vk_failure("vkCreatePipelineCache", r))?,
        };
        state.driver_cache = Some(cache);
        Ok(cache)
    }

    /// Borrows a descriptor set for set `set_index` of `pipeline`, recycling a released set when
    /// one is available and growing the pools otherwise.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for a foreign pipeline or out-of-range set,
    /// [`ErrorCode::AllocationFailed`] when the pool limit is reached, and
    /// [`ErrorCode::PipelineCreationFailed`] when the driver fails.
    pub fn allocate_descriptor_set(
        &self,
        pipeline: &PipelineRef,
        set_index: u32,
    ) -> Result<DescriptorSetLease> {
        if pipeline.owner != self.shared.id {
            return Err(pipeline_error(
                ErrorCode::InvalidArgument,
                "pipeline does not belong to this manager",
            ));
        }
        let layout = pipeline.layout.sets.get(set_index as usize).ok_or_else(|| {
            pipeline_error(ErrorCode::InvalidArgument, "descriptor set index is out of range")
        })?;
        layout.allocate()
    }

    /// Writes the driver pipeline cache to the cache file atomically. A no-op without a cache file
    /// or before the first compilation.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::IoError`] when the cache cannot be read from the driver or written.
    pub fn save_pipeline_cache(&self) -> Status {
        let state = lock(&self.shared.state);
        self.shared.save(state.driver_cache)
    }
}

impl Shared {
    fn save(&self, cache: Option<vk::PipelineCache>) -> Status {
        let (Some(cache), Some(path)) = (cache, self.cache_file.as_deref()) else {
            return Ok(());
        };
        // SAFETY: the cache is live and internally synchronized.
        let payload = unsafe { self.device.raw().get_pipeline_cache_data(cache) }.map_err(|r| {
            pipeline_error(
                ErrorCode::IoError,
                format!("vkGetPipelineCacheData failed with {r:?}"),
            )
        })?;
        if payload.is_empty() {
            return Ok(());
        }
        if payload.len() > MAXIMUM_CACHE_FILE_SIZE - CACHE_HEADER_SIZE {
            return Err(pipeline_error(
                ErrorCode::IoError,
                "driver pipeline cache exceeds the 64 MiB persistence limit",
            ));
        }
        fsutil::write_atomic(path, &encode_cache_file(&self.identity, &payload))
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let cache = state.driver_cache.take();
        if let Err(error) = self.save(cache) {
            crate::kora_log!(Device, Warn, "{error}");
        }
        if let Some(cache) = cache {
            // SAFETY: compilation borrows the manager, so no thread is using the cache now.
            unsafe { self.device.raw().destroy_pipeline_cache(cache, None) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{KernelAccess, PushConstantFieldMetadata};

    fn metadata() -> KernelMetadata {
        KernelMetadata {
            kernel_id: "pipeline-key-test".into(),
            local_size: [8, 2, 1],
            bindings: vec![KernelBindingMetadata {
                set: 0,
                binding: 1,
                name: "output".into(),
                kind: KernelBindingKind::StorageBuffer,
                access: KernelAccess::Write,
            }],
            push_constant_size: 4,
            push_constants: vec![PushConstantFieldMetadata {
                name: "count".into(),
                type_name: "u32".into(),
                offset: 0,
                size: 4,
            }],
            specialization_constants: vec![SpecializationConstantMetadata {
                id: 9,
                name: "factor".into(),
                type_name: "u32".into(),
                default_value: SpecializationValue::U64(3),
            }],
            ..KernelMetadata::default()
        }
    }

    fn key(module: &[u32], metadata: &KernelMetadata, entry: &str) -> PipelineKey {
        PipelineKey::new(module, metadata, &[], entry).unwrap()
    }

    #[test]
    fn key_is_stable_and_covers_module_entry_and_interface() {
        let module = [0x0723_0203, 0x0001_0000, 0, 1];
        let base = metadata();
        let first = key(&module, &base, "main");
        assert_eq!(first, key(&module, &base, "main"));
        assert_eq!(first.digest(), key(&module, &base, "main").digest());

        let mut changed = module;
        changed[3] = 2;
        assert_ne!(first, key(&changed, &base, "main"));
        assert_ne!(first, key(&module[..3], &base, "main"));
        assert_ne!(first, key(&module, &base, "other"));

        let variants: [fn(&mut KernelMetadata); 7] = [
            |m| m.local_size[0] = 4,
            |m| m.specialization_constants[0].default_value = SpecializationValue::U64(4),
            |m| m.bindings[0].access = KernelAccess::ReadWrite,
            |m| m.bindings[0].name = "renamed".into(),
            |m| m.push_constants[0].offset = 4,
            |m| m.push_constant_size = 8,
            |m| m.bindings[0].kind = KernelBindingKind::UniformBuffer,
        ];
        for mutate in variants {
            let mut m = base.clone();
            mutate(&mut m);
            assert_ne!(first, key(&module, &m, "main"));
        }
        let mut renamed_kernel = base.clone();
        renamed_kernel.kernel_id = "other-id".into();
        assert_eq!(
            first,
            key(&module, &renamed_kernel, "main"),
            "kernel_id does not affect pipelines"
        );
    }

    #[test]
    fn overrides_change_the_key_and_must_match_declared_constants() {
        let module = [1, 2, 3];
        let base = metadata();
        let mut four = base.specialization_constants.clone();
        four[0].default_value = SpecializationValue::U64(4);
        let overridden = PipelineKey::new(&module, &base, &four, "main").unwrap();
        assert_ne!(overridden, key(&module, &base, "main"));
        let mut redundant = base.specialization_constants.clone();
        redundant[0].name = "ignored".into();
        assert_eq!(
            PipelineKey::new(&module, &base, &redundant, "main").unwrap(),
            key(&module, &base, "main"),
            "an override equal to the default resolves to the same pipeline"
        );

        let mut unknown = four.clone();
        unknown[0].id = 99;
        let error = PipelineKey::new(&module, &base, &unknown, "main").unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidArgument);
        let mut mistyped = four;
        mistyped[0].type_name = "i32".into();
        assert!(PipelineKey::new(&module, &base, &mistyped, "main").is_err());
    }

    #[test]
    fn module_hash_distinguishes_order_and_length() {
        assert_ne!(module_hash(&[1, 2]), module_hash(&[2, 1]));
        assert_ne!(module_hash(&[0]), module_hash(&[0, 0]));
        assert_ne!(module_hash(&[]), module_hash(&[0]));
    }

    fn identity() -> PipelineCacheIdentity {
        PipelineCacheIdentity {
            vendor_id: 42, // kora:identity-check
            device_id: 7,
            driver_version: 11,
            device_uuid: [1; vk::UUID_SIZE],
            driver_uuid: [2; vk::UUID_SIZE],
        }
    }

    #[test]
    fn cache_file_round_trips_and_rejects_foreign_or_corrupt_files() {
        let payload = b"driver-cache-bytes".to_vec();
        let file = encode_cache_file(&identity(), &payload);
        assert_eq!(file.len(), CACHE_HEADER_SIZE + payload.len());
        assert_eq!(decode_cache_file(&file, &identity()), Some(payload.as_slice()));

        let foreign = [
            PipelineCacheIdentity {
                device_id: 8,
                ..identity()
            },
            PipelineCacheIdentity {
                driver_version: 12,
                ..identity()
            },
            PipelineCacheIdentity {
                device_uuid: [9; vk::UUID_SIZE],
                ..identity()
            },
            PipelineCacheIdentity {
                driver_uuid: [9; vk::UUID_SIZE],
                ..identity()
            },
            PipelineCacheIdentity {
                vendor_id: 43, // kora:identity-check
                ..identity()
            },
        ];
        for other in foreign {
            assert_eq!(decode_cache_file(&file, &other), None, "{other:?}");
        }
        for corrupt_at in [0, 8, 56, 64, CACHE_HEADER_SIZE] {
            let mut corrupt = file.clone();
            corrupt[corrupt_at] ^= 0xff;
            assert_eq!(
                decode_cache_file(&corrupt, &identity()),
                None,
                "byte {corrupt_at}"
            );
        }
        assert_eq!(decode_cache_file(&file[..file.len() - 1], &identity()), None);
        assert_eq!(decode_cache_file(&file[..10], &identity()), None);
        assert_eq!(decode_cache_file(b"not a KORA pipeline cache", &identity()), None);
        assert_eq!(
            decode_cache_file(&encode_cache_file(&identity(), &[]), &identity()),
            None
        );
    }

    #[test]
    fn cache_file_layout_matches_the_v0_format() {
        // Files written by KORA 0.x stay readable: fixed offsets, little endian, FNV-1a checksum.
        let file = encode_cache_file(&identity(), b"x");
        assert_eq!(&file[..8], b"KORAPCH1");
        assert_eq!(file[8..12], 1_u32.to_le_bytes());
        assert_eq!(file[12..16], 42_u32.to_le_bytes());
        assert_eq!(file[56..64], 1_u64.to_le_bytes());
        assert_eq!(file[64..72], fnv1a(b"x").to_le_bytes());
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn specialization_data_uses_declared_widths() {
        let constant = |id, type_name: &str, value| SpecializationConstantMetadata {
            id,
            name: format!("c{id}"),
            type_name: type_name.into(),
            default_value: value,
        };
        let constants = [
            constant(0, "bool", SpecializationValue::Bool(true)),
            constant(1, "i32", SpecializationValue::I64(-2)),
            constant(2, "u32", SpecializationValue::U64(7)),
            constant(3, "f32", SpecializationValue::F64(1.5)),
            constant(4, "i64", SpecializationValue::I64(-3)),
            constant(5, "u64", SpecializationValue::U64(u64::MAX)),
            constant(6, "f64", SpecializationValue::F64(0.25)),
        ];
        let (entries, data) = specialization_data(&constants);
        let layout: Vec<_> = entries
            .iter()
            .map(|e| (e.constant_id, e.offset, e.size))
            .collect();
        assert_eq!(
            layout,
            [
                (0, 0, 4),
                (1, 4, 4),
                (2, 8, 4),
                (3, 12, 4),
                (4, 16, 8),
                (5, 24, 8),
                (6, 32, 8)
            ]
        );
        assert_eq!(data[0..4], 1_u32.to_le_bytes());
        assert_eq!(data[4..8], (-2_i32).to_le_bytes());
        assert_eq!(data[8..12], 7_u32.to_le_bytes());
        assert_eq!(data[12..16], 1.5_f32.to_le_bytes());
        assert_eq!(data[16..24], (-3_i64).to_le_bytes());
        assert_eq!(data[24..32], u64::MAX.to_le_bytes());
        assert_eq!(data[32..40], 0.25_f64.to_le_bytes());
        assert_eq!(specialization_data(&[]).1.len(), 0);
    }

    #[test]
    fn descriptor_types_cover_every_binding_kind() {
        assert_eq!(
            descriptor_type(KernelBindingKind::StorageBuffer),
            vk::DescriptorType::STORAGE_BUFFER
        );
        assert_eq!(
            descriptor_type(KernelBindingKind::UniformBuffer),
            vk::DescriptorType::UNIFORM_BUFFER
        );
        assert_eq!(
            descriptor_type(KernelBindingKind::StorageImage),
            vk::DescriptorType::STORAGE_IMAGE
        );
        assert_eq!(
            descriptor_type(KernelBindingKind::SampledImage),
            vk::DescriptorType::SAMPLED_IMAGE
        );
    }
}
