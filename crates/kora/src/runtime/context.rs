//! Per-device runtime state: queues, memory pools, transfers, pipelines, and timestamps.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use ash::vk;

use super::replica::Replica;
use crate::buffer::{BufferHandle, WeakBufferHandle};
use crate::calibration::CalibrationRecord;
use crate::device::DeviceInfo;
use crate::error::{Error, ErrorCode, Result, Status};
use crate::ids::DeviceId;
use crate::kernel::{
    DeviceCapabilities, PipelineCacheIdentity, PipelineManager, PipelineManagerOptions, Trial,
    WorkgroupAutotuner,
};
use crate::memory::{
    MemoryManager, MemoryPolicy, MemoryPool, MemoryPoolConfig, MemoryTypeRequirements, MemoryUsageClass,
    PoolAllocationRequest, PoolBudget, QueueTimelineUse, SharedPool, TransferBuffer, TransferEngine,
    TransferOptions,
};
use crate::options::{ResolvedOptions, ShutdownMode, TelemetryLevel};
use crate::performance_db::PerformanceDatabase;
use crate::profiling::TimestampRing;
use crate::scheduler::ScoreBreakdown;
use crate::vulkan::{
    CoalescingOptions, CommandPoolSet, Device, DeviceFaultState, DeviceOptions, Instance, PhysicalDeviceInfo,
    QueueContext, QueueContextOptions, QueueFamilyPolicy, device_info,
};

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Budget ledger shared by a device's pools.
struct HeapHooks {
    heap: u32,
    memory: Arc<Mutex<MemoryManager>>,
    device: Device,
    fraction: f64,
    last_refresh: Arc<Mutex<Option<Instant>>>,
}

impl PoolBudget for HeapHooks {
    fn reserve(&self, bytes: u64) -> Status {
        lock(&self.memory).reserve(self.heap, bytes)
    }
    fn release(&self, bytes: u64) {
        let _ = lock(&self.memory).release(self.heap, bytes);
    }
    fn allocation_failed(&self) {
        let _ = lock(&self.memory).record_allocation_failure(self.heap, 0.9);
    }
    fn refresh_budget(&self) -> Status {
        refresh_budget(
            &self.device,
            &self.memory,
            self.fraction,
            &self.last_refresh,
            true,
        )
    }
}

fn refresh_budget(
    device: &Device,
    memory: &Mutex<MemoryManager>,
    fraction: f64,
    last_refresh: &Mutex<Option<Instant>>,
    force: bool,
) -> Status {
    let mut last = lock(last_refresh);
    if !force && last.is_some_and(|t| t.elapsed() < Duration::from_secs(1)) {
        return Ok(());
    }
    let budget = device.memory_budget();
    lock(memory).refresh_budget(&device.physical().memory_properties, budget.as_ref(), fraction)?;
    *last = Some(Instant::now());
    Ok(())
}

/// An autotuning trial awaiting its GPU timestamp.
pub(crate) struct PendingAutotune {
    pub(crate) tuner: Arc<WorkgroupAutotuner>,
    pub(crate) trial: Trial,
    pub(crate) key: String,
}

/// Host-side phase durations of one dispatch (nanoseconds; zero when unmeasured).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct PhaseProfile {
    pub(crate) submission_id: u64,
    pub(crate) dispatch_validation_ns: u64,
    pub(crate) binding_preflight_ns: u64,
    pub(crate) pipeline_lookup_ns: u64,
    pub(crate) buffer_residency_ns: u64,
    pub(crate) submission_assembly_ns: u64,
    pub(crate) host_submission_ns: u64,
    pub(crate) queue_enqueue_ns: u64,
    pub(crate) command_buffer_acquire_ns: u64,
    pub(crate) descriptor_setup_ns: u64,
    pub(crate) command_recording_ns: u64,
    pub(crate) queue_completion_ns: u64,
    pub(crate) completion_worker_delay_ns: u64,
    pub(crate) device_completion_wait_ns: u64,
    pub(crate) timestamp_collection_ns: u64,
    pub(crate) completion_callback_ns: u64,
}

/// Elapsed nanoseconds, at least 1 so a measured phase is never reported as unmeasured.
pub(crate) fn elapsed_ns(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
}

/// The adaptive scheduler's score for a traced dispatch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TraceScore {
    pub(crate) device_index: u32,
    pub(crate) score: ScoreBreakdown,
}

const TRACE_SCORE_HISTORY: usize = 4096;

#[derive(Debug, Default)]
pub(crate) struct TraceScores {
    pending: HashMap<u64, TraceScore>,
    pub(crate) completed: HashMap<u64, TraceScore>,
    order: VecDeque<u64>,
}

impl TraceScores {
    pub(crate) fn add_pending(&mut self, submission: u64, score: TraceScore) {
        if self.pending.len() >= TRACE_SCORE_HISTORY {
            self.pending.clear();
        }
        self.pending.insert(submission, score);
    }
    fn complete(&mut self, submission: u64) {
        if let Some(score) = self.pending.remove(&submission) {
            self.completed.insert(submission, score);
            self.order.push_back(submission);
            while self.order.len() > TRACE_SCORE_HISTORY {
                if let Some(oldest) = self.order.pop_front() {
                    self.completed.remove(&oldest);
                }
            }
        }
    }
}

/// Timestamp sampling period for a telemetry level (0 = never).
pub(crate) const fn timestamp_interval(level: TelemetryLevel) -> u32 {
    match level {
        TelemetryLevel::Off => 0,
        TelemetryLevel::Basic => 16,
        TelemetryLevel::Profile | TelemetryLevel::Trace => 1,
    }
}

/// Everything the runtime owns for one selected device.
pub(crate) struct DeviceContext {
    /// Unique across all runtimes; keys buffer replicas.
    pub(crate) key: u64,
    pub(crate) runtime: u64,
    pub(crate) id: DeviceId,
    pub(crate) info: DeviceInfo,
    pub(crate) device: Device,
    pub(crate) fault_state: Arc<DeviceFaultState>,
    pub(crate) queue: Arc<QueueContext>,
    pub(crate) transfer_queue: Arc<QueueContext>,
    pub(crate) families: QueueFamilyPolicy,
    pub(crate) transfers: TransferEngine,
    pub(crate) pipelines: PipelineManager,
    pub(crate) capabilities: DeviceCapabilities,
    memory: Arc<Mutex<MemoryManager>>,
    pools: Vec<SharedPool>,
    fraction: f64,
    last_refresh: Arc<Mutex<Option<Instant>>>,
    /// Unified memory: idle host-visible replicas are written and read through their mapping.
    pub(crate) direct_host_access: bool,
    pub(crate) atom_size: u64,
    pub(crate) commands: Mutex<Option<CommandPoolSet>>,
    pub(crate) timestamps: Option<TimestampRing>,
    pub(crate) timestamp_interval: u32,
    pub(crate) timestamp_counter: AtomicU64,
    pub(crate) autotuning: bool,
    pub(crate) performance: Arc<PerformanceDatabase>,
    pub(crate) performance_path: PathBuf,
    pub(crate) pending_autotune: Mutex<HashMap<u64, PendingAutotune>>,
    /// Adaptive dispatches awaiting their execution timestamp: submission → database key.
    pub(crate) pending_observations: Mutex<HashMap<u64, String>>,
    pub(crate) trace_scores: Mutex<TraceScores>,
    pub(crate) phase_profile: bool,
    pub(crate) trace: bool,
    pub(crate) latest_profile: Mutex<PhaseProfile>,
    pub(crate) calibration: Mutex<CalibrationRecord>,
    pub(crate) calibration_path: Option<PathBuf>,
    /// Orders hazard bookkeeping with queue enqueue so barrier decisions match submission order.
    pub(crate) submit_lock: Mutex<()>,
    pub(crate) accepting: AtomicBool,
    pub(crate) quarantined: AtomicBool,
    pub(crate) uploads: AtomicU64,
    pub(crate) downloads: AtomicU64,
    replicated: Mutex<Vec<WeakBufferHandle>>,
    stopped: AtomicBool,
    shutdown_mode: ShutdownMode,
}

impl std::fmt::Debug for DeviceContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceContext")
            .field("id", &self.id)
            .field("name", &self.info.name)
            .finish()
    }
}

static NEXT_CONTEXT: AtomicU64 = AtomicU64::new(1);

fn pool_config(info: &DeviceInfo, memory: &MemoryManager, index: u32) -> Result<MemoryPoolConfig> {
    let memory_type = info.memory_types[index as usize];
    let heap = memory_type.heap_index;
    let budget = memory.budget(heap)?.limit_bytes;
    Ok(MemoryPoolConfig {
        memory_type_index: index,
        heap_index: heap,
        usage: MemoryUsageClass::General,
        host_visible: memory_type.host_visible,
        host_coherent: memory_type.host_coherent,
        heap_size: info.memory_heaps[heap as usize].size_bytes,
        budget,
        block_size_hint: budget.clamp(1 << 20, 16 << 20),
        ..MemoryPoolConfig::default()
    })
}

impl DeviceContext {
    /// Creates the logical device and everything that hangs off it.
    pub(crate) fn new(
        instance: &Instance,
        physical: &PhysicalDeviceInfo,
        id: DeviceId,
        runtime: u64,
        options: &ResolvedOptions,
        performance: Arc<PerformanceDatabase>,
    ) -> Result<Arc<Self>> {
        // Runtime transfers run on the compute queue so dispatches are ordered after their uploads
        // without cross-queue semaphores; overlapping copies on a dedicated transfer queue is a
        // tracked performance follow-up.
        let _ = options.disable_multi_queue;
        let device_options = DeviceOptions {
            enable_dedicated_transfer_queue: false,
            enable_timeline_semaphore: !options.force_legacy_sync,
            enable_synchronization2: !options.force_legacy_sync,
            ..DeviceOptions::default()
        };
        let device = instance.create_device(physical, &device_options)?;
        let info = device_info(physical);
        let fault_state = Arc::new(DeviceFaultState::default());
        if device.supports_fault_diagnostics() {
            let query_device = device.clone();
            fault_state.set_diagnostics_query(Some(Arc::new(move || query_device.fault_snapshot())));
        }
        let queue_options = |coalescing: bool| QueueContextOptions {
            force_fence_fallback: options.force_legacy_sync,
            coalescing: CoalescingOptions {
                enabled: coalescing,
                // Coalesce work already queued together, but never delay an isolated dispatch.
                maximum_wait_ns: 0,
                ..CoalescingOptions::default()
            },
            fault_state: Some(Arc::clone(&fault_state)),
            ..QueueContextOptions::default()
        };
        let queue = Arc::new(QueueContext::new(
            device.clone(),
            device.compute_queue(),
            queue_options(true),
        )?);
        let families = QueueFamilyPolicy::new(device.compute_queue_family(), device.transfer_queue_family());
        let transfer_queue = if families.compute_family == families.transfer_family {
            Arc::clone(&queue)
        } else {
            Arc::new(QueueContext::new(
                device.clone(),
                device.transfer_queue(),
                queue_options(false),
            )?)
        };

        let fraction = options.memory_budget_fraction;
        let memory = Arc::new(Mutex::new(MemoryManager::new(
            &info,
            &physical.memory_properties,
            physical.memory_budget.as_ref(),
            fraction,
        )?));
        let last_refresh = Arc::new(Mutex::new(Some(Instant::now())));
        let mut pools = Vec::with_capacity(info.memory_types.len());
        for index in 0..info.memory_types.len() as u32 {
            let mut config = pool_config(&info, &lock(&memory), index)?;
            config.budget_hooks = Some(Arc::new(HeapHooks {
                heap: config.heap_index,
                memory: Arc::clone(&memory),
                device: device.clone(),
                fraction,
                last_refresh: Arc::clone(&last_refresh),
            }));
            pools.push(Arc::new(Mutex::new(MemoryPool::new(device.clone(), config)?)));
        }
        let host_pool = |policy| {
            let required = MemoryTypeRequirements {
                host_visible: true,
                ..MemoryTypeRequirements::default()
            };
            lock(&memory)
                .select_memory_type(policy, u32::MAX, &required)
                .map(|index| Arc::clone(&pools[index as usize]))
                .map_err(|_| {
                    Error::new(
                        ErrorCode::UnsupportedFeature,
                        "selected Vulkan device has no host-visible memory for transfers",
                    )
                })
        };
        let upload_pool = host_pool(MemoryPolicy::Streaming)?;
        let readback_pool = host_pool(MemoryPolicy::Readback)?;
        let atom_size = physical.properties.limits.non_coherent_atom_size.max(1);
        let transfers = TransferEngine::new(
            device.clone(),
            Arc::clone(&transfer_queue),
            upload_pool,
            readback_pool,
            TransferOptions {
                queue_family: families.transfer_family,
                // Always keep staging rings; unified-memory contexts bypass them per replica.
                topology: crate::device::MemoryTopology::Discrete,
                non_coherent_atom_size: atom_size,
                cross_queue_concurrent: families.uses_concurrent_sharing(),
                ..TransferOptions::default()
            },
        )?;
        let capabilities = DeviceCapabilities::from_device(&info);
        let cache_key = if id.uuid.is_empty() {
            format!("device-{}", id.index)
        } else {
            id.uuid.to_string()
        };
        let pipelines = PipelineManager::new(
            device.clone(),
            capabilities,
            PipelineCacheIdentity::from_physical(physical),
            Some(
                PathBuf::from(&options.cache_dir)
                    .join("pipelines")
                    .join(format!("{cache_key}.bin")),
            ),
            PipelineManagerOptions::default(),
        )?;
        let timestamps = info
            .queue_families
            .get(families.compute_family as usize)
            .filter(|f| f.timestamp_valid_bits != 0)
            .and_then(|f| {
                TimestampRing::new(
                    device.clone(),
                    f.timestamp_valid_bits,
                    info.timestamp_period_nanoseconds,
                )
                .ok()
            });
        if let Some(ring) = &timestamps {
            ring.enable_trace_history(options.telemetry == TelemetryLevel::Trace);
        }
        let calibration = CalibrationRecord::new(id.uuid.to_string(), info.driver_version);
        let calibration_path = (!id.uuid.is_empty()).then(|| {
            PathBuf::from(&options.cache_dir)
                .join("calibration")
                .join(format!("{}.json", id.uuid))
        });
        let context = Arc::new(Self {
            key: NEXT_CONTEXT.fetch_add(1, Ordering::Relaxed),
            runtime,
            id,
            direct_host_access: info.memory_topology == crate::device::MemoryTopology::Uma,
            info,
            device,
            fault_state,
            queue,
            transfer_queue,
            families,
            transfers,
            pipelines,
            capabilities,
            memory,
            pools,
            fraction,
            last_refresh,
            atom_size,
            commands: Mutex::new(None),
            timestamps,
            timestamp_interval: timestamp_interval(options.telemetry),
            timestamp_counter: AtomicU64::new(0),
            autotuning: !options.disable_autotune,
            performance_path: PathBuf::from(&options.cache_dir).join("performance-db.json"),
            performance,
            pending_autotune: Mutex::new(HashMap::new()),
            pending_observations: Mutex::new(HashMap::new()),
            trace_scores: Mutex::default(),
            phase_profile: matches!(options.telemetry, TelemetryLevel::Profile | TelemetryLevel::Trace),
            trace: options.telemetry == TelemetryLevel::Trace,
            latest_profile: Mutex::default(),
            calibration: Mutex::new(calibration),
            calibration_path,
            submit_lock: Mutex::new(()),
            accepting: AtomicBool::new(true),
            quarantined: AtomicBool::new(false),
            uploads: AtomicU64::new(0),
            downloads: AtomicU64::new(0),
            replicated: Mutex::new(Vec::new()),
            stopped: AtomicBool::new(false),
            shutdown_mode: options.shutdown_mode,
        });
        Ok(context)
    }

    /// The pool for a memory type.
    pub(crate) fn pool(&self, memory_type: u32) -> Option<&SharedPool> {
        self.pools.get(memory_type as usize)
    }

    /// All pools.
    pub(crate) fn pools(&self) -> &[SharedPool] {
        &self.pools
    }

    /// Budget state of every heap.
    pub(crate) fn memory_manager(&self) -> MutexGuard<'_, MemoryManager> {
        lock(&self.memory)
    }

    /// Re-queries the driver budget at most once a second (always with `force`).
    pub(crate) fn refresh_budget(&self, force: bool) -> Status {
        refresh_budget(
            &self.device,
            &self.memory,
            self.fraction,
            &self.last_refresh,
            force,
        )
    }

    /// Validates that the context can take new work.
    pub(crate) fn check_accepting(&self) -> Status {
        if !self.accepting.load(Ordering::Acquire) {
            return Err(Error::new(
                ErrorCode::Cancelled,
                "the runtime device is shutting down",
            ));
        }
        if self.quarantined.load(Ordering::Acquire) {
            return Err(Error::new(
                ErrorCode::ValidationError,
                "device is quarantined after a verification mismatch",
            ));
        }
        Ok(())
    }

    /// The replica of `buffer` on this device, creating its `VkBuffer` and memory on first use.
    pub(crate) fn replica(&self, buffer: &BufferHandle) -> Result<Arc<Replica>> {
        if let Some(existing) = buffer.replica::<Replica>(self.key) {
            return Ok(existing);
        }
        self.check_accepting()?;
        let created = buffer.replica_or_insert(self.key, || self.create_replica(buffer.size_bytes()))?;
        let mut replicated = lock(&self.replicated);
        // Prune dead entries whenever the list doubles, keeping tracking amortized O(1).
        if replicated.len() >= 64 && replicated.len().is_power_of_two() {
            replicated.retain(WeakBufferHandle::is_alive);
        }
        replicated.push(buffer.downgrade());
        Ok(created)
    }

    fn create_replica(&self, size: u64) -> Result<Replica> {
        if size > self.info.limits.max_storage_buffer_range {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "buffer size exceeds maxStorageBufferRange",
            ));
        }
        self.refresh_budget(false)?;
        let families = [self.families.compute_family, self.families.transfer_family];
        let mut create = vk::BufferCreateInfo::default().size(size).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        );
        create = if self.families.uses_concurrent_sharing() {
            create
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families)
        } else {
            create.sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        let raw = self.device.raw();
        // SAFETY: `create` and `families` are valid for the call; the device is live.
        let buffer = unsafe { raw.create_buffer(&create, None) }.map_err(|r| {
            Error::new(
                ErrorCode::AllocationFailed,
                "vkCreateBuffer failed for a runtime buffer",
            )
            .with_vk(r.as_raw())
        })?;
        let destroy = |error: Error| {
            // SAFETY: the buffer was created above and never bound or used.
            unsafe { raw.destroy_buffer(buffer, None) };
            error
        };
        // SAFETY: `buffer` is a live buffer of this device.
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let memory_type = MemoryManager::select_memory_type_from(
            &self.info.memory_types,
            &self.info.memory_type_preferences,
            MemoryPolicy::Auto,
            requirements.memory_type_bits,
            &MemoryTypeRequirements::default(),
        )
        .map_err(destroy)?;
        let pool = Arc::clone(self.pool(memory_type).ok_or_else(|| {
            destroy(Error::new(
                ErrorCode::UnsupportedFeature,
                "selected Vulkan buffer memory type has no runtime pool",
            ))
        })?);
        let allocation = lock(&pool)
            .allocate(&PoolAllocationRequest::new(
                requirements.size,
                requirements.alignment,
            ))
            .map_err(destroy)?;
        // SAFETY: the allocation satisfies the buffer's requirements and is not bound elsewhere.
        if let Err(result) = unsafe { raw.bind_buffer_memory(buffer, allocation.memory, allocation.offset) } {
            let _ = lock(&pool).deallocate(&allocation);
            return Err(destroy(
                Error::new(
                    ErrorCode::AllocationFailed,
                    "vkBindBufferMemory failed for a runtime buffer",
                )
                .with_vk(result.as_raw()),
            ));
        }
        let memory_type_info = self.info.memory_types[memory_type as usize];
        let block = lock(&pool).block_size();
        let transfer = TransferBuffer {
            buffer,
            size,
            memory: allocation.memory,
            memory_offset: allocation.offset,
            memory_allocation_size: if allocation.dedicated {
                allocation.allocated_size
            } else {
                block
            },
            mapped: allocation.mapped,
            host_visible: memory_type_info.host_visible,
            host_coherent: memory_type_info.host_coherent,
            queue_use: Some(Arc::new(QueueTimelineUse::default())),
            keepalive: None,
        };
        Ok(Replica::new(self.device.clone(), pool, allocation, transfer))
    }

    /// Next dispatch should record GPU timestamps.
    pub(crate) fn sample_timestamps(&self) -> bool {
        self.timestamps.is_some()
            && self.timestamp_interval != 0
            && self
                .timestamp_counter
                .fetch_add(1, Ordering::Relaxed)
                .is_multiple_of(u64::from(self.timestamp_interval))
    }

    /// Consumes completed timestamp samples: autotuning trials and adaptive executions are
    /// recorded in the performance database, and traced scheduler scores move to history.
    pub(crate) fn collect_timestamps(&self, wait: Duration) -> Status {
        let Some(ring) = &self.timestamps else {
            return Ok(());
        };
        let completed = self.queue.completed_value()?;
        let samples = ring.collect(completed, wait)?;
        if samples.is_empty() {
            return Ok(());
        }
        let now = crate::performance_db::unix_ms();
        let mut save = false;
        for sample in samples {
            let duration = sample.duration_nanoseconds as f64;
            if let Some(pending) = lock(&self.pending_autotune).remove(&sample.submission_id)
                && pending.tuner.observe(&pending.trial, duration).is_ok()
                && !pending.trial.warmup
                && self.performance.observe(&pending.key, duration, now).is_ok()
            {
                save = true;
            }
            if let Some(key) = lock(&self.pending_observations).remove(&sample.submission_id)
                && self.performance.observe(&key, duration, now).is_ok()
            {
                // Persist at the first sample, when the record becomes confident, then every 8.
                let count = self.performance.query(&key).map_or(0, |s| s.count);
                save |= count == 1 || count == 5 || count.is_multiple_of(8);
            }
            lock(&self.trace_scores).complete(sample.submission_id);
        }
        if save {
            // Persistence is best effort: a read-only cache directory must not fail dispatches.
            let _ = self.performance.save(&self.performance_path);
        }
        Ok(())
    }

    /// Stops accepting work, drains or cancels queued work, and releases this device's replicas
    /// of every buffer it touched. Idempotent.
    pub(crate) fn shutdown(&self, mode: ShutdownMode) {
        if self.stopped.swap(true, Ordering::AcqRel) {
            return;
        }
        self.accepting.store(false, Ordering::Release);
        let _ = self.queue.shutdown(mode);
        if !Arc::ptr_eq(&self.queue, &self.transfer_queue) {
            let _ = self.transfer_queue.shutdown(mode);
        }
        let lost = !self.fault_state.healthy();
        for weak in std::mem::take(&mut *lock(&self.replicated)) {
            if let Some(buffer) = weak.upgrade() {
                // Results that only this lost device held are gone with it.
                if lost
                    && let Some(replica) = buffer.replica::<Replica>(self.key)
                    && replica.state().device_newer
                    && replica.state().resident_version == Some(buffer.host_version())
                {
                    buffer.mark_data_lost();
                }
                buffer.remove_replica(self.key);
            }
        }
        for (_, pending) in lock(&self.pending_autotune).drain() {
            pending.tuner.cancel(&pending.trial);
        }
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        self.shutdown(self.shutdown_mode);
    }
}
