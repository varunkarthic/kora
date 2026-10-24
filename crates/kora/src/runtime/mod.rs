//! The owning runtime: selected Vulkan devices, logical buffers, kernels, and dispatches.
//!
//! # Examples
//!
//! ```no_run
//! use kora::{
//!     KernelAccess, KernelBindingKind, KernelBindingMetadata, KernelMetadata, PushConstantFieldMetadata,
//!     Runtime, RuntimeOptions,
//! };
//!
//! # fn main() -> kora::Result<()> {
//! let runtime = Runtime::new(&RuntimeOptions::default())?;
//! let binding = |binding, name: &str, access| KernelBindingMetadata {
//!     set: 0,
//!     binding,
//!     name: name.into(),
//!     kind: KernelBindingKind::StorageBuffer,
//!     access,
//! };
//! let metadata = KernelMetadata {
//!     kernel_id: "example.vector_add".into(),
//!     local_size: [64, 1, 1],
//!     bindings: vec![
//!         binding(0, "a", KernelAccess::Read),
//!         binding(1, "b", KernelAccess::Read),
//!         binding(2, "out", KernelAccess::Write),
//!     ],
//!     push_constant_size: 4,
//!     push_constants: vec![PushConstantFieldMetadata {
//!         name: "count".into(),
//!         type_name: "u32".into(),
//!         offset: 0,
//!         size: 4,
//!     }],
//!     ..Default::default()
//! };
//! let kernel = runtime.load_kernel_file(std::path::Path::new("vector_add.spv"), "main", metadata)?;
//! let a = runtime.create_buffer::<f32>(1024)?;
//! let b = runtime.create_buffer::<f32>(1024)?;
//! let out = runtime.create_buffer::<f32>(1024)?;
//! a.write(&vec![1.0; 1024])?;
//! b.write(&vec![2.0; 1024])?;
//! runtime
//!     .dispatch(&kernel)
//!     .global(1024)
//!     .read(&a)
//!     .read(&b)
//!     .write(&out)
//!     .push(&1024_u32)
//!     .submit()?
//!     .wait()?;
//! assert!(out.read()?.iter().all(|&x| x == 3.0));
//! # Ok(())
//! # }
//! ```

mod calibrate;
mod context;
mod cpu;
mod dispatch;
mod kernel;
mod replica;
mod verify;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytemuck::Pod;

use self::context::{DeviceContext, lock};
use self::replica::Replica;
use crate::buffer::{Buffer, BufferHandle, BufferManager, BufferUsage};
use crate::calibration::CalibrationRecord;
use crate::cpu_executor::{CpuExecutor, CpuExecutorOptions, CpuTaskPriority};
use crate::device::{DeviceCandidate, DeviceInfo, initialize_selected_devices, select_devices};
use crate::error::{Error, ErrorCode, Result, Status};
use crate::event::{Event, EventPromise};
use crate::ids::DeviceId;
use crate::json::{self, Value};
use crate::kernel::spirv::words_from_bytes;
use crate::kernel::validate_and_reflect;
use crate::log::{self, Category, Level};
use crate::memory::{MemoryHeapStats, MemoryManager, MemoryStats};
use crate::metadata::{KernelAccess, KernelMetadata};
use crate::options::{ResolvedOptions, RuntimeOptions};
use crate::performance_db::{PerformanceDatabase, unix_ms};
use crate::placement::{AdaptiveDispatchWorkload, PlacementCandidate, select_adaptive_device};
use crate::vulkan::{
    DeviceFaultAddressRecord, DeviceFaultVendorRecord, InstanceOptions, Loader, ash::vk, device_info,
};

pub use self::cpu::{CpuFallback, CpuTaskContext};
pub use self::dispatch::{DispatchBuilder, DispatchPriority};
pub use self::kernel::Kernel;

/// Transfers issued by a runtime since creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RuntimeTransferStats {
    /// Host-to-device uploads (staged or direct).
    pub uploads: u64,
    /// Device-to-host readbacks.
    pub downloads: u64,
}

/// GPU dispatch timing collected from completed submissions without waiting for pending work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RuntimeProfilingStats {
    /// The device supports GPU timestamp queries (sample fields stay zero without them).
    pub gpu_timestamps: bool,
    /// Samples collected.
    pub samples_collected: u64,
    /// Samples dropped because the ring was full or a query failed.
    pub dropped_samples: u64,
    /// Submission ID of the latest sample.
    pub last_submission_id: u64,
    /// GPU duration of the latest sample.
    pub last_duration_nanoseconds: u64,
    /// Calibrated host time of the latest sample (0 without calibrated timestamps).
    pub last_host_timestamp_nanoseconds: u64,
    /// Calibration deviation of the latest sample.
    pub last_calibration_deviation_nanoseconds: u64,
    /// Submission of the latest host phase profile (0 without `Profile` or `Trace` telemetry).
    pub last_profile_submission_id: u64,
    /// Binding, size, and limit validation.
    pub last_dispatch_validation_nanoseconds: u64,
    /// Device copies and cross-device checks.
    pub last_binding_preflight_nanoseconds: u64,
    /// Pipeline cache lookup (or compilation).
    pub last_pipeline_lookup_nanoseconds: u64,
    /// Input uploads (enqueue, not completion).
    pub last_buffer_residency_nanoseconds: u64,
    /// Building the queue submission.
    pub last_submission_assembly_nanoseconds: u64,
    /// Host time in `submit` excluding residency.
    pub last_host_submission_nanoseconds: u64,
    /// Handing the submission to the queue.
    pub last_queue_enqueue_nanoseconds: u64,
    /// Command buffer acquisition on the submission thread.
    pub last_command_buffer_acquire_nanoseconds: u64,
    /// Descriptor set allocation and update.
    pub last_descriptor_setup_nanoseconds: u64,
    /// Command recording.
    pub last_command_recording_nanoseconds: u64,
    /// Enqueue to completion callback.
    pub last_queue_completion_nanoseconds: u64,
    /// Completion worker wake-up delay.
    pub last_completion_worker_delay_nanoseconds: u64,
    /// Time the completion worker waited on the device.
    pub last_device_completion_wait_nanoseconds: u64,
    /// Timestamp result collection.
    pub last_timestamp_collection_nanoseconds: u64,
    /// The completion callback itself.
    pub last_completion_callback_nanoseconds: u64,
}

/// Optional `VK_EXT_device_fault` details captured after device loss. Addresses are returned only
/// to the caller and are never logged or persisted by KORA.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceFaultReport {
    /// The device supports fault queries.
    pub supported: bool,
    /// A query was made.
    pub queried: bool,
    /// Records were truncated.
    pub truncated: bool,
    /// `VkResult` that reported the loss (0 while healthy).
    pub loss_result: i32,
    /// `VkResult` of the fault query.
    pub query_result: i32,
    /// Driver description.
    pub description: String,
    /// Address records.
    pub addresses: Vec<DeviceFaultAddressRecord>,
    /// Vendor records.
    pub vendor_records: Vec<DeviceFaultVendorRecord>,
}

#[derive(Default)]
struct CpuState {
    fallbacks: HashMap<String, CpuFallback>,
    executor: Option<Arc<CpuExecutor>>,
}

static NEXT_RUNTIME: AtomicU64 = AtomicU64::new(1);

/// Owns the Vulkan instance, the selected devices, and their memory, queues, and pipelines.
///
/// `Runtime` is `Send + Sync`; share it behind an `Arc` to dispatch from several threads.
/// Dropping it drains (or cancels, per [`ShutdownMode`](crate::ShutdownMode)) queued work.
pub struct Runtime {
    id: u64,
    requested: RuntimeOptions,
    options: ResolvedOptions,
    contexts: Arc<[Arc<DeviceContext>]>,
    quarantined: Vec<DeviceId>,
    buffers: BufferManager,
    performance: Arc<PerformanceDatabase>,
    cpu: Mutex<CpuState>,
    // Declared last: the instance must outlive every device context.
    _instance: crate::vulkan::Instance,
    _loader: Loader,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("devices", &self.devices())
            .field("quarantined", &self.quarantined)
            .finish()
    }
}

impl Runtime {
    /// Loads Vulkan, selects devices per `options` (and `KORA_*` environment overrides), and
    /// initializes each one.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::NoVulkanImplementation`] without a usable Vulkan loader or driver,
    /// [`ErrorCode::NoEligibleDevice`] when no device matches, and initialization errors of the
    /// selected device (with `AllEligible`, failing devices are quarantined instead).
    pub fn new(options: &RuntimeOptions) -> Result<Runtime> {
        let runtime = Self::without_calibration(options)?;
        if runtime.options.enable_startup_calibration {
            runtime.run_calibration(false);
        }
        Ok(runtime)
    }

    fn without_calibration(requested: &RuntimeOptions) -> Result<Runtime> {
        let options = requested.resolve();
        if options.debug_safety {
            log::set_category_level(Category::Sync, Level::Trace);
        }
        let loader = Loader::load()?;
        let instance = loader.create_instance(&InstanceOptions {
            enable_validation: options.enable_validation || options.debug_safety,
            ..InstanceOptions::default()
        })?;
        let physical = instance.physical_devices()?;
        let candidates: Vec<DeviceCandidate> = physical
            .iter()
            .enumerate()
            .map(|(index, device)| {
                let info = device_info(device);
                DeviceCandidate {
                    id: DeviceId::new(index as u32, info.uuid.as_str()),
                    kind: info.kind,
                    tier: info.tier,
                    device_local_memory_bytes: info
                        .memory_heaps
                        .iter()
                        .filter(|h| h.device_local)
                        .map(|h| h.size_bytes)
                        .sum(),
                    compute_queue_count: device
                        .queue_families
                        .iter()
                        .filter(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                        .map(|f| f.queue_count)
                        .sum(),
                    satisfies_requirements: true,
                }
            })
            .collect();
        let selected = select_devices(&options, &candidates)?;
        let id = NEXT_RUNTIME.fetch_add(1, Ordering::Relaxed);
        let performance = Arc::new(PerformanceDatabase::default());
        let performance_path = Path::new(&options.cache_dir).join("performance-db.json");
        // A missing or corrupt database never blocks startup.
        let _ = performance.load(&performance_path, unix_ms());

        let mut contexts = Vec::with_capacity(selected.len());
        let mut quarantined = Vec::new();
        initialize_selected_devices(
            &selected,
            options.device_policy == crate::options::DevicePolicy::AllEligible,
            |device| {
                let context = DeviceContext::new(
                    &instance,
                    &physical[device.index as usize],
                    device.clone(),
                    id,
                    &options,
                    Arc::clone(&performance),
                )?;
                contexts.push(context);
                Ok(())
            },
            |device| quarantined.push(device.clone()),
            |device, error| {
                crate::kora_log!(
                    Recovery,
                    Warn,
                    "quarantined device {} during AllEligible initialization: {error}",
                    device.index
                );
            },
        )?;
        Ok(Runtime {
            id,
            requested: requested.clone(),
            options,
            contexts: contexts.into(),
            quarantined,
            buffers: BufferManager::new(),
            performance,
            cpu: Mutex::default(),
            _instance: instance,
            _loader: loader,
        })
    }

    /// Effective options after environment overrides.
    pub fn options(&self) -> &ResolvedOptions {
        &self.options
    }

    /// Initialized devices, in selection order. IDs keep their Vulkan enumeration index.
    pub fn devices(&self) -> Vec<DeviceId> {
        self.contexts.iter().map(|c| c.id.clone()).collect()
    }

    /// Devices skipped during `AllEligible` initialization.
    pub fn quarantined_devices(&self) -> Vec<DeviceId> {
        let mut devices = self.quarantined.clone();
        devices.extend(
            self.contexts
                .iter()
                .filter(|c| c.quarantined.load(Ordering::Acquire))
                .map(|c| c.id.clone()),
        );
        devices
    }

    fn context(&self, device: &DeviceId) -> Result<&Arc<DeviceContext>> {
        self.contexts
            .iter()
            .find(|c| c.id == *device)
            .ok_or_else(|| Error::new(ErrorCode::InvalidArgument, "device is not owned by this runtime"))
    }

    fn first(&self) -> &Arc<DeviceContext> {
        &self.contexts[0]
    }

    /// Capabilities of an initialized device.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for a device this runtime does not own.
    pub fn device_info(&self, device: &DeviceId) -> Result<&DeviceInfo> {
        Ok(&self.context(device)?.info)
    }

    /// The manager every runtime buffer comes from.
    pub fn buffer_manager(&self) -> &BufferManager {
        &self.buffers
    }

    /// Creates a zero-filled buffer of `count` elements.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for zero elements; [`ErrorCode::Overflow`] when the byte
    /// size overflows.
    pub fn create_buffer<T: Pod>(&self, count: u64) -> Result<Buffer<T>> {
        self.buffers.create_typed(count)
    }

    /// Creates a zero-filled untyped buffer.
    ///
    /// # Errors
    ///
    /// As [`BufferManager::create`].
    pub fn create_buffer_bytes(&self, size_bytes: u64, usage: BufferUsage) -> Result<BufferHandle> {
        self.buffers.create(size_bytes, usage)
    }

    /// Validates `words` against `metadata` and compiles the kernel for the first device.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidKernel`] for invalid modules or metadata mismatches, and
    /// [`ErrorCode::PipelineCreationFailed`] when the driver rejects the pipeline.
    pub fn load_kernel(&self, words: &[u32], entry_point: &str, metadata: KernelMetadata) -> Result<Kernel> {
        kernel::load(self.first(), words, entry_point, metadata)
    }

    /// [`load_kernel`](Self::load_kernel) for a specific device.
    ///
    /// # Errors
    ///
    /// As [`load_kernel`](Self::load_kernel); [`ErrorCode::InvalidArgument`] for a device this
    /// runtime does not own.
    pub fn load_kernel_on(
        &self,
        device: &DeviceId,
        words: &[u32],
        entry_point: &str,
        metadata: KernelMetadata,
    ) -> Result<Kernel> {
        kernel::load(self.context(device)?, words, entry_point, metadata)
    }

    /// Reads a `.spv` file (at most 16 MiB) and loads it for the first device.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidKernel`] for unreadable or malformed files; otherwise as
    /// [`load_kernel`](Self::load_kernel).
    pub fn load_kernel_file(
        &self,
        path: &Path,
        entry_point: &str,
        metadata: KernelMetadata,
    ) -> Result<Kernel> {
        let bytes = crate::fsutil::read_bounded(path, 16 << 20).map_err(|error| {
            Error::new(
                ErrorCode::InvalidKernel,
                format!("cannot read SPIR-V module {}", path.display()),
            )
            .with_cause(error)
        })?;
        self.load_kernel(&words_from_bytes(&bytes)?, entry_point, metadata)
    }

    /// Starts a dispatch of `kernel` on the device it was loaded for.
    pub fn dispatch(&self, kernel: &Kernel) -> DispatchBuilder {
        DispatchBuilder::new(kernel.clone(), Arc::clone(&self.contexts))
    }

    /// Chooses the device with the lowest predicted finish time for `workload` from startup
    /// calibration and confident online observations, and returns a dispatch bound to it. Without
    /// complete calibration on every eligible device the kernel's own device is kept. At
    /// submission, a dispatch whose inputs hold unread results on the kernel's own device returns
    /// there instead of moving them.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for zero kernel work or a kernel from another runtime;
    /// [`ErrorCode::NoEligibleDevice`] when the kernel's device is not eligible.
    pub fn dispatch_adaptive(
        &self,
        kernel: &Kernel,
        workload: &AdaptiveDispatchWorkload,
    ) -> Result<DispatchBuilder> {
        if !self.owns(kernel) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "adaptive dispatch kernel does not belong to this runtime",
            ));
        }
        if workload.kernel_work == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "adaptive dispatch kernel work must be nonzero",
            ));
        }
        let state = &kernel.state;
        let current = &state.context;
        let now = unix_ms();
        let mut candidates = Vec::with_capacity(self.contexts.len());
        for context in self.contexts.iter() {
            let is_current = Arc::ptr_eq(context, current);
            let healthy = context.check_accepting().is_ok() && context.fault_state.healthy();
            if !healthy && !is_current {
                continue;
            }
            // An autotuned kernel stays on the device whose tuner owns it.
            let supported = (is_current || state.autotuner.is_none())
                && validate_and_reflect(
                    &state.words,
                    &state.metadata,
                    &context.capabilities,
                    &state.entry_point,
                )
                .is_ok();
            let available = self.memory_stats(&context.id).map_or(0, |stats| {
                stats
                    .heaps
                    .iter()
                    .filter(|h| {
                        context
                            .info
                            .memory_heaps
                            .get(h.heap_index as usize)
                            .is_some_and(|heap| heap.device_local)
                    })
                    .map(|h| h.available_bytes)
                    .max()
                    .unwrap_or(0)
            });
            candidates.push(PlacementCandidate {
                id: context.id.clone(),
                driver_version: context.info.driver_version,
                calibration: lock(&context.calibration).clone(),
                online_execution: dispatch::execution_key(context, state.semantic_identity, workload)
                    .and_then(|key| self.performance.query_confident(&key, now, 5)),
                available_device_memory_bytes: available,
                supported,
                healthy,
                allowed: true,
            });
        }
        let decision = select_adaptive_device(&candidates, &current.id, workload)?;
        let target = self
            .contexts
            .iter()
            .find(|c| c.id == decision.device)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InternalInvariantViolation,
                    "adaptive scheduler selected an unknown device",
                )
            })?;
        let mut builder = DispatchBuilder::new(kernel.on(target)?, Arc::clone(&self.contexts));
        builder.adaptive = Some(Arc::new(dispatch::AdaptiveState {
            original: kernel.clone(),
            workload: *workload,
            score: decision.score,
        }));
        Ok(builder)
    }

    fn run_calibration(&self, force: bool) {
        let budget = Duration::from_millis(u64::from(self.options.calibration_budget_ms));
        for context in self.contexts.iter() {
            calibrate::calibrate_device(&self.buffers, context, budget, force);
        }
    }

    /// Re-runs calibration on every device, ignoring cached records. Probes that cannot run
    /// within the budget stay zero.
    pub fn calibrate(&self) {
        self.run_calibration(true);
    }

    /// The calibration record of a device (measurements are zero until calibrated).
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for a device this runtime does not own.
    pub fn calibration(&self, device: &DeviceId) -> Result<CalibrationRecord> {
        Ok(lock(&self.context(device)?.calibration).clone())
    }

    /// Test hook: replaces a device's calibration record.
    #[doc(hidden)]
    pub fn set_calibration(&self, device: &DeviceId, mut record: CalibrationRecord) -> Status {
        let context = self.context(device)?;
        record.device_uuid = context.id.uuid.to_string();
        record.driver_version = context.info.driver_version;
        *lock(&context.calibration) = record;
        Ok(())
    }

    /// Recreates every device after an observed device loss. Buffers keep their host contents
    /// (results that only the lost device held are marked lost); kernels must be loaded again.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] when no device was lost; otherwise as
    /// [`rebuild_devices`](Self::rebuild_devices).
    pub fn recover(&mut self) -> Status {
        if self.contexts.iter().all(|c| c.fault_state.healthy()) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "runtime recovery requires an observed device loss",
            ));
        }
        self.rebuild_devices()
    }

    /// Re-enumerates and recreates every device from the original options, picking up devices
    /// that appeared or disappeared. Buffers, CPU fallbacks, and quarantine decisions are kept;
    /// existing kernels are invalidated. Call only when no work is in flight. On failure the
    /// runtime is unchanged.
    ///
    /// # Errors
    ///
    /// The errors of [`Runtime::new`].
    pub fn rebuild_devices(&mut self) -> Status {
        let mut replacement = Self::without_calibration(&self.requested)?;
        let quarantined: Vec<DeviceId> = self.quarantined_devices();
        for context in replacement.contexts.iter() {
            if quarantined.contains(&context.id) {
                context.quarantined.store(true, Ordering::Release);
            }
        }
        replacement.quarantined.extend(
            quarantined
                .into_iter()
                .filter(|id| !replacement.contexts.iter().any(|c| c.id == *id)),
        );
        std::mem::swap(&mut self.id, &mut replacement.id);
        std::mem::swap(&mut self.contexts, &mut replacement.contexts);
        std::mem::swap(&mut self.quarantined, &mut replacement.quarantined);
        std::mem::swap(&mut self.performance, &mut replacement.performance);
        std::mem::swap(&mut self._instance, &mut replacement._instance);
        std::mem::swap(&mut self._loader, &mut replacement._loader);
        // `replacement` now owns the old devices; dropping it drains their queues and releases
        // their copies of every buffer, marking results that only a lost device held.
        drop(replacement);
        if self.options.enable_startup_calibration {
            self.run_calibration(false);
        }
        Ok(())
    }

    /// Completed GPU dispatch samples as Chrome Trace Event JSON (open in Perfetto). Needs
    /// `TelemetryLevel::Trace`; history is bounded to 4096 samples per device. Adaptive dispatches
    /// include the scheduler score that selected their device.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] without trace telemetry.
    pub fn export_trace_json(&self) -> Result<String> {
        if self.options.telemetry != crate::options::TelemetryLevel::Trace {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "trace export requires TelemetryLevel::Trace",
            ));
        }
        let mut events = Vec::new();
        for context in self.contexts.iter() {
            let Some(ring) = &context.timestamps else {
                continue;
            };
            context.collect_timestamps(Duration::ZERO)?;
            let scores = lock(&context.trace_scores).completed.clone();
            for sample in ring.history() {
                let mut args = vec![
                    ("submission_id".to_owned(), Value::uint(sample.submission_id)),
                    ("queue_value".to_owned(), Value::uint(sample.queue_value)),
                    (
                        "host_clock_calibrated".to_owned(),
                        Value::Bool(sample.host_timestamp_nanoseconds != 0),
                    ),
                    (
                        "calibration_deviation_ns".to_owned(),
                        Value::uint(sample.calibration_deviation_nanoseconds),
                    ),
                ];
                if let Some(selected) = scores.get(&sample.submission_id) {
                    let s = &selected.score;
                    let fields = [
                        ("ready_delay_ns", s.ready_delay_ns),
                        ("queue_delay_ns", s.queue_delay_ns),
                        ("input_transfer_ns", s.input_transfer_ns),
                        ("execute_ns", s.execute_ns),
                        ("required_output_ns", s.required_output_ns),
                        ("synchronization_ns", s.synchronization_ns),
                        ("memory_penalty_ns", s.memory_penalty_ns),
                        ("uncertainty_penalty_ns", s.uncertainty_penalty_ns),
                        ("locality_bonus_ns", s.locality_bonus_ns),
                        ("downstream_movement_penalty_ns", s.downstream_movement_penalty_ns),
                        ("predicted_finish_ns", s.predicted_finish_ns),
                    ];
                    let mut score = vec![(
                        "selected_device_index".to_owned(),
                        Value::uint(u64::from(selected.device_index)),
                    )];
                    score.extend(fields.iter().map(|(k, v)| ((*k).to_owned(), Value::float(*v))));
                    args.push(("scheduler_score".to_owned(), Value::Object(score)));
                }
                events.push(Value::Object(vec![
                    ("name".to_owned(), Value::string("gpu_dispatch")),
                    ("cat".to_owned(), Value::string("kora.gpu")),
                    ("ph".to_owned(), Value::string("X")),
                    (
                        "ts".to_owned(),
                        Value::uint(sample.host_timestamp_nanoseconds / 1000),
                    ),
                    ("dur".to_owned(), Value::uint(sample.duration_nanoseconds / 1000)),
                    ("pid".to_owned(), Value::uint(1)),
                    ("tid".to_owned(), Value::uint(u64::from(context.id.index))),
                    ("args".to_owned(), Value::Object(args)),
                ]));
            }
        }
        let document = Value::Object(vec![
            ("traceEvents".to_owned(), Value::Array(events)),
            ("displayTimeUnit".to_owned(), Value::string("us")),
            (
                "metadata".to_owned(),
                Value::Object(vec![
                    ("format".to_owned(), Value::string("Chrome Trace Event")),
                    (
                        "time_origin".to_owned(),
                        Value::string("calibrated_monotonic_host_timestamp_when_available"),
                    ),
                ]),
            ),
        ]);
        let mut out = String::new();
        json::write_value(&mut out, &document);
        Ok(out)
    }

    /// Makes the newest device-resident contents of `buffer` visible in its host shadow. Completes
    /// immediately when the host is already current.
    ///
    /// # Errors
    ///
    /// Transfer errors.
    pub fn readback(&self, buffer: &impl AsRef<BufferHandle>) -> Result<Event> {
        let buffer = buffer.as_ref();
        for context in self.contexts.iter() {
            if let Some(event) = readback_from(context, buffer)? {
                return Ok(event);
            }
        }
        Ok(Event::ready(Ok(())))
    }

    /// [`readback`](Self::readback) from one device only.
    ///
    /// # Errors
    ///
    /// As [`readback`](Self::readback); [`ErrorCode::InvalidArgument`] for a device this runtime
    /// does not own.
    pub fn readback_from(&self, buffer: &impl AsRef<BufferHandle>, device: &DeviceId) -> Result<Event> {
        Ok(readback_from(self.context(device)?, buffer.as_ref())?.unwrap_or_else(|| Event::ready(Ok(()))))
    }

    /// A handle that reads byte ranges of device replicas on `device` after the runtime lock that
    /// produced it is released (used by split cluster dispatches).
    pub(crate) fn replica_reader(&self, device: &DeviceId) -> Result<ReplicaReader> {
        Ok(ReplicaReader(Arc::clone(self.context(device)?)))
    }

    /// Registers (or replaces) a CPU implementation under `key`.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for an empty key.
    pub fn register_cpu_fallback(
        &self,
        key: impl Into<String>,
        function: impl Fn(&mut CpuTaskContext<'_>) -> Status + Send + Sync + 'static,
    ) -> Status {
        let key = key.into();
        if key.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "CPU fallback key must be non-empty",
            ));
        }
        lock(&self.cpu).fallbacks.insert(key, Arc::new(function));
        Ok(())
    }

    /// Runs the CPU implementation registered under `key` on the CPU executor once every readable
    /// buffer's device results are visible on the host.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidKernel`] for an unknown key, [`ErrorCode::InvalidKernelBinding`] when a
    /// buffer is declared twice, and readback errors.
    pub fn run_cpu_fallback(
        &self,
        key: &str,
        bindings: &[(&BufferHandle, KernelAccess)],
        push_constants: &[u8],
    ) -> Result<Event> {
        let (function, executor) = {
            let mut cpu = lock(&self.cpu);
            let function = cpu.fallbacks.get(key).cloned().ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidKernel,
                    format!("no CPU fallback is registered for key '{key}'"),
                )
            })?;
            if cpu.executor.is_none() {
                cpu.executor = Some(Arc::new(CpuExecutor::new(CpuExecutorOptions::default())?));
            }
            (
                function,
                Arc::clone(cpu.executor.as_ref().expect("created above")),
            )
        };
        let mut seen = HashSet::new();
        let mut prerequisites = Vec::new();
        for (buffer, access) in bindings {
            if !seen.insert(buffer.id()) {
                return Err(Error::new(
                    ErrorCode::InvalidKernelBinding,
                    "CPU task cannot declare the same buffer more than once",
                ));
            }
            if access.reads() {
                prerequisites.push(self.readback(*buffer)?);
            }
        }
        let owned: Vec<(BufferHandle, KernelAccess)> =
            bindings.iter().map(|(b, a)| ((*b).clone(), *a)).collect();
        let push = push_constants.to_vec();
        let promise = EventPromise::new();
        let event = promise.event();
        let ready = Event::when_all(&prerequisites)?;
        ready.on_complete(move |status| {
            if let Err(error) = status {
                promise.complete(Err(error));
                return;
            }
            let task = move || run_cpu_task(&function, &owned, &push);
            match executor.submit(CpuTaskPriority::Normal, task) {
                Ok(done) => {
                    let _ = done.on_complete(move |s| promise.complete(s));
                }
                Err(error) => promise.complete(Err(error)),
            }
        })?;
        Ok(event)
    }

    /// Pool totals and per-heap budgets of a device.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for a device this runtime does not own.
    pub fn memory_stats(&self, device: &DeviceId) -> Result<MemoryStats> {
        let context = self.context(device)?;
        context.refresh_budget(true)?;
        let pool_stats: Vec<_> = context.pools().iter().map(|p| lock(p).stats()).collect();
        let mut stats = MemoryManager::aggregate_stats(&pool_stats)?;
        let memory = context.memory_manager();
        for heap in 0..memory.heap_count() as u32 {
            let budget = memory.budget(heap)?;
            stats.heaps.push(MemoryHeapStats {
                heap_index: heap,
                budget_bytes: budget.limit_bytes,
                reserved_bytes: budget.reserved_bytes,
                available_bytes: budget.available_bytes,
                externally_used_bytes: budget.externally_used_bytes,
                budget_queried: budget.queried,
            });
        }
        Ok(stats)
    }

    /// Uploads and readbacks issued so far.
    pub fn transfer_stats(&self) -> RuntimeTransferStats {
        self.contexts
            .iter()
            .fold(RuntimeTransferStats::default(), |total, c| RuntimeTransferStats {
                uploads: total.uploads + c.uploads.load(Ordering::Relaxed),
                downloads: total.downloads + c.downloads.load(Ordering::Relaxed),
            })
    }

    /// Completed GPU timestamp samples of a device (waiting at most 2 ms for results the driver
    /// publishes after the queue signal).
    ///
    /// # Errors
    ///
    /// [`ErrorCode::UnsupportedFeature`] when the device has no timestamp queries.
    pub fn profiling_stats(&self, device: &DeviceId) -> Result<RuntimeProfilingStats> {
        let context = self.context(device)?;
        if context.timestamps.is_none() && !context.phase_profile {
            return Err(Error::new(
                ErrorCode::UnsupportedFeature,
                "GPU timestamp profiling is unavailable for this device",
            ));
        }
        let mut stats = RuntimeProfilingStats::default();
        if let Some(ring) = &context.timestamps {
            stats.gpu_timestamps = true;
            context.collect_timestamps(Duration::from_millis(2))?;
            let latest = ring.latest_sample();
            stats.samples_collected = ring.samples_collected();
            stats.dropped_samples = ring.dropped_samples();
            stats.last_submission_id = latest.submission_id;
            stats.last_duration_nanoseconds = latest.duration_nanoseconds;
            stats.last_host_timestamp_nanoseconds = latest.host_timestamp_nanoseconds;
            stats.last_calibration_deviation_nanoseconds = latest.calibration_deviation_nanoseconds;
        }
        let p = *lock(&context.latest_profile);
        stats.last_profile_submission_id = p.submission_id;
        stats.last_dispatch_validation_nanoseconds = p.dispatch_validation_ns;
        stats.last_binding_preflight_nanoseconds = p.binding_preflight_ns;
        stats.last_pipeline_lookup_nanoseconds = p.pipeline_lookup_ns;
        stats.last_buffer_residency_nanoseconds = p.buffer_residency_ns;
        stats.last_submission_assembly_nanoseconds = p.submission_assembly_ns;
        stats.last_host_submission_nanoseconds = p.host_submission_ns;
        stats.last_queue_enqueue_nanoseconds = p.queue_enqueue_ns;
        stats.last_command_buffer_acquire_nanoseconds = p.command_buffer_acquire_ns;
        stats.last_descriptor_setup_nanoseconds = p.descriptor_setup_ns;
        stats.last_command_recording_nanoseconds = p.command_recording_ns;
        stats.last_queue_completion_nanoseconds = p.queue_completion_ns;
        stats.last_completion_worker_delay_nanoseconds = p.completion_worker_delay_ns;
        stats.last_device_completion_wait_nanoseconds = p.device_completion_wait_ns;
        stats.last_timestamp_collection_nanoseconds = p.timestamp_collection_ns;
        stats.last_completion_callback_nanoseconds = p.completion_callback_ns;
        Ok(stats)
    }

    /// Fault details captured after a device loss.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for a device this runtime does not own.
    pub fn device_fault_report(&self, device: &DeviceId) -> Result<DeviceFaultReport> {
        let context = self.context(device)?;
        let snapshot = context.fault_state.diagnostics();
        Ok(DeviceFaultReport {
            supported: snapshot.supported || context.device.supports_fault_diagnostics(),
            queried: snapshot.queried,
            truncated: snapshot.truncated,
            loss_result: context.fault_state.last_result().as_raw(),
            query_result: snapshot.query_result,
            description: snapshot.description,
            addresses: snapshot.addresses,
            vendor_records: snapshot.vendor_records,
        })
    }

    /// The performance database shared by the devices (autotuning and placement records).
    pub fn performance_database(&self) -> &PerformanceDatabase {
        &self.performance
    }

    /// Test hook: marks a device lost as if a submission had returned `VK_ERROR_DEVICE_LOST`.
    #[doc(hidden)]
    pub fn simulate_device_loss(&self, device: &DeviceId) -> Status {
        self.context(device)?
            .fault_state
            .mark_lost(vk::Result::ERROR_DEVICE_LOST, false);
        Ok(())
    }

    /// Debug check that a runtime ID matches (used by bindings).
    #[doc(hidden)]
    pub fn owns(&self, kernel: &Kernel) -> bool {
        kernel.state.context.runtime == self.id
    }
}

/// Reads replica ranges of one device; see [`Runtime::replica_reader`].
#[derive(Debug, Clone)]
pub(crate) struct ReplicaReader(Arc<DeviceContext>);

impl ReplicaReader {
    pub(crate) fn read(
        &self,
        buffer: &BufferHandle,
        offset: u64,
        size: u64,
        sink: impl FnOnce(&[u8]) -> Status + Send + 'static,
    ) -> Result<Event> {
        dispatch::read_replica_range(&self.0, buffer, offset, size, sink)
    }
}

fn readback_from(context: &Arc<DeviceContext>, buffer: &BufferHandle) -> Result<Option<Event>> {
    let Some(replica) = buffer.replica::<Replica>(context.key) else {
        return Ok(None);
    };
    let version = buffer.host_version();
    {
        let mut state = replica.state();
        if !state.device_newer {
            return Ok(None);
        }
        if state.resident_version != Some(version) {
            // The host was rewritten after the device produced its results: the host wins.
            state.device_newer = false;
            return Ok(None);
        }
    }
    let direct = context.direct_host_access
        && replica.directly_accessible(0)
        && replica
            .transfer()
            .queue_use
            .as_ref()
            .is_some_and(|u| u.host_readable.load(Ordering::Acquire));
    dispatch::read_to_host(context, buffer, &replica, version, direct)
        .map(|event| Some(event.unwrap_or_else(|| Event::ready(Ok(())))))
}

fn run_cpu_task(function: &CpuFallback, bindings: &[(BufferHandle, KernelAccess)], push: &[u8]) -> Status {
    let mut mappings = Vec::with_capacity(bindings.len());
    for (buffer, access) in bindings {
        mappings.push(if *access == KernelAccess::Read {
            cpu::Mapping::Read(buffer.map_read()?)
        } else {
            cpu::Mapping::Write(buffer.map_write()?)
        });
    }
    let mut context = CpuTaskContext {
        mappings,
        accesses: bindings.iter().map(|(_, a)| *a).collect(),
        push,
    };
    function(&mut context)
}

impl Drop for Runtime {
    fn drop(&mut self) {
        for context in self.contexts.iter() {
            context.shutdown(self.options.shutdown_mode);
        }
        if let Some(executor) = lock(&self.cpu).executor.take() {
            let _ = executor.shutdown(self.options.shutdown_mode);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::spirv::words_from_bytes;
    use crate::{KernelBindingKind, KernelBindingMetadata, PushConstantFieldMetadata};

    fn split_saxpy(runtime: &Runtime) -> Kernel {
        let binding = |binding, name: &str, access| KernelBindingMetadata {
            set: 0,
            binding,
            name: name.into(),
            kind: KernelBindingKind::StorageBuffer,
            access,
        };
        let field = |name: &str, type_name: &str, offset| PushConstantFieldMetadata {
            name: name.into(),
            type_name: type_name.into(),
            offset,
            size: 4,
        };
        let metadata = KernelMetadata {
            kernel_id: "unit.runtime.split_saxpy".into(),
            local_size: [64, 1, 1],
            bindings: vec![
                binding(0, "input_x", KernelAccess::Read),
                binding(1, "input_y", KernelAccess::ReadWrite),
            ],
            push_constant_size: 12,
            push_constants: vec![
                field("scale", "f32", 0),
                field("count", "u32", 4),
                field("base", "u32", 8),
            ],
            ..KernelMetadata::default()
        };
        let words = words_from_bytes(include_bytes!("../../tests/fixtures/split_saxpy.spv")).unwrap();
        runtime.load_kernel(&words, "main", metadata).unwrap()
    }

    fn read_range(reader: &ReplicaReader, buffer: &BufferHandle, offset: u64, size: u64) -> Vec<f32> {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&bytes);
        reader
            .read(buffer, offset, size, move |b| {
                *lock(&sink) = b.to_vec();
                Ok(())
            })
            .unwrap()
            .wait()
            .unwrap();
        let bytes = lock(&bytes);
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// Two slices of one split on the same device (a failed-over slice) must not re-upload the
    /// host copy over the first slice's range before that range is read back.
    #[test]
    fn slices_of_one_split_share_a_replica_and_other_splits_reupload() {
        let options = RuntimeOptions {
            enable_startup_calibration: Some(false),
            ..RuntimeOptions::default()
        };
        let runtime = match Runtime::new(&options) {
            Ok(runtime) => runtime,
            Err(error) => return eprintln!("skipping: {error}"),
        };
        let kernel = split_saxpy(&runtime);
        let (count, half) = (1024_u32, 512_u64);
        let x = runtime.create_buffer::<f32>(u64::from(count)).unwrap();
        let y = runtime.create_buffer::<f32>(u64::from(count)).unwrap();
        x.write(&vec![1.0; count as usize]).unwrap();
        y.write(&vec![0.0; count as usize]).unwrap();
        let slice = |split: u64, base: u32| {
            let push: Vec<u8> = [1.0_f32.to_bits(), count, base]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            runtime
                .dispatch(&kernel)
                .global(half)
                .read(&x)
                .read_write(&y)
                .push_bytes(&push)
                .partial_outputs(split)
                .submit()
                .unwrap()
                .wait()
                .unwrap();
        };
        let reader = runtime.replica_reader(&runtime.devices()[0]).unwrap();
        let bytes = half * 4;
        slice(1, 0);
        slice(1, half as u32);
        assert!(
            read_range(&reader, y.handle(), 0, bytes)
                .iter()
                .all(|&v| v == 1.0)
        );
        assert!(
            read_range(&reader, y.handle(), bytes, bytes)
                .iter()
                .all(|&v| v == 1.0)
        );
        // Another split at the same host version starts from the host copy again.
        slice(2, half as u32);
        assert!(
            read_range(&reader, y.handle(), 0, bytes)
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(y.read().unwrap().iter().all(|&v| v == 0.0)); // Slices never commit by themselves.
    }
}
