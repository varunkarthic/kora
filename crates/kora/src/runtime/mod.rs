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

mod context;
mod cpu;
mod dispatch;
mod kernel;
mod replica;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytemuck::Pod;

use self::context::{DeviceContext, lock};
use self::replica::Replica;
use crate::buffer::{Buffer, BufferHandle, BufferManager, BufferUsage};
use crate::cpu_executor::{CpuExecutor, CpuExecutorOptions, CpuTaskPriority};
use crate::device::{DeviceCandidate, DeviceInfo, initialize_selected_devices, select_devices};
use crate::error::{Error, ErrorCode, Result, Status};
use crate::event::{Event, EventPromise};
use crate::ids::DeviceId;
use crate::kernel::spirv::words_from_bytes;
use crate::log::{self, Category, Level};
use crate::memory::{MemoryHeapStats, MemoryManager, MemoryStats};
use crate::metadata::{KernelAccess, KernelMetadata};
use crate::options::{ResolvedOptions, RuntimeOptions};
use crate::performance_db::{PerformanceDatabase, unix_ms};
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
    options: ResolvedOptions,
    contexts: Vec<Arc<DeviceContext>>,
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
        let options = options.resolve();
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
            options,
            contexts,
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
        DispatchBuilder::new(kernel.clone())
    }

    /// Makes the newest device-resident contents of `buffer` visible in its host shadow. Completes
    /// immediately when the host is already current.
    ///
    /// # Errors
    ///
    /// Transfer errors.
    pub fn readback(&self, buffer: &impl AsRef<BufferHandle>) -> Result<Event> {
        let buffer = buffer.as_ref();
        for context in &self.contexts {
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
        let ring = context.timestamps.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::UnsupportedFeature,
                "GPU timestamp profiling is unavailable for this device",
            )
        })?;
        context.collect_timestamps(Duration::from_millis(2))?;
        let latest = ring.latest_sample();
        Ok(RuntimeProfilingStats {
            samples_collected: ring.samples_collected(),
            dropped_samples: ring.dropped_samples(),
            last_submission_id: latest.submission_id,
            last_duration_nanoseconds: latest.duration_nanoseconds,
            last_host_timestamp_nanoseconds: latest.host_timestamp_nanoseconds,
            last_calibration_deviation_nanoseconds: latest.calibration_deviation_nanoseconds,
        })
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
        for context in &self.contexts {
            context.shutdown(self.options.shutdown_mode);
        }
        if let Some(executor) = lock(&self.cpu).executor.take() {
            let _ = executor.shutdown(self.options.shutdown_mode);
        }
    }
}
