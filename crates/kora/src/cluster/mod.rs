//! One host [`Runtime`] plus any number of hot-swappable remote KORA workers.
//!
//! Buffers always live on the host. Each dispatch runs where it is expected to finish first (a
//! local device or a connected [`WorkerServer`](crate::remote::WorkerServer)), and any remote
//! failure transparently re-runs it locally, so results are never lost or corrupted when a worker
//! is unplugged mid-stream.

mod link;
mod status;

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bytemuck::Pod;

use self::link::WorkerLink;
use crate::buffer::{Buffer, BufferHandle, BufferUsage, WeakBufferHandle};
use crate::error::{Error, ErrorCode, Result, Status};
use crate::event::{Event, EventPromise};
use crate::ids::{BufferId, DeviceId};
use crate::metadata::{KernelAccess, KernelMetadata};
use crate::options::DevicePolicy;
use crate::remote::buffer_key;
use crate::remote::socket::{Endpoint, listen};
use crate::runtime::{Kernel, Runtime};

pub use crate::options::RuntimeOptions;

const MAX_EVENTS: usize = 64;
const RELEASE_SWEEP: Duration = Duration::from_millis(250);

/// Where a [`Cluster`] may run each dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffloadPolicy {
    /// Pick the target with the lowest measured latency for this kernel and data size, exploring
    /// unmeasured connected workers occasionally.
    #[default]
    Auto,
    /// Never offload; workers stay connected but idle.
    LocalOnly,
    /// Offload to a connected worker whenever one can run the kernel.
    PreferRemote,
}

/// Configuration for [`Cluster::new`].
#[derive(Debug, Clone)]
pub struct ClusterOptions {
    /// Options for the local runtime, which also owns every buffer's authoritative host copy.
    /// When `local.device_policy` (and `KORA_DEVICE`) is unset, every eligible local device is
    /// used.
    pub local: RuntimeOptions,
    /// Worker endpoints (`host:port`). `KORA_WORKERS` (comma-separated) adds more. Absent workers
    /// are retried in the background, so they may start before or after the Cluster.
    pub workers: Vec<String>,
    /// Handshake secret; empty uses `KORA_WORKER_TOKEN`.
    pub token: String,
    /// Placement policy.
    pub policy: OffloadPolicy,
    /// Deadline for connecting to a worker.
    pub connect_timeout: Duration,
    /// Delay between reconnection attempts.
    pub reconnect_interval: Duration,
    /// Deadline for one remote request; exceeding it drops the link and re-runs the job locally.
    pub request_timeout: Duration,
    /// Idle time after which a link sends a heartbeat.
    pub heartbeat_interval: Duration,
    /// Largest accepted protocol frame.
    pub max_message_bytes: u64,
    /// How often to check whether local GPUs appeared or disappeared; zero disables the check. A
    /// change rebuilds the local runtime at the next idle point, keeping every buffer.
    pub device_rescan_interval: Duration,
    /// Split dispatches smaller than this many elements always run on one target.
    pub split_min_elements: u64,
    /// Fixed cost assumed per extra slice when deciding whether splitting pays.
    pub split_overhead: Duration,
    /// Loopback `host:port` on which to stream status snapshots for `kora-top`, one JSON object
    /// per line every `status_interval`. Empty disables it; port 0 picks a free port
    /// ([`Cluster::status_port`]). `KORA_STATUS_LISTEN` sets it when empty. Only loopback
    /// addresses are accepted, because snapshots name devices and endpoints.
    pub status_listen: String,
    /// Interval between status snapshots.
    pub status_interval: Duration,
}

impl Default for ClusterOptions {
    fn default() -> Self {
        Self {
            local: RuntimeOptions::default(),
            workers: Vec::new(),
            token: String::new(),
            policy: OffloadPolicy::Auto,
            connect_timeout: Duration::from_secs(1),
            reconnect_interval: Duration::from_millis(500),
            request_timeout: Duration::from_secs(30),
            heartbeat_interval: Duration::from_secs(1),
            max_message_bytes: 1 << 30,
            device_rescan_interval: Duration::from_secs(5),
            split_min_elements: 65_536,
            split_overhead: Duration::from_micros(150),
            status_listen: String::new(),
            status_interval: Duration::from_millis(250),
        }
    }
}

/// One local device the Cluster can run work on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LocalDeviceStatus {
    /// Device identity.
    pub id: DeviceId,
    /// Device name.
    pub name: String,
    /// Whole dispatches completed on this device.
    pub dispatches: u64,
    /// Split slices completed on this device.
    pub slices: u64,
    /// Jobs running on this device.
    pub inflight: u32,
    /// Device memory KORA has allocated on this device.
    pub allocated_bytes: u64,
    /// Budget the driver reports (zero when it cannot report one).
    pub budget_bytes: u64,
}

/// One attached worker.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkerStatus {
    /// Endpoint as attached.
    pub endpoint: String,
    /// A session is established.
    pub connected: bool,
    /// Name the worker reported.
    pub name: String,
    /// Device the worker reported.
    pub device: String,
    /// Dispatches the worker completed.
    pub dispatches: u64,
    /// Jobs re-run locally after this worker failed them.
    pub failovers: u64,
    /// Sessions established.
    pub connects: u64,
    /// Sessions lost.
    pub disconnects: u64,
    /// Jobs queued or running on this worker.
    pub inflight: u32,
    /// Bytes of buffers the worker holds at the host's current version, so they are not resent.
    pub cached_bytes: u64,
}

/// Something that changed where work runs: a worker connecting or leaving, a failover, a device
/// rescan, or a kernel moving to a different target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterEvent {
    /// Milliseconds since the Cluster was created, of the latest occurrence.
    pub at_ms: u64,
    /// Description.
    pub message: String,
    /// Consecutive identical events are merged into one entry.
    pub repeats: u32,
}

/// Cluster-wide counters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClusterStats {
    /// Whole dispatches completed locally.
    pub local_dispatches: u64,
    /// Whole dispatches completed by workers.
    pub remote_dispatches: u64,
    /// Remote jobs re-executed locally after a link failure, timeout, or worker error.
    pub failovers: u64,
    /// Local device losses recovered by recreating the local devices.
    pub local_recoveries: u64,
    /// Dispatches that ran as several slices on different targets.
    pub split_dispatches: u64,
    /// Slices those dispatches used.
    pub slices: u64,
    /// Times the local device set changed and the local devices were rebuilt.
    pub device_rescans: u64,
}

#[derive(Default)]
struct LocalKernels {
    generation: u64,
    loaded: HashMap<String, Kernel>,
    unsupported: HashSet<String>,
}

struct KernelState {
    key: u64,
    words: Vec<u32>,
    entry_point: String,
    metadata: KernelMetadata,
    metadata_json: String,
    /// Push-constant offset of the `u32` field named `base`, if any.
    base_offset: Option<u32>,
    /// Per-device kernels, reloaded after the local devices change.
    local: Mutex<LocalKernels>,
}

/// A kernel loaded once and runnable on the local runtime or any compatible worker. Cheap to
/// clone.
#[derive(Clone)]
pub struct ClusterKernel {
    state: Arc<KernelState>,
}

impl std::fmt::Debug for ClusterKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterKernel")
            .field("kernel_id", &self.state.metadata.kernel_id)
            .field("entry_point", &self.state.entry_point)
            .finish()
    }
}

/// What a job computes; slices of a split dispatch copy and narrow it.
#[derive(Clone)]
struct Work {
    kernel: Arc<KernelState>,
    global: [u64; 3],
    buffers: Vec<BufferHandle>,
    accesses: Vec<KernelAccess>,
    push: Vec<u8>,
    bytes: u64,
}

struct Slice {
    split: Arc<SplitState>,
    base: u64,
    count: u64,
}

enum Completion {
    Whole(EventPromise),
    Slice(Slice),
}

struct Job {
    work: Work,
    completion: Completion,
    dependencies: Vec<Event>,
    force_local: bool,
    local_retried: bool,
    /// Local generation at which a local attempt observed device loss (0: never).
    local_generation: u64,
    /// Non-zero when the caller allowed splitting.
    split_element_bytes: u32,
    started: Instant,
}

impl Job {
    fn slice(&self) -> Option<&Slice> {
        match &self.completion {
            Completion::Slice(slice) => Some(slice),
            Completion::Whole(_) => None,
        }
    }
}

struct SplitProgress {
    remaining: usize,
    error: Option<Error>,
    /// Pre-dispatch contents of each written buffer, overwritten slice by slice.
    assembled: Vec<Vec<u8>>,
}

/// Collects the slices of one split dispatch and commits the merged outputs once.
struct SplitState {
    /// Unique per split dispatch; lets its slices share a device replica.
    id: u64,
    work: Work,
    promise: Mutex<Option<EventPromise>>,
    /// Indices of written bindings.
    outputs: Vec<usize>,
    /// Host versions the commit expects.
    versions: Vec<u64>,
    element_bytes: u32,
    started: Instant,
    progress: Mutex<SplitProgress>,
}

/// Byte range of a slice within one written buffer.
fn slice_bytes(slice: &Slice, buffer_size: u64) -> (u64, u64) {
    let element = u64::from(slice.split.element_bytes);
    let begin = (slice.base * element).min(buffer_size);
    let end = ((slice.base + slice.count) * element).min(buffer_size);
    (begin, end - begin)
}

/// A place a job can run: one local device, or one connected worker.
#[derive(Clone)]
enum Target {
    Local { device: DeviceId, key: String },
    Remote(Arc<WorkerLink>),
    None,
}

impl Target {
    fn key(&self) -> &str {
        match self {
            Target::Local { key, .. } => key,
            Target::Remote(link) => &link.endpoint,
            Target::None => "none",
        }
    }
    fn label(&self) -> String {
        match self {
            Target::Local { device, .. } => local_label(device),
            Target::Remote(link) => format!("worker {}", link.endpoint),
            Target::None => "no target".into(),
        }
    }
}

fn device_key(id: &DeviceId) -> String {
    format!("local:{}:{}", id.index, id.uuid)
}

fn local_label(id: &DeviceId) -> String {
    format!("local {}", id.index)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn read<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

fn write<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

fn fnv1a(bytes: &[u8], mut hash: u64) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

#[derive(Debug, Clone)]
struct LocalTarget {
    id: DeviceId,
    key: String,
    name: String,
}

#[derive(Default)]
struct BufferAccess {
    last_write: Option<Event>,
    reads: Vec<Event>,
}

#[derive(Default)]
struct SubmitState {
    access: HashMap<BufferId, BufferAccess>,
    outstanding: Vec<Event>,
    /// Buffers the Cluster has seen, so workers can be told when they are released.
    known: HashMap<BufferId, WeakBufferHandle>,
}

#[derive(Default)]
struct QueueState {
    jobs: VecDeque<Box<Job>>,
    shut_down: bool,
    stopping: bool,
    rescan_requested: bool,
}

#[derive(Default, Clone, Copy)]
struct DeviceCounters {
    dispatches: u64,
    slices: u64,
    inflight: u32,
}

#[derive(Default)]
struct Model {
    /// Nanoseconds per dispatch, per target, kernel, and size bucket.
    latency: HashMap<String, f64>,
    /// Elements per nanosecond, per target and kernel.
    throughput: HashMap<String, f64>,
    devices: HashMap<String, DeviceCounters>,
    explore: u64,
    split_declines: HashMap<String, u32>,
}

fn model_key(target: &str, work: &Work) -> String {
    let bucket = if work.bytes == 0 { 0 } else { work.bytes.ilog2() };
    format!("{target}|{}|{bucket}", work.kernel.key)
}

#[derive(Default)]
struct Counters {
    local_dispatches: AtomicU64,
    remote_dispatches: AtomicU64,
    failovers: AtomicU64,
    local_recoveries: AtomicU64,
    split_dispatches: AtomicU64,
    slices: AtomicU64,
    device_rescans: AtomicU64,
}

#[derive(Default)]
struct Threads {
    dispatcher: Option<JoinHandle<()>>,
    monitor: Option<JoinHandle<()>>,
    status: Option<JoinHandle<()>>,
}

struct Shared {
    options: ClusterOptions,
    /// Shared for local work; exclusive while the local devices are rebuilt.
    runtime: RwLock<Runtime>,
    targets: RwLock<Vec<LocalTarget>>,
    local_generation: AtomicU64,
    local_inflight: AtomicU32,
    submit: Mutex<SubmitState>,
    queue: Mutex<QueueState>,
    queue_cv: Condvar,
    /// Wakes the device monitor at shutdown (separate so `notify_one` always reaches the
    /// dispatcher).
    monitor_cv: Condvar,
    workers: Mutex<Vec<Arc<WorkerLink>>>,
    counters: Counters,
    model: Mutex<Model>,
    created: Instant,
    events: Mutex<VecDeque<ClusterEvent>>,
    status_port: u16,
    status_stop: AtomicBool,
    threads: Mutex<Threads>,
}

/// One host runtime plus hot-swappable remote workers. Dropping the Cluster waits for submitted
/// work (failed-over jobs still run locally) and disconnects every worker.
pub struct Cluster {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for Cluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cluster")
            .field("policy", &self.shared.options.policy)
            .field("local_devices", &read(&self.shared.targets).len())
            .field("workers", &lock(&self.shared.workers).len())
            .finish()
    }
}

impl Cluster {
    /// Creates the local runtime and starts connecting to the configured workers.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for a malformed worker endpoint or a non-loopback
    /// `status_listen`, [`ErrorCode::IoError`] when the status port cannot be bound, and the
    /// errors of [`Runtime::new`].
    pub fn new(mut options: ClusterOptions) -> Result<Cluster> {
        if options.token.is_empty() {
            options.token = std::env::var("KORA_WORKER_TOKEN").unwrap_or_default();
        }
        if options.status_listen.is_empty() {
            options.status_listen = std::env::var("KORA_STATUS_LISTEN").unwrap_or_default();
        }
        if let Ok(extra) = std::env::var("KORA_WORKERS") {
            options.workers.extend(
                extra
                    .split(',')
                    .map(str::trim)
                    .filter(|e| !e.is_empty())
                    .map(String::from),
            );
        }
        for endpoint in &options.workers {
            Endpoint::parse(endpoint)?;
        }
        let status_listener = if options.status_listen.is_empty() {
            None
        } else {
            let endpoint = Endpoint::parse(&options.status_listen)?;
            if !endpoint.is_loopback() {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    "status_listen must be a loopback address (127.0.0.1, ::1, or localhost)",
                ));
            }
            Some(listen(&endpoint)?)
        };
        let status_port = match &status_listener {
            Some(listener) => listener
                .local_addr()
                .map_err(|e| crate::remote::socket::io_error("cannot read the status port", &e))?
                .port(),
            None => 0,
        };
        // Use every eligible local device unless the caller (or KORA_DEVICE) chose one.
        let all_local_devices =
            options.local.device_policy.is_none() && std::env::var_os("KORA_DEVICE").is_none();
        if all_local_devices {
            options.local.device_policy = Some(DevicePolicy::AllEligible);
        }
        let runtime = Runtime::new(&options.local)?;
        let workers = options.workers.clone();
        let monitor = all_local_devices && !options.device_rescan_interval.is_zero();
        let shared = Arc::new(Shared {
            options,
            runtime: RwLock::new(runtime),
            targets: RwLock::default(),
            local_generation: AtomicU64::new(1),
            local_inflight: AtomicU32::new(0),
            submit: Mutex::default(),
            queue: Mutex::default(),
            queue_cv: Condvar::new(),
            monitor_cv: Condvar::new(),
            workers: Mutex::default(),
            counters: Counters::default(),
            model: Mutex::default(),
            created: Instant::now(),
            events: Mutex::default(),
            status_port,
            status_stop: AtomicBool::new(false),
            threads: Mutex::default(),
        });
        shared.refresh_local_targets(&read(&shared.runtime));
        let cluster = Cluster { shared };
        cluster.start_threads(status_listener, monitor)?;
        for endpoint in workers {
            cluster.shared.start_link(&endpoint)?;
        }
        Ok(cluster)
    }

    fn start_threads(&self, status_listener: Option<std::net::TcpListener>, monitor: bool) -> Status {
        let spawn = |name: &str, body: Box<dyn FnOnce() + Send>| {
            std::thread::Builder::new()
                .name(name.into())
                .spawn(body)
                .map_err(|e| Error::new(ErrorCode::IoError, format!("cannot start {name}: {e}")))
        };
        let shared = &self.shared;
        let dispatcher = {
            let shared = Arc::clone(shared);
            spawn("kora-cluster", Box::new(move || shared.dispatcher_loop()))?
        };
        lock(&shared.threads).dispatcher = Some(dispatcher);
        if monitor {
            let thread = {
                let shared = Arc::clone(shared);
                spawn("kora-cluster-monitor", Box::new(move || shared.monitor_loop()))?
            };
            lock(&shared.threads).monitor = Some(thread);
        }
        if let Some(listener) = status_listener {
            let thread = {
                let shared = Arc::clone(shared);
                spawn(
                    "kora-cluster-status",
                    Box::new(move || shared.status_loop(&listener)),
                )?
            };
            lock(&shared.threads).status = Some(thread);
        }
        Ok(())
    }

    /// The local runtime. Do not call [`Runtime::recover`] on it directly; the Cluster recovers
    /// device loss. Holding the guard delays a pending local device rebuild.
    pub fn runtime(&self) -> RwLockReadGuard<'_, Runtime> {
        read(&self.shared.runtime)
    }

    /// Creates a typed buffer in the local runtime.
    ///
    /// # Errors
    ///
    /// As [`Runtime::create_buffer`].
    pub fn create_buffer<T: Pod>(&self, count: u64) -> Result<Buffer<T>> {
        self.runtime().create_buffer(count)
    }

    /// Creates an untyped buffer in the local runtime.
    ///
    /// # Errors
    ///
    /// As [`Runtime::create_buffer_bytes`].
    pub fn create_buffer_bytes(&self, size_bytes: u64, usage: BufferUsage) -> Result<BufferHandle> {
        self.runtime().create_buffer_bytes(size_bytes, usage)
    }

    /// Loads a kernel from SPIR-V words. It is validated on the first local device now; other
    /// devices and workers load it when they are first chosen.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidKernel`] for an empty or oversized module, and the errors of
    /// [`Runtime::load_kernel`].
    pub fn load_kernel(
        &self,
        words: &[u32],
        entry_point: &str,
        metadata: KernelMetadata,
    ) -> Result<ClusterKernel> {
        if words.is_empty() || words.len() > 4 * 1024 * 1024 {
            return Err(Error::new(
                ErrorCode::InvalidKernel,
                "SPIR-V module is empty or exceeds 16 MiB",
            ));
        }
        let metadata_json = metadata.to_json();
        let word_bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut key = fnv1a(&word_bytes, 0xCBF2_9CE4_8422_2325);
        key = fnv1a(metadata_json.as_bytes(), key);
        key = fnv1a(entry_point.as_bytes(), key);
        let base_offset = metadata
            .push_constants
            .iter()
            .find(|f| f.name == "base" && f.type_name == "u32" && f.size == 4)
            .map(|f| f.offset);
        let state = Arc::new(KernelState {
            key,
            words: words.to_vec(),
            entry_point: entry_point.to_string(),
            metadata,
            metadata_json,
            base_offset,
            local: Mutex::default(),
        });
        let runtime = self.runtime();
        let first = read(&self.shared.targets)
            .first()
            .map(|t| t.id.clone())
            .ok_or_else(|| Error::new(ErrorCode::NoEligibleDevice, "local runtime has no device"))?;
        self.shared.local_kernel(&runtime, &state, &first)?;
        Ok(ClusterKernel { state })
    }

    /// Reads a `.spv` file (at most 16 MiB) and loads it like [`load_kernel`](Self::load_kernel).
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidKernel`] for unreadable or malformed files; otherwise as
    /// [`load_kernel`](Self::load_kernel).
    pub fn load_kernel_file(
        &self,
        path: impl AsRef<Path>,
        entry_point: &str,
        metadata: KernelMetadata,
    ) -> Result<ClusterKernel> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| {
            Error::new(
                ErrorCode::InvalidKernel,
                format!("cannot read SPIR-V module {}: {e}", path.display()),
            )
        })?;
        let words = crate::kernel::spirv::words_from_bytes(&bytes)?;
        self.load_kernel(&words, entry_point, metadata)
    }

    /// Starts describing a dispatch of `kernel`.
    pub fn dispatch(&self, kernel: &ClusterKernel) -> ClusterDispatch {
        ClusterDispatch {
            shared: Arc::clone(&self.shared),
            kernel: Arc::clone(&kernel.state),
            global: [1, 1, 1],
            buffers: Vec::new(),
            accesses: Vec::new(),
            push: Vec::new(),
            split_element_bytes: 0,
        }
    }

    /// Adds a worker endpoint; it is connected in the background.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for a malformed or already attached endpoint.
    pub fn attach(&self, endpoint: &str) -> Status {
        self.shared.start_link(endpoint)
    }

    /// Removes a worker; its queued jobs run locally.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] when the endpoint is not attached.
    pub fn detach(&self, endpoint: &str) -> Status {
        let link = {
            let mut workers = lock(&self.shared.workers);
            let index = workers
                .iter()
                .position(|l| l.endpoint == endpoint)
                .ok_or_else(|| Error::new(ErrorCode::InvalidArgument, "worker is not attached"))?;
            workers.remove(index)
        };
        link.stop_and_join();
        Ok(())
    }

    /// Waits for every submitted dispatch to complete.
    ///
    /// # Errors
    ///
    /// The first failure among them.
    pub fn wait_idle(&self) -> Status {
        let pending = lock(&self.shared.submit).outstanding.clone();
        pending.iter().try_for_each(Event::wait)
    }

    /// Status of every attached worker.
    pub fn workers(&self) -> Vec<WorkerStatus> {
        self.shared.worker_status()
    }

    /// Status of every local device.
    pub fn local_devices(&self) -> Vec<LocalDeviceStatus> {
        self.shared.local_status()
    }

    /// Cluster-wide counters.
    pub fn stats(&self) -> ClusterStats {
        self.shared.stats()
    }

    /// The most recent events, oldest first (at most 64).
    pub fn events(&self) -> Vec<ClusterEvent> {
        lock(&self.shared.events).iter().cloned().collect()
    }

    /// Port of the `status_listen` endpoint, or 0 when it is disabled.
    pub fn status_port(&self) -> u16 {
        self.shared.status_port
    }
}

impl Drop for Cluster {
    fn drop(&mut self) {
        self.shared.shutdown();
    }
}

/// Describes one Cluster dispatch; finish with [`submit`](Self::submit).
pub struct ClusterDispatch {
    shared: Arc<Shared>,
    kernel: Arc<KernelState>,
    global: [u64; 3],
    buffers: Vec<BufferHandle>,
    accesses: Vec<KernelAccess>,
    push: Vec<u8>,
    split_element_bytes: u32,
}

impl std::fmt::Debug for ClusterDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterDispatch")
            .field("kernel_id", &self.kernel.metadata.kernel_id)
            .field("global", &self.global)
            .field("bindings", &self.buffers.len())
            .finish()
    }
}

impl ClusterDispatch {
    /// One-dimensional global size.
    pub fn global(self, x: u64) -> Self {
        self.global_3d(x, 1, 1)
    }

    /// Three-dimensional global size, in invocations.
    pub fn global_3d(mut self, x: u64, y: u64, z: u64) -> Self {
        self.global = [x, y, z];
        self
    }

    fn bind(mut self, buffer: &impl AsRef<BufferHandle>, access: KernelAccess) -> Self {
        self.buffers.push(buffer.as_ref().clone());
        self.accesses.push(access);
        self
    }

    /// Binds the next buffer read-only.
    pub fn read(self, buffer: &impl AsRef<BufferHandle>) -> Self {
        self.bind(buffer, KernelAccess::Read)
    }

    /// Binds the next buffer write-only.
    pub fn write(self, buffer: &impl AsRef<BufferHandle>) -> Self {
        self.bind(buffer, KernelAccess::Write)
    }

    /// Binds the next buffer read-write.
    pub fn read_write(self, buffer: &impl AsRef<BufferHandle>) -> Self {
        self.bind(buffer, KernelAccess::ReadWrite)
    }

    /// Push constants from a `Pod` value.
    pub fn push<T: Pod>(self, value: &T) -> Self {
        self.push_bytes(bytemuck::bytes_of(value))
    }

    /// Push constants from raw bytes.
    pub fn push_bytes(mut self, bytes: &[u8]) -> Self {
        self.push = bytes.to_vec();
        self
    }

    /// Allows the Cluster to split this dispatch along x across several devices. The kernel must
    /// declare a `u32` push-constant field named `base` and handle element
    /// `base + gl_GlobalInvocationID.x`, and every written buffer must hold `output_element_bytes`
    /// per element. Each device computes one range; the merged result is committed once, so a
    /// split dispatch is still all-or-nothing. Small dispatches, or ones where splitting is not
    /// expected to pay, run on one target.
    pub fn split(mut self, output_element_bytes: u32) -> Self {
        self.split_element_bytes = output_element_bytes;
        self
    }

    /// Enqueues the dispatch behind earlier Cluster dispatches that touch the same buffers. The
    /// event completes after outputs are in the host copies.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidKernelBinding`] when the buffer count does not match the metadata,
    /// [`ErrorCode::InvalidArgument`] for a push-constant size mismatch, a `split` without a
    /// `base` field or beyond 2^32 elements, or a shut-down Cluster.
    pub fn submit(self) -> Result<Event> {
        let kernel = &self.kernel;
        if self.buffers.len() != kernel.metadata.bindings.len() {
            return Err(Error::new(
                ErrorCode::InvalidKernelBinding,
                "dispatch buffer count does not match kernel metadata",
            ));
        }
        if self.push.len() != kernel.metadata.push_constant_size as usize {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "push-constant byte size does not match kernel metadata",
            ));
        }
        if self.split_element_bytes != 0 && kernel.base_offset.is_none() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "split() requires a u32 push-constant field named 'base'",
            ));
        }
        if self.split_element_bytes != 0 && self.global[0] > u64::from(u32::MAX) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "split() supports at most 2^32 elements",
            ));
        }
        let shared = self.shared;
        if lock(&shared.queue).shut_down {
            return Err(Error::new(ErrorCode::InvalidArgument, "cluster is shut down"));
        }
        let promise = EventPromise::new();
        let event = promise.event();
        let mut job = Box::new(Job {
            work: Work {
                bytes: self.buffers.iter().map(BufferHandle::size_bytes).sum(),
                kernel: self.kernel,
                global: self.global,
                buffers: self.buffers,
                accesses: self.accesses,
                push: self.push,
            },
            completion: Completion::Whole(promise),
            dependencies: Vec::new(),
            force_local: false,
            local_retried: false,
            local_generation: 0,
            split_element_bytes: self.split_element_bytes,
            started: Instant::now(),
        });
        {
            // Order this job after earlier jobs that touch the same buffers (RAW, WAR, WAW).
            let mut submit = lock(&shared.submit);
            submit.outstanding.retain(|e| !e.is_done());
            submit.outstanding.push(event.clone());
            // Collect every dependency before recording this job, so a buffer bound twice never
            // makes the job wait on itself.
            for buffer in &job.work.buffers {
                let id = buffer.id();
                submit.known.entry(id).or_insert_with(|| buffer.downgrade());
            }
            for (buffer, access) in job.work.buffers.iter().zip(&job.work.accesses) {
                let state = submit.access.entry(buffer.id()).or_default();
                state.reads.retain(|e| !e.is_done());
                if let Some(last) = state.last_write.as_ref().filter(|e| !e.is_done()) {
                    job.dependencies.push(last.clone());
                }
                if *access != KernelAccess::Read {
                    job.dependencies.extend(state.reads.iter().cloned());
                }
            }
            for (buffer, access) in job.work.buffers.iter().zip(&job.work.accesses) {
                let state = submit.access.entry(buffer.id()).or_default();
                if *access == KernelAccess::Read {
                    state.reads.push(event.clone());
                } else {
                    state.reads.clear();
                    state.last_write = Some(event.clone());
                }
            }
        }
        shared.enqueue(job, false);
        Ok(event)
    }
}

/// Targets each kernel used in the current and previous one-second windows; dispatcher only.
#[derive(Default)]
struct Routing {
    windows: HashMap<u64, (String, BTreeSet<String>, BTreeSet<String>)>,
    labels: HashMap<String, String>,
    window_start: Option<Instant>,
}

impl Shared {
    fn note(&self, message: impl Into<String>) {
        let message = message.into();
        let at_ms = self.created.elapsed().as_millis() as u64;
        let mut events = lock(&self.events);
        if let Some(last) = events.back_mut().filter(|e| e.message == message) {
            last.at_ms = at_ms;
            last.repeats += 1;
            return;
        }
        events.push_back(ClusterEvent {
            at_ms,
            message,
            repeats: 1,
        });
        if events.len() > MAX_EVENTS {
            events.pop_front();
        }
    }

    /// Reports when the set of targets a kernel runs on changes, once per second at most, so
    /// steady alternation between equivalent targets does not flood the event log.
    fn note_route(&self, routing: &mut Routing, work: &Work, key: &str, label: impl FnOnce() -> String) {
        if !routing.labels.contains_key(key) {
            routing.labels.insert(key.to_string(), label());
        }
        let window = routing.windows.entry(work.kernel.key).or_insert_with(|| {
            (
                work.kernel.metadata.kernel_id.clone(),
                BTreeSet::new(),
                BTreeSet::new(),
            )
        });
        window.1.insert(key.to_string());
        let now = Instant::now();
        let start = *routing.window_start.get_or_insert(now);
        if now - start < Duration::from_secs(1) {
            return;
        }
        routing.window_start = Some(now);
        for (kernel_id, current, previous) in routing.windows.values_mut() {
            if current.is_empty() {
                continue; // Idle kernels keep their last known routing.
            }
            if current != previous {
                let targets: Vec<&str> = current
                    .iter()
                    .map(|t| routing.labels.get(t).map_or(t.as_str(), String::as_str))
                    .collect();
                self.note(format!("{kernel_id} now runs on {}", targets.join(", ")));
            }
            *previous = std::mem::take(current);
        }
    }

    fn enqueue(&self, job: Box<Job>, front: bool) {
        {
            let mut queue = lock(&self.queue);
            if front {
                queue.jobs.push_front(job);
            } else {
                queue.jobs.push_back(job);
            }
        }
        self.queue_cv.notify_all();
    }

    fn fail_over(&self, mut job: Box<Job>, link: Option<&WorkerLink>) {
        self.counters.failovers.fetch_add(1, Ordering::Relaxed);
        if let Some(link) = link {
            link.state().status.failovers += 1;
            self.note(format!(
                "worker {} failed a dispatch; re-running it locally",
                link.endpoint
            ));
        }
        job.force_local = true;
        self.enqueue(job, true); // Its dependents are already ordered behind its Event.
    }

    fn refresh_local_targets(&self, runtime: &Runtime) {
        let targets = runtime
            .devices()
            .into_iter()
            .map(|id| LocalTarget {
                key: device_key(&id),
                name: runtime
                    .device_info(&id)
                    .map_or_else(|_| format!("device {}", id.index), |info| info.name.clone()),
                id,
            })
            .collect();
        *write(&self.targets) = targets;
    }

    fn local_kernel(&self, runtime: &Runtime, kernel: &KernelState, device: &DeviceId) -> Result<Kernel> {
        let generation = self.local_generation.load(Ordering::Acquire);
        let key = device_key(device);
        let mut local = lock(&kernel.local);
        if local.generation != generation {
            *local = LocalKernels {
                generation,
                ..LocalKernels::default()
            };
        }
        if let Some(loaded) = local.loaded.get(&key) {
            return Ok(loaded.clone());
        }
        if local.unsupported.contains(&key) {
            return Err(Error::new(
                ErrorCode::UnsupportedFeature,
                "kernel is not supported on this local device",
            ));
        }
        match runtime.load_kernel_on(
            device,
            &kernel.words,
            &kernel.entry_point,
            kernel.metadata.clone(),
        ) {
            Ok(loaded) => {
                local.loaded.insert(key, loaded.clone());
                Ok(loaded)
            }
            Err(error) => {
                local.unsupported.insert(key);
                Err(error)
            }
        }
    }

    fn candidates(&self, job: &Job) -> Vec<Target> {
        let generation = self.local_generation.load(Ordering::Acquire);
        let mut result: Vec<Target> = {
            let local = lock(&job.work.kernel.local);
            read(&self.targets)
                .iter()
                .filter(|t| !(local.generation == generation && local.unsupported.contains(&t.key)))
                .map(|t| Target::Local {
                    device: t.id.clone(),
                    key: t.key.clone(),
                })
                .collect()
        };
        if job.force_local || self.options.policy == OffloadPolicy::LocalOnly {
            return result;
        }
        for link in lock(&self.workers).iter() {
            if link.connected.load(Ordering::Acquire)
                && !link.state().unsupported_kernels.contains(&job.work.kernel.key)
            {
                result.push(Target::Remote(Arc::clone(link)));
            }
        }
        result
    }

    fn inflight(&self, target: &Target) -> f64 {
        match target {
            Target::Remote(link) => f64::from(link.inflight.load(Ordering::Acquire)),
            Target::Local { key, .. } => {
                f64::from(lock(&self.model).devices.get(key).map_or(0, |c| c.inflight))
            }
            Target::None => 0.0,
        }
    }

    fn predicted(&self, target: &str, work: &Work) -> Option<f64> {
        lock(&self.model).latency.get(&model_key(target, work)).copied()
    }

    fn throughput(&self, target: &str, work: &Work) -> Option<f64> {
        lock(&self.model)
            .throughput
            .get(&format!("{target}|{}", work.kernel.key))
            .copied()
    }

    fn record_latency(&self, target: &str, work: &Work, elapsed: Duration) {
        let ns = elapsed.as_nanos() as f64;
        lock(&self.model)
            .latency
            .entry(model_key(target, work))
            .and_modify(|ema| *ema = 0.8 * *ema + 0.2 * ns)
            .or_insert(ns);
    }

    fn record_throughput(&self, target: &str, work: &Work, elements: u64, elapsed: Duration) {
        let ns = elapsed.as_nanos() as f64;
        if ns <= 0.0 || elements == 0 {
            return;
        }
        let rate = elements as f64 / ns;
        lock(&self.model)
            .throughput
            .entry(format!("{target}|{}", work.kernel.key))
            .and_modify(|ema| *ema = 0.7 * *ema + 0.3 * rate)
            .or_insert(rate);
    }

    fn count_device(&self, key: &str, slice: bool, starting: bool) {
        let mut model = lock(&self.model);
        let counters = model.devices.entry(key.to_string()).or_default();
        if starting {
            counters.inflight += 1;
        } else {
            counters.inflight = counters.inflight.saturating_sub(1);
            if slice {
                counters.slices += 1;
            } else {
                counters.dispatches += 1;
            }
        }
    }

    fn choose_target(&self, job: &Job) -> Target {
        let options = self.candidates(job);
        if options.is_empty() {
            return Target::None;
        }
        if self.options.policy == OffloadPolicy::PreferRemote {
            let remote = options
                .iter()
                .filter(|t| matches!(t, Target::Remote(_)))
                .min_by(|a, b| self.inflight(a).total_cmp(&self.inflight(b)));
            if let Some(remote) = remote {
                return remote.clone();
            }
        }
        // Measure every local device once, explore each unmeasured worker occasionally, then pick
        // the lowest expected completion time including work already queued on each target.
        if let Some(unmeasured) = options
            .iter()
            .find(|t| matches!(t, Target::Local { .. }) && self.predicted(t.key(), &job.work).is_none())
        {
            return unmeasured.clone();
        }
        let mut best: Option<(&Target, f64)> = None;
        for target in &options {
            let Some(latency) = self.predicted(target.key(), &job.work) else {
                let mut model = lock(&self.model);
                model.explore += 1;
                if (model.explore - 1).is_multiple_of(8) {
                    return target.clone();
                }
                continue;
            };
            let cost = latency * (1.0 + self.inflight(target));
            if best.is_none_or(|(_, best_cost)| cost < best_cost) {
                best = Some((target, cost));
            }
        }
        best.map_or_else(|| options[0].clone(), |(target, _)| target.clone())
    }

    /// Brings any newer device-resident contents of the job's buffers back to the host copies.
    fn ensure_host_current(&self, work: &Work) {
        let runtime = read(&self.runtime);
        for buffer in &work.buffers {
            if let Ok(event) = runtime.readback(buffer) {
                let _ = event.wait();
            }
        }
    }

    /// Splits `job` across targets when that is expected to pay; returns it untouched otherwise.
    fn try_split(
        self: &Arc<Self>,
        job: Box<Job>,
        routing: &mut Routing,
    ) -> std::result::Result<(), Box<Job>> {
        if job.split_element_bytes == 0 || job.slice().is_some() || job.force_local {
            return Err(job);
        }
        let count = job.work.global[0];
        if count < self.options.split_min_elements || job.work.global[1] != 1 || job.work.global[2] != 1 {
            return Err(job);
        }
        let targets = self.candidates(&job);
        if targets.len() < 2 {
            return Err(job);
        }
        // Weight targets by measured throughput; unmeasured ones get the average so they are
        // explored.
        let measured: Vec<Option<f64>> = targets
            .iter()
            .map(|t| self.throughput(t.key(), &job.work))
            .collect();
        let known: Vec<f64> = measured.iter().flatten().copied().collect();
        let fallback = if known.is_empty() {
            1.0
        } else {
            known.iter().sum::<f64>() / known.len() as f64
        };
        let rates: Vec<f64> = measured
            .iter()
            .map(|r| r.filter(|r| *r > 0.0).unwrap_or(fallback))
            .collect();
        let mut order: Vec<usize> = (0..targets.len()).collect();
        order.sort_by(|a, b| rates[*b].total_cmp(&rates[*a]));
        // Use the k fastest targets that minimize the estimated makespan, if that beats one target.
        // ponytail: linear model (count / summed rate + fixed overhead per extra slice); replace
        // with measured per-target fixed latency if slices of very different sizes mispredict.
        let overhead = self.options.split_overhead.as_nanos() as f64;
        let single = count as f64 / rates[order[0]];
        let (mut best_k, mut best_time, mut rate_sum) = (1, single, 0.0);
        for (k, index) in order.iter().enumerate() {
            rate_sum += rates[*index];
            let time = count as f64 / rate_sum + overhead * k as f64;
            if time < best_time {
                best_time = time;
                best_k = k + 1;
            }
        }
        if best_k < 2 || best_time > 0.95 * single {
            return Err(job);
        }
        // The model assumes devices run independently; two drivers of one GPU, or a CPU renderer
        // sharing memory bandwidth, do not. Split only after one device has been measured alone,
        // and stop once finished splits measure no faster, re-probing every 16th time in case load
        // changes.
        let Some(alone) = targets
            .iter()
            .filter_map(|t| self.predicted(t.key(), &job.work))
            .min_by(f64::total_cmp)
        else {
            return Err(job);
        };
        if self
            .predicted("split", &job.work)
            .is_some_and(|split| split >= 0.95 * alone)
        {
            let mut model = lock(&self.model);
            let declines = model
                .split_declines
                .entry(model_key("split", &job.work))
                .or_default();
            *declines += 1;
            if !declines.is_multiple_of(16) {
                return Err(job);
            }
        }

        // Slice boundaries are multiples of the workgroup size so no two devices write one element.
        let granule = u64::from(job.work.kernel.metadata.local_size[0].max(1));
        let used_rate: f64 = order[..best_k].iter().map(|i| rates[*i]).sum();
        let mut plan = Vec::new(); // (target index, element count)
        let mut assigned = 0;
        for (k, index) in order[..best_k].iter().enumerate() {
            let share = if k + 1 == best_k {
                count - assigned
            } else {
                ((count as f64 * rates[*index] / used_rate) as u64 / granule * granule).min(count - assigned)
            };
            if share != 0 {
                plan.push((*index, share));
                assigned += share;
            }
        }
        if plan.len() < 2 {
            return Err(job);
        }
        let split_key = plan.iter().fold("split".to_string(), |key, (index, _)| {
            format!("{key}|{}", targets[*index].key())
        });
        self.note_route(routing, &job.work, &split_key, || {
            let labels: Vec<String> = plan.iter().map(|(index, _)| targets[*index].label()).collect();
            format!("split({})", labels.join(" + "))
        });

        self.ensure_host_current(&job.work);
        let mut outputs = Vec::new();
        let mut versions = Vec::new();
        let mut assembled = Vec::new();
        for (index, (buffer, access)) in job.work.buffers.iter().zip(&job.work.accesses).enumerate() {
            if *access == KernelAccess::Read {
                continue;
            }
            let version = buffer.host_version();
            let Ok(contents) = buffer.read_bytes() else {
                return Err(job);
            };
            outputs.push(index);
            versions.push(version);
            assembled.push(contents);
        }
        let Job {
            work,
            completion: Completion::Whole(promise),
            split_element_bytes,
            ..
        } = *job
        else {
            unreachable!("slices are never split again");
        };
        let base_offset = work.kernel.base_offset.expect("validated at submit") as usize;
        static NEXT_SPLIT: AtomicU64 = AtomicU64::new(1);
        let split = Arc::new(SplitState {
            id: NEXT_SPLIT.fetch_add(1, Ordering::Relaxed),
            work: work.clone(),
            promise: Mutex::new(Some(promise)),
            outputs,
            versions,
            element_bytes: split_element_bytes,
            started: Instant::now(),
            progress: Mutex::new(SplitProgress {
                remaining: plan.len(),
                error: None,
                assembled,
            }),
        });
        self.counters.split_dispatches.fetch_add(1, Ordering::Relaxed);
        let mut base = 0_u64;
        for (index, share) in plan {
            let mut slice_work = work.clone();
            slice_work.global = [share, 1, 1];
            slice_work.push[base_offset..base_offset + 4].copy_from_slice(&(base as u32).to_le_bytes());
            let slice = Box::new(Job {
                work: slice_work,
                completion: Completion::Slice(Slice {
                    split: Arc::clone(&split),
                    base,
                    count: share,
                }),
                dependencies: Vec::new(),
                force_local: false,
                local_retried: false,
                local_generation: 0,
                split_element_bytes: 0,
                started: Instant::now(),
            });
            base += share;
            self.start_on(slice, &targets[index]);
        }
        Ok(())
    }

    fn start_on(self: &Arc<Self>, mut job: Box<Job>, target: &Target) {
        job.started = Instant::now();
        match target {
            Target::Remote(link) => link.push(job),
            Target::Local { device, .. } => self.run_local(job, device),
            Target::None => self.fail(
                *job,
                Error::new(
                    ErrorCode::NoEligibleDevice,
                    "no local or remote target can run this kernel",
                ),
            ),
        }
    }

    /// Completes a job (or its slice) with an error.
    fn fail(&self, job: Job, error: Error) {
        match job.completion {
            Completion::Whole(promise) => promise.complete(Err(error)),
            Completion::Slice(slice) => self.finish_slice(slice, &job.work, Err(error), None),
        }
    }

    /// Records one slice's outcome; the last slice commits the merged outputs once, or nothing.
    fn finish_slice(
        &self,
        slice: Slice,
        work: &Work,
        result: Result<Vec<Vec<u8>>>,
        target: Option<(&str, Instant)>,
    ) {
        let split = &slice.split;
        if result.is_ok() {
            self.counters.slices.fetch_add(1, Ordering::Relaxed);
            if let Some((key, started)) = target {
                self.record_throughput(key, work, slice.count, started.elapsed());
            }
        }
        let (assembled, error) = {
            let mut progress = lock(&split.progress);
            match result {
                Ok(ranges) => {
                    for (k, range) in ranges.iter().enumerate().take(split.outputs.len()) {
                        let buffer = &mut progress.assembled[k];
                        let (offset, size) = slice_bytes(&slice, buffer.len() as u64);
                        let (offset, size) = (offset as usize, (size as usize).min(range.len()));
                        buffer[offset..offset + size].copy_from_slice(&range[..size]);
                    }
                }
                Err(error) => {
                    progress.error.get_or_insert(error);
                }
            }
            progress.remaining -= 1;
            if progress.remaining != 0 {
                return;
            }
            (std::mem::take(&mut progress.assembled), progress.error.take())
        };
        let Some(promise) = lock(&split.promise).take() else {
            return;
        };
        if let Some(error) = error {
            promise.complete(Err(error));
            return;
        }
        for (k, bytes) in assembled.iter().enumerate() {
            let _ = split.work.buffers[split.outputs[k]].write_bytes_if_version(bytes, split.versions[k]);
        }
        self.record_latency("split", &split.work, split.started.elapsed());
        promise.complete(Ok(()));
    }

    /// Retries a job once locally after device loss (after the devices are recreated); fails it
    /// otherwise.
    fn local_failed(&self, mut job: Box<Job>, error: Error, generation: u64) {
        if error.code() == ErrorCode::DeviceLost && !job.local_retried {
            job.local_retried = true;
            job.local_generation = generation;
            job.force_local = true;
            self.enqueue(job, true);
            return;
        }
        self.fail(*job, error);
    }

    fn local_done(&self, key: &str, slice: bool) {
        self.local_inflight.fetch_sub(1, Ordering::AcqRel);
        self.count_device(key, slice, false);
        self.queue_cv.notify_all(); // A pending rebuild may now proceed.
    }

    fn run_local(self: &Arc<Self>, job: Box<Job>, device: &DeviceId) {
        let started = Instant::now();
        let generation = self.local_generation.load(Ordering::Acquire);
        let key = device_key(device);
        let split = job.slice().map(|slice| slice.split.id);
        let slice = split.is_some();
        let submitted = {
            let runtime = read(&self.runtime);
            self.local_kernel(&runtime, &job.work.kernel, device)
                .and_then(|kernel| {
                    let mut builder = runtime
                        .dispatch(&kernel)
                        .global_3d(job.work.global[0], job.work.global[1], job.work.global[2])
                        .push_bytes(&job.work.push);
                    for (buffer, access) in job.work.buffers.iter().zip(&job.work.accesses) {
                        builder = match access {
                            KernelAccess::Read => builder.read(buffer),
                            KernelAccess::Write => builder.write(buffer),
                            KernelAccess::ReadWrite => builder.read_write(buffer),
                        };
                    }
                    // A slice leaves its outputs on the device; only its own range is read back.
                    let reader = if let Some(split) = split {
                        builder = builder.partial_outputs(split);
                        Some(runtime.replica_reader(device)?)
                    } else {
                        None
                    };
                    Ok((builder.submit()?, reader))
                })
        };
        let (event, reader) = match submitted {
            Ok(submitted) => submitted,
            Err(error) => return self.local_failed(job, error, generation),
        };
        self.local_inflight.fetch_add(1, Ordering::AcqRel);
        self.count_device(&key, slice, true);
        let weak = Arc::downgrade(self);
        let _ = event.on_complete(move |status| {
            let Some(shared) = weak.upgrade() else { return };
            if let Err(error) = status {
                shared.local_done(&key, slice);
                return shared.local_failed(job, error, generation);
            }
            let Some(reader) = reader else {
                shared.local_done(&key, slice);
                shared.counters.local_dispatches.fetch_add(1, Ordering::Relaxed);
                shared.record_latency(&key, &job.work, started.elapsed());
                if let Completion::Whole(promise) = job.completion {
                    promise.complete(Ok(()));
                }
                return;
            };
            shared.read_slice(*job, &reader, key, started);
        });
    }

    /// Reads back this slice's range of every written buffer, without waiting here.
    fn read_slice(
        self: &Arc<Self>,
        job: Job,
        reader: &crate::runtime::ReplicaReader,
        key: String,
        started: Instant,
    ) {
        let Completion::Slice(slice) = job.completion else {
            unreachable!("only slices defer their readback")
        };
        let ranges = Arc::new(Mutex::new(vec![Vec::new(); slice.split.outputs.len()]));
        let mut reads = Vec::new();
        for (k, index) in slice.split.outputs.iter().enumerate() {
            let buffer = &job.work.buffers[*index];
            let (offset, size) = slice_bytes(&slice, buffer.size_bytes());
            let ranges = Arc::clone(&ranges);
            match reader.read(buffer, offset, size, move |bytes| {
                lock(&ranges)[k] = bytes.to_vec();
                Ok(())
            }) {
                Ok(event) => reads.push(event),
                Err(error) => {
                    self.local_done(&key, true);
                    return self.finish_slice(slice, &job.work, Err(error), None);
                }
            }
        }
        let all = match Event::when_all(&reads) {
            Ok(all) => all,
            Err(error) => {
                self.local_done(&key, true);
                return self.finish_slice(slice, &job.work, Err(error), None);
            }
        };
        let weak = Arc::downgrade(self);
        let work = job.work;
        let _ = all.on_complete(move |status| {
            let Some(shared) = weak.upgrade() else { return };
            shared.local_done(&key, true);
            let result = status.map(|()| std::mem::take(&mut *lock(&ranges)));
            shared.finish_slice(slice, &work, result, Some((&key, started)));
        });
    }

    /// Waits until no local dispatch is in flight (the runtime must be idle to rebuild).
    fn wait_local_idle(&self) {
        let mut queue = lock(&self.queue);
        while self.local_inflight.load(Ordering::Acquire) != 0 && !queue.stopping {
            queue = self
                .queue_cv
                .wait_timeout(queue, Duration::from_millis(20))
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// Recreates the local devices after device loss (`observed` is the generation the loss was
    /// seen at) or after the device set changed (`observed` is `None`).
    fn rebuild_local(&self, observed: Option<u64>) {
        self.wait_local_idle();
        let mut runtime = write(&self.runtime);
        if observed.is_some_and(|g| g != self.local_generation.load(Ordering::Acquire)) {
            return; // Already recovered.
        }
        let rebuilt = if observed.is_some() {
            runtime.recover()
        } else {
            runtime.rebuild_devices()
        };
        if let Err(error) = rebuilt {
            drop(runtime);
            self.note(format!("could not recreate local devices: {}", error.message()));
            return;
        }
        self.local_generation.fetch_add(1, Ordering::AcqRel);
        self.refresh_local_targets(&runtime);
        let count = read(&self.targets).len();
        drop(runtime);
        if observed.is_some() {
            self.counters.local_recoveries.fetch_add(1, Ordering::Relaxed);
            self.note(format!("local device lost; recreated {count} local devices"));
        } else {
            self.counters.device_rescans.fetch_add(1, Ordering::Relaxed);
            self.note(format!("local GPUs changed; now using {count} local devices"));
        }
    }

    /// Forgets released buffers and tells every worker to drop its copies.
    fn sweep_released(&self) {
        let released: Vec<u64> = {
            let mut submit = lock(&self.submit);
            let dead: Vec<BufferId> = submit
                .known
                .iter()
                .filter(|(_, weak)| !weak.is_alive())
                .map(|(id, _)| *id)
                .collect();
            for id in &dead {
                submit.known.remove(id);
                submit.access.remove(id);
            }
            dead.into_iter().map(buffer_key).collect()
        };
        if released.is_empty() {
            return;
        }
        for link in lock(&self.workers).iter() {
            link.release(&released);
        }
    }

    fn dispatcher_loop(self: Arc<Self>) {
        let mut routing = Routing::default();
        let mut last_sweep = Instant::now();
        loop {
            let (job, rescan) = {
                let mut queue = lock(&self.queue);
                loop {
                    if queue.rescan_requested {
                        queue.rescan_requested = false;
                        break (None, true);
                    }
                    if let Some(job) = queue.jobs.pop_front() {
                        break (Some(job), false);
                    }
                    if queue.stopping {
                        return;
                    }
                    if last_sweep.elapsed() >= RELEASE_SWEEP {
                        break (None, false);
                    }
                    queue = self
                        .queue_cv
                        .wait_timeout(queue, RELEASE_SWEEP)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
            };
            if last_sweep.elapsed() >= RELEASE_SWEEP {
                self.sweep_released();
                last_sweep = Instant::now();
            }
            if rescan {
                self.rebuild_local(None);
            }
            let Some(mut job) = job else { continue };
            // Never block this thread on dependencies: a dependency may be a remote job that fails
            // over and needs this thread to run it locally. Re-queue the job once they complete.
            job.dependencies.retain(|e| !e.is_done());
            if !job.dependencies.is_empty() {
                let dependencies = std::mem::take(&mut job.dependencies);
                if let Ok(all) = Event::when_all(&dependencies) {
                    let weak = Arc::downgrade(&self);
                    let _ = all.on_complete(move |_| {
                        if let Some(shared) = weak.upgrade() {
                            shared.enqueue(job, false);
                        }
                    });
                    continue;
                }
            }
            if job.local_generation != 0
                && job.local_generation == self.local_generation.load(Ordering::Acquire)
            {
                // A local attempt observed device loss: recreate the devices before retrying.
                self.rebuild_local(Some(job.local_generation));
            }
            let job = match self.try_split(job, &mut routing) {
                Ok(()) => continue,
                Err(job) => job,
            };
            let target = self.choose_target(&job);
            self.note_route(&mut routing, &job.work, target.key(), || target.label());
            self.start_on(job, &target);
        }
    }

    fn monitor_loop(self: Arc<Self>) {
        let mut fingerprint = physical_device_fingerprint();
        loop {
            {
                let queue = lock(&self.queue);
                let (queue, _) = self
                    .monitor_cv
                    .wait_timeout_while(queue, self.options.device_rescan_interval, |q| {
                        !q.shut_down && !q.stopping
                    })
                    .unwrap_or_else(PoisonError::into_inner);
                if queue.shut_down || queue.stopping {
                    return;
                }
            }
            let now = physical_device_fingerprint();
            if now.is_empty() || now == fingerprint {
                continue;
            }
            fingerprint = now;
            lock(&self.queue).rescan_requested = true;
            self.queue_cv.notify_all();
        }
    }

    fn start_link(self: &Arc<Self>, endpoint: &str) -> Status {
        let address = Endpoint::parse(endpoint)?;
        let mut workers = lock(&self.workers);
        if workers.iter().any(|l| l.endpoint == endpoint) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "worker is already attached",
            ));
        }
        let link = WorkerLink::start(self, endpoint, address)?;
        workers.push(link);
        Ok(())
    }

    fn worker_status(&self) -> Vec<WorkerStatus> {
        lock(&self.workers).iter().map(|link| link.status()).collect()
    }

    fn local_status(&self) -> Vec<LocalDeviceStatus> {
        let runtime = read(&self.runtime);
        let targets = read(&self.targets).clone();
        targets
            .into_iter()
            .map(|target| {
                let mut status = LocalDeviceStatus {
                    name: target.name,
                    ..LocalDeviceStatus::default()
                };
                if let Ok(memory) = runtime.memory_stats(&target.id) {
                    status.allocated_bytes = memory.allocated_bytes;
                    status.budget_bytes = memory
                        .heaps
                        .iter()
                        .filter(|h| h.budget_queried)
                        .map(|h| h.budget_bytes)
                        .sum();
                }
                if let Some(counters) = lock(&self.model).devices.get(&target.key) {
                    status.dispatches = counters.dispatches;
                    status.slices = counters.slices;
                    status.inflight = counters.inflight;
                }
                status.id = target.id;
                status
            })
            .collect()
    }

    fn stats(&self) -> ClusterStats {
        let c = &self.counters;
        ClusterStats {
            local_dispatches: c.local_dispatches.load(Ordering::Relaxed),
            remote_dispatches: c.remote_dispatches.load(Ordering::Relaxed),
            failovers: c.failovers.load(Ordering::Relaxed),
            local_recoveries: c.local_recoveries.load(Ordering::Relaxed),
            split_dispatches: c.split_dispatches.load(Ordering::Relaxed),
            slices: c.slices.load(Ordering::Relaxed),
            device_rescans: c.device_rescans.load(Ordering::Relaxed),
        }
    }

    fn shutdown(&self) {
        {
            let mut queue = lock(&self.queue);
            if queue.shut_down {
                return;
            }
            queue.shut_down = true;
        }
        self.status_stop.store(true, Ordering::Release);
        let (status, monitor) = {
            let mut threads = lock(&self.threads);
            (threads.status.take(), threads.monitor.take())
        };
        if let Some(thread) = status {
            let _ = thread.join();
        }
        let links = std::mem::take(&mut *lock(&self.workers));
        for link in &links {
            link.request_stop();
        }
        self.monitor_cv.notify_all();
        if let Some(thread) = monitor {
            let _ = thread.join();
        }
        for link in &links {
            link.join();
        }
        let pending = lock(&self.submit).outstanding.clone();
        for event in pending {
            let _ = event.wait(); // Failed-over jobs still run locally.
        }
        lock(&self.queue).stopping = true;
        self.queue_cv.notify_all();
        let dispatcher = lock(&self.threads).dispatcher.take();
        if let Some(thread) = dispatcher {
            let _ = thread.join();
        }
    }
}

/// Identity of the physical devices present now; it changes when a GPU appears or disappears.
fn physical_device_fingerprint() -> String {
    let Ok(loader) = crate::vulkan::Loader::load() else {
        return String::new();
    };
    let Ok(instance) = loader.create_instance(&crate::vulkan::InstanceOptions::default()) else {
        return String::new();
    };
    let Ok(devices) = instance.physical_devices() else {
        return String::new();
    };
    let mut ids: Vec<String> = devices
        .iter()
        .map(|device| {
            let info = crate::vulkan::device_info(device);
            format!("{}/{}", info.uuid, info.name)
        })
        .collect();
    ids.sort();
    ids.join(";")
}
