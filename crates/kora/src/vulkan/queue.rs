//! Serialized queue submission with a separate GPU-completion thread.
//!
//! One submission thread exclusively owns host access to a `VkQueue`: it records commands, assigns
//! private timeline values, and submits. A completion thread waits on the timeline (or on pooled
//! fences when timeline semaphores are unavailable) and runs completion callbacks outside every
//! lock. Compatible dependency-free submissions can be coalesced into one `vkQueueSubmit` while
//! keeping per-submission IDs and callbacks.
//!
//! [`QueueContext`] is generic over a [`QueueBackend`] so the protocol can be tested against
//! deterministic fakes; production uses [`Device`].

use std::cell::Cell;
use std::collections::VecDeque;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ash::vk;

use super::Device;
use super::fault_state::DeviceFaultState;
use crate::error::{Error, ErrorCode, Result, Status};
use crate::options::ShutdownMode;

/// One semaphore wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueWait {
    /// Semaphore (timeline or binary).
    pub semaphore: vk::Semaphore,
    /// Timeline value (0 for binary semaphores).
    pub value: u64,
    /// Destination stage.
    pub stage: vk::PipelineStageFlags,
}

/// A resolved submission handed to the backend.
#[derive(Debug, Clone, Copy)]
pub struct SubmitDescription<'a> {
    /// Waits.
    pub waits: &'a [QueueWait],
    /// Command buffers.
    pub command_buffers: &'a [vk::CommandBuffer],
    /// Signals `(semaphore, value)`; values are 0 for binary semaphores.
    pub signals: &'a [(vk::Semaphore, u64)],
    /// Timeline values are present (chain `VkTimelineSemaphoreSubmitInfo`).
    pub timeline: bool,
    /// Use `vkQueueSubmit2`.
    pub synchronization2: bool,
}

/// The Vulkan calls a [`QueueContext`] makes.
pub trait QueueBackend: Send + Sync + 'static {
    /// Timeline semaphores are enabled.
    fn timeline_supported(&self) -> bool;
    /// synchronization2 is enabled.
    fn synchronization2_supported(&self) -> bool;
    /// Creates a timeline semaphore with initial value 0.
    fn create_timeline_semaphore(&self) -> std::result::Result<vk::Semaphore, vk::Result>;
    /// Destroys a semaphore.
    fn destroy_semaphore(&self, semaphore: vk::Semaphore);
    /// Current timeline value.
    fn semaphore_value(&self, semaphore: vk::Semaphore) -> std::result::Result<u64, vk::Result>;
    /// Waits for a timeline value (`Err(TIMEOUT)` on timeout).
    fn wait_semaphore(
        &self,
        semaphore: vk::Semaphore,
        value: u64,
        timeout_ns: u64,
    ) -> std::result::Result<(), vk::Result>;
    /// Creates an unsignalled fence.
    fn create_fence(&self) -> std::result::Result<vk::Fence, vk::Result>;
    /// Destroys a fence.
    fn destroy_fence(&self, fence: vk::Fence);
    /// Waits for a fence (`Err(TIMEOUT)` on timeout).
    fn wait_fence(&self, fence: vk::Fence, timeout_ns: u64) -> std::result::Result<(), vk::Result>;
    /// Resets a fence.
    fn reset_fence(&self, fence: vk::Fence) -> std::result::Result<(), vk::Result>;
    /// Submits one batch.
    fn submit(
        &self,
        queue: vk::Queue,
        description: &SubmitDescription<'_>,
        fence: vk::Fence,
    ) -> std::result::Result<(), vk::Result>;
}

impl QueueBackend for Device {
    fn timeline_supported(&self) -> bool {
        self.enabled_features().timeline_semaphore
    }
    fn synchronization2_supported(&self) -> bool {
        self.enabled_features().synchronization2
    }
    fn create_timeline_semaphore(&self) -> std::result::Result<vk::Semaphore, vk::Result> {
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: valid create info on a live device with timeline semaphores enabled.
        unsafe { self.raw().create_semaphore(&info, None) }
    }
    fn destroy_semaphore(&self, semaphore: vk::Semaphore) {
        // SAFETY: the queue context destroys its semaphore after both worker threads stopped.
        unsafe { self.raw().destroy_semaphore(semaphore, None) }
    }
    fn semaphore_value(&self, semaphore: vk::Semaphore) -> std::result::Result<u64, vk::Result> {
        self.semaphore_counter_value(semaphore)
    }
    fn wait_semaphore(
        &self,
        semaphore: vk::Semaphore,
        value: u64,
        timeout_ns: u64,
    ) -> std::result::Result<(), vk::Result> {
        let semaphores = [semaphore];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        self.wait_semaphores(&info, timeout_ns)
    }
    fn create_fence(&self) -> std::result::Result<vk::Fence, vk::Result> {
        // SAFETY: valid create info on a live device.
        unsafe { self.raw().create_fence(&vk::FenceCreateInfo::default(), None) }
    }
    fn destroy_fence(&self, fence: vk::Fence) {
        // SAFETY: pooled fences are destroyed only after the workers stopped.
        unsafe { self.raw().destroy_fence(fence, None) }
    }
    fn wait_fence(&self, fence: vk::Fence, timeout_ns: u64) -> std::result::Result<(), vk::Result> {
        // SAFETY: `fence` is a live fence of this device.
        unsafe { self.raw().wait_for_fences(&[fence], true, timeout_ns) }
    }
    fn reset_fence(&self, fence: vk::Fence) -> std::result::Result<(), vk::Result> {
        // SAFETY: `fence` is signalled and not in use by any pending submission.
        unsafe { self.raw().reset_fences(&[fence]) }
    }
    fn submit(
        &self,
        queue: vk::Queue,
        d: &SubmitDescription<'_>,
        fence: vk::Fence,
    ) -> std::result::Result<(), vk::Result> {
        if d.synchronization2 {
            let waits: Vec<_> = d
                .waits
                .iter()
                .map(|w| {
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(w.semaphore)
                        .value(w.value)
                        .stage_mask(vk::PipelineStageFlags2::from_raw(u64::from(w.stage.as_raw())))
                })
                .collect();
            let signals: Vec<_> = d
                .signals
                .iter()
                .map(|&(s, v)| {
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(s)
                        .value(v)
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                })
                .collect();
            let commands: Vec<_> = d
                .command_buffers
                .iter()
                .map(|&c| vk::CommandBufferSubmitInfo::default().command_buffer(c))
                .collect();
            let info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&waits)
                .command_buffer_infos(&commands)
                .signal_semaphore_infos(&signals);
            return self.queue_submit2(queue, &[info], fence);
        }
        let wait_semaphores: Vec<_> = d.waits.iter().map(|w| w.semaphore).collect();
        let wait_stages: Vec<_> = d.waits.iter().map(|w| w.stage).collect();
        let wait_values: Vec<_> = d.waits.iter().map(|w| w.value).collect();
        let signal_semaphores: Vec<_> = d.signals.iter().map(|s| s.0).collect();
        let signal_values: Vec<_> = d.signals.iter().map(|s| s.1).collect();
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let mut info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(d.command_buffers)
            .signal_semaphores(&signal_semaphores);
        if d.timeline {
            info = info.push_next(&mut timeline);
        }
        self.queue_submit(queue, &[info], fence)
    }
}

/// Optional batching of compatible submissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoalescingOptions {
    /// Enable coalescing.
    pub enabled: bool,
    /// Maximum submissions per batch.
    pub maximum_batch_size: u32,
    /// Maximum time to wait for more compatible submissions (0 batches only those already queued).
    pub maximum_wait_ns: u64,
}

impl Default for CoalescingOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            maximum_batch_size: 8,
            maximum_wait_ns: 100_000,
        }
    }
}

/// How the completion thread waits for GPU work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CompletionWaitOptions {
    /// Busy-poll the timeline for up to this long before blocking (0 = always block).
    pub spin_ns: u64,
}

/// One resource touched by a submission (for coalescing conflict checks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceAccess {
    /// Buffer slot.
    pub slot: u32,
    /// Buffer generation.
    pub generation: u64,
    /// The submission writes it.
    pub write: bool,
}

/// Records command buffers on the submission thread given the assigned timeline value.
pub type RecordCommands = Box<dyn FnOnce(u64, &mut Vec<vk::CommandBuffer>) -> Status + Send>;
/// Called on the submission thread after `vkQueueSubmit` (or after a recording failure).
pub type OnSubmitted = Box<dyn FnOnce(u64, &Status) + Send>;
/// Called on the completion thread with the submission ID and final status.
pub type OnComplete = Box<dyn FnOnce(u64, Status) + Send>;
/// Diagnostic hook: completion-worker wake delay and device-wait duration in nanoseconds.
pub type OnCompletionProfile = Box<dyn FnOnce(u64, u64) + Send>;

/// One owned submission record.
#[derive(Default)]
pub struct Submission {
    /// Semaphore waits.
    pub waits: Vec<QueueWait>,
    /// Pre-recorded command buffers.
    pub command_buffers: Vec<vk::CommandBuffer>,
    /// Extra semaphore signals.
    pub signals: Vec<(vk::Semaphore, u64)>,
    /// External fence (timeline mode only).
    pub fence: vk::Fence,
    /// May join a coalesced batch.
    pub coalescible: bool,
    /// Has no dependency on earlier submissions.
    pub dependency_free: bool,
    /// Only submissions with equal non-zero keys coalesce.
    pub coalescing_key: u64,
    /// 0 = low, 1 = normal, 2+ = high (high never coalesces).
    pub priority: u8,
    /// Resources for conflict checks.
    pub resources: Vec<ResourceAccess>,
    /// Command recording.
    pub record: Option<RecordCommands>,
    /// Submission hook.
    pub on_submitted: Option<OnSubmitted>,
    /// Completion callback.
    pub on_complete: Option<OnComplete>,
    /// Profiling hook.
    pub on_completion_profile: Option<OnCompletionProfile>,
}

impl fmt::Debug for Submission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Submission")
            .field("waits", &self.waits.len())
            .field("command_buffers", &self.command_buffers.len())
            .field("coalescible", &self.coalescible)
            .finish()
    }
}

struct Queued {
    id: u64,
    submission: Submission,
}

struct CompletionCallback {
    id: u64,
    on_complete: Option<OnComplete>,
    on_profile: Option<OnCompletionProfile>,
}

struct Completion {
    timeline_value: u64,
    status: Status,
    callbacks: Vec<CompletionCallback>,
    fence: vk::Fence,
    pooled_fence: bool,
    enqueued_at: Instant,
}

struct ReadyState {
    queue: VecDeque<Queued>,
    next_id: u64,
    accepting: bool,
    stop_requested: bool,
    cancel_pending: bool,
}

struct CompletionState {
    queue: VecDeque<Completion>,
    submission_finished: bool,
}

struct FenceState {
    available: VecDeque<vk::Fence>,
    failed: bool,
}

struct Shared<B: QueueBackend> {
    backend: B,
    queue: vk::Queue,
    timeline: vk::Semaphore,
    use_timeline: bool,
    use_synchronization2: bool,
    coalescing: CoalescingOptions,
    wait: CompletionWaitOptions,
    fault_state: Option<Arc<DeviceFaultState>>,
    ready: Mutex<ReadyState>,
    ready_changed: Condvar,
    completions: Mutex<CompletionState>,
    completion_changed: Condvar,
    fence_pool: Vec<vk::Fence>,
    fences: Mutex<FenceState>,
    fence_changed: Condvar,
    fallback_completed: AtomicU64,
    next_timeline_value: AtomicU64,
    submit_active: AtomicBool,
}

const COMPLETION_POLL_NS: u64 = 250_000_000;
const FENCE_POOL_SIZE: usize = 8;

thread_local! {
    static ACTIVE_CONTEXT: Cell<usize> = const { Cell::new(0) };
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|p| p.into_inner())
}

fn vk_status(operation: &str, result: vk::Result) -> Error {
    let lost = result == vk::Result::ERROR_DEVICE_LOST;
    let error = Error::new(
        if lost {
            ErrorCode::DeviceLost
        } else {
            ErrorCode::TransferFailed
        },
        operation,
    )
    .with_vk(result.as_raw());
    if lost {
        error.with_recovery("call Runtime::recover() after observing device loss")
    } else {
        error
    }
}

fn device_lost(fault: &DeviceFaultState, message: &str) -> Error {
    Error::new(ErrorCode::DeviceLost, message).with_vk(fault.last_result().as_raw())
}

fn run_guarded<R>(f: impl FnOnce() -> R) -> Option<R> {
    catch_unwind(AssertUnwindSafe(f)).ok()
}

impl<B: QueueBackend> Shared<B> {
    fn identity(self: &Arc<Self>) -> usize {
        Arc::as_ptr(self) as *const u8 as usize
    }

    fn enqueue_completion(&self, completion: Completion) {
        lock(&self.completions).queue.push_back(completion);
        self.completion_changed.notify_one();
    }

    fn fail(&self, mut items: Vec<Queued>, status: Status) {
        for queued in items.drain(..) {
            self.enqueue_completion(Completion {
                timeline_value: 0,
                status: status.clone(),
                callbacks: vec![CompletionCallback {
                    id: queued.id,
                    on_complete: queued.submission.on_complete,
                    on_profile: queued.submission.on_completion_profile,
                }],
                fence: vk::Fence::null(),
                pooled_fence: false,
                enqueued_at: Instant::now(),
            });
        }
    }

    fn fail_queued_device_lost(&self) {
        let Some(fault) = self.fault_state.as_ref().filter(|f| !f.healthy()) else {
            return;
        };
        let pending: Vec<Queued> = lock(&self.ready).queue.drain(..).collect();
        self.fail(
            pending,
            Err(device_lost(
                fault,
                "logical device is lost; queued work was not submitted",
            )),
        );
    }

    fn mark_lost(&self, result: vk::Result) {
        if let Some(fault) = &self.fault_state {
            fault.mark_lost(result, true);
            self.fail_queued_device_lost();
        }
    }

    fn acquire_fence(&self) -> Result<vk::Fence> {
        let mut fences = lock(&self.fences);
        while !fences.failed && fences.available.is_empty() {
            fences = self.fence_changed.wait(fences).unwrap_or_else(|p| p.into_inner());
        }
        if fences.failed {
            return Err(Error::new(
                ErrorCode::DeviceLost,
                "QueueContext fence pool is unavailable",
            ));
        }
        Ok(fences.available.pop_front().expect("checked non-empty"))
    }

    fn release_fence(&self, fence: vk::Fence, reset: bool) {
        let mut fences = lock(&self.fences);
        if reset && self.backend.reset_fence(fence).is_err() {
            fences.failed = true;
        } else {
            fences.available.push_back(fence);
        }
        drop(fences);
        self.fence_changed.notify_all();
    }

    /// Pops the next item (and compatible followers) or returns `None` to stop.
    fn next_batch(&self) -> Option<(Vec<Queued>, bool)> {
        let mut ready = lock(&self.ready);
        while !ready.stop_requested && ready.queue.is_empty() {
            ready = self.ready_changed.wait(ready).unwrap_or_else(|p| p.into_inner());
        }
        let first = ready.queue.pop_front()?;
        if ready.stop_requested && ready.cancel_pending {
            return Some((vec![first], true));
        }
        let mut batch = vec![first];
        let c = self.coalescing;
        let head = &batch[0].submission;
        let head_ok = |s: &Submission| {
            c.enabled
                && self.use_timeline
                && c.maximum_batch_size > 1
                && s.coalescible
                && s.dependency_free
                && s.priority <= 1
                && s.coalescing_key != 0
                && s.signals.is_empty()
                && s.fence == vk::Fence::null()
        };
        if !head_ok(head) {
            return Some((batch, false));
        }
        let key = head.coalescing_key;
        let waits = head.waits.clone();
        let mut resources = head.resources.clone();
        let deadline = Instant::now() + Duration::from_nanos(c.maximum_wait_ns);
        while batch.len() < c.maximum_batch_size as usize {
            if ready.queue.is_empty() {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                let (guard, timeout) = self
                    .ready_changed
                    .wait_timeout_while(ready, deadline - now, |r| !r.stop_requested && r.queue.is_empty())
                    .unwrap_or_else(|p| p.into_inner());
                ready = guard;
                if timeout.timed_out() && ready.queue.is_empty() {
                    break;
                }
            }
            let Some(next) = ready.queue.front() else { break };
            if ready.stop_requested && ready.cancel_pending {
                break;
            }
            let s = &next.submission;
            let conflicts = s.resources.iter().any(|a| {
                resources
                    .iter()
                    .any(|e| e.slot == a.slot && e.generation == a.generation && (e.write || a.write))
            });
            if !head_ok(s) || s.coalescing_key != key || s.waits != waits || conflicts {
                break;
            }
            let next = ready.queue.pop_front().expect("peeked");
            resources.extend_from_slice(&next.submission.resources);
            batch.push(next);
        }
        Some((batch, false))
    }

    fn submission_loop(self: Arc<Self>) {
        ACTIVE_CONTEXT.with(|c| c.set(self.identity()));
        while let Some((batch, cancel)) = self.next_batch() {
            if cancel {
                let status = match self.fault_state.as_ref().filter(|f| !f.healthy()) {
                    Some(fault) => Err(device_lost(
                        fault,
                        "logical device is lost; queued work was not submitted",
                    )),
                    None => Err(Error::new(
                        ErrorCode::Cancelled,
                        "Submission cancelled before queue submit",
                    )),
                };
                self.fail(batch, status);
                continue;
            }
            if let Some(fault) = self.fault_state.as_ref().filter(|f| !f.healthy()) {
                self.fail(
                    batch,
                    Err(device_lost(
                        fault,
                        "logical device is lost; queued work was not submitted",
                    )),
                );
                self.fail_queued_device_lost();
                continue;
            }
            self.submit_batch(batch);
        }
        lock(&self.completions).submission_finished = true;
        self.completion_changed.notify_one();
        ACTIVE_CONTEXT.with(|c| c.set(0));
    }

    fn submit_batch(&self, batch: Vec<Queued>) {
        let count = batch.len() as u64;
        let next = self.next_timeline_value.load(Ordering::Relaxed);
        let Some(timeline_value) = next.checked_add(count - 1).filter(|_| next != 0) else {
            self.fail(
                batch,
                Err(Error::new(
                    ErrorCode::Overflow,
                    "QueueContext timeline value space is exhausted",
                )),
            );
            return;
        };
        self.next_timeline_value
            .store(timeline_value.checked_add(1).unwrap_or(0), Ordering::Relaxed);

        let mut parts: Vec<(u64, Submission)> = batch.into_iter().map(|q| (q.id, q.submission)).collect();
        let mut command_buffers = Vec::new();
        let mut record_status: Status = Ok(());
        for (_, part) in parts.iter_mut() {
            if let Some(record) = part.record.take() {
                let mut commands = std::mem::take(&mut part.command_buffers);
                record_status = run_guarded(|| record(timeline_value, &mut commands)).unwrap_or_else(|| {
                    Err(Error::new(
                        ErrorCode::InternalInvariantViolation,
                        "QueueContext command-recording callback panicked",
                    ))
                });
                part.command_buffers = commands;
                if record_status.is_err() {
                    break;
                }
            }
            command_buffers.extend_from_slice(&part.command_buffers);
        }
        let complete_all =
            |parts: Vec<(u64, Submission)>, status: Status, value: u64, fence, pooled| Completion {
                timeline_value: value,
                status,
                callbacks: parts
                    .into_iter()
                    .map(|(id, s)| CompletionCallback {
                        id,
                        on_complete: s.on_complete,
                        on_profile: s.on_completion_profile,
                    })
                    .collect(),
                fence,
                pooled_fence: pooled,
                enqueued_at: Instant::now(),
            };
        let notify_submitted = |parts: &mut Vec<(u64, Submission)>, status: &Status| {
            for (_, part) in parts.iter_mut() {
                if let Some(on_submitted) = part.on_submitted.take() {
                    let _ = run_guarded(|| on_submitted(timeline_value, status));
                }
            }
        };
        if record_status.is_err() {
            notify_submitted(&mut parts, &record_status);
            self.enqueue_completion(complete_all(parts, record_status, 0, vk::Fence::null(), false));
            return;
        }
        // Merge waits (identical for coalesced batches) and extra signals.
        let mut waits: Vec<QueueWait> = Vec::new();
        for (_, part) in &parts {
            for wait in &part.waits {
                match waits.iter_mut().find(|w| w.semaphore == wait.semaphore) {
                    Some(existing) => {
                        existing.stage |= wait.stage;
                        existing.value = existing.value.max(wait.value);
                    }
                    None => waits.push(*wait),
                }
            }
        }
        let mut signals: Vec<(vk::Semaphore, u64)> = parts
            .iter()
            .flat_map(|(_, p)| p.signals.iter().copied())
            .collect();
        if !self.use_timeline && (waits.iter().any(|w| w.value != 0) || signals.iter().any(|s| s.1 != 0)) {
            let error = Err(Error::new(
                ErrorCode::UnsupportedFeature,
                "Fence fallback accepts binary semaphore waits and signals only",
            ));
            self.enqueue_completion(complete_all(parts, error, 0, vk::Fence::null(), false));
            return;
        }
        if self.use_timeline {
            signals.push((self.timeline, timeline_value));
        }
        let external_fence = parts[0].1.fence;
        let (fence, pooled) = if self.use_timeline || external_fence != vk::Fence::null() {
            (external_fence, false)
        } else {
            match self.acquire_fence() {
                Ok(fence) => (fence, true),
                Err(error) => {
                    self.enqueue_completion(complete_all(parts, Err(error), 0, vk::Fence::null(), false));
                    return;
                }
            }
        };
        let description = SubmitDescription {
            waits: &waits,
            command_buffers: &command_buffers,
            signals: &signals,
            timeline: self.use_timeline,
            synchronization2: self.use_synchronization2,
        };
        let was_active = self.submit_active.swap(true, Ordering::Acquire);
        debug_assert!(!was_active, "concurrent host submission to one Vulkan queue");
        let result = match &self.fault_state {
            Some(fault) => {
                let _guard = fault.submission_lock();
                if fault.healthy() {
                    let result = self.backend.submit(self.queue, &description, fence);
                    if result == Err(vk::Result::ERROR_DEVICE_LOST) {
                        fault.mark_lost(vk::Result::ERROR_DEVICE_LOST, true);
                    }
                    result
                } else {
                    Err(vk::Result::ERROR_DEVICE_LOST)
                }
            }
            None => self.backend.submit(self.queue, &description, fence),
        };
        self.submit_active.store(false, Ordering::Release);
        if result == Err(vk::Result::ERROR_DEVICE_LOST) {
            self.fail_queued_device_lost();
        }
        let operation = if self.use_synchronization2 {
            "vkQueueSubmit2 failed"
        } else {
            "vkQueueSubmit failed"
        };
        let status = result.map_err(|r| vk_status(operation, r));
        notify_submitted(&mut parts, &status);
        match status {
            Ok(()) => self.enqueue_completion(complete_all(parts, Ok(()), timeline_value, fence, pooled)),
            Err(error) => {
                if pooled {
                    self.release_fence(fence, false);
                }
                self.enqueue_completion(complete_all(parts, Err(error), 0, vk::Fence::null(), false));
            }
        }
    }

    fn wait_timeline(&self, value: u64) -> Status {
        if self.wait.spin_ns != 0 {
            let deadline = Instant::now() + Duration::from_nanos(self.wait.spin_ns);
            while Instant::now() < deadline {
                match self.backend.semaphore_value(self.timeline) {
                    Ok(completed) if completed >= value => return Ok(()),
                    Ok(_) => std::hint::spin_loop(),
                    Err(_) => break,
                }
            }
        }
        loop {
            match self
                .backend
                .wait_semaphore(self.timeline, value, COMPLETION_POLL_NS)
            {
                Ok(()) => return Ok(()),
                Err(vk::Result::TIMEOUT) => {
                    // A timeout is only a poll boundary: never deliver completion (or release
                    // resources) until the timeline confirms the work finished.
                    match self.backend.semaphore_value(self.timeline) {
                        Ok(completed) if completed >= value => return Ok(()),
                        Ok(_) => continue,
                        Err(result) => {
                            if result == vk::Result::ERROR_DEVICE_LOST {
                                self.mark_lost(result);
                            }
                            return Err(vk_status("Polling queue completion timeline failed", result));
                        }
                    }
                }
                Err(result) => {
                    if result == vk::Result::ERROR_DEVICE_LOST {
                        self.mark_lost(result);
                    }
                    return Err(vk_status("Waiting for queue completion failed", result));
                }
            }
        }
    }

    fn wait_completion_fence(&self, completion: &Completion) -> Status {
        loop {
            match self.backend.wait_fence(completion.fence, COMPLETION_POLL_NS) {
                Ok(()) => {
                    self.fallback_completed
                        .store(completion.timeline_value, Ordering::Release);
                    if completion.pooled_fence {
                        self.release_fence(completion.fence, true);
                    } else {
                        let _guard = lock(&self.fences);
                        self.fence_changed.notify_all();
                    }
                    return Ok(());
                }
                Err(vk::Result::TIMEOUT) => continue,
                Err(result) => {
                    if result == vk::Result::ERROR_DEVICE_LOST {
                        self.mark_lost(result);
                    }
                    if completion.pooled_fence {
                        lock(&self.fences).failed = true;
                        self.fence_changed.notify_all();
                    }
                    return Err(vk_status("Waiting for queue completion fence failed", result));
                }
            }
        }
    }

    fn completion_loop(self: Arc<Self>) {
        ACTIVE_CONTEXT.with(|c| c.set(self.identity()));
        loop {
            let mut completion = {
                let mut state = lock(&self.completions);
                while !state.submission_finished && state.queue.is_empty() {
                    state = self
                        .completion_changed
                        .wait(state)
                        .unwrap_or_else(|p| p.into_inner());
                }
                match state.queue.pop_front() {
                    Some(completion) => completion,
                    None => break,
                }
            };
            let started = Instant::now();
            if completion.timeline_value != 0 && completion.status.is_ok() {
                completion.status = if self.use_timeline {
                    self.wait_timeline(completion.timeline_value)
                } else {
                    self.wait_completion_fence(&completion)
                };
            }
            let finished = Instant::now();
            if completion.status.is_ok()
                && let Some(fault) = self.fault_state.as_ref().filter(|f| !f.healthy())
            {
                completion.status = Err(vk_status(
                    "Logical device was lost while work was in flight",
                    fault.last_result(),
                ));
            }
            let delay = started
                .saturating_duration_since(completion.enqueued_at)
                .as_nanos() as u64;
            let wait = finished.saturating_duration_since(started).as_nanos() as u64;
            for callback in completion.callbacks {
                if let Some(profile) = callback.on_profile {
                    let _ = run_guarded(|| profile(delay.max(1), wait));
                }
                if let Some(on_complete) = callback.on_complete {
                    let status = completion.status.clone();
                    let _ = run_guarded(|| on_complete(callback.id, status));
                }
            }
        }
        ACTIVE_CONTEXT.with(|c| c.set(0));
    }
}

/// Serializes submissions for one `VkQueue`.
pub struct QueueContext<B: QueueBackend = Device> {
    shared: Arc<Shared<B>>,
    threads: Mutex<Option<(JoinHandle<()>, JoinHandle<()>)>>,
}

impl<B: QueueBackend> fmt::Debug for QueueContext<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueueContext")
            .field("timeline", &self.shared.use_timeline)
            .field("synchronization2", &self.shared.use_synchronization2)
            .finish()
    }
}

/// Context creation options.
#[derive(Debug, Clone, Default)]
pub struct QueueContextOptions {
    /// Use pooled fences even when timeline semaphores are available (test hook).
    pub force_fence_fallback: bool,
    /// Coalescing.
    pub coalescing: CoalescingOptions,
    /// Completion wait strategy.
    pub completion_wait: CompletionWaitOptions,
    /// Device-loss state shared with sibling queues.
    pub fault_state: Option<Arc<DeviceFaultState>>,
}

impl<B: QueueBackend> QueueContext<B> {
    /// Creates the timeline (or fence pool) and starts both worker threads.
    pub fn new(backend: B, queue: vk::Queue, options: QueueContextOptions) -> Result<QueueContext<B>> {
        if queue == vk::Queue::null() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "QueueContext requires a queue",
            ));
        }
        if options.coalescing.maximum_batch_size > u32::MAX / 2 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "QueueContext coalescing limits exceed supported bounds",
            ));
        }
        let mut use_timeline = backend.timeline_supported() && !options.force_fence_fallback;
        let mut timeline = vk::Semaphore::null();
        if use_timeline {
            match backend.create_timeline_semaphore() {
                Ok(semaphore) => timeline = semaphore,
                Err(_) => use_timeline = false,
            }
        }
        let mut fence_pool = Vec::new();
        if !use_timeline {
            for _ in 0..FENCE_POOL_SIZE {
                match backend.create_fence() {
                    Ok(fence) => fence_pool.push(fence),
                    Err(result) => {
                        for fence in fence_pool {
                            backend.destroy_fence(fence);
                        }
                        return Err(Error::new(
                            ErrorCode::UnsupportedFeature,
                            "Creating QueueContext fence pool failed",
                        )
                        .with_vk(result.as_raw()));
                    }
                }
            }
        }
        let use_synchronization2 = !options.force_fence_fallback && backend.synchronization2_supported();
        let shared = Arc::new(Shared {
            backend,
            queue,
            timeline,
            use_timeline,
            use_synchronization2,
            coalescing: options.coalescing,
            wait: options.completion_wait,
            fault_state: options.fault_state,
            ready: Mutex::new(ReadyState {
                queue: VecDeque::new(),
                next_id: 1,
                accepting: true,
                stop_requested: false,
                cancel_pending: false,
            }),
            ready_changed: Condvar::new(),
            completions: Mutex::new(CompletionState {
                queue: VecDeque::new(),
                submission_finished: false,
            }),
            completion_changed: Condvar::new(),
            fences: Mutex::new(FenceState {
                available: fence_pool.iter().copied().collect(),
                failed: false,
            }),
            fence_pool,
            fence_changed: Condvar::new(),
            fallback_completed: AtomicU64::new(0),
            next_timeline_value: AtomicU64::new(1),
            submit_active: AtomicBool::new(false),
        });
        let completion = {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("kora-queue-completion".into())
                .spawn(move || shared.completion_loop())
        };
        let submission = {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("kora-queue-submit".into())
                .spawn(move || shared.submission_loop())
        };
        let context = QueueContext {
            shared,
            threads: Mutex::new(None),
        };
        match (completion, submission) {
            (Ok(c), Ok(s)) => *lock(&context.threads) = Some((s, c)),
            (completion, submission) => {
                lock(&context.shared.ready).stop_requested = true;
                context.shared.ready_changed.notify_all();
                lock(&context.shared.completions).submission_finished = true;
                context.shared.completion_changed.notify_all();
                if let Ok(handle) = submission {
                    let _ = handle.join();
                }
                if let Ok(handle) = completion {
                    let _ = handle.join();
                }
                return Err(Error::new(
                    ErrorCode::InternalInvariantViolation,
                    "Failed to start QueueContext worker threads",
                ));
            }
        }
        Ok(context)
    }

    /// Enqueues a submission; the returned ID is unique for this context's lifetime.
    pub fn submit(&self, submission: Submission) -> Result<u64> {
        let mut ready = lock(&self.shared.ready);
        if !ready.accepting {
            return Err(Error::new(
                ErrorCode::Cancelled,
                "QueueContext is shutting down and rejects new work",
            ));
        }
        if let Some(fault) = self.shared.fault_state.as_ref().filter(|f| !f.healthy()) {
            return Err(device_lost(
                fault,
                "logical device is lost; queue submission was rejected",
            ));
        }
        let id = ready.next_id;
        ready.next_id = id.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::Overflow,
                "QueueContext submission ID space is exhausted",
            )
        })?;
        ready.queue.push_back(Queued { id, submission });
        drop(ready);
        self.shared.ready_changed.notify_one();
        Ok(id)
    }

    /// Completed value of this context's private timeline.
    pub fn completed_value(&self) -> Result<u64> {
        if !self.shared.use_timeline {
            return Ok(self.shared.fallback_completed.load(Ordering::Acquire));
        }
        self.shared
            .backend
            .semaphore_value(self.shared.timeline)
            .map_err(|r| vk_status("Reading queue completion timeline failed", r))
    }

    /// Timeline semaphore for cross-queue dependencies (null in fence mode).
    pub fn completion_semaphore(&self) -> vk::Semaphore {
        if self.shared.use_timeline {
            self.shared.timeline
        } else {
            vk::Semaphore::null()
        }
    }

    /// Whether submissions use `vkQueueSubmit2`.
    pub fn uses_synchronization2(&self) -> bool {
        self.shared.use_synchronization2
    }

    /// Whether completion uses a timeline semaphore.
    pub fn uses_timeline(&self) -> bool {
        self.shared.use_timeline
    }

    /// Waits until the timeline reaches `value` (`Timeout` after `timeout_ns`).
    pub fn wait_for_value(&self, value: u64, timeout_ns: u64) -> Status {
        if value == 0 {
            return Ok(());
        }
        let shared = &self.shared;
        if !shared.use_timeline {
            let fences = lock(&shared.fences);
            let reached = |_: &mut FenceState| shared.fallback_completed.load(Ordering::Acquire) < value;
            if timeout_ns == u64::MAX {
                drop(
                    shared
                        .fence_changed
                        .wait_while(fences, reached)
                        .unwrap_or_else(|p| p.into_inner()),
                );
                return Ok(());
            }
            let (guard, result) = shared
                .fence_changed
                .wait_timeout_while(fences, Duration::from_nanos(timeout_ns), reached)
                .unwrap_or_else(|p| p.into_inner());
            drop(guard);
            if result.timed_out() && shared.fallback_completed.load(Ordering::Acquire) < value {
                return Err(Error::new(
                    ErrorCode::Timeout,
                    "Queue completion fence wait timed out",
                ));
            }
            return Ok(());
        }
        match shared.backend.wait_semaphore(shared.timeline, value, timeout_ns) {
            Ok(()) => Ok(()),
            Err(vk::Result::TIMEOUT) => {
                Err(Error::new(ErrorCode::Timeout, "Queue completion wait timed out"))
            }
            Err(result) => {
                if result == vk::Result::ERROR_DEVICE_LOST {
                    shared.mark_lost(result);
                }
                Err(vk_status("Waiting for queue completion timeline failed", result))
            }
        }
    }

    /// Stops admission, then drains (`Graceful`) or cancels (`CancelPending`) work that has not
    /// reached `vkQueueSubmit`, waits for in-flight completions, and joins both workers. Rejected
    /// from this context's own callbacks; repeated and concurrent calls are safe.
    pub fn shutdown(&self, mode: ShutdownMode) -> Status {
        let identity = self.shared.identity();
        if ACTIVE_CONTEXT.with(|c| c.get()) == identity {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "QueueContext cannot be shut down from one of its callbacks",
            ));
        }
        let mut threads = lock(&self.threads);
        {
            let mut ready = lock(&self.shared.ready);
            ready.accepting = false;
            ready.stop_requested = true;
            if mode == ShutdownMode::CancelPending {
                ready.cancel_pending = true;
            }
        }
        self.shared.ready_changed.notify_all();
        if let Some((submission, completion)) = threads.take() {
            let _ = submission.join();
            self.shared.completion_changed.notify_all();
            let _ = completion.join();
        }
        Ok(())
    }
}

impl<B: QueueBackend> Drop for QueueContext<B> {
    fn drop(&mut self) {
        if self.shutdown(ShutdownMode::Graceful).is_err() {
            // Dropped from one of this context's own callbacks: request shutdown without joining.
            // The workers exit on their own and the last of them releases the Vulkan objects.
            let mut ready = lock(&self.shared.ready);
            ready.accepting = false;
            ready.stop_requested = true;
            drop(ready);
            self.shared.ready_changed.notify_all();
        }
    }
}

impl<B: QueueBackend> Drop for Shared<B> {
    fn drop(&mut self) {
        // Runs after both worker threads have released their references, so no submission or
        // wait can still use the timeline semaphore or pooled fences.
        if self.use_timeline {
            self.backend.destroy_semaphore(self.timeline);
        }
        for &fence in &self.fence_pool {
            self.backend.destroy_fence(fence);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use ash::vk::Handle;
    use std::sync::atomic::{AtomicU32, AtomicU64};

    #[derive(Default)]
    pub(crate) struct Mock {
        pub submit_calls: AtomicU64,
        pub wait_calls: AtomicU64,
        pub timeout_next_wait: AtomicBool,
        pub device_lost_next_wait: AtomicBool,
        pub destroyed_semaphores: AtomicU64,
        pub completed: AtomicU64,
        pub active: AtomicU32,
        pub maximum_active: AtomicU32,
        pub malformed: AtomicU32,
        pub submitted_command_buffers: AtomicU32,
        pub non_monotonic: AtomicU32,
        pub non_null_fences: AtomicU32,
        pub external_waits: AtomicU32,
        pub external_wait_value: AtomicU64,
        pub created_fences: AtomicU32,
        pub destroyed_fences: AtomicU32,
        pub waited_fences: AtomicU32,
        pub reset_fences: AtomicU32,
        pub next_fence: AtomicU64,
        pub gate: Mutex<(bool, bool)>,
        pub gate_changed: Condvar,
        pub block_first_submit: bool,
        pub fail_first_submit: bool,
        pub timeline: bool,
        pub synchronization2: bool,
    }

    impl Mock {
        pub(crate) fn new() -> Self {
            Self {
                timeline: true,
                next_fence: AtomicU64::new(100),
                ..Default::default()
            }
        }
        fn wait_until_first_submit(&self) {
            let mut gate = self.gate.lock().unwrap();
            while !gate.0 {
                gate = self.gate_changed.wait(gate).unwrap();
            }
        }
        fn release_first_submit(&self) {
            self.gate.lock().unwrap().1 = true;
            self.gate_changed.notify_all();
        }
    }

    pub(crate) const TIMELINE: u64 = 77;

    impl QueueBackend for Arc<Mock> {
        fn timeline_supported(&self) -> bool {
            self.timeline
        }
        fn synchronization2_supported(&self) -> bool {
            self.synchronization2
        }
        fn create_timeline_semaphore(&self) -> std::result::Result<vk::Semaphore, vk::Result> {
            Ok(vk::Semaphore::from_raw(TIMELINE))
        }
        fn destroy_semaphore(&self, _: vk::Semaphore) {
            self.destroyed_semaphores.fetch_add(1, Ordering::SeqCst);
        }
        fn semaphore_value(&self, _: vk::Semaphore) -> std::result::Result<u64, vk::Result> {
            Ok(self.completed.load(Ordering::SeqCst))
        }
        fn wait_semaphore(&self, _: vk::Semaphore, _: u64, _: u64) -> std::result::Result<(), vk::Result> {
            self.wait_calls.fetch_add(1, Ordering::SeqCst);
            if self.timeout_next_wait.swap(false, Ordering::SeqCst) {
                return Err(vk::Result::TIMEOUT);
            }
            if self.device_lost_next_wait.swap(false, Ordering::SeqCst) {
                return Err(vk::Result::ERROR_DEVICE_LOST);
            }
            Ok(())
        }
        fn create_fence(&self) -> std::result::Result<vk::Fence, vk::Result> {
            self.created_fences.fetch_add(1, Ordering::SeqCst);
            Ok(vk::Fence::from_raw(
                self.next_fence.fetch_add(1, Ordering::SeqCst),
            ))
        }
        fn destroy_fence(&self, _: vk::Fence) {
            self.destroyed_fences.fetch_add(1, Ordering::SeqCst);
        }
        fn wait_fence(&self, _: vk::Fence, _: u64) -> std::result::Result<(), vk::Result> {
            self.waited_fences.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn reset_fence(&self, _: vk::Fence) -> std::result::Result<(), vk::Result> {
            self.reset_fences.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn submit(
            &self,
            _: vk::Queue,
            d: &SubmitDescription<'_>,
            fence: vk::Fence,
        ) -> std::result::Result<(), vk::Result> {
            for wait in d.waits {
                if wait.semaphore == vk::Semaphore::from_raw(88)
                    && wait.stage == vk::PipelineStageFlags::COMPUTE_SHADER
                {
                    self.external_waits.fetch_add(1, Ordering::SeqCst);
                    self.external_wait_value.store(wait.value, Ordering::SeqCst);
                } else if wait.semaphore != vk::Semaphore::from_raw(TIMELINE) {
                    self.malformed.fetch_add(1, Ordering::SeqCst);
                }
            }
            if fence != vk::Fence::null() {
                self.non_null_fences.fetch_add(1, Ordering::SeqCst);
            }
            self.submitted_command_buffers
                .fetch_add(d.command_buffers.len() as u32, Ordering::SeqCst);
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.maximum_active.fetch_max(active, Ordering::SeqCst);
            if d.timeline {
                match d.signals.last() {
                    Some(&(semaphore, value))
                        if semaphore == vk::Semaphore::from_raw(TIMELINE) && value != 0 =>
                    {
                        if value <= self.completed.load(Ordering::SeqCst) {
                            self.non_monotonic.fetch_add(1, Ordering::SeqCst);
                        }
                        self.completed.store(value, Ordering::SeqCst);
                    }
                    _ => {
                        self.malformed.fetch_add(1, Ordering::SeqCst);
                    }
                }
            } else if fence == vk::Fence::null() || !d.signals.is_empty() {
                self.malformed.fetch_add(1, Ordering::SeqCst);
            }
            let call = self.submit_calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.block_first_submit && call == 1 {
                let mut gate = self.gate.lock().unwrap();
                gate.0 = true;
                self.gate_changed.notify_all();
                while !gate.1 {
                    gate = self.gate_changed.wait(gate).unwrap();
                }
            }
            std::thread::sleep(Duration::from_micros(20));
            self.active.fetch_sub(1, Ordering::SeqCst);
            if self.fail_first_submit && call == 1 {
                Err(vk::Result::ERROR_DEVICE_LOST)
            } else {
                Ok(())
            }
        }
    }

    fn context(mock: &Arc<Mock>, options: QueueContextOptions) -> QueueContext<Arc<Mock>> {
        QueueContext::new(Arc::clone(mock), vk::Queue::from_raw(22), options).unwrap()
    }

    fn counting(counter: &Arc<AtomicU64>, errors: &Arc<AtomicU64>) -> OnComplete {
        let (counter, errors) = (Arc::clone(counter), Arc::clone(errors));
        Box::new(move |_, status| {
            if status.is_err() {
                errors.fetch_add(1, Ordering::SeqCst);
            }
            counter.fetch_add(1, Ordering::SeqCst);
        })
    }

    #[test]
    fn serializes_sixteen_concurrent_producers_and_completes_callbacks() {
        let mock = Arc::new(Mock::new());
        let ctx = context(&mock, QueueContextOptions::default());
        let (callbacks, errors, accepted) = (
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            AtomicU64::new(0),
        );
        std::thread::scope(|scope| {
            for _ in 0..16 {
                scope.spawn(|| {
                    for _ in 0..128 {
                        let submission = Submission {
                            on_complete: Some(counting(&callbacks, &errors)),
                            ..Default::default()
                        };
                        if ctx.submit(submission).is_ok() {
                            accepted.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                });
            }
        });
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        let accepted = accepted.load(Ordering::SeqCst);
        assert_eq!(accepted, 16 * 128);
        assert_eq!(mock.submit_calls.load(Ordering::SeqCst), accepted);
        assert_eq!(callbacks.load(Ordering::SeqCst), accepted);
        assert_eq!(errors.load(Ordering::SeqCst), 0);
        assert_eq!(mock.non_monotonic.load(Ordering::SeqCst), 0);
        assert_eq!(mock.non_null_fences.load(Ordering::SeqCst), 0);
        assert_eq!(mock.wait_calls.load(Ordering::SeqCst), accepted);
        assert_eq!(mock.maximum_active.load(Ordering::SeqCst), 1);
        assert_eq!(mock.malformed.load(Ordering::SeqCst), 0);
        assert_eq!(
            ctx.submit(Submission::default()).unwrap_err().code(),
            ErrorCode::Cancelled
        );
        drop(ctx);
        assert_eq!(mock.destroyed_semaphores.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cancel_pending_cancels_queued_work_but_drains_inflight_completion() {
        let mock = Arc::new(Mock {
            block_first_submit: true,
            ..Mock::new()
        });
        let ctx = context(&mock, QueueContextOptions::default());
        let succeeded = Arc::new(AtomicU64::new(0));
        let cancelled = Arc::new(AtomicU64::new(0));
        for _ in 0..32 {
            let (s, c) = (Arc::clone(&succeeded), Arc::clone(&cancelled));
            ctx.submit(Submission {
                on_complete: Some(Box::new(move |_, status| match status {
                    Ok(()) => drop(s.fetch_add(1, Ordering::SeqCst)),
                    Err(e) if e.code() == ErrorCode::Cancelled => drop(c.fetch_add(1, Ordering::SeqCst)),
                    Err(_) => {}
                })),
                ..Default::default()
            })
            .unwrap();
        }
        mock.wait_until_first_submit();
        std::thread::scope(|scope| {
            scope.spawn(|| ctx.shutdown(ShutdownMode::CancelPending).unwrap());
            std::thread::sleep(Duration::from_millis(5));
            mock.release_first_submit();
        });
        assert_eq!(mock.submit_calls.load(Ordering::SeqCst), 1);
        assert_eq!(succeeded.load(Ordering::SeqCst), 1);
        assert_eq!(cancelled.load(Ordering::SeqCst), 31);
        assert_eq!(mock.wait_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn forwards_external_timeline_wait_for_cross_queue_dependency() {
        let mock = Arc::new(Mock::new());
        let ctx = context(&mock, QueueContextOptions::default());
        ctx.submit(Submission {
            waits: vec![QueueWait {
                semaphore: vk::Semaphore::from_raw(88),
                value: 42,
                stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            }],
            ..Default::default()
        })
        .unwrap();
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        assert_eq!(mock.external_waits.load(Ordering::SeqCst), 1);
        assert_eq!(mock.external_wait_value.load(Ordering::SeqCst), 42);
        assert_eq!(mock.malformed.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn fence_fallback_reuses_bounded_pool_and_emulates_completion_values() {
        let mock = Arc::new(Mock::new());
        let ctx = context(
            &mock,
            QueueContextOptions {
                force_fence_fallback: true,
                ..Default::default()
            },
        );
        assert!(!ctx.uses_timeline());
        let (callbacks, errors) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
        let mut last = 0;
        for _ in 0..32 {
            last = ctx
                .submit(Submission {
                    on_complete: Some(counting(&callbacks, &errors)),
                    ..Default::default()
                })
                .unwrap();
        }
        ctx.wait_for_value(last, 5_000_000_000).unwrap();
        assert_eq!(ctx.completed_value().unwrap(), last);
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        assert_eq!(callbacks.load(Ordering::SeqCst), 32);
        assert_eq!(errors.load(Ordering::SeqCst), 0);
        assert_eq!(mock.submit_calls.load(Ordering::SeqCst), 32);
        assert_eq!(mock.created_fences.load(Ordering::SeqCst), 8);
        assert_eq!(mock.waited_fences.load(Ordering::SeqCst), 32);
        assert_eq!(mock.reset_fences.load(Ordering::SeqCst), 32);
        assert_eq!(mock.malformed.load(Ordering::SeqCst), 0);
        drop(ctx);
        assert_eq!(mock.destroyed_fences.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn device_lost_fails_queued_submissions_and_prevents_further_queue_use() {
        let mock = Arc::new(Mock {
            fail_first_submit: true,
            block_first_submit: true,
            ..Mock::new()
        });
        let fault = Arc::new(DeviceFaultState::default());
        let options = QueueContextOptions {
            fault_state: Some(Arc::clone(&fault)),
            ..Default::default()
        };
        let ctx = context(&mock, options.clone());
        let sibling = context(&mock, options);
        let results = Arc::new(Mutex::new(Vec::new()));
        let record = |results: &Arc<Mutex<Vec<(ErrorCode, i32)>>>| -> OnComplete {
            let results = Arc::clone(results);
            Box::new(move |_, status| {
                let error = status.unwrap_err();
                results.lock().unwrap().push((error.code(), error.vk_result()));
            })
        };
        ctx.submit(Submission {
            on_complete: Some(record(&results)),
            ..Default::default()
        })
        .unwrap();
        mock.wait_until_first_submit();
        ctx.submit(Submission {
            on_complete: Some(record(&results)),
            ..Default::default()
        })
        .unwrap();
        mock.release_first_submit();
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        let results = results.lock().unwrap().clone();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(
            |(code, vk)| *code == ErrorCode::DeviceLost && *vk == vk::Result::ERROR_DEVICE_LOST.as_raw()
        ));
        assert!(!fault.healthy());
        assert_eq!(fault.last_result(), vk::Result::ERROR_DEVICE_LOST);
        assert_eq!(
            sibling.submit(Submission::default()).unwrap_err().code(),
            ErrorCode::DeviceLost
        );
        assert_eq!(
            ctx.submit(Submission::default()).unwrap_err().code(),
            ErrorCode::Cancelled
        );
        assert_eq!(mock.submit_calls.load(Ordering::SeqCst), 1);
        assert_eq!(mock.wait_calls.load(Ordering::SeqCst), 0);
        sibling.shutdown(ShutdownMode::Graceful).unwrap();
    }

    #[test]
    fn concurrent_shutdown_callers_join_workers_safely() {
        let mock = Arc::new(Mock::new());
        let ctx = context(&mock, QueueContextOptions::default());
        let (callbacks, errors) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
        for _ in 0..8 {
            ctx.submit(Submission {
                on_complete: Some(counting(&callbacks, &errors)),
                ..Default::default()
            })
            .unwrap();
        }
        std::thread::scope(|scope| {
            scope.spawn(|| ctx.shutdown(ShutdownMode::Graceful).unwrap());
            scope.spawn(|| ctx.shutdown(ShutdownMode::Graceful).unwrap());
        });
        assert_eq!(callbacks.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn records_on_submission_thread_and_reports_timeline_across_callbacks() {
        let mock = Arc::new(Mock::new());
        let ctx = context(&mock, QueueContextOptions::default());
        let state = Arc::new(Mutex::new((0u64, 0u64, false, 0u64, 0u64, 0u32)));
        let (s1, s2, s3, s4) = (
            Arc::clone(&state),
            Arc::clone(&state),
            Arc::clone(&state),
            Arc::clone(&state),
        );
        let id = ctx
            .submit(Submission {
                record: Some(Box::new(move |value, commands| {
                    s1.lock().unwrap().0 = value;
                    commands.push(vk::CommandBuffer::from_raw(88));
                    Ok(())
                })),
                on_submitted: Some(Box::new(move |value, status| {
                    let mut s = s2.lock().unwrap();
                    s.1 = value;
                    s.2 = status.is_ok();
                })),
                on_complete: Some(Box::new(move |id, status| {
                    assert!(status.is_ok());
                    s3.lock().unwrap().3 = id;
                })),
                on_completion_profile: Some(Box::new(move |delay, _| {
                    let mut s = s4.lock().unwrap();
                    s.4 = delay;
                    s.5 += 1;
                })),
                ..Default::default()
            })
            .unwrap();
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        let s = *state.lock().unwrap();
        assert_eq!((s.0, s.1, s.2, s.3, s.5), (1, 1, true, id, 1));
        assert!(s.4 > 0);
        assert_eq!(mock.submit_calls.load(Ordering::SeqCst), 1);
        assert_eq!(mock.submitted_command_buffers.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn record_failure_skips_queue_submit_and_reports_failure() {
        let mock = Arc::new(Mock::new());
        let ctx = context(&mock, QueueContextOptions::default());
        let flags = Arc::new(Mutex::new((false, false)));
        let (a, b) = (Arc::clone(&flags), Arc::clone(&flags));
        ctx.submit(Submission {
            record: Some(Box::new(|_, _| {
                Err(Error::new(ErrorCode::TransferFailed, "mock recording failed"))
            })),
            on_submitted: Some(Box::new(move |_, status| a.lock().unwrap().0 = status.is_err())),
            on_complete: Some(Box::new(move |_, status| b.lock().unwrap().1 = status.is_err())),
            ..Default::default()
        })
        .unwrap();
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        assert_eq!(*flags.lock().unwrap(), (true, true));
        assert_eq!(mock.submit_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn coalesces_compatible_dispatches_into_one_timeline_submission() {
        let mock = Arc::new(Mock::new());
        let coalescing = CoalescingOptions {
            enabled: true,
            maximum_batch_size: 8,
            maximum_wait_ns: 5_000_000,
        };
        let ctx = context(
            &mock,
            QueueContextOptions {
                coalescing,
                ..Default::default()
            },
        );
        let values = Arc::new(Mutex::new(vec![0u64; 32]));
        let ids = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::new(AtomicU32::new(0));
        for index in 0..32u32 {
            let (values, ids, recorded) = (Arc::clone(&values), Arc::clone(&ids), Arc::clone(&recorded));
            ctx.submit(Submission {
                coalescible: true,
                dependency_free: true,
                coalescing_key: 1234,
                resources: vec![ResourceAccess {
                    slot: index,
                    generation: 1,
                    write: false,
                }],
                record: Some(Box::new(move |_, commands| {
                    recorded.fetch_add(1, Ordering::SeqCst);
                    commands.push(vk::CommandBuffer::from_raw(200 + u64::from(index)));
                    Ok(())
                })),
                on_submitted: Some(Box::new(move |value, status| {
                    assert!(status.is_ok());
                    values.lock().unwrap()[index as usize] = value;
                })),
                on_complete: Some(Box::new(move |id, status| {
                    assert!(status.is_ok());
                    ids.lock().unwrap().push(id);
                })),
                ..Default::default()
            })
            .unwrap();
        }
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        assert!(mock.submit_calls.load(Ordering::SeqCst) < 32);
        assert_eq!(mock.submitted_command_buffers.load(Ordering::SeqCst), 32);
        assert_eq!(recorded.load(Ordering::SeqCst), 32);
        let mut ids = ids.lock().unwrap().clone();
        ids.sort_unstable();
        assert_eq!(ids, (1..=32).collect::<Vec<u64>>());
        let values = values.lock().unwrap();
        assert_eq!(values[0], values[1]);
        assert_eq!(values[1], values[2]);
        assert!(values[0] < values[31]);
    }

    #[test]
    fn does_not_coalesce_high_priority_conflicting_or_waiting_dispatches() {
        let mock = Arc::new(Mock::new());
        let coalescing = CoalescingOptions {
            enabled: true,
            maximum_batch_size: 8,
            maximum_wait_ns: 5_000_000,
        };
        let ctx = context(
            &mock,
            QueueContextOptions {
                coalescing,
                ..Default::default()
            },
        );
        for index in 0..4u32 {
            ctx.submit(Submission {
                coalescible: true,
                dependency_free: index < 2,
                coalescing_key: 1234,
                priority: if index < 2 { 2 } else { 1 },
                resources: vec![ResourceAccess {
                    slot: if index < 2 { 1 } else { 9 },
                    generation: 1,
                    write: index >= 2,
                }],
                ..Default::default()
            })
            .unwrap();
        }
        for value in 1..=2u64 {
            ctx.submit(Submission {
                coalescible: true,
                dependency_free: true,
                coalescing_key: 5678,
                resources: vec![ResourceAccess {
                    slot: 20 + value as u32,
                    generation: 1,
                    write: false,
                }],
                waits: vec![QueueWait {
                    semaphore: vk::Semaphore::from_raw(TIMELINE),
                    value,
                    stage: vk::PipelineStageFlags::COMPUTE_SHADER,
                }],
                ..Default::default()
            })
            .unwrap();
        }
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        assert_eq!(mock.submit_calls.load(Ordering::SeqCst), 6);
    }

    #[test]
    fn completion_timeout_polls_counter_before_delivering_callback() {
        let mock = Arc::new(Mock::new());
        mock.timeout_next_wait.store(true, Ordering::SeqCst);
        let fault = Arc::new(DeviceFaultState::default());
        let ctx = context(
            &mock,
            QueueContextOptions {
                fault_state: Some(Arc::clone(&fault)),
                ..Default::default()
            },
        );
        let success = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&success);
        ctx.submit(Submission {
            on_complete: Some(Box::new(move |_, s| flag.store(s.is_ok(), Ordering::SeqCst))),
            ..Default::default()
        })
        .unwrap();
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        assert!(success.load(Ordering::SeqCst));
        assert_eq!(mock.wait_calls.load(Ordering::SeqCst), 1);
        assert!(fault.healthy());
    }

    #[test]
    fn explicit_wait_distinguishes_timeout_from_device_loss() {
        let mock = Arc::new(Mock::new());
        let fault = Arc::new(DeviceFaultState::default());
        let ctx = context(
            &mock,
            QueueContextOptions {
                fault_state: Some(Arc::clone(&fault)),
                ..Default::default()
            },
        );
        mock.timeout_next_wait.store(true, Ordering::SeqCst);
        assert_eq!(ctx.wait_for_value(1, 1).unwrap_err().code(), ErrorCode::Timeout);
        assert!(fault.healthy());
        mock.device_lost_next_wait.store(true, Ordering::SeqCst);
        assert_eq!(
            ctx.wait_for_value(1, 1).unwrap_err().code(),
            ErrorCode::DeviceLost
        );
        assert!(!fault.healthy());
        assert_eq!(fault.last_result(), vk::Result::ERROR_DEVICE_LOST);
        assert_eq!(
            ctx.submit(Submission::default()).unwrap_err().code(),
            ErrorCode::DeviceLost
        );
        assert_eq!(mock.submit_calls.load(Ordering::SeqCst), 0);
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        assert!(ctx.wait_for_value(0, 0).is_ok());
    }

    #[test]
    fn spin_wait_completes_without_blocking_when_value_is_reached() {
        let mock = Arc::new(Mock::new());
        let ctx = context(
            &mock,
            QueueContextOptions {
                completion_wait: CompletionWaitOptions { spin_ns: 50_000_000 },
                ..Default::default()
            },
        );
        let (callbacks, errors) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
        for _ in 0..16 {
            ctx.submit(Submission {
                on_complete: Some(counting(&callbacks, &errors)),
                ..Default::default()
            })
            .unwrap();
        }
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        assert_eq!(callbacks.load(Ordering::SeqCst), 16);
        assert_eq!(
            mock.wait_calls.load(Ordering::SeqCst),
            0,
            "the mock completes on submit, so spinning never blocks"
        );
    }

    #[test]
    fn shutdown_from_a_callback_is_rejected() {
        let mock = Arc::new(Mock::new());
        let ctx = Arc::new(context(&mock, QueueContextOptions::default()));
        let observed = Arc::new(Mutex::new(None));
        let (inner, seen) = (Arc::clone(&ctx), Arc::clone(&observed));
        ctx.submit(Submission {
            on_complete: Some(Box::new(move |_, _| {
                *seen.lock().unwrap() = Some(inner.shutdown(ShutdownMode::Graceful).map_err(|e| e.code()));
            })),
            ..Default::default()
        })
        .unwrap();
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        assert_eq!(*observed.lock().unwrap(), Some(Err(ErrorCode::InvalidArgument)));
    }
}
