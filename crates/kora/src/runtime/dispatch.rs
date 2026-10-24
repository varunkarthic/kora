//! Dispatch assembly, residency, hazards, and completion.

use std::any::Any;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ash::vk;
use bytemuck::Pod;

use super::context::{DeviceContext, PendingAutotune, PhaseProfile, TraceScore, elapsed_ns, lock};
use super::kernel::Kernel;
use super::replica::{DeviceAccess, Replica};
use crate::buffer::BufferHandle;
use crate::error::{Error, ErrorCode, Result, Status};
use crate::event::{Event, EventPromise};
use crate::kernel::{DescriptorSetLease, PipelineRef, Trial, WorkgroupAutotuner};
use crate::memory::{PendingUses, maintain_mapped_range};
use crate::metadata::{KernelAccess, KernelBindingKind, SpecializationConstantMetadata, SpecializationValue};
use crate::performance_db::{
    PerformanceObservationKey, PerformanceTimingDomain, performance_observation_key,
};
use crate::placement::AdaptiveDispatchWorkload;
use crate::scheduler::ScoreBreakdown;
use crate::verification::{VerificationDataType, VerificationPolicy};
use crate::vulkan::{CommandPoolSet, Submission};

/// Scheduling hint for a dispatch. `High` and `RealtimeHint` are never delayed for coalescing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DispatchPriority {
    /// Background work.
    Low,
    /// Default.
    #[default]
    Normal,
    /// Latency sensitive.
    High,
    /// Latency critical (a hint; Vulkan offers no real-time guarantee).
    RealtimeHint,
}

/// Builds one dispatch of a [`Kernel`]. Bind buffers in the kernel's binding order with the
/// access each binding declares, then [`submit`](Self::submit).
///
/// Inputs are uploaded only when their host contents changed since the device copy was made.
/// Outputs are copied back to the host when the dispatch completes, unless
/// [`defer_readback`](Self::defer_readback) keeps them on the device for a following dispatch.
#[must_use = "a dispatch does nothing until submitted"]
#[derive(Debug, Clone)]
pub struct DispatchBuilder {
    pub(crate) kernel: Kernel,
    pub(crate) peers: Arc<[Arc<DeviceContext>]>,
    pub(crate) global: [u64; 3],
    pub(crate) bindings: Vec<(BufferHandle, KernelAccess)>,
    pub(crate) push: Vec<u8>,
    pub(crate) defer_readback: bool,
    pub(crate) priority: DispatchPriority,
    pub(crate) adaptive: Option<Arc<AdaptiveState>>,
    pub(crate) verification: Option<(VerificationPolicy, VerificationDataType)>,
    /// A verification duplicate: the other participants hold divergent copies by design.
    pub(crate) duplicate: bool,
    /// A slice of a split dispatch: its outputs are partial, so they never become the buffer's
    /// newest contents; the caller reads its range back with a replica reader.
    pub(crate) partial: bool,
}

/// Placement made by [`Runtime::dispatch_adaptive`](super::Runtime::dispatch_adaptive).
#[derive(Debug)]
pub(crate) struct AdaptiveState {
    /// The kernel as loaded by the caller (its device produced device-resident inputs).
    pub(crate) original: Kernel,
    pub(crate) workload: AdaptiveDispatchWorkload,
    pub(crate) score: Option<ScoreBreakdown>,
}

/// Performance-database key for the device execution time of `identity` under `workload`.
pub(crate) fn execution_key(
    ctx: &DeviceContext,
    identity: u64,
    workload: &AdaptiveDispatchWorkload,
) -> Option<String> {
    if ctx.id.uuid.is_empty() {
        return None;
    }
    performance_observation_key(&PerformanceObservationKey {
        device_uuid: ctx.id.uuid.to_string(),
        driver_version: ctx.info.driver_version,
        runtime_schema: crate::version::SCHEMA_VERSION,
        kernel_identity: identity,
        workload_class: workload.kernel_kind as u8,
        workload_bucket: workload.kernel_work.max(1).ilog2() as u8,
        timing_domain: PerformanceTimingDomain::DeviceExecution,
    })
    .ok()
}

impl DispatchBuilder {
    pub(crate) fn new(kernel: Kernel, peers: Arc<[Arc<DeviceContext>]>) -> Self {
        let bindings = Vec::with_capacity(kernel.state.metadata.bindings.len());
        Self {
            kernel,
            peers,
            global: [1, 1, 1],
            bindings,
            push: Vec::new(),
            defer_readback: false,
            priority: DispatchPriority::Normal,
            adaptive: None,
            verification: None,
            duplicate: false,
            partial: false,
        }
    }

    /// Runs the dispatch on every initialized, non-quarantined device that supports the kernel
    /// and compares each writable output across devices with `policy`. Matching results are
    /// published to the host; on a mismatch the outputs are marked lost, every participant is
    /// quarantined, and the event fails with [`ErrorCode::ValidationError`].
    pub fn verify_with(mut self, policy: VerificationPolicy, data_type: VerificationDataType) -> Self {
        self.verification = Some((policy, data_type));
        self
    }

    /// One-dimensional global size in invocations (rounded up to whole workgroups).
    pub fn global(self, x: u64) -> Self {
        self.global_3d(x, 1, 1)
    }

    /// Three-dimensional global size in invocations.
    pub fn global_3d(mut self, x: u64, y: u64, z: u64) -> Self {
        self.global = [x, y, z];
        self
    }

    fn bind(mut self, buffer: &impl AsRef<BufferHandle>, access: KernelAccess) -> Self {
        self.bindings.push((buffer.as_ref().clone(), access));
        self
    }

    /// Binds the next buffer as read-only.
    pub fn read(self, buffer: &impl AsRef<BufferHandle>) -> Self {
        self.bind(buffer, KernelAccess::Read)
    }

    /// Binds the next buffer as write-only (its host contents are not uploaded).
    pub fn write(self, buffer: &impl AsRef<BufferHandle>) -> Self {
        self.bind(buffer, KernelAccess::Write)
    }

    /// Binds the next buffer as read-write.
    pub fn read_write(self, buffer: &impl AsRef<BufferHandle>) -> Self {
        self.bind(buffer, KernelAccess::ReadWrite)
    }

    /// Sets the push-constant block from a `Pod` value.
    pub fn push<T: Pod>(self, value: &T) -> Self {
        self.push_bytes(bytemuck::bytes_of(value))
    }

    /// Sets the push-constant block from raw bytes.
    pub fn push_bytes(mut self, bytes: &[u8]) -> Self {
        self.push.clear();
        self.push.extend_from_slice(bytes);
        self
    }

    /// Keeps device-written outputs resident instead of copying them to the host on completion.
    /// Use [`Runtime::readback`](super::Runtime::readback) when the host needs them.
    pub fn defer_readback(mut self) -> Self {
        self.defer_readback = true;
        self
    }

    /// Marks this dispatch as one slice of a split dispatch (see [`DispatchBuilder::partial`]).
    pub(crate) fn partial_outputs(mut self) -> Self {
        self.partial = true;
        self.defer_readback = true;
        self
    }

    /// Scheduling hint.
    pub fn priority(mut self, priority: DispatchPriority) -> Self {
        self.priority = priority;
        self
    }

    /// The device this dispatch targets. An adaptive dispatch may still return to the kernel's
    /// original device at submission to keep device-resident inputs local.
    pub fn device(&self) -> &crate::ids::DeviceId {
        self.kernel.device()
    }

    /// Keeps adaptive dispatches next to device-resident inputs: if exactly one device holds
    /// newer results for an input, the dispatch must run there.
    fn resolve_adaptive_inputs(&mut self, adaptive: &AdaptiveState) -> Status {
        let mut producers: Vec<&Arc<DeviceContext>> = Vec::new();
        for (buffer, access) in &self.bindings {
            if !access.reads() {
                continue;
            }
            let version = buffer.host_version();
            for peer in self.peers.iter() {
                let newer = buffer.replica::<Replica>(peer.key).is_some_and(|r| {
                    let state = r.state();
                    state.device_newer && state.resident_version == Some(version)
                });
                if newer && !producers.iter().any(|p| p.key == peer.key) {
                    producers.push(peer);
                }
            }
        }
        match producers.as_slice() {
            [] => Ok(()),
            [producer] if producer.key == self.kernel.state.context.key => Ok(()),
            [producer] if producer.key == adaptive.original.state.context.key => {
                self.kernel = adaptive.original.clone();
                Ok(())
            }
            [_] => Err(Error::new(
                ErrorCode::UnsupportedFeature,
                "adaptive dispatch input is newer on another device; read it back before migration",
            )),
            _ => Err(Error::new(
                ErrorCode::UnsupportedFeature,
                "adaptive dispatch inputs have newer copies on multiple devices; read them back first",
            )),
        }
    }

    /// Validates, uploads stale inputs, and enqueues the dispatch. The returned event completes
    /// once the GPU work (and, unless deferred, the output readback) is done.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidKernelBinding`] for a binding mismatch, [`ErrorCode::InvalidDispatch`]
    /// for sizes the device cannot dispatch, [`ErrorCode::InvalidArgument`] for push-constant
    /// mismatches and for an input whose newest contents live on another device, plus allocation
    /// and transfer errors.
    pub fn submit(mut self) -> Result<Event> {
        if self.verification.is_some() {
            if self.adaptive.is_some() {
                return Err(Error::new(
                    ErrorCode::UnsupportedFeature,
                    "adaptive dispatch cannot be combined with duplicate verification",
                ));
            }
            return super::verify::submit_verified(self);
        }
        let started = Instant::now();
        if let Some(adaptive) = self.adaptive.clone() {
            self.resolve_adaptive_inputs(&adaptive)?;
        }
        let state = Arc::clone(&self.kernel.state);
        let ctx = &state.context;
        ctx.check_accepting()?;
        let mut profile = PhaseProfile::default();
        let mut mark = Instant::now();
        let mut phase = |slot: &mut u64| {
            if ctx.phase_profile {
                *slot = elapsed_ns(mark);
                mark = Instant::now();
            }
        };
        let metadata = &state.metadata;
        if self.bindings.len() != metadata.bindings.len() {
            return Err(Error::new(
                ErrorCode::InvalidKernelBinding,
                format!(
                    "dispatch binds {} buffers but kernel '{}' declares {}",
                    self.bindings.len(),
                    metadata.kernel_id,
                    metadata.bindings.len()
                ),
            ));
        }
        for (index, ((_, access), declared)) in self.bindings.iter().zip(&metadata.bindings).enumerate() {
            if *access != declared.access {
                return Err(Error::new(
                    ErrorCode::InvalidKernelBinding,
                    format!(
                        "binding {index} ('{}') is declared {} but dispatched as {}",
                        declared.name,
                        declared.access.name(),
                        access.name()
                    ),
                ));
            }
            if declared.kind != KernelBindingKind::StorageBuffer {
                return Err(Error::new(
                    ErrorCode::UnsupportedFeature,
                    "runtime dispatch currently supports storage-buffer bindings only",
                ));
            }
        }
        if self.push.len() != metadata.push_constant_size as usize {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                format!(
                    "push constants are {} bytes but the kernel declares {}",
                    self.push.len(),
                    metadata.push_constant_size
                ),
            ));
        }
        if metadata.push_constant_size > ctx.info.limits.max_push_constants_size {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "push-constant block exceeds the selected device limit",
            ));
        }

        let mut trial = TrialGuard::reserve(ctx, state.autotuner.as_ref());
        let local_size = trial.local_size().unwrap_or(state.reflection.local_size);
        let groups = group_counts(self.global, local_size, &ctx.info.limits)?;
        let overrides = local_size_overrides(&state, local_size);
        phase(&mut profile.dispatch_validation_ns);
        let pipeline = ctx
            .pipelines
            .get_or_create(&state.words, metadata, &overrides, &state.entry_point)?;
        phase(&mut profile.pipeline_lookup_ns);

        let replicas = self
            .bindings
            .iter()
            .map(|(buffer, _)| ctx.replica(buffer))
            .collect::<Result<Vec<_>>>()?;
        for (buffer, access) in &self.bindings {
            if access.reads() && !self.duplicate && !self.partial {
                reject_newer_elsewhere(ctx, buffer)?;
            }
        }
        phase(&mut profile.binding_preflight_ns);
        let mut bound_versions = Vec::with_capacity(self.bindings.len());
        let mut uploads = Vec::new();
        for ((buffer, access), replica) in self.bindings.iter().zip(&replicas) {
            if access.reads() {
                let (version, upload) = make_resident(ctx, buffer, replica)?;
                bound_versions.push(version);
                uploads.extend(upload);
            } else {
                bound_versions.push(buffer.host_version());
            }
        }
        phase(&mut profile.buffer_residency_ns);

        let host_read_barrier = ctx.direct_host_access && !self.defer_readback;
        let promise = EventPromise::new();
        let event = promise.event();
        let work = Arc::new(Mutex::new(Work::default()));
        let profile_work = Arc::clone(&work);
        let mut keepalive: Vec<Arc<dyn Any + Send + Sync>> = self
            .bindings
            .iter()
            .map(|(buffer, _)| buffer.keepalive())
            .collect();
        keepalive.extend(
            replicas
                .iter()
                .map(|r| Arc::clone(r) as Arc<dyn Any + Send + Sync>),
        );
        let pending = PendingUses::new(
            replicas
                .iter()
                .filter_map(|r| r.transfer().queue_use.clone())
                .collect(),
        );
        let outputs: Vec<usize> = (0..self.bindings.len())
            .filter(|&i| self.bindings[i].1.writes())
            .collect();
        let observation = self
            .adaptive
            .as_ref()
            .and_then(|a| execution_key(ctx, state.semantic_identity, &a.workload));
        let trace_score = self
            .adaptive
            .as_ref()
            .and_then(|a| a.score)
            .filter(|_| ctx.trace)
            .map(|score| TraceScore {
                device_index: ctx.id.index,
                score,
            });
        let timestamped =
            ctx.sample_timestamps() || trial.tracked() || observation.is_some() || trace_score.is_some();

        for &index in &outputs {
            if let Some(use_) = &replicas[index].transfer().queue_use {
                use_.host_readable.store(host_read_barrier, Ordering::Release);
            }
        }
        let _order = lock(&ctx.submit_lock);
        let mut barrier = false;
        for ((_, access), replica) in self.bindings.iter().zip(&replicas) {
            let previous = replica.state().access;
            barrier |= match previous {
                DeviceAccess::Write => true,
                DeviceAccess::Read => access.writes(),
                DeviceAccess::None => false,
            };
        }
        for (index, ((_, access), replica)) in self.bindings.iter().zip(&replicas).enumerate() {
            let mut replica_state = replica.state();
            replica_state.access = if access.writes() {
                DeviceAccess::Write
            } else {
                DeviceAccess::Read
            };
            if access.writes() {
                replica_state.resident_version = (!self.partial).then_some(bound_versions[index]);
                replica_state.device_newer = !self.partial;
            }
        }

        let record = Recording {
            ctx: Arc::clone(ctx),
            pipeline: Arc::clone(&pipeline),
            buffers: replicas.iter().map(|r| r.buffer()).collect(),
            push: self.push,
            groups,
            barrier,
            host_read_barrier,
            timestamped,
            work: Arc::clone(&work),
        };
        let output_replicas: Vec<Arc<Replica>> = outputs.iter().map(|&i| Arc::clone(&replicas[i])).collect();
        let submitted = Submitted {
            ctx: Arc::clone(ctx),
            work: Arc::clone(&work),
            replicas: replicas.clone(),
            outputs: outputs.clone(),
            trial: trial
                .take()
                .map(|(tuner, trial)| (tuner, trial, state.autotune_keys[trial.candidate_index].clone())),
            observation,
            trace_score,
        };
        let completed = Completed {
            ctx: Arc::clone(ctx),
            work,
            bindings: self.bindings,
            replicas,
            outputs: outputs.clone(),
            bound_versions,
            uploads,
            defer_readback: self.defer_readback,
            host_read_barrier,
            promise,
            pending,
            keepalive,
            pipeline,
        };
        let submission = Submission {
            coalescible: true,
            dependency_free: !barrier,
            coalescing_key: Arc::as_ptr(&completed.pipeline) as u64,
            priority: self.priority as u8,
            resources: completed
                .bindings
                .iter()
                .map(|(buffer, access)| crate::vulkan::ResourceAccess {
                    slot: buffer.id().slot,
                    generation: buffer.id().generation,
                    write: access.writes(),
                })
                .collect(),
            record: Some(Box::new(move |id, commands| record.run(id, commands))),
            on_submitted: Some(Box::new(move |timeline, status| submitted.run(timeline, status))),
            on_complete: Some(Box::new(move |_, status| completed.run(status))),
            on_completion_profile: ctx.phase_profile.then(|| {
                let work = Arc::clone(&profile_work);
                Box::new(move |delay: u64, wait: u64| {
                    let mut work = lock(&work);
                    work.profile.completion_worker_delay_ns = delay.max(1);
                    work.profile.device_completion_wait_ns = wait.max(1);
                }) as crate::vulkan::OnCompletionProfile
            }),
            ..Submission::default()
        };
        if ctx.phase_profile {
            phase(&mut profile.submission_assembly_ns);
            profile.host_submission_ns = elapsed_ns(started)
                .saturating_sub(profile.buffer_residency_ns)
                .max(1);
            let mut work = lock(&profile_work);
            work.profile = PhaseProfile {
                command_buffer_acquire_ns: work.profile.command_buffer_acquire_ns,
                descriptor_setup_ns: work.profile.descriptor_setup_ns,
                command_recording_ns: work.profile.command_recording_ns,
                ..profile
            };
        }
        lock(&profile_work).queued_at = Some(Instant::now());
        let enqueue_started = Instant::now();
        let submitted = ctx.queue.submit(submission);
        if ctx.phase_profile {
            let enqueue = elapsed_ns(enqueue_started);
            lock(&profile_work).profile.queue_enqueue_ns = enqueue;
            let mut latest = lock(&ctx.latest_profile);
            if latest.submission_id != 0 && latest.submission_id == lock(&profile_work).submission_id {
                latest.queue_enqueue_ns = enqueue;
            }
        }
        match submitted {
            Ok(_) => Ok(event),
            Err(error) => {
                // The queue dropped the submission without running its callbacks.
                for replica in &output_replicas {
                    let mut state = replica.state();
                    state.resident_version = None;
                    state.device_newer = false;
                }
                Err(error)
            }
        }
    }
}

#[derive(Default)]
struct Work {
    command: vk::CommandBuffer,
    submission_id: u64,
    timestamped: bool,
    leases: Vec<DescriptorSetLease>,
    profile: PhaseProfile,
    queued_at: Option<Instant>,
}

/// Cancels an autotuning reservation unless the submission took ownership of it.
struct TrialGuard {
    tuner: Option<Arc<WorkgroupAutotuner>>,
    trial: Option<Trial>,
}

impl TrialGuard {
    fn reserve(ctx: &DeviceContext, tuner: Option<&Arc<WorkgroupAutotuner>>) -> Self {
        let tuner = tuner
            .filter(|_| ctx.autotuning && ctx.timestamps.is_some())
            .cloned();
        let trial = tuner.as_ref().map(|t| t.reserve());
        Self { tuner, trial }
    }

    fn tracked(&self) -> bool {
        self.trial.is_some_and(|t| t.tracked)
    }

    /// The workgroup size of the reserved trial, or the tuned size once tuning finished.
    fn local_size(&self) -> Option<[u32; 3]> {
        let tuner = self.tuner.as_ref()?;
        let trial = self.trial?;
        Some(if trial.tracked {
            tuner.candidates()[trial.candidate_index].local_size
        } else {
            tuner.selected_local_size()
        })
    }

    fn take(&mut self) -> Option<(Arc<WorkgroupAutotuner>, Trial)> {
        let trial = self.trial.take().filter(|t| t.tracked)?;
        Some((Arc::clone(self.tuner.as_ref()?), trial))
    }
}

impl Drop for TrialGuard {
    fn drop(&mut self) {
        if let (Some(tuner), Some(trial)) = (&self.tuner, self.trial) {
            tuner.cancel(&trial);
        }
    }
}

/// `LocalSizeId` specialization for a workgroup size. Unlike the C++ runtime, the tuned size is
/// specialized for every dispatch once tuning finishes, not only for trial dispatches, so the
/// group count and the shader's workgroup size always agree.
fn local_size_overrides(
    state: &super::kernel::KernelState,
    local_size: [u32; 3],
) -> Vec<SpecializationConstantMetadata> {
    if !state.reflection.local_size_id_based || state.autotuner.is_none() {
        return Vec::new();
    }
    let ids = state.reflection.local_size_spec_ids;
    state
        .metadata
        .specialization_constants
        .iter()
        .filter_map(|spec| {
            let dimension = ids.iter().position(|&id| id == spec.id)?;
            Some(SpecializationConstantMetadata {
                default_value: SpecializationValue::U64(u64::from(local_size[dimension])),
                ..spec.clone()
            })
        })
        .collect()
}

pub(crate) fn group_counts(
    global: [u64; 3],
    local: [u32; 3],
    limits: &crate::device::DeviceLimits,
) -> Result<[u32; 3]> {
    let mut groups = [0_u32; 3];
    let mut invocations = 1_u64;
    for dimension in 0..3 {
        if global[dimension] == 0 {
            return Err(Error::new(
                ErrorCode::InvalidDispatch,
                "global dispatch dimensions must all be nonzero",
            ));
        }
        let size = u64::from(local[dimension]);
        if local[dimension] == 0 || local[dimension] > limits.max_compute_work_group_size[dimension] {
            return Err(Error::new(
                ErrorCode::InvalidDispatch,
                "kernel local size exceeds a selected-device dimension limit",
            ));
        }
        invocations *= size;
        let count = global[dimension].div_ceil(size);
        groups[dimension] = u32::try_from(count)
            .ok()
            .filter(|&c| c <= limits.max_compute_work_group_count[dimension])
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidDispatch,
                    "dispatch group count exceeds a selected-device limit",
                )
            })?;
    }
    if invocations > u64::from(limits.max_compute_work_group_invocations) {
        return Err(Error::new(
            ErrorCode::InvalidDispatch,
            "kernel local size exceeds maxComputeWorkGroupInvocations",
        ));
    }
    Ok(groups)
}

/// Refuses to upload stale host bytes over results that only another device holds.
fn reject_newer_elsewhere(ctx: &DeviceContext, buffer: &BufferHandle) -> Status {
    let version = buffer.host_version();
    for owner in buffer.replica_owners() {
        if owner == ctx.key {
            continue;
        }
        if let Some(other) = buffer.replica::<Replica>(owner) {
            let state = other.state();
            if state.device_newer && state.resident_version == Some(version) {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    format!(
                        "buffer {}:{} has newer contents on another device; read it back before using it here",
                        buffer.id().slot,
                        buffer.id().generation
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Ensures `replica` holds the buffer's current host contents. Staged uploads are enqueued on the
/// compute queue ahead of the dispatch and are not waited for; the transfer records the barriers
/// that order it before compute.
fn make_resident(
    ctx: &DeviceContext,
    buffer: &BufferHandle,
    replica: &Arc<Replica>,
) -> Result<(u64, Option<Event>)> {
    let mut state = replica.state();
    buffer.with_contents(|bytes, version| -> Result<(u64, Option<Event>)> {
        if state.resident_version == Some(version) {
            return Ok((version, None));
        }
        let event = if ctx.direct_host_access && replica.directly_accessible(0) {
            let mapped = replica
                .transfer()
                .mapped
                .expect("directly accessible replicas are mapped");
            // SAFETY: the mapping covers the whole replica (`bytes.len()` equals its size), and no
            // queued or executing submission uses the replica (`pending == 0`).
            unsafe { mapped.slice_mut(bytes.len()).copy_from_slice(bytes) };
            maintain_mapped_range(
                &ctx.device,
                replica.transfer(),
                0,
                replica.size(),
                ctx.atom_size,
                true,
            )?;
            None
        } else {
            Some(ctx.transfers.upload(&replica.transfer_view(), 0, bytes)?)
        };
        ctx.uploads.fetch_add(1, Ordering::Relaxed);
        state.resident_version = Some(version);
        state.device_newer = false;
        // Staged copies end with a barrier to compute access; direct writes precede the submit.
        state.access = DeviceAccess::None;
        Ok((version, event))
    })?
}

struct Recording {
    ctx: Arc<DeviceContext>,
    pipeline: PipelineRef,
    buffers: Vec<vk::Buffer>,
    push: Vec<u8>,
    groups: [u32; 3],
    barrier: bool,
    host_read_barrier: bool,
    timestamped: bool,
    work: Arc<Mutex<Work>>,
}

impl Recording {
    fn run(self, submission_id: u64, commands: &mut Vec<vk::CommandBuffer>) -> Status {
        let ctx = &self.ctx;
        let acquire_started = Instant::now();
        let mut slot = lock(&ctx.commands);
        if slot.is_none() {
            *slot = Some(CommandPoolSet::new(
                ctx.device.clone(),
                ctx.families.compute_family,
                32,
            )?);
        }
        let pool = slot.as_mut().expect("created above");
        let mut completed = ctx.queue.completed_value()?;
        let command = loop {
            match pool.acquire(completed) {
                Ok(command) => break command,
                Err(error)
                    if error.code() == ErrorCode::OutOfMemoryBudget && pool.oldest_pending_value() != 0 =>
                {
                    ctx.queue.wait_for_value(pool.oldest_pending_value(), u64::MAX)?;
                    completed = ctx.queue.completed_value()?;
                }
                Err(error) => return Err(error),
            }
        };
        let mut work = lock(&self.work);
        work.command = command;
        work.submission_id = submission_id;
        if ctx.phase_profile {
            work.profile.command_buffer_acquire_ns = elapsed_ns(acquire_started);
        }
        let recorded = self.record(command, submission_id, &mut work);
        match recorded.and_then(|()| pool.end_recording(command)) {
            Ok(()) => {
                commands.push(command);
                Ok(())
            }
            Err(error) => {
                let _ = pool.discard(command);
                work.command = vk::CommandBuffer::null();
                if work.timestamped
                    && let Some(ring) = &ctx.timestamps
                {
                    ring.cancel(submission_id);
                }
                Err(error)
            }
        }
    }

    fn record(&self, command: vk::CommandBuffer, submission_id: u64, work: &mut Work) -> Status {
        let ctx = &self.ctx;
        let device = ctx.device.raw();
        work.timestamped = self.timestamped
            && ctx
                .timestamps
                .as_ref()
                .is_some_and(|ring| ring.record_start(command, submission_id));
        let descriptors_started = Instant::now();
        for set in 0..self.pipeline.descriptor_set_count() {
            work.leases
                .push(ctx.pipelines.allocate_descriptor_set(&self.pipeline, set)?);
        }
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = self
            .buffers
            .iter()
            .map(|&buffer| {
                [vk::DescriptorBufferInfo {
                    buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                }]
            })
            .collect();
        let writes: Vec<_> = self
            .pipeline
            .bindings()
            .iter()
            .zip(&infos)
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(work.leases[binding.set as usize].handle())
                    .dst_binding(binding.binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        let sets: Vec<vk::DescriptorSet> = work.leases.iter().map(DescriptorSetLease::handle).collect();
        // SAFETY: the descriptor sets are leased to this submission and not used by pending work;
        // the buffers they reference are kept alive by the submission until completion.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        let recording_started = Instant::now();
        if ctx.phase_profile {
            work.profile.descriptor_setup_ns = elapsed_ns(descriptors_started);
        }
        // SAFETY: the command buffer is recording on the submission thread; the pipeline, layout,
        // and descriptor sets are kept alive by the submission until completion.
        unsafe {
            if self.barrier {
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
            }
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline.handle());
            if !sets.is_empty() {
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline.layout(),
                    0,
                    &sets,
                    &[],
                );
            }
            if !self.push.is_empty() {
                device.cmd_push_constants(
                    command,
                    self.pipeline.layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &self.push,
                );
            }
            device.cmd_dispatch(command, self.groups[0], self.groups[1], self.groups[2]);
            if self.host_read_barrier {
                // Makes shader writes visible to host reads of mapped replicas after completion.
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
            }
        }
        if work.timestamped
            && let Some(ring) = &ctx.timestamps
        {
            ring.record_end(command, submission_id);
        }
        if ctx.phase_profile {
            work.profile.command_recording_ns = elapsed_ns(recording_started);
        }
        Ok(())
    }
}

struct Submitted {
    ctx: Arc<DeviceContext>,
    work: Arc<Mutex<Work>>,
    replicas: Vec<Arc<Replica>>,
    outputs: Vec<usize>,
    trial: Option<(Arc<WorkgroupAutotuner>, Trial, String)>,
    observation: Option<String>,
    trace_score: Option<TraceScore>,
}

impl Drop for Submitted {
    fn drop(&mut self) {
        // A submission dropped before `run` (for example, refused by the queue) releases its
        // autotuning reservation so the schedule cannot stall.
        if let Some((tuner, trial, _)) = self.trial.take() {
            tuner.cancel(&trial);
        }
    }
}

impl Submitted {
    fn run(mut self, timeline: u64, status: &Status) {
        let ctx = &self.ctx;
        let mut work = lock(&self.work);
        let command = std::mem::replace(&mut work.command, vk::CommandBuffer::null());
        if command != vk::CommandBuffer::null()
            && let Some(pool) = lock(&ctx.commands).as_mut()
        {
            let _ = if status.is_ok() {
                pool.submitted(command, timeline)
            } else {
                pool.discard(command)
            };
        }
        if let Some(ring) = ctx.timestamps.as_ref().filter(|_| work.timestamped) {
            if status.is_ok() {
                ring.submitted(work.submission_id, timeline);
            } else {
                ring.cancel(work.submission_id);
            }
        }
        if let Some((tuner, trial, key)) = self.trial.take() {
            if status.is_ok() && work.timestamped {
                lock(&ctx.pending_autotune).insert(work.submission_id, PendingAutotune { tuner, trial, key });
            } else {
                tuner.cancel(&trial);
            }
        }
        if status.is_ok() && work.timestamped {
            if let Some(key) = self.observation.take() {
                lock(&ctx.pending_observations).insert(work.submission_id, key);
            }
            if let Some(score) = self.trace_score.take() {
                lock(&ctx.trace_scores).add_pending(work.submission_id, score);
            }
        }
        for replica in &self.replicas {
            let mut state = replica.state();
            if status.is_ok() {
                state.last_use = state.last_use.max(timeline);
            }
        }
        if status.is_err() {
            for &index in &self.outputs {
                let mut state = self.replicas[index].state();
                state.resident_version = None;
                state.device_newer = false;
            }
        }
    }
}

struct Completed {
    ctx: Arc<DeviceContext>,
    work: Arc<Mutex<Work>>,
    bindings: Vec<(BufferHandle, KernelAccess)>,
    replicas: Vec<Arc<Replica>>,
    outputs: Vec<usize>,
    bound_versions: Vec<u64>,
    uploads: Vec<Event>,
    defer_readback: bool,
    host_read_barrier: bool,
    promise: EventPromise,
    pending: PendingUses,
    keepalive: Vec<Arc<dyn Any + Send + Sync>>,
    pipeline: PipelineRef,
}

impl Completed {
    fn run(self, mut status: Status) {
        let Completed {
            ctx,
            work,
            bindings,
            replicas,
            outputs,
            bound_versions,
            uploads,
            defer_readback,
            host_read_barrier,
            promise,
            pending,
            keepalive,
            pipeline,
        } = self;
        let callback_started = Instant::now();
        // Descriptor sets return to their pool now that the GPU is done with them.
        let (mut profile, queued_at) = {
            let mut work = lock(&work);
            drop(std::mem::take(&mut work.leases));
            let mut profile = work.profile;
            profile.submission_id = work.submission_id;
            (profile, work.queued_at)
        };
        drop(work);
        if let Some(queued_at) = queued_at.filter(|_| status.is_ok()) {
            profile.queue_completion_ns = elapsed_ns(queued_at);
        }
        if let Err(error) = &status
            && error.code() == ErrorCode::DeviceLost
        {
            for &index in &outputs {
                bindings[index].0.mark_data_lost();
            }
        }
        if status.is_ok()
            && let Some(failed) = uploads
                .iter()
                .filter(|e| e.is_done())
                .find_map(|e| e.status().err())
        {
            status = Err(failed);
        }
        drop(uploads);
        let collect_started = Instant::now();
        let _ = ctx.collect_timestamps(std::time::Duration::ZERO);
        profile.timestamp_collection_ns = elapsed_ns(collect_started);
        let mut events = Vec::new();
        if status.is_err() {
            for &index in &outputs {
                let mut state = replicas[index].state();
                state.resident_version = None;
                state.device_newer = false;
            }
        } else if !defer_readback {
            for &index in &outputs {
                let (buffer, _) = &bindings[index];
                let replica = &replicas[index];
                // This submission still counts as one pending use of its replicas.
                let direct = host_read_barrier && replica.directly_accessible(1);
                match read_to_host(&ctx, buffer, replica, bound_versions[index], direct) {
                    Ok(Some(event)) => events.push(event),
                    Ok(None) => {}
                    Err(error) => {
                        status = Err(error);
                        break;
                    }
                }
            }
        }
        if ctx.phase_profile && status.is_ok() {
            profile.completion_callback_ns = elapsed_ns(callback_started);
            *lock(&ctx.latest_profile) = profile;
        }
        // Release every resource of the finished dispatch before anyone can observe completion.
        drop((pending, keepalive, pipeline, replicas, bindings, ctx));
        if status.is_err() || events.is_empty() {
            promise.complete(status);
            return;
        }
        match Event::when_all(&events) {
            Ok(all) => {
                if let Err(error) = all.on_complete(move |s| promise.complete(s)) {
                    drop(error);
                }
            }
            Err(error) => promise.complete(Err(error)),
        }
    }
}

/// Copies a replica's results into the host shadow if the host still has `expected` (the version
/// the device work started from). With `direct`, the caller guarantees the producing work finished
/// with a host-read barrier and nothing else uses the replica.
pub(crate) fn read_to_host(
    ctx: &Arc<DeviceContext>,
    buffer: &BufferHandle,
    replica: &Arc<Replica>,
    expected: u64,
    direct: bool,
) -> Result<Option<Event>> {
    let publish = {
        let buffer = buffer.clone();
        let replica = Arc::clone(replica);
        let ctx = Arc::clone(ctx);
        move |bytes: &[u8]| -> Status {
            let written = buffer.write_bytes_if_version(bytes, expected)?;
            ctx.downloads.fetch_add(1, Ordering::Relaxed);
            let mut state = replica.state();
            if written.written {
                state.resident_version = Some(written.host_version);
            }
            state.device_newer = false;
            Ok(())
        }
    };
    if direct {
        maintain_mapped_range(
            &ctx.device,
            replica.transfer(),
            0,
            replica.size(),
            ctx.atom_size,
            false,
        )?;
        let mapped = replica
            .transfer()
            .mapped
            .expect("directly accessible replicas are mapped");
        let len =
            usize::try_from(replica.size()).map_err(|_| Error::new(ErrorCode::Overflow, "replica size"))?;
        // SAFETY: the producing work completed with a shader-write to host-read barrier, the
        // mapping covers the replica, and no other queued work can write it (pending uses).
        let bytes = unsafe { mapped.slice(len) };
        publish(bytes)?;
        return Ok(None);
    }
    let _order = lock(&ctx.submit_lock);
    let event = ctx
        .transfers
        .download(&replica.transfer_view(), 0, replica.size(), None, publish)?;
    // The copy reads the replica on the device; a later writer must wait for it.
    let mut state = replica.state();
    if state.access == DeviceAccess::None {
        state.access = DeviceAccess::Read;
    }
    Ok(Some(event))
}

/// Copies `size` bytes at `offset` of `buffer`'s replica on `ctx` to `sink` without touching the
/// host shadow. Call only after the work that produced the replica completed.
pub(crate) fn read_replica_range(
    ctx: &Arc<DeviceContext>,
    buffer: &BufferHandle,
    offset: u64,
    size: u64,
    sink: impl FnOnce(&[u8]) -> Status + Send + 'static,
) -> Result<Event> {
    let replica = buffer
        .replica::<Replica>(ctx.key)
        .ok_or_else(|| Error::new(ErrorCode::InvalidArgument, "buffer has no replica on this device"))?;
    if size == 0 {
        sink(&[])?;
        return Ok(Event::ready(Ok(())));
    }
    let _order = lock(&ctx.submit_lock);
    let event = ctx
        .transfers
        .download(&replica.transfer_view(), offset, size, None, sink)?;
    let mut state = replica.state();
    if state.access == DeviceAccess::None {
        state.access = DeviceAccess::Read;
    }
    Ok(event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceLimits;

    fn limits() -> DeviceLimits {
        DeviceLimits {
            max_compute_work_group_count: [65_535, 65_535, 65_535],
            max_compute_work_group_size: [1024, 1024, 64],
            max_compute_work_group_invocations: 1024,
            ..DeviceLimits::default()
        }
    }

    #[test]
    fn group_counts_round_up_and_respect_limits() {
        assert_eq!(
            group_counts([257, 1, 1], [64, 1, 1], &limits()).unwrap(),
            [5, 1, 1]
        );
        assert_eq!(
            group_counts([64, 64, 3], [8, 8, 1], &limits()).unwrap(),
            [8, 8, 3]
        );
        let code = |global, local| group_counts(global, local, &limits()).unwrap_err().code();
        assert_eq!(code([0, 1, 1], [64, 1, 1]), ErrorCode::InvalidDispatch);
        assert_eq!(code([1, 1, 1], [2048, 1, 1]), ErrorCode::InvalidDispatch);
        assert_eq!(code([1, 1, 1], [64, 64, 1]), ErrorCode::InvalidDispatch);
        assert_eq!(
            code([65_536 * 64 + 1, 1, 1], [64, 1, 1]),
            ErrorCode::InvalidDispatch
        );
        assert_eq!(code([u64::MAX, 1, 1], [1, 1, 1]), ErrorCode::InvalidDispatch);
        assert_eq!(code([1, 1, 1], [0, 1, 1]), ErrorCode::InvalidDispatch);
    }
}
