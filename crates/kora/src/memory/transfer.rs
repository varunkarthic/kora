//! Asynchronous upload, download, copy, and fill over a [`QueueContext`].
//!
//! Discrete memory stages through persistent rings; unified memory with host-visible buffers
//! reads and writes the mapping directly. Every transfer records precise hazard barriers, holds its
//! staging bytes and buffer keep-alives until queue completion, and publishes its status through an
//! [`Event`]. Downloads hand the mapped bytes to a completion closure, so no caller-owned output
//! buffer must stay borrowed across the asynchronous operation.

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ash::vk;

use super::pool::MappedPtr;
use super::staging::{SharedPool, StagingDirection, StagingOptions, StagingRing, StagingSlice};
use crate::checked;
use crate::device::MemoryTopology;
use crate::error::{Error, ErrorCode, Result, Status};
use crate::event::{Event, EventPromise};
use crate::vulkan::{
    AccessKind, CommandPoolSet, Device, HazardTracker, QueueContext, QueueWait, Submission,
    record_transition, vk_error,
};

/// Per-buffer queue bookkeeping shared by every [`TransferBuffer`] view of one runtime buffer.
#[derive(Debug, Default)]
pub struct QueueTimelineUse {
    /// Last transfer-queue timeline value that touched a cross-queue buffer.
    pub transfer_value: AtomicU64,
    /// GPU operations queued or executing on the buffer; host code may touch the mapping directly
    /// only while this is zero.
    pub pending: AtomicU32,
    /// The last writer ended with a shader-write to host-read barrier.
    pub host_readable: AtomicBool,
}

/// Raises [`QueueTimelineUse::pending`] for its buffers until dropped.
#[derive(Debug)]
pub struct PendingUses(Vec<Arc<QueueTimelineUse>>);

impl PendingUses {
    /// Marks `uses` as pending.
    pub fn new(uses: Vec<Arc<QueueTimelineUse>>) -> Self {
        for use_ in &uses {
            use_.pending.fetch_add(1, Ordering::AcqRel);
        }
        Self(uses)
    }
}

impl Drop for PendingUses {
    fn drop(&mut self) {
        for use_ in &self.0 {
            use_.pending.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// A Vulkan buffer view used by transfers.
#[derive(Clone, Default)]
pub struct TransferBuffer {
    /// Buffer.
    pub buffer: vk::Buffer,
    /// Size.
    pub size: u64,
    /// Backing memory (for non-coherent maintenance).
    pub memory: vk::DeviceMemory,
    /// Offset of byte 0 in `memory`.
    pub memory_offset: u64,
    /// Size of `memory`.
    pub memory_allocation_size: u64,
    /// Mapped address of byte 0 (host-visible buffers).
    pub mapped: Option<MappedPtr>,
    /// Host-visible.
    pub host_visible: bool,
    /// Host-coherent.
    pub host_coherent: bool,
    /// Shared queue bookkeeping.
    pub queue_use: Option<Arc<QueueTimelineUse>>,
    /// Kept alive until the transfer completes (the owner of `buffer` and its memory).
    pub keepalive: Option<Arc<dyn Any + Send + Sync>>,
}

impl fmt::Debug for TransferBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransferBuffer")
            .field("buffer", &self.buffer)
            .field("size", &self.size)
            .field("host_visible", &self.host_visible)
            .finish()
    }
}

/// Flushes or invalidates `[offset, offset + size)` of a mapped non-coherent buffer (no-op when
/// coherent).
pub fn maintain_mapped_range(
    device: &Device,
    buffer: &TransferBuffer,
    offset: u64,
    size: u64,
    atom: u64,
    flush: bool,
) -> Status {
    if buffer.host_coherent || size == 0 {
        return Ok(());
    }
    if !buffer.host_visible
        || buffer.memory == vk::DeviceMemory::null()
        || buffer.memory_allocation_size == 0
        || !checked::is_pow2(atom)
    {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "non-coherent transfer memory metadata is incomplete",
        ));
    }
    let overflow = || Error::new(ErrorCode::Overflow, "mapped transfer range overflowed");
    let relative_end = offset
        .checked_add(size)
        .filter(|&end| end <= buffer.size)
        .ok_or_else(overflow)?;
    let begin = buffer.memory_offset.checked_add(offset).ok_or_else(overflow)?;
    let end = buffer
        .memory_offset
        .checked_add(relative_end)
        .ok_or_else(overflow)?;
    if end > buffer.memory_allocation_size {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "mapped transfer range exceeds its memory allocation",
        ));
    }
    let start = checked::align_down(begin, atom).ok_or_else(overflow)?;
    let rounded_end = checked::align_up(end, atom).ok_or_else(overflow)?;
    let range_size = if rounded_end > buffer.memory_allocation_size {
        vk::WHOLE_SIZE
    } else {
        rounded_end - start
    };
    let range = [vk::MappedMemoryRange::default()
        .memory(buffer.memory)
        .offset(start)
        .size(range_size)];
    use super::dirty::MappedMemoryOps;
    let result = if flush {
        device.flush_ranges(&range)
    } else {
        device.invalidate_ranges(&range)
    };
    result.map_err(|r| {
        vk_error(
            ErrorCode::TransferFailed,
            if flush {
                "Flushing mapped upload bytes failed"
            } else {
                "Invalidating mapped readback bytes failed"
            },
            r,
        )
    })
}

/// Engine configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferOptions {
    /// Queue family of the submitting queue.
    pub queue_family: u32,
    /// Command buffer ring size.
    pub command_buffer_count: usize,
    /// Initial staging capacity per direction.
    pub staging_initial_capacity: u64,
    /// Maximum staging capacity per direction.
    pub staging_maximum_capacity: u64,
    /// Device memory topology.
    pub topology: MemoryTopology,
    /// Staging measurably helps on this device.
    pub measured_staging_benefit: bool,
    /// `nonCoherentAtomSize`.
    pub non_coherent_atom_size: u64,
    /// Transfers run on a separate queue family from compute (concurrent sharing).
    pub cross_queue_concurrent: bool,
    /// How long a producer waits for pending staging reservations to be submitted.
    pub staging_submission_timeout: Duration,
}

impl Default for TransferOptions {
    fn default() -> Self {
        Self {
            queue_family: 0,
            command_buffer_count: 32,
            staging_initial_capacity: 4 << 20,
            staging_maximum_capacity: 128 << 20,
            topology: MemoryTopology::Unknown,
            measured_staging_benefit: false,
            non_coherent_atom_size: 1,
            cross_queue_concurrent: false,
            staging_submission_timeout: Duration::from_secs(30),
        }
    }
}

type Record = Box<dyn FnOnce(&Device, vk::CommandBuffer) -> Status + Send>;
type DownloadSink = Box<dyn FnOnce(&[u8]) -> Status + Send>;
type Complete = Box<dyn FnOnce(&Device, Option<&StagingSlice>, Option<&StagingRing>) -> Status + Send>;

struct Inner {
    device: Device,
    queue: Arc<QueueContext<Device>>,
    options: TransferOptions,
    upload_ring: Option<StagingRing>,
    readback_ring: Option<StagingRing>,
    commands: Mutex<Option<CommandPoolSet<Device>>>,
}

/// Asynchronous transfer engine.
#[derive(Clone)]
pub struct TransferEngine {
    inner: Arc<Inner>,
}

impl fmt::Debug for TransferEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransferEngine")
            .field("staged", &self.inner.upload_ring.is_some())
            .finish()
    }
}

struct TransferJob {
    record: Record,
    complete: Option<Complete>,
    slice: Option<(StagingSlice, StagingRing)>,
    dependency: Option<QueueWait>,
    on_submitted: Option<Box<dyn FnOnce(u64) + Send>>,
    uses: Vec<Option<Arc<QueueTimelineUse>>>,
    keepalive: Vec<Option<Arc<dyn Any + Send + Sync>>>,
}

impl TransferJob {
    fn new(record: Record, buffers: &[&TransferBuffer]) -> Self {
        Self {
            record,
            complete: None,
            slice: None,
            dependency: None,
            on_submitted: None,
            uses: buffers.iter().map(|b| b.queue_use.clone()).collect(),
            keepalive: buffers.iter().map(|b| b.keepalive.clone()).collect(),
        }
    }
}

struct Work {
    slice: Option<StagingSlice>,
    command_buffer: vk::CommandBuffer,
}

fn ready_event() -> Event {
    Event::ready(Ok(()))
}

impl TransferEngine {
    /// Creates an engine on `queue`. Staging rings are created when the topology (or measured
    /// benefit) selects staging.
    pub fn new(
        device: Device,
        queue: Arc<QueueContext<Device>>,
        upload_pool: SharedPool,
        readback_pool: SharedPool,
        options: TransferOptions,
    ) -> Result<TransferEngine> {
        if options.command_buffer_count == 0
            || options.staging_initial_capacity == 0
            || options.staging_maximum_capacity < options.staging_initial_capacity
            || !checked::is_pow2(options.non_coherent_atom_size)
        {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "TransferEngine ring size or atom size is invalid",
            ));
        }
        let (upload_ring, readback_ring) =
            if StagingRing::should_use_staging(options.topology, options.measured_staging_benefit) {
                let upload = StagingOptions {
                    direction: StagingDirection::Upload,
                    initial_capacity: options.staging_initial_capacity,
                    maximum_capacity: options.staging_maximum_capacity,
                    topology: options.topology,
                    measured_staging_benefit: options.measured_staging_benefit,
                    non_coherent_atom_size: options.non_coherent_atom_size,
                    ..Default::default()
                };
                let readback = StagingOptions {
                    direction: StagingDirection::Readback,
                    ..upload
                };
                (
                    Some(StagingRing::new(device.clone(), upload_pool, upload)?),
                    Some(StagingRing::new(device.clone(), readback_pool, readback)?),
                )
            } else {
                (None, None)
            };
        Ok(TransferEngine {
            inner: Arc::new(Inner {
                device,
                queue,
                options,
                upload_ring,
                readback_ring,
                commands: Mutex::new(None),
            }),
        })
    }

    /// Whether transfers stage through rings.
    pub fn uses_staging(&self) -> bool {
        self.inner.upload_ring.is_some()
    }

    /// The submitting queue.
    pub fn queue(&self) -> &Arc<QueueContext<Device>> {
        &self.inner.queue
    }

    /// Largest single staged transfer: a quarter of the ring so several stay in flight.
    fn staging_chunk(&self) -> u64 {
        (self.inner.options.staging_maximum_capacity / 4).max(1 << 16)
    }

    fn reserve(&self, ring: &StagingRing, size: u64) -> Result<StagingSlice> {
        let queue = &self.inner.queue;
        let completed = queue.completed_value()?;
        ring.reclaim_completed(completed);
        let mut wait = |value: u64| -> Result<u64> {
            queue.wait_for_value(value, u64::MAX)?;
            queue.completed_value()
        };
        ring.reserve_blocking(
            size,
            1,
            completed,
            Some(&mut wait),
            self.inner.options.staging_submission_timeout,
        )
    }

    fn enqueue(&self, job: TransferJob) -> Result<Event> {
        let TransferJob {
            record,
            complete,
            slice,
            dependency,
            on_submitted,
            uses,
            keepalive,
        } = job;
        let inner = Arc::clone(&self.inner);
        let (slice, ring) = match slice {
            Some((slice, ring)) => (Some(slice), Some(ring)),
            None => (None, None),
        };
        let work = Arc::new(Mutex::new(Work {
            slice,
            command_buffer: vk::CommandBuffer::null(),
        }));
        let promise = EventPromise::new();
        let event = promise.event();
        let pending = PendingUses::new(uses.into_iter().flatten().collect());

        let record_inner = Arc::clone(&inner);
        let record_work = Arc::clone(&work);
        let submit_inner = Arc::clone(&inner);
        let submit_work = Arc::clone(&work);
        let submit_ring = ring.clone();
        let complete_inner = Arc::clone(&inner);
        let submission = Submission {
            waits: dependency.into_iter().collect(),
            record: Some(Box::new(move |_, commands| {
                let inner = record_inner;
                let mut pool_slot = inner.commands.lock().unwrap_or_else(|p| p.into_inner());
                if pool_slot.is_none() {
                    *pool_slot = Some(CommandPoolSet::new(
                        inner.device.clone(),
                        inner.options.queue_family,
                        inner.options.command_buffer_count,
                    )?);
                }
                let pool = pool_slot.as_mut().expect("created above");
                let mut completed = inner.queue.completed_value()?;
                let command = loop {
                    match pool.acquire(completed) {
                        Ok(command) => break command,
                        Err(error)
                            if error.code() == ErrorCode::OutOfMemoryBudget
                                && pool.oldest_pending_value() != 0 =>
                        {
                            inner
                                .queue
                                .wait_for_value(pool.oldest_pending_value(), u64::MAX)?;
                            completed = inner.queue.completed_value()?;
                        }
                        Err(error) => return Err(error),
                    }
                };
                record_work
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .command_buffer = command;
                if let Err(error) = record(&inner.device, command) {
                    let _ = pool.discard(command);
                    record_work
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .command_buffer = vk::CommandBuffer::null();
                    return Err(error);
                }
                pool.end_recording(command)?;
                commands.push(command);
                Ok(())
            })),
            on_submitted: Some(Box::new(move |timeline, status| {
                let mut work = submit_work.lock().unwrap_or_else(|p| p.into_inner());
                let command = std::mem::replace(&mut work.command_buffer, vk::CommandBuffer::null());
                let mut pool_slot = submit_inner.commands.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(pool) = pool_slot
                    .as_mut()
                    .filter(|_| command != vk::CommandBuffer::null())
                {
                    let _ = if status.is_ok() {
                        pool.submitted(command, timeline)
                    } else {
                        pool.discard(command)
                    };
                }
                drop(pool_slot);
                match (status, work.slice.as_ref(), submit_ring.as_ref()) {
                    (Ok(()), Some(slice), Some(ring)) => drop(ring.retire(slice, timeline)),
                    (Err(_), Some(_), Some(ring)) => {
                        if let Some(slice) = work.slice.take() {
                            let _ = ring.cancel(slice);
                        }
                    }
                    _ => {}
                }
                if status.is_ok()
                    && let Some(hook) = on_submitted
                {
                    hook(timeline);
                }
            })),
            on_complete: Some(Box::new(move |_, status| {
                let _pending = pending;
                let _keepalive = keepalive;
                let slice = work.lock().unwrap_or_else(|p| p.into_inner()).slice.take();
                let status = match (status, complete) {
                    (Ok(()), Some(complete)) => {
                        complete(&complete_inner.device, slice.as_ref(), ring.as_ref())
                    }
                    (status, _) => status,
                };
                // Dropping the slice releases (or makes reclaimable) its staging range.
                drop(slice);
                promise.complete(status);
            })),
            ..Default::default()
        };
        self.inner.queue.submit(submission)?;
        Ok(event)
    }

    /// Uploads `source` to `destination[offset..]`. Bytes are copied before this returns.
    pub fn upload(&self, destination: &TransferBuffer, offset: u64, source: &[u8]) -> Result<Event> {
        let size = source.len() as u64;
        if destination.buffer == vk::Buffer::null() || !checked::in_range(offset, size, destination.size) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "upload destination range is invalid",
            ));
        }
        if source.is_empty() {
            return Ok(ready_event());
        }
        let inner = &self.inner;
        let chunk = self.staging_chunk();
        if inner.upload_ring.is_some() && size > chunk {
            // Larger than a ring can hold: pipeline ring-sized pieces.
            let events = source
                .chunks(chunk as usize)
                .enumerate()
                .map(|(index, piece)| self.upload(destination, offset + index as u64 * chunk, piece))
                .collect::<Result<Vec<_>>>()?;
            return Event::when_all(&events);
        }
        let cross_queue = inner.options.cross_queue_concurrent;
        let direct = inner.upload_ring.is_none() && destination.host_visible && destination.mapped.is_some();
        let staged = if direct {
            // SAFETY: the destination mapping covers `offset + size` (checked above) and the caller
            // guarantees no device work accesses the range until this transfer orders after it.
            unsafe {
                destination
                    .mapped
                    .expect("direct path")
                    .add(offset)
                    .slice_mut(source.len())
                    .copy_from_slice(source)
            };
            maintain_mapped_range(
                &inner.device,
                destination,
                offset,
                size,
                inner.options.non_coherent_atom_size,
                true,
            )?;
            None
        } else {
            let ring = inner.upload_ring.as_ref().ok_or_else(|| {
                Error::new(
                    ErrorCode::UnsupportedFeature,
                    "transfer policy selected direct access but this buffer has no host-visible mapping",
                )
            })?;
            let mut slice = self.reserve(ring, size)?;
            // SAFETY: the slice is reserved and unsubmitted, so no device work touches it.
            unsafe { slice.bytes_mut().copy_from_slice(source) };
            ring.flush(&slice)?;
            Some((slice, ring.clone()))
        };
        let (staging_buffer, staging_offset) = staged
            .as_ref()
            .map_or((vk::Buffer::null(), 0), |(s, _)| (s.buffer, s.offset));
        let target = destination.buffer;
        let record: Record = Box::new(move |device, command| {
            // SAFETY: `command` is recording on the submission thread; every buffer outlives the
            // transfer through the keep-alive and staging-slice ownership held by the submission.
            unsafe {
                if direct {
                    let mut hazards = HazardTracker::new();
                    hazards.access(AccessKind::HostWrite);
                    if cross_queue {
                        record_transition(device, command, &mut hazards, AccessKind::TransferRead);
                    } else {
                        record_transition(device, command, &mut hazards, AccessKind::ComputeRead);
                        record_transition(device, command, &mut hazards, AccessKind::ComputeWrite);
                    }
                    return Ok(());
                }
                let mut staging = HazardTracker::new();
                staging.access(AccessKind::HostWrite);
                record_transition(device, command, &mut staging, AccessKind::TransferRead);
                let mut dest = HazardTracker::new();
                if cross_queue {
                    dest.access(AccessKind::TransferWrite);
                } else {
                    dest.access(AccessKind::ComputeRead);
                    dest.access(AccessKind::ComputeWrite);
                }
                record_transition(device, command, &mut dest, AccessKind::TransferWrite);
                let region = [vk::BufferCopy {
                    src_offset: staging_offset,
                    dst_offset: offset,
                    size,
                }];
                device
                    .raw()
                    .cmd_copy_buffer(command, staging_buffer, target, &region);
                if !cross_queue {
                    record_transition(device, command, &mut dest, AccessKind::ComputeRead);
                    record_transition(device, command, &mut dest, AccessKind::ComputeWrite);
                }
            }
            Ok(())
        });
        let hook = destination
            .queue_use
            .clone()
            .map(|use_| -> Box<dyn FnOnce(u64) + Send> {
                Box::new(move |value| use_.transfer_value.store(value, Ordering::Release))
            });
        self.enqueue(TransferJob {
            slice: staged,
            on_submitted: hook,
            ..TransferJob::new(record, &[destination])
        })
    }

    /// Downloads `source[offset..offset + size]` and calls `sink` with the bytes on the completion
    /// thread before the returned event completes. An optional `dependency` orders the copy after
    /// another queue's work.
    pub fn download(
        &self,
        source: &TransferBuffer,
        offset: u64,
        size: u64,
        dependency: Option<QueueWait>,
        sink: impl FnOnce(&[u8]) -> Status + Send + 'static,
    ) -> Result<Event> {
        self.download_boxed(source, offset, size, dependency, Box::new(sink))
    }

    fn download_boxed(
        &self,
        source: &TransferBuffer,
        offset: u64,
        size: u64,
        dependency: Option<QueueWait>,
        sink: DownloadSink,
    ) -> Result<Event> {
        if source.buffer == vk::Buffer::null() || !checked::in_range(offset, size, source.size) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "download source range is invalid",
            ));
        }
        let length = checked::to_usize(size, "download size")?;
        if size == 0 {
            sink(&[])?;
            return Ok(ready_event());
        }
        let inner = &self.inner;
        let chunk = self.staging_chunk();
        if inner.readback_ring.is_some() && size > chunk {
            // Larger than a ring can hold: gather ring-sized pieces, then hand over all bytes once.
            let gathered = Arc::new(Mutex::new(vec![0u8; length]));
            let mut events = Vec::new();
            let mut start = 0u64;
            while start < size {
                let piece = chunk.min(size - start);
                let target = Arc::clone(&gathered);
                let at = start as usize;
                events.push(self.download_boxed(
                    source,
                    offset + start,
                    piece,
                    dependency,
                    Box::new(move |bytes| {
                        target.lock().unwrap_or_else(|p| p.into_inner())[at..at + bytes.len()]
                            .copy_from_slice(bytes);
                        Ok(())
                    }),
                )?);
                start += piece;
            }
            return Event::when_all(&events)?.then(move |status| {
                status?;
                let bytes = gathered.lock().unwrap_or_else(|p| p.into_inner());
                sink(&bytes)
            });
        }
        let cross_queue = inner.options.cross_queue_concurrent;
        let direct = inner.readback_ring.is_none() && source.host_visible && source.mapped.is_some();
        let staged = if direct {
            None
        } else {
            let ring = inner.readback_ring.as_ref().ok_or_else(|| {
                Error::new(
                    ErrorCode::UnsupportedFeature,
                    "transfer policy selected direct access but this source has no host-visible mapping",
                )
            })?;
            Some((self.reserve(ring, size)?, ring.clone()))
        };
        let (staging_buffer, staging_offset) = staged
            .as_ref()
            .map_or((vk::Buffer::null(), 0), |(s, _)| (s.buffer, s.offset));
        let source_buffer = source.buffer;
        let record: Record = Box::new(move |device, command| {
            // SAFETY: as for uploads.
            unsafe {
                if direct {
                    if !cross_queue {
                        let mut hazards = HazardTracker::new();
                        hazards.access(AccessKind::ComputeWrite);
                        record_transition(device, command, &mut hazards, AccessKind::HostRead);
                    }
                    return Ok(());
                }
                if !cross_queue {
                    let mut hazards = HazardTracker::new();
                    hazards.access(AccessKind::ComputeWrite);
                    record_transition(device, command, &mut hazards, AccessKind::TransferRead);
                }
                let mut staging = HazardTracker::new();
                staging.access(AccessKind::HostRead);
                record_transition(device, command, &mut staging, AccessKind::TransferWrite);
                let region = [vk::BufferCopy {
                    src_offset: offset,
                    dst_offset: staging_offset,
                    size,
                }];
                device
                    .raw()
                    .cmd_copy_buffer(command, source_buffer, staging_buffer, &region);
                record_transition(device, command, &mut staging, AccessKind::HostRead);
            }
            Ok(())
        });
        let view = source.clone();
        let atom = inner.options.non_coherent_atom_size;
        let complete: Complete = Box::new(move |device, slice, ring| {
            match (slice, ring) {
                (Some(slice), Some(ring)) => {
                    ring.invalidate(slice)?;
                    // SAFETY: the queue timeline completed and the range was invalidated, so the
                    // device no longer writes these bytes.
                    sink(unsafe { slice.bytes() })
                }
                _ => {
                    maintain_mapped_range(device, &view, offset, size, atom, false)?;
                    // SAFETY: the copy's completion ordered the device writes before this host read
                    // (barrier to HOST_READ plus timeline completion); the mapping covers the range.
                    sink(unsafe { view.mapped.expect("direct path").add(offset).slice(length) })
                }
            }
        });
        self.enqueue(TransferJob {
            complete: Some(complete),
            slice: staged,
            dependency,
            ..TransferJob::new(record, &[source])
        })
    }

    /// Copies `size` bytes between device buffers (same-buffer ranges must not overlap).
    pub fn copy(
        &self,
        source: &TransferBuffer,
        source_offset: u64,
        destination: &TransferBuffer,
        destination_offset: u64,
        size: u64,
    ) -> Result<Event> {
        for (buffer, offset, what) in [
            (source, source_offset, "device-copy source"),
            (destination, destination_offset, "device-copy destination"),
        ] {
            if buffer.buffer == vk::Buffer::null() {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    "transfer requires a valid Vulkan buffer",
                ));
            }
            checked::range_end(offset, size, buffer.size, what)?;
        }
        if size == 0 {
            return Ok(ready_event());
        }
        if source.buffer == destination.buffer
            && source_offset < destination_offset + size
            && destination_offset < source_offset + size
        {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "overlapping same-buffer copy ranges are not supported",
            ));
        }
        let (from, to) = (source.buffer, destination.buffer);
        let record: Record = Box::new(move |device, command| {
            // SAFETY: as for uploads.
            unsafe {
                let mut src = HazardTracker::new();
                src.access(AccessKind::ComputeWrite);
                record_transition(device, command, &mut src, AccessKind::TransferRead);
                let mut dst = HazardTracker::new();
                dst.access(AccessKind::ComputeWrite);
                record_transition(device, command, &mut dst, AccessKind::TransferWrite);
                device.raw().cmd_copy_buffer(
                    command,
                    from,
                    to,
                    &[vk::BufferCopy {
                        src_offset: source_offset,
                        dst_offset: destination_offset,
                        size,
                    }],
                );
                let mut src_after = HazardTracker::new();
                src_after.access(AccessKind::TransferRead);
                record_transition(device, command, &mut src_after, AccessKind::ComputeRead);
                record_transition(device, command, &mut src_after, AccessKind::ComputeWrite);
                let mut dst_after = HazardTracker::new();
                dst_after.access(AccessKind::TransferWrite);
                record_transition(device, command, &mut dst_after, AccessKind::ComputeRead);
                record_transition(device, command, &mut dst_after, AccessKind::ComputeWrite);
            }
            Ok(())
        });
        self.enqueue(TransferJob::new(record, &[source, destination]))
    }

    /// Fills `[offset, offset + size)` with a repeated 32-bit `value` (4-byte aligned).
    pub fn fill(&self, destination: &TransferBuffer, offset: u64, size: u64, value: u32) -> Result<Event> {
        if destination.buffer == vk::Buffer::null() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "transfer requires a valid Vulkan buffer",
            ));
        }
        checked::range_end(offset, size, destination.size, "buffer fill")?;
        if size == 0 || !offset.is_multiple_of(4) || !size.is_multiple_of(4) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "vkCmdFillBuffer requires a nonzero size and 4-byte alignment",
            ));
        }
        let target = destination.buffer;
        let record: Record = Box::new(move |device, command| {
            // SAFETY: as for uploads.
            unsafe {
                let mut hazards = HazardTracker::new();
                hazards.access(AccessKind::ComputeWrite);
                record_transition(device, command, &mut hazards, AccessKind::TransferWrite);
                device.raw().cmd_fill_buffer(command, target, offset, size, value);
                record_transition(device, command, &mut hazards, AccessKind::ComputeRead);
                record_transition(device, command, &mut hazards, AccessKind::ComputeWrite);
            }
            Ok(())
        });
        self.enqueue(TransferJob::new(record, &[destination]))
    }
}
