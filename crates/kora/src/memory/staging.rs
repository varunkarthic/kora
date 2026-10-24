//! Timeline-safe staging rings over persistently mapped transfer buffers.
//!
//! A reserved range stays owned until it is cancelled, or until its release timeline value has
//! completed (and, for readback, its bytes were consumed). The ring reclaims completed ranges,
//! grows within its maximum capacity, or waits for the oldest submitted range; it never overwrites
//! in-flight bytes. Slices keep the ring's memory alive, so a mapped slice can never dangle.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ash::vk;

use super::dirty::MappedMemoryOps;
use super::pool::{MappedPtr, MemoryPool, PoolAllocation, PoolAllocationRequest};
use crate::checked;
use crate::device::MemoryTopology;
use crate::error::{Error, ErrorCode, Result, Status};
use crate::vulkan::{Device, vk_error};

/// A memory pool shared between owners.
pub type SharedPool = Arc<Mutex<MemoryPool>>;

/// Transfer direction of a ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StagingDirection {
    /// Host to device.
    #[default]
    Upload,
    /// Device to host.
    Readback,
}

/// Ring configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagingOptions {
    /// Direction.
    pub direction: StagingDirection,
    /// First block size.
    pub initial_capacity: u64,
    /// Total capacity limit.
    pub maximum_capacity: u64,
    /// Extra buffer usage flags.
    pub additional_usage: vk::BufferUsageFlags,
    /// Device memory topology.
    pub topology: MemoryTopology,
    /// Staging was measured to help on this device.
    pub measured_staging_benefit: bool,
    /// `nonCoherentAtomSize`.
    pub non_coherent_atom_size: u64,
}

impl Default for StagingOptions {
    fn default() -> Self {
        Self {
            direction: StagingDirection::Upload,
            initial_capacity: 4 << 20,
            maximum_capacity: 64 << 20,
            additional_usage: vk::BufferUsageFlags::empty(),
            topology: MemoryTopology::Unknown,
            measured_staging_benefit: false,
            non_coherent_atom_size: 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Segment {
    offset: u64,
    size: u64,
    reserved_size: u64,
    release_timeline: Option<u64>,
    /// Readback bytes not yet copied to the host; never reclaimed until consumed.
    host_pending: bool,
}

struct Block {
    buffer: vk::Buffer,
    allocation: PoolAllocation,
    mapped: MappedPtr,
    capacity: u64,
    cursor: u64,
    segments: BTreeMap<u64, Segment>,
}

/// Hashes sequential reservation IDs with one multiply (SipHash dominated reservation cost).
#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0 ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    fn write_u64(&mut self, value: u64) {
        self.0 = value.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

#[derive(Default)]
struct RingState {
    blocks: Vec<Block>,
    next_reservation: u64,
    capacity: u64,
    /// Reservation ID -> (block, offset).
    ids: HashMap<u64, (usize, u64), BuildHasherDefault<IdHasher>>,
    /// Submitted segments by release timeline (lazily cleaned).
    pending: BinaryHeap<Reverse<(u64, u64)>>,
}

struct RingInner {
    device: Device,
    pool: SharedPool,
    options: StagingOptions,
    host_coherent: bool,
    state: Mutex<RingState>,
    /// Signalled whenever a reservation is retired, cancelled, consumed, reclaimed, or dropped.
    changed: Condvar,
}

impl Drop for RingInner {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(|p| p.into_inner());
        let mut pool = self.pool.lock().unwrap_or_else(|p| p.into_inner());
        for block in state.blocks.drain(..) {
            // SAFETY: every slice holds an Arc to this ring, so no slice (and no recorded copy that
            // a live slice describes) can still reference the buffer when the ring drops.
            unsafe { self.device.raw().destroy_buffer(block.buffer, None) };
            let _ = pool.deallocate(&block.allocation);
        }
    }
}

/// A shareable staging ring.
#[derive(Clone)]
pub struct StagingRing {
    inner: Arc<RingInner>,
}

impl fmt::Debug for StagingRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StagingRing")
            .field("direction", &self.inner.options.direction)
            .field("capacity", &self.capacity())
            .finish()
    }
}

/// An exclusively owned staging range.
pub struct StagingSlice {
    ring: Arc<RingInner>,
    /// Reservation ID.
    pub reservation: u64,
    /// Staging buffer.
    pub buffer: vk::Buffer,
    /// Backing memory.
    pub memory: vk::DeviceMemory,
    /// Offset in `buffer`.
    pub offset: u64,
    /// Requested size.
    pub size: u64,
    /// Reserved size (atom aligned).
    pub reserved_size: u64,
    /// Offset in `memory`.
    pub memory_offset: u64,
    mapped: MappedPtr,
}

impl fmt::Debug for StagingSlice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StagingSlice")
            .field("reservation", &self.reservation)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .finish()
    }
}

impl Drop for StagingSlice {
    fn drop(&mut self) {
        // An abandoned unsubmitted reservation is released; a dropped readback slice no longer
        // needs its bytes, so it becomes reclaimable once its timeline completes.
        let mut state = lock(&self.ring.state);
        let Some(segment) = state.segment_mut(self.reservation) else {
            return;
        };
        match segment.release_timeline {
            None => state.remove(self.reservation),
            Some(timeline) if segment.host_pending => {
                segment.host_pending = false;
                state.pending.push(Reverse((timeline, self.reservation)));
            }
            Some(_) => return,
        }
        drop(state);
        self.ring.changed.notify_all();
    }
}

impl StagingSlice {
    /// The slice's mapped bytes.
    ///
    /// # Safety
    /// No device work may write this range concurrently (for readback: the release timeline has
    /// completed and the range was invalidated).
    pub unsafe fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping lives as long as the ring, which this slice keeps alive; the range is
        // exclusively reserved for this slice; device access is excluded by the caller.
        unsafe { self.mapped.slice(self.size as usize) }
    }

    /// The slice's mapped bytes, mutably.
    ///
    /// # Safety
    /// No device work may read or write this range concurrently (for upload: before submission).
    pub unsafe fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as for `bytes`; `&mut self` makes the host access exclusive.
        unsafe { self.mapped.slice_mut(self.size as usize) }
    }
}

fn lock(state: &Mutex<RingState>) -> MutexGuard<'_, RingState> {
    state.lock().unwrap_or_else(|p| p.into_inner())
}

fn try_gap(block: &Block, begin: u64, reserved: u64, alignment: u64) -> Option<u64> {
    let capacity = block.capacity;
    let mut candidate = checked::align_up(begin, alignment)?;
    if candidate > capacity || reserved > capacity - candidate {
        return None;
    }
    let start_key = block
        .segments
        .range(..=candidate)
        .next_back()
        .map_or(0, |(&key, _)| key);
    for segment in block.segments.range(start_key..).map(|(_, s)| s) {
        let (start, end) = (segment.offset, segment.offset + segment.reserved_size);
        if end <= candidate {
            continue;
        }
        if candidate <= start && reserved <= start - candidate {
            return Some(candidate);
        }
        if candidate < end {
            candidate = checked::align_up(end, alignment)?;
        }
    }
    (candidate <= capacity && reserved <= capacity - candidate).then_some(candidate)
}

impl RingState {
    fn segment(&self, reservation: u64) -> Option<(usize, &Segment)> {
        let &(block, offset) = self.ids.get(&reservation)?;
        Some((block, self.blocks[block].segments.get(&offset)?))
    }

    fn segment_mut(&mut self, reservation: u64) -> Option<&mut Segment> {
        let &(block, offset) = self.ids.get(&reservation)?;
        self.blocks[block].segments.get_mut(&offset)
    }

    fn remove(&mut self, reservation: u64) {
        if let Some((block, offset)) = self.ids.remove(&reservation) {
            self.blocks[block].segments.remove(&offset);
        }
    }

    /// Pops stale heap entries; returns the oldest reclaimable timeline.
    fn oldest_reclaimable(&mut self) -> Option<u64> {
        while let Some(&Reverse((timeline, id))) = self.pending.peek() {
            match self.segment(id) {
                Some((_, s)) if s.release_timeline == Some(timeline) && !s.host_pending => {
                    return Some(timeline);
                }
                _ => {
                    self.pending.pop();
                }
            }
        }
        None
    }

    fn reclaim(&mut self, completed: u64) {
        while let Some(timeline) = self.oldest_reclaimable() {
            if timeline > completed {
                break;
            }
            let Reverse((_, id)) = self.pending.pop().expect("peeked");
            self.remove(id);
        }
    }
}

impl StagingRing {
    /// `true` when transfers should stage through a ring on this topology: discrete memory
    /// stages; unified, BAR-like, and unknown memory stage only with measured benefit.
    pub fn should_use_staging(topology: MemoryTopology, measured_staging_benefit: bool) -> bool {
        measured_staging_benefit || topology == MemoryTopology::Discrete
    }

    /// Creates a ring with one initial block from a host-visible pool.
    pub fn new(device: Device, pool: SharedPool, options: StagingOptions) -> Result<StagingRing> {
        let (host_visible, host_coherent) = {
            let pool = pool.lock().unwrap_or_else(|p| p.into_inner());
            (pool.host_visible(), pool.host_coherent())
        };
        if !host_visible {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "StagingRing requires a host-visible memory pool",
            ));
        }
        if options.initial_capacity == 0
            || options.maximum_capacity < options.initial_capacity
            || !checked::is_pow2(options.non_coherent_atom_size)
        {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "StagingRing capacity limits are invalid",
            ));
        }
        if !Self::should_use_staging(options.topology, options.measured_staging_benefit) {
            return Err(Error::new(
                ErrorCode::UnsupportedFeature,
                "transfer policy selected direct host-visible access instead of staging",
            ));
        }
        let ring = StagingRing {
            inner: Arc::new(RingInner {
                device,
                pool,
                options,
                host_coherent,
                state: Mutex::new(RingState {
                    next_reservation: 1,
                    ..Default::default()
                }),
                changed: Condvar::new(),
            }),
        };
        {
            let mut state = lock(&ring.inner.state);
            ring.add_block(&mut state, options.initial_capacity)?;
        }
        Ok(ring)
    }

    fn add_block(&self, state: &mut RingState, capacity: u64) -> Status {
        let inner = &self.inner;
        let usage = inner.options.additional_usage
            | match inner.options.direction {
                StagingDirection::Upload => vk::BufferUsageFlags::TRANSFER_SRC,
                StagingDirection::Readback => vk::BufferUsageFlags::TRANSFER_DST,
            };
        let info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let raw = inner.device.raw();
        // SAFETY: valid create info on a live device.
        let buffer = unsafe { raw.create_buffer(&info, None) }.map_err(|r| {
            vk_error(
                ErrorCode::AllocationFailed,
                "vkCreateBuffer failed for staging ring",
                r,
            )
        })?;
        let destroy = || {
            // SAFETY: the buffer was created above and is not bound or used yet.
            unsafe { raw.destroy_buffer(buffer, None) }
        };
        // SAFETY: `buffer` is a live buffer of this device.
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let mut pool = inner.pool.lock().unwrap_or_else(|p| p.into_inner());
        let type_index = pool.memory_type_index();
        if requirements.size == 0
            || !checked::is_pow2(requirements.alignment)
            || type_index >= 32
            || requirements.memory_type_bits & (1 << type_index) == 0
        {
            destroy();
            return Err(Error::new(
                ErrorCode::UnsupportedFeature,
                "StagingRing pool memory type is incompatible with its Vulkan buffer",
            ));
        }
        let atom = inner.options.non_coherent_atom_size;
        let Some(storage) = checked::align_up(requirements.size, atom) else {
            destroy();
            return Err(Error::new(
                ErrorCode::Overflow,
                "StagingRing memory requirement alignment overflowed",
            ));
        };
        let allocation = match pool.allocate(&PoolAllocationRequest::new(
            storage,
            requirements.alignment.max(atom),
        )) {
            Ok(allocation) => allocation,
            Err(error) => {
                destroy();
                return Err(error);
            }
        };
        let Some(mapped) = allocation.mapped else {
            destroy();
            let _ = pool.deallocate(&allocation);
            return Err(Error::new(
                ErrorCode::InternalInvariantViolation,
                "host-visible staging allocation is not mapped",
            ));
        };
        // SAFETY: the allocation satisfies the buffer's size, alignment, and type requirements.
        if let Err(r) = unsafe { raw.bind_buffer_memory(buffer, allocation.memory, allocation.offset) } {
            destroy();
            let _ = pool.deallocate(&allocation);
            return Err(vk_error(
                ErrorCode::AllocationFailed,
                "vkBindBufferMemory failed for staging ring",
                r,
            ));
        }
        state.blocks.push(Block {
            buffer,
            allocation,
            mapped,
            capacity,
            cursor: 0,
            segments: BTreeMap::new(),
        });
        state.capacity += capacity;
        Ok(())
    }

    /// Reserves a non-overlapping range: reclaims completed ranges, grows within the maximum
    /// capacity, or calls `wait_for(oldest timeline)` and retries. Unsubmitted reservations are
    /// never waited on, so a ring full of them fails with `AllocationFailed`.
    pub fn reserve(
        &self,
        size: u64,
        alignment: u64,
        completed_timeline: u64,
        wait_for: Option<&mut dyn FnMut(u64) -> Result<u64>>,
    ) -> Result<StagingSlice> {
        self.reserve_with(size, alignment, completed_timeline, wait_for, None)
    }

    /// Like [`reserve`](Self::reserve), but when the ring is full only of reservations that are
    /// still waiting to be submitted (or consumed), waits up to `pending_timeout` for another
    /// thread (normally the queue's submission worker) to retire or release them. This lets
    /// producers pipeline staged transfers instead of serializing on each other.
    pub fn reserve_blocking(
        &self,
        size: u64,
        alignment: u64,
        completed_timeline: u64,
        wait_for: Option<&mut dyn FnMut(u64) -> Result<u64>>,
        pending_timeout: Duration,
    ) -> Result<StagingSlice> {
        self.reserve_with(
            size,
            alignment,
            completed_timeline,
            wait_for,
            Some(pending_timeout),
        )
    }

    fn reserve_with(
        &self,
        size: u64,
        alignment: u64,
        completed_timeline: u64,
        mut wait_for: Option<&mut dyn FnMut(u64) -> Result<u64>>,
        pending_timeout: Option<Duration>,
    ) -> Result<StagingSlice> {
        let pending_deadline = pending_timeout.map(|t| Instant::now() + t);
        if size == 0 || !checked::is_pow2(alignment) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "staging reservation requires nonzero size and power-of-two alignment",
            ));
        }
        let options = &self.inner.options;
        let atom = options.non_coherent_atom_size;
        let reserved = checked::align_up(size, atom)
            .ok_or_else(|| Error::new(ErrorCode::Overflow, "staging reservation alignment overflowed"))?;
        let alignment = alignment.max(atom);
        let mut state = lock(&self.inner.state);
        let mut completed = completed_timeline;
        loop {
            state.reclaim(completed);
            for index in 0..state.blocks.len() {
                let block = &state.blocks[index];
                if reserved > block.capacity {
                    continue;
                }
                let offset = try_gap(block, block.cursor, reserved, alignment).or_else(|| {
                    if block.cursor != 0 {
                        try_gap(block, 0, reserved, alignment)
                    } else {
                        None
                    }
                });
                let Some(offset) = offset else { continue };
                let id = state.next_reservation;
                state.next_reservation = id.checked_add(1).ok_or_else(|| {
                    Error::new(ErrorCode::Overflow, "staging reservation ID space is exhausted")
                })?;
                let block = &mut state.blocks[index];
                block.segments.insert(
                    offset,
                    Segment {
                        offset,
                        size,
                        reserved_size: reserved,
                        release_timeline: None,
                        host_pending: options.direction == StagingDirection::Readback,
                    },
                );
                block.cursor = (offset + reserved) % block.capacity;
                // SAFETY: `offset + reserved <= capacity`, inside the block's mapping.
                let mapped = unsafe { block.mapped.add(offset) };
                let slice = StagingSlice {
                    ring: Arc::clone(&self.inner),
                    reservation: id,
                    buffer: block.buffer,
                    memory: block.allocation.memory,
                    offset,
                    size,
                    reserved_size: reserved,
                    memory_offset: block.allocation.offset + offset,
                    mapped,
                };
                state.ids.insert(id, (index, offset));
                return Ok(slice);
            }
            if size <= options.maximum_capacity - state.capacity {
                let growth = options
                    .initial_capacity
                    .max(size)
                    .min(options.maximum_capacity - state.capacity);
                match self.add_block(&mut state, growth) {
                    Ok(()) => continue,
                    Err(error)
                        if !matches!(
                            error.code(),
                            ErrorCode::AllocationFailed | ErrorCode::OutOfMemoryBudget
                        ) =>
                    {
                        return Err(error);
                    }
                    Err(_) => {}
                }
            }
            let oldest = state.oldest_reclaimable();
            let (Some(oldest), Some(wait)) = (oldest, wait_for.as_mut()) else {
                if let Some(deadline) = pending_deadline
                    && !state.ids.is_empty()
                {
                    let now = Instant::now();
                    if now < deadline {
                        // Wait for the submission worker to retire (or a consumer to release) a
                        // reservation, then retry the whole search.
                        state = self
                            .inner
                            .changed
                            .wait_timeout(state, deadline - now)
                            .unwrap_or_else(|p| p.into_inner())
                            .0;
                        continue;
                    }
                }
                return Err(Error::new(
                    ErrorCode::AllocationFailed,
                    "staging ring is full with unsubmitted or unconsumed reservations, or has no wait callback",
                ));
            };
            drop(state);
            let waited = wait(oldest)?;
            state = lock(&self.inner.state);
            completed = completed.max(waited);
        }
    }

    fn check_owner(&self, slice: &StagingSlice) -> Status {
        if !Arc::ptr_eq(&self.inner, &slice.ring) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "staging slice belongs to a different ring",
            ));
        }
        Ok(())
    }

    /// Marks a reservation as submitted; it is reclaimed once `release_timeline` completes.
    pub fn retire(&self, slice: &StagingSlice, release_timeline: u64) -> Status {
        self.check_owner(slice)?;
        if release_timeline == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "release timeline value must be nonzero",
            ));
        }
        let mut state = lock(&self.inner.state);
        let segment = state.segment_mut(slice.reservation).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidArgument,
                "staging reservation is unknown or was already reclaimed",
            )
        })?;
        if segment.release_timeline.is_some() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "staging reservation was already retired",
            ));
        }
        segment.release_timeline = Some(release_timeline);
        state.pending.push(Reverse((release_timeline, slice.reservation)));
        drop(state);
        self.inner.changed.notify_all();
        Ok(())
    }

    /// Marks a readback slice's bytes as copied to the host, allowing reclamation.
    pub fn consumed(&self, slice: &StagingSlice) -> Status {
        self.check_owner(slice)?;
        let mut state = lock(&self.inner.state);
        let segment = state.segment_mut(slice.reservation).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidArgument,
                "staging reservation is unknown or already released",
            )
        })?;
        segment.host_pending = false;
        if let Some(timeline) = segment.release_timeline {
            state.pending.push(Reverse((timeline, slice.reservation)));
        }
        drop(state);
        self.inner.changed.notify_all();
        Ok(())
    }

    /// Releases an unsubmitted reservation immediately.
    pub fn cancel(&self, slice: StagingSlice) -> Status {
        self.check_owner(&slice)?;
        let mut state = lock(&self.inner.state);
        let segment = state.segment(slice.reservation).map(|(_, s)| *s).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidArgument,
                "staging reservation is unknown or already released",
            )
        })?;
        if segment.release_timeline.is_some() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "submitted staging reservation cannot be cancelled",
            ));
        }
        if let Some(&(index, _)) = state.ids.get(&slice.reservation) {
            // Cancelling the newest reservation hands its space straight back: the cursor is only
            // a search hint, so rewinding it never allows an overlap.
            let block = &mut state.blocks[index];
            if block.cursor == (segment.offset + segment.reserved_size) % block.capacity {
                block.cursor = segment.offset;
            }
        }
        state.remove(slice.reservation);
        drop(state);
        self.inner.changed.notify_all();
        Ok(())
    }

    fn maintain(&self, slice: &StagingSlice, submitted: bool, flush: bool) -> Status {
        self.check_owner(slice)?;
        let state = lock(&self.inner.state);
        let valid = state.segment(slice.reservation).is_some_and(|(_, s)| {
            s.offset == slice.offset && s.size == slice.size && s.release_timeline.is_some() == submitted
        });
        if !valid {
            let what = if flush {
                "staging upload slice is unknown or already submitted"
            } else {
                "staging readback slice is unknown or not submitted"
            };
            return Err(Error::new(ErrorCode::InvalidArgument, what));
        }
        if self.inner.host_coherent {
            return Ok(());
        }
        let range = [vk::MappedMemoryRange::default()
            .memory(slice.memory)
            .offset(slice.memory_offset)
            .size(slice.reserved_size)];
        let device = &self.inner.device;
        let result = if flush {
            device.flush_ranges(&range)
        } else {
            device.invalidate_ranges(&range)
        };
        result.map_err(|r| {
            let what = if flush {
                "Flushing staging upload bytes failed"
            } else {
                "Invalidating staging readback bytes failed"
            };
            vk_error(ErrorCode::TransferFailed, what, r)
        })
    }

    /// Makes host writes to an unsubmitted upload slice visible to the device.
    pub fn flush(&self, slice: &StagingSlice) -> Status {
        self.maintain(slice, false, true)
    }

    /// Makes device writes to a submitted readback slice visible to the host.
    pub fn invalidate(&self, slice: &StagingSlice) -> Status {
        self.maintain(slice, true, false)
    }

    /// Reclaims every submitted reservation whose release timeline is at or below `completed`.
    pub fn reclaim_completed(&self, completed: u64) {
        lock(&self.inner.state).reclaim(completed);
        self.inner.changed.notify_all();
    }

    /// Total capacity.
    pub fn capacity(&self) -> u64 {
        lock(&self.inner.state).capacity
    }

    /// Direction.
    pub fn direction(&self) -> StagingDirection {
        self.inner.options.direction
    }

    /// Live (unreclaimed) reservations.
    pub fn live_reservations(&self) -> usize {
        lock(&self.inner.state).ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_policy_uses_topology_and_measured_benefit_without_vendor_rules() {
        use MemoryTopology::*;
        assert!(StagingRing::should_use_staging(Discrete, false));
        assert!(!StagingRing::should_use_staging(Uma, false));
        assert!(StagingRing::should_use_staging(Uma, true));
        assert!(!StagingRing::should_use_staging(DiscreteWithBar, false));
        assert!(StagingRing::should_use_staging(DiscreteWithBar, true));
        assert!(!StagingRing::should_use_staging(Unknown, false));
        assert!(StagingRing::should_use_staging(Unknown, true));
    }
}
