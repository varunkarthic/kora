//! Per-queue bounded GPU timestamp ring.
//!
//! Each sampled submission records a start/end timestamp pair. Results are read only after the
//! owning queue value completed, without `VK_QUERY_RESULT_WAIT_BIT`. Tick deltas respect the queue
//! family's valid timestamp bits; host correlation uses calibrated timestamps when available.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ash::vk;

use crate::error::{Error, ErrorCode, Result, Status};
use crate::vulkan::Device;

/// One completed sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuTimestampSample {
    /// Queue submission ID.
    pub submission_id: u64,
    /// Queue timeline value.
    pub queue_value: u64,
    /// Start ticks.
    pub start_ticks: u64,
    /// End ticks.
    pub end_ticks: u64,
    /// Duration.
    pub duration_nanoseconds: u64,
    /// Calibrated host monotonic time at recording (0 when unavailable).
    pub host_timestamp_nanoseconds: u64,
    /// Calibration deviation.
    pub calibration_deviation_nanoseconds: u64,
}

/// Wrap-aware tick difference at `valid_bits` width.
pub fn timestamp_delta(start: u64, end: u64, valid_bits: u32) -> Result<u64> {
    match valid_bits {
        0 | 65.. => Err(Error::new(
            ErrorCode::InvalidArgument,
            "timestamp valid-bit count must be in [1, 64]",
        )),
        64 => Ok(end.wrapping_sub(start)),
        bits => Ok(end.wrapping_sub(start) & ((1u64 << bits) - 1)),
    }
}

/// Ticks to nanoseconds with overflow checks.
pub fn timestamp_nanoseconds(ticks: u64, period_nanoseconds: f32) -> Result<u64> {
    if !(period_nanoseconds.is_finite() && period_nanoseconds > 0.0) {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "timestamp period must be finite and positive",
        ));
    }
    let duration = (ticks as f64 * f64::from(period_nanoseconds)).round();
    if duration >= u64::MAX as f64 {
        return Err(Error::new(
            ErrorCode::Overflow,
            "GPU timestamp duration exceeds nanosecond representation",
        ));
    }
    Ok(duration as u64)
}

/// The Vulkan calls a [`TimestampRing`] makes.
pub trait TimestampBackend: Send + Sync {
    /// Creates a timestamp query pool.
    fn create_query_pool(&self, count: u32) -> std::result::Result<vk::QueryPool, vk::Result>;
    /// Destroys it.
    fn destroy_query_pool(&self, pool: vk::QueryPool);
    /// Records a query reset.
    fn cmd_reset_query_pool(
        &self,
        command_buffer: vk::CommandBuffer,
        pool: vk::QueryPool,
        first: u32,
        count: u32,
    );
    /// Records a timestamp write.
    fn cmd_write_timestamp(
        &self,
        command_buffer: vk::CommandBuffer,
        stage: vk::PipelineStageFlags,
        pool: vk::QueryPool,
        query: u32,
    );
    /// Reads two 64-bit results (`Err(NOT_READY)` when unavailable).
    fn query_results(
        &self,
        pool: vk::QueryPool,
        first: u32,
        values: &mut [u64; 2],
    ) -> std::result::Result<(), vk::Result>;
    /// Calibrated `(host monotonic ns, max deviation)` when supported.
    fn calibrated_host_timestamp(&self) -> Option<(u64, u64)>;
}

impl TimestampBackend for Device {
    fn create_query_pool(&self, count: u32) -> std::result::Result<vk::QueryPool, vk::Result> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);
        // SAFETY: valid create info on a live device.
        unsafe { self.raw().create_query_pool(&info, None) }
    }
    fn destroy_query_pool(&self, pool: vk::QueryPool) {
        // SAFETY: the ring is dropped only after its queue drained.
        unsafe { self.raw().destroy_query_pool(pool, None) }
    }
    fn cmd_reset_query_pool(
        &self,
        command_buffer: vk::CommandBuffer,
        pool: vk::QueryPool,
        first: u32,
        count: u32,
    ) {
        // SAFETY: the command buffer is recording on the calling (submission) thread.
        unsafe {
            self.raw()
                .cmd_reset_query_pool(command_buffer, pool, first, count)
        }
    }
    fn cmd_write_timestamp(
        &self,
        command_buffer: vk::CommandBuffer,
        stage: vk::PipelineStageFlags,
        pool: vk::QueryPool,
        query: u32,
    ) {
        // SAFETY: as above; the query was reset earlier in the same command buffer.
        unsafe { self.raw().cmd_write_timestamp(command_buffer, stage, pool, query) }
    }
    fn query_results(
        &self,
        pool: vk::QueryPool,
        first: u32,
        values: &mut [u64; 2],
    ) -> std::result::Result<(), vk::Result> {
        self.get_query_pool_results(pool, first, values, vk::QueryResultFlags::TYPE_64)
    }
    fn calibrated_host_timestamp(&self) -> Option<(u64, u64)> {
        let domains = &self.physical().calibrated_time_domains;
        if !domains.contains(&vk::TimeDomainKHR::DEVICE)
            || !domains.contains(&vk::TimeDomainKHR::CLOCK_MONOTONIC)
        {
            return None;
        }
        self.calibrated_timestamps(vk::TimeDomainKHR::CLOCK_MONOTONIC)
            .map(|(_, host, deviation)| (host, deviation))
    }
}

/// Ring capacity (samples in flight).
const CAPACITY: usize = 128;
const HISTORY: usize = 4096;

#[derive(Debug, Clone, Copy)]
struct Slot {
    submission_id: u64,
    queue_value: u64,
    query_index: u32,
    host_timestamp: u64,
    host_deviation: u64,
    recording: bool,
    submitted: bool,
}

#[derive(Default)]
struct RingState {
    slots: VecDeque<Slot>,
    dropped: u64,
    collected: u64,
    latest: GpuTimestampSample,
    trace_history: bool,
    history: VecDeque<GpuTimestampSample>,
}

/// Bounded timestamp ring.
pub struct TimestampRing<B: TimestampBackend = Device> {
    backend: B,
    pool: vk::QueryPool,
    valid_bits: u32,
    period_nanoseconds: f32,
    state: Mutex<RingState>,
}

impl<B: TimestampBackend> fmt::Debug for TimestampRing<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimestampRing")
            .field("samples", &self.samples_collected())
            .field("dropped", &self.dropped_samples())
            .finish()
    }
}

impl<B: TimestampBackend> TimestampRing<B> {
    /// Creates the query pool. Fails with `UnsupportedFeature` when timestamps are unusable.
    pub fn new(backend: B, valid_bits: u32, period_nanoseconds: f32) -> Result<Self> {
        if valid_bits == 0 || valid_bits > 64 || !(period_nanoseconds.is_finite() && period_nanoseconds > 0.0)
        {
            return Err(Error::new(
                ErrorCode::UnsupportedFeature,
                "GPU timestamp profiling is unavailable on this queue",
            ));
        }
        let pool = backend.create_query_pool((CAPACITY * 2) as u32).map_err(|r| {
            Error::new(
                ErrorCode::UnsupportedFeature,
                "Creating GPU timestamp query pool failed",
            )
            .with_vk(r.as_raw())
        })?;
        Ok(Self {
            backend,
            pool,
            valid_bits,
            period_nanoseconds,
            state: Mutex::new(RingState::default()),
        })
    }

    fn lock(&self) -> MutexGuard<'_, RingState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Records the start timestamp; returns `false` (and counts a drop) when the ring is full.
    pub fn record_start(&self, command_buffer: vk::CommandBuffer, submission_id: u64) -> bool {
        let mut state = self.lock();
        if command_buffer == vk::CommandBuffer::null() || state.slots.len() == CAPACITY {
            state.dropped += 1;
            return false;
        }
        let query_index = ((submission_id % CAPACITY as u64) * 2) as u32;
        let (host_timestamp, host_deviation) = self.backend.calibrated_host_timestamp().unwrap_or((0, 0));
        self.backend
            .cmd_reset_query_pool(command_buffer, self.pool, query_index, 2);
        self.backend.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            self.pool,
            query_index,
        );
        state.slots.push_back(Slot {
            submission_id,
            queue_value: 0,
            query_index,
            host_timestamp,
            host_deviation,
            recording: true,
            submitted: false,
        });
        true
    }

    /// Records the end timestamp of a started sample.
    pub fn record_end(&self, command_buffer: vk::CommandBuffer, submission_id: u64) {
        let state = self.lock();
        if let Some(slot) = state
            .slots
            .iter()
            .rev()
            .find(|s| s.submission_id == submission_id && s.recording)
        {
            self.backend.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                slot.query_index + 1,
            );
        }
    }

    /// Drops an unsubmitted sample.
    pub fn cancel(&self, submission_id: u64) {
        let mut state = self.lock();
        if let Some(position) = state
            .slots
            .iter()
            .position(|s| s.submission_id == submission_id && !s.submitted)
        {
            state.slots.remove(position);
        }
    }

    /// Marks a sample as submitted at `queue_value`.
    pub fn submitted(&self, submission_id: u64, queue_value: u64) {
        let mut state = self.lock();
        if let Some(slot) = state
            .slots
            .iter_mut()
            .find(|s| s.submission_id == submission_id && s.recording)
        {
            slot.queue_value = queue_value;
            slot.recording = false;
            slot.submitted = true;
        }
    }

    /// Reads completed samples in order, polling up to `ready_wait` for drivers that publish
    /// results shortly after the queue signal (hot paths pass zero and never block).
    pub fn collect(
        &self,
        completed_queue_value: u64,
        ready_wait: Duration,
    ) -> Result<Vec<GpuTimestampSample>> {
        let mut samples = Vec::new();
        let mut state = self.lock();
        while let Some(slot) = state.slots.front().copied() {
            if !slot.submitted || slot.queue_value > completed_queue_value {
                break;
            }
            let mut values = [0u64; 2];
            let mut result = self
                .backend
                .query_results(self.pool, slot.query_index, &mut values);
            let deadline = Instant::now() + ready_wait;
            while result == Err(vk::Result::NOT_READY) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_micros(20));
                result = self
                    .backend
                    .query_results(self.pool, slot.query_index, &mut values);
            }
            match result {
                Err(vk::Result::NOT_READY) => break,
                Err(r) => {
                    return Err(Error::new(
                        ErrorCode::TransferFailed,
                        "Reading completed GPU timestamps failed",
                    )
                    .with_vk(r.as_raw()));
                }
                Ok(()) => {}
            }
            let ticks = timestamp_delta(values[0], values[1], self.valid_bits)?;
            let sample = GpuTimestampSample {
                submission_id: slot.submission_id,
                queue_value: slot.queue_value,
                start_ticks: values[0],
                end_ticks: values[1],
                duration_nanoseconds: timestamp_nanoseconds(ticks, self.period_nanoseconds)?,
                host_timestamp_nanoseconds: slot.host_timestamp,
                calibration_deviation_nanoseconds: slot.host_deviation,
            };
            state.latest = sample;
            state.collected += 1;
            if state.trace_history {
                if state.history.len() == HISTORY {
                    state.history.pop_front();
                }
                state.history.push_back(sample);
            }
            samples.push(sample);
            state.slots.pop_front();
        }
        Ok(samples)
    }

    /// Enables bounded history retention for trace export.
    pub fn enable_trace_history(&self, enabled: bool) {
        let mut state = self.lock();
        state.trace_history = enabled;
        if !enabled {
            state.history.clear();
        }
    }

    /// Samples dropped because the ring was full.
    pub fn dropped_samples(&self) -> u64 {
        self.lock().dropped
    }
    /// Samples collected.
    pub fn samples_collected(&self) -> u64 {
        self.lock().collected
    }
    /// Latest sample.
    pub fn latest_sample(&self) -> GpuTimestampSample {
        self.lock().latest
    }
    /// Retained history (trace mode).
    pub fn history(&self) -> Vec<GpuTimestampSample> {
        self.lock().history.iter().copied().collect()
    }
    /// Pending (recorded, not yet collected) samples.
    pub fn pending(&self) -> usize {
        self.lock().slots.len()
    }

    /// Validates readiness; kept for API symmetry with fallible initialization.
    pub fn check(&self) -> Status {
        Ok(())
    }
}

impl<B: TimestampBackend> Drop for TimestampRing<B> {
    fn drop(&mut self) {
        self.backend.destroy_query_pool(self.pool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Default)]
    struct Fixture {
        reads: AtomicU32,
        not_ready_reads: AtomicU32,
    }

    impl TimestampBackend for &Fixture {
        fn create_query_pool(&self, count: u32) -> std::result::Result<vk::QueryPool, vk::Result> {
            assert_eq!(count, 256);
            Ok(vk::QueryPool::from_raw(1))
        }
        fn destroy_query_pool(&self, _: vk::QueryPool) {}
        fn cmd_reset_query_pool(&self, _: vk::CommandBuffer, _: vk::QueryPool, _: u32, count: u32) {
            assert_eq!(count, 2);
        }
        fn cmd_write_timestamp(
            &self,
            _: vk::CommandBuffer,
            _: vk::PipelineStageFlags,
            _: vk::QueryPool,
            _: u32,
        ) {
        }
        fn query_results(
            &self,
            _: vk::QueryPool,
            _: u32,
            values: &mut [u64; 2],
        ) -> std::result::Result<(), vk::Result> {
            if self.not_ready_reads.load(Ordering::SeqCst) > 0 {
                self.not_ready_reads.fetch_sub(1, Ordering::SeqCst);
                return Err(vk::Result::NOT_READY);
            }
            self.reads.fetch_add(1, Ordering::SeqCst);
            *values = [100, 130];
            Ok(())
        }
        fn calibrated_host_timestamp(&self) -> Option<(u64, u64)> {
            None
        }
    }

    #[test]
    fn handles_wrap_at_valid_bit_width() {
        assert_eq!(timestamp_delta(250, 5, 8).unwrap(), 11);
        assert_eq!(timestamp_delta(u64::MAX - 3, 2, 64).unwrap(), 6);
        assert!(timestamp_delta(0, 1, 0).is_err());
        assert!(timestamp_delta(0, 1, 65).is_err());
    }

    #[test]
    fn converts_ticks_and_checks_overflow() {
        assert_eq!(timestamp_nanoseconds(40, 2.5).unwrap(), 100);
        assert!(timestamp_nanoseconds(1, 0.0).is_err());
        assert!(timestamp_nanoseconds(1, f32::NAN).is_err());
        assert!(timestamp_nanoseconds(u64::MAX, 2.0).is_err());
    }

    #[test]
    fn reads_only_completed_slots_without_waiting() {
        let fixture = Fixture::default();
        let ring = TimestampRing::new(&fixture, 8, 2.5).unwrap();
        let cb = vk::CommandBuffer::from_raw(3);
        assert!(ring.record_start(cb, 7));
        ring.record_end(cb, 7);
        ring.submitted(7, 12);
        assert!(ring.collect(11, Duration::ZERO).unwrap().is_empty());
        assert_eq!(fixture.reads.load(Ordering::SeqCst), 0);
        let completed = ring.collect(12, Duration::ZERO).unwrap();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].duration_nanoseconds, 75);
        assert_eq!(ring.samples_collected(), 1);
        assert!(ring.history().is_empty());
        ring.enable_trace_history(true);
        for id in 8..=4103u64 {
            assert!(ring.record_start(cb, id));
            ring.record_end(cb, id);
            ring.submitted(id, id);
            assert_eq!(ring.collect(id, Duration::ZERO).unwrap().len(), 1);
        }
        let history = ring.history();
        assert_eq!(history.len(), 4096);
        assert_eq!(history.first().unwrap().submission_id, 8);
        assert_eq!(history.last().unwrap().submission_id, 4103);
    }

    #[test]
    fn full_ring_drops_samples_and_cancel_releases_slots() {
        let fixture = Fixture::default();
        let ring = TimestampRing::new(&fixture, 64, 1.0).unwrap();
        let cb = vk::CommandBuffer::from_raw(3);
        for id in 0..CAPACITY as u64 {
            assert!(ring.record_start(cb, id));
        }
        assert!(!ring.record_start(cb, 999));
        assert_eq!(ring.dropped_samples(), 1);
        ring.cancel(0);
        assert!(ring.record_start(cb, 1000));
        assert!(!ring.record_start(vk::CommandBuffer::null(), 1001));
        assert_eq!(ring.pending(), CAPACITY);
    }

    #[test]
    fn not_ready_results_are_retried_within_the_budget_or_left_pending() {
        let fixture = Fixture::default();
        let ring = TimestampRing::new(&fixture, 64, 1.0).unwrap();
        let cb = vk::CommandBuffer::from_raw(3);
        ring.record_start(cb, 1);
        ring.submitted(1, 1);
        fixture.not_ready_reads.store(1, Ordering::SeqCst);
        assert!(ring.collect(1, Duration::ZERO).unwrap().is_empty());
        assert_eq!(ring.pending(), 1);
        fixture.not_ready_reads.store(2, Ordering::SeqCst);
        assert_eq!(ring.collect(1, Duration::from_millis(50)).unwrap().len(), 1);
        assert!(TimestampRing::new(&fixture, 0, 1.0).is_err());
    }
}
