//! A thread-owned ring of reusable primary command buffers.
//!
//! Buffers are associated with nondecreasing completion values of one queue timeline (equal values
//! are allowed for the buffers of one coalesced submission). The whole pool is reset at once after
//! every slot's value completed, so steady-state recording never allocates or frees.

use std::fmt;

use ash::vk;

use super::Device;
use crate::error::{Error, ErrorCode, Result, Status};

/// The Vulkan calls a [`CommandPoolSet`] makes.
pub trait CommandBackend {
    /// Creates a transient, individually resettable pool.
    fn create_command_pool(&self, queue_family: u32) -> std::result::Result<vk::CommandPool, vk::Result>;
    /// Destroys a pool (and its buffers).
    fn destroy_command_pool(&self, pool: vk::CommandPool);
    /// Resets a pool.
    fn reset_command_pool(&self, pool: vk::CommandPool) -> std::result::Result<(), vk::Result>;
    /// Allocates primary buffers.
    fn allocate_command_buffers(
        &self,
        pool: vk::CommandPool,
        count: u32,
    ) -> std::result::Result<Vec<vk::CommandBuffer>, vk::Result>;
    /// Begins one-time-submit recording.
    fn begin_command_buffer(&self, buffer: vk::CommandBuffer) -> std::result::Result<(), vk::Result>;
    /// Ends recording.
    fn end_command_buffer(&self, buffer: vk::CommandBuffer) -> std::result::Result<(), vk::Result>;
    /// Resets one buffer.
    fn reset_command_buffer(&self, buffer: vk::CommandBuffer) -> std::result::Result<(), vk::Result>;
}

impl CommandBackend for Device {
    fn create_command_pool(&self, queue_family: u32) -> std::result::Result<vk::CommandPool, vk::Result> {
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);
        // SAFETY: valid create info on a live device.
        unsafe { self.raw().create_command_pool(&info, None) }
    }
    fn destroy_command_pool(&self, pool: vk::CommandPool) {
        // SAFETY: the owning set is dropped only after its buffers' submissions completed.
        unsafe { self.raw().destroy_command_pool(pool, None) }
    }
    fn reset_command_pool(&self, pool: vk::CommandPool) -> std::result::Result<(), vk::Result> {
        // SAFETY: called only when every buffer's completion value has been reached.
        unsafe {
            self.raw()
                .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())
        }
    }
    fn allocate_command_buffers(
        &self,
        pool: vk::CommandPool,
        count: u32,
    ) -> std::result::Result<Vec<vk::CommandBuffer>, vk::Result> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);
        // SAFETY: `pool` is a live pool owned by the calling set.
        unsafe { self.raw().allocate_command_buffers(&info) }
    }
    fn begin_command_buffer(&self, buffer: vk::CommandBuffer) -> std::result::Result<(), vk::Result> {
        let info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the buffer is in the initial state (never used or reset after completion).
        unsafe { self.raw().begin_command_buffer(buffer, &info) }
    }
    fn end_command_buffer(&self, buffer: vk::CommandBuffer) -> std::result::Result<(), vk::Result> {
        // SAFETY: the buffer is recording on the owning thread.
        unsafe { self.raw().end_command_buffer(buffer) }
    }
    fn reset_command_buffer(&self, buffer: vk::CommandBuffer) -> std::result::Result<(), vk::Result> {
        // SAFETY: the buffer was never submitted (discard path).
        unsafe {
            self.raw()
                .reset_command_buffer(buffer, vk::CommandBufferResetFlags::empty())
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    buffer: vk::CommandBuffer,
    completion_value: u64,
    acquired: bool,
    ended: bool,
}

/// Ring of reusable command buffers. `&mut self` methods make single-thread ownership explicit.
pub struct CommandPoolSet<B: CommandBackend = Device> {
    backend: B,
    pool: vk::CommandPool,
    slots: Vec<Slot>,
    cursor: usize,
    last_signal_value: u64,
    completed_batch_value: u64,
}

impl<B: CommandBackend> fmt::Debug for CommandPoolSet<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandPoolSet")
            .field("capacity", &self.slots.len())
            .field("oldest_pending", &self.oldest_pending_value())
            .finish()
    }
}

impl<B: CommandBackend> CommandPoolSet<B> {
    /// Creates a pool on `queue_family` and preallocates `ring_size` buffers.
    pub fn new(backend: B, queue_family: u32, ring_size: usize) -> Result<Self> {
        let count = u32::try_from(ring_size)
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::new(ErrorCode::InvalidArgument, "command ring size is invalid"))?;
        let pool = backend.create_command_pool(queue_family).map_err(|r| {
            Error::new(ErrorCode::NoVulkanImplementation, "Failed to create command pool").with_vk(r.as_raw())
        })?;
        let buffers = match backend.allocate_command_buffers(pool, count) {
            Ok(buffers) => buffers,
            Err(r) => {
                backend.destroy_command_pool(pool);
                return Err(Error::new(
                    ErrorCode::NoVulkanImplementation,
                    "Failed to allocate command buffers",
                )
                .with_vk(r.as_raw()));
            }
        };
        Ok(Self {
            backend,
            pool,
            slots: buffers
                .into_iter()
                .map(|buffer| Slot {
                    buffer,
                    ..Default::default()
                })
                .collect(),
            cursor: 0,
            last_signal_value: 0,
            completed_batch_value: 0,
        })
    }

    /// Acquires a buffer in the recording state. `completed_value` comes from the owning timeline.
    /// Fails with `OutOfMemoryBudget` when every slot is still in flight.
    pub fn acquire(&mut self, completed_value: u64) -> Result<vk::CommandBuffer> {
        let has_submitted = self.slots.iter().any(|s| s.completion_value != 0);
        let all_complete = self
            .slots
            .iter()
            .all(|s| !s.acquired && s.completion_value <= completed_value);
        if has_submitted && all_complete && completed_value > self.completed_batch_value {
            self.backend.reset_command_pool(self.pool).map_err(|r| {
                Error::new(
                    ErrorCode::NoVulkanImplementation,
                    "Failed to batch-reset the completed command pool",
                )
                .with_vk(r.as_raw())
            })?;
            for slot in &mut self.slots {
                slot.completion_value = 0;
            }
            self.completed_batch_value = completed_value;
        }
        let count = self.slots.len();
        for offset in 0..count {
            let index = (self.cursor + offset) % count;
            let slot = &mut self.slots[index];
            if slot.acquired || slot.completion_value != 0 {
                continue;
            }
            self.backend.begin_command_buffer(slot.buffer).map_err(|r| {
                Error::new(
                    ErrorCode::NoVulkanImplementation,
                    "Failed to begin a recycled command buffer",
                )
                .with_vk(r.as_raw())
            })?;
            slot.acquired = true;
            slot.ended = false;
            self.cursor = (index + 1) % count;
            return Ok(slot.buffer);
        }
        Err(Error::new(
            ErrorCode::OutOfMemoryBudget,
            "Command buffer ring is full; submitted work has not completed",
        ))
    }

    fn slot(&mut self, buffer: vk::CommandBuffer) -> Option<&mut Slot> {
        self.slots.iter_mut().find(|s| s.buffer == buffer)
    }

    /// Ends recording before submission.
    pub fn end_recording(&mut self, buffer: vk::CommandBuffer) -> Status {
        let backend_end = {
            let Some(slot) = self
                .slots
                .iter_mut()
                .find(|s| s.buffer == buffer && s.acquired && !s.ended)
            else {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    "Command buffer was not recording in this pool set",
                ));
            };
            slot.ended = true;
            slot.buffer
        };
        if let Err(r) = self.backend.end_command_buffer(backend_end) {
            if let Some(slot) = self.slot(buffer) {
                slot.acquired = false;
                slot.ended = false;
            }
            return Err(Error::new(
                ErrorCode::NoVulkanImplementation,
                "Failed to end a command buffer",
            )
            .with_vk(r.as_raw()));
        }
        Ok(())
    }

    /// Returns an unsubmitted buffer to the ring after preparation or submission failed.
    pub fn discard(&mut self, buffer: vk::CommandBuffer) -> Status {
        if !self.slots.iter().any(|s| s.buffer == buffer && s.acquired) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "Command buffer was not acquired from this pool set",
            ));
        }
        self.backend.reset_command_buffer(buffer).map_err(|r| {
            Error::new(
                ErrorCode::NoVulkanImplementation,
                "Failed to reset a discarded command buffer",
            )
            .with_vk(r.as_raw())
        })?;
        let slot = self.slot(buffer).expect("checked above");
        *slot = Slot {
            buffer,
            ..Default::default()
        };
        Ok(())
    }

    /// Associates a recorded buffer with its nonzero, nondecreasing completion value.
    pub fn submitted(&mut self, buffer: vk::CommandBuffer, signal_value: u64) -> Status {
        if signal_value == 0 || signal_value < self.last_signal_value {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "Submission timeline values must be nonzero and nondecreasing",
            ));
        }
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.buffer == buffer && s.acquired && s.ended)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidArgument,
                    "Command buffer was not acquired from this pool set",
                )
            })?;
        slot.completion_value = signal_value;
        slot.acquired = false;
        slot.ended = false;
        self.last_signal_value = signal_value;
        Ok(())
    }

    /// Ring size.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Lowest pending completion value, or 0.
    pub fn oldest_pending_value(&self) -> u64 {
        self.slots
            .iter()
            .map(|s| s.completion_value)
            .filter(|&v| v != 0)
            .min()
            .unwrap_or(0)
    }
}

impl<B: CommandBackend> Drop for CommandPoolSet<B> {
    fn drop(&mut self) {
        self.backend.destroy_command_pool(self.pool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use std::cell::Cell;

    #[derive(Default)]
    struct Mock {
        allocations: Cell<u32>,
        resets: Cell<u32>,
        begins: Cell<u32>,
        ends: Cell<u32>,
        command_resets: Cell<u32>,
        destroyed: Cell<u32>,
    }

    impl CommandBackend for &Mock {
        fn create_command_pool(&self, _: u32) -> std::result::Result<vk::CommandPool, vk::Result> {
            Ok(vk::CommandPool::from_raw(1))
        }
        fn destroy_command_pool(&self, _: vk::CommandPool) {
            self.destroyed.set(self.destroyed.get() + 1);
        }
        fn reset_command_pool(&self, _: vk::CommandPool) -> std::result::Result<(), vk::Result> {
            self.resets.set(self.resets.get() + 1);
            Ok(())
        }
        fn allocate_command_buffers(
            &self,
            _: vk::CommandPool,
            count: u32,
        ) -> std::result::Result<Vec<vk::CommandBuffer>, vk::Result> {
            self.allocations.set(self.allocations.get() + 1);
            Ok((0..count)
                .map(|i| vk::CommandBuffer::from_raw(u64::from(i) + 2))
                .collect())
        }
        fn begin_command_buffer(&self, _: vk::CommandBuffer) -> std::result::Result<(), vk::Result> {
            self.begins.set(self.begins.get() + 1);
            Ok(())
        }
        fn end_command_buffer(&self, _: vk::CommandBuffer) -> std::result::Result<(), vk::Result> {
            self.ends.set(self.ends.get() + 1);
            Ok(())
        }
        fn reset_command_buffer(&self, _: vk::CommandBuffer) -> std::result::Result<(), vk::Result> {
            self.command_resets.set(self.command_resets.get() + 1);
            Ok(())
        }
    }

    #[test]
    fn reuses_ring_without_steady_state_allocation_or_free() {
        let mock = Mock::default();
        let mut pools = CommandPoolSet::new(&mock, 0, 4).unwrap();
        assert_eq!(pools.capacity(), 4);
        for i in 1..=10_000u64 {
            let done = if i > 4 { (i - 1) / 4 * 4 } else { 0 };
            let buffer = pools.acquire(done).unwrap();
            pools.end_recording(buffer).unwrap();
            pools.submitted(buffer, i).unwrap();
        }
        assert_eq!(mock.allocations.get(), 1);
        assert_eq!(mock.resets.get(), 2499);
        assert_eq!(mock.begins.get(), 10_000);
        assert_eq!(mock.ends.get(), 10_000);
        drop(pools);
        assert_eq!(mock.destroyed.get(), 1);
    }

    #[test]
    fn rejects_unacquired_buffers_and_zero_or_decreasing_values() {
        let mock = Mock::default();
        let mut pools = CommandPoolSet::new(&mock, 0, 2).unwrap();
        assert_eq!(
            pools
                .submitted(vk::CommandBuffer::from_raw(88), 1)
                .unwrap_err()
                .code(),
            ErrorCode::InvalidArgument
        );
        let first = pools.acquire(0).unwrap();
        assert!(pools.submitted(first, 0).is_err());
        pools.end_recording(first).unwrap();
        pools.submitted(first, 1).unwrap();
        let second = pools.acquire(0).unwrap();
        pools.end_recording(second).unwrap();
        pools.submitted(second, 1).unwrap();
        let third = pools.acquire(1).unwrap();
        pools.end_recording(third).unwrap();
        assert!(pools.submitted(third, 0).is_err());
        assert!(pools.end_recording(third).is_err());
        assert_eq!(
            pools.oldest_pending_value(),
            0,
            "the batch reset cleared completed slots"
        );
    }

    #[test]
    fn full_ring_reports_out_of_memory_budget_until_work_completes() {
        let mock = Mock::default();
        let mut pools = CommandPoolSet::new(&mock, 0, 2).unwrap();
        for value in 1..=2 {
            let b = pools.acquire(0).unwrap();
            pools.end_recording(b).unwrap();
            pools.submitted(b, value).unwrap();
        }
        assert_eq!(pools.acquire(1).unwrap_err().code(), ErrorCode::OutOfMemoryBudget);
        assert_eq!(pools.oldest_pending_value(), 1);
        assert!(pools.acquire(2).is_ok());
    }

    #[test]
    fn discards_a_recording_after_preparation_fails() {
        let mock = Mock::default();
        let mut pools = CommandPoolSet::new(&mock, 0, 1).unwrap();
        let buffer = pools.acquire(0).unwrap();
        pools.discard(buffer).unwrap();
        assert_eq!(mock.command_resets.get(), 1);
        assert!(pools.acquire(0).is_ok());
        assert!(pools.discard(vk::CommandBuffer::from_raw(99)).is_err());
        assert!(CommandPoolSet::new(&mock, 0, 0).is_err());
    }
}
