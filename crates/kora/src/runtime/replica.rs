//! Device copies of logical buffers.

use std::any::Any;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ash::vk;

use crate::memory::{PoolAllocation, SharedPool, TransferBuffer};
use crate::vulkan::Device;

/// How the device touched a replica since the last memory barrier that ordered it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum DeviceAccess {
    /// Nothing outstanding (or transfers already ordered it for compute).
    #[default]
    None,
    /// Read by a dispatch or a download.
    Read,
    /// Written by a dispatch.
    Write,
}

/// Mutable replica bookkeeping, guarded per replica.
#[derive(Debug, Default)]
pub(crate) struct ReplicaState {
    /// Host version whose bytes the replica holds (after pending work completes).
    pub(crate) resident_version: Option<u64>,
    /// `(host version, split ID)` after a slice of that split wrote its range of a replica holding
    /// that version: only later slices of the same split may treat the replica as resident.
    pub(crate) partial_version: Option<(u64, u64)>,
    /// The replica holds results newer than the host shadow (deferred readback).
    pub(crate) device_newer: bool,
    /// Compute-queue timeline value of the last submission that used the replica.
    pub(crate) last_use: u64,
    /// Outstanding device access for hazard barriers.
    pub(crate) access: DeviceAccess,
}

/// One buffer's `VkBuffer` and memory on one device. Destroyed when the logical buffer and every
/// in-flight submission that uses it are gone.
pub(crate) struct Replica {
    device: Device,
    pool: SharedPool,
    allocation: PoolAllocation,
    transfer: TransferBuffer,
    state: Mutex<ReplicaState>,
}

impl std::fmt::Debug for Replica {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Replica")
            .field("size", &self.transfer.size)
            .field("host_visible", &self.transfer.host_visible)
            .finish()
    }
}

impl Replica {
    pub(crate) fn new(
        device: Device,
        pool: SharedPool,
        allocation: PoolAllocation,
        transfer: TransferBuffer,
    ) -> Self {
        Self {
            device,
            pool,
            allocation,
            transfer,
            state: Mutex::default(),
        }
    }

    pub(crate) fn state(&self) -> MutexGuard<'_, ReplicaState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.transfer.buffer
    }

    pub(crate) fn size(&self) -> u64 {
        self.transfer.size
    }

    /// The mapping is usable for direct host access: host visible, mapped, and no queued or
    /// executing work beyond `own_uses` of the caller's submission.
    pub(crate) fn directly_accessible(&self, own_uses: u32) -> bool {
        self.transfer.host_visible
            && self.transfer.mapped.is_some()
            && self
                .transfer
                .queue_use
                .as_ref()
                .is_some_and(|u| u.pending.load(Ordering::Acquire) == own_uses)
    }

    pub(crate) fn transfer(&self) -> &TransferBuffer {
        &self.transfer
    }

    /// A transfer view that keeps this replica alive until the transfer completes.
    pub(crate) fn transfer_view(self: &Arc<Self>) -> TransferBuffer {
        let mut view = self.transfer.clone();
        view.keepalive = Some(Arc::clone(self) as Arc<dyn Any + Send + Sync>);
        view
    }
}

impl Drop for Replica {
    fn drop(&mut self) {
        // SAFETY: the last reference is gone; every submission that used the buffer held a clone
        // (through its keep-alive) until completion, so the device no longer accesses it.
        unsafe { self.device.raw().destroy_buffer(self.transfer.buffer, None) };
        let _ = self
            .pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .deallocate(&self.allocation);
    }
}
