//! Per-buffer read/write hazard tracking with precise legacy stage/access masks.

use ash::vk;

use super::Device;

/// A queue-independent access class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessKind {
    /// Host read of mapped memory.
    HostRead,
    /// Host write of mapped memory.
    HostWrite,
    /// Transfer source.
    TransferRead,
    /// Transfer destination.
    TransferWrite,
    /// Shader read.
    ComputeRead,
    /// Shader write.
    ComputeWrite,
}

/// The Vulkan synchronization scope of an access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AccessScope {
    /// Pipeline stage.
    pub stage: vk::PipelineStageFlags,
    /// Access mask.
    pub access: vk::AccessFlags,
    /// The access writes.
    pub writes: bool,
}

/// One memory barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BarrierPlan {
    /// Source stages.
    pub source_stage: vk::PipelineStageFlags,
    /// Source accesses.
    pub source_access: vk::AccessFlags,
    /// Destination stages.
    pub destination_stage: vk::PipelineStageFlags,
    /// Destination accesses.
    pub destination_access: vk::AccessFlags,
}

/// Maps an access class to its precise scope.
pub const fn access_scope(kind: AccessKind) -> AccessScope {
    use vk::{AccessFlags as A, PipelineStageFlags as S};
    let (stage, access, writes) = match kind {
        AccessKind::HostRead => (S::HOST, A::HOST_READ, false),
        AccessKind::HostWrite => (S::HOST, A::HOST_WRITE, true),
        AccessKind::TransferRead => (S::TRANSFER, A::TRANSFER_READ, false),
        AccessKind::TransferWrite => (S::TRANSFER, A::TRANSFER_WRITE, true),
        AccessKind::ComputeRead => (S::COMPUTE_SHADER, A::SHADER_READ, false),
        AccessKind::ComputeWrite => (S::COMPUTE_SHADER, A::SHADER_WRITE, true),
    };
    AccessScope {
        stage,
        access,
        writes,
    }
}

/// Two accesses to one buffer conflict when either writes.
pub const fn accesses_conflict(previous: AccessScope, next: AccessScope) -> bool {
    previous.writes || next.writes
}

/// Tracks the last writer and the readers since it, for one buffer.
#[derive(Debug, Clone, Default)]
pub struct HazardTracker {
    writer: Option<AccessScope>,
    readers: Vec<AccessScope>,
}

impl HazardTracker {
    /// A tracker with no prior accesses.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an access and returns the barrier (if any) ordering it after conflicting accesses.
    pub fn access(&mut self, kind: AccessKind) -> Option<BarrierPlan> {
        let next = access_scope(kind);
        let mut source_stage = vk::PipelineStageFlags::empty();
        let mut source_access = vk::AccessFlags::empty();
        if let Some(writer) = self.writer.filter(|w| accesses_conflict(*w, next)) {
            source_stage |= writer.stage;
            source_access |= writer.access;
        }
        if next.writes {
            for reader in self.readers.iter().filter(|r| accesses_conflict(**r, next)) {
                source_stage |= reader.stage;
                source_access |= reader.access;
            }
            self.writer = Some(next);
            self.readers.clear();
        } else {
            self.readers.push(next);
        }
        (!source_stage.is_empty()).then_some(BarrierPlan {
            source_stage,
            source_access,
            destination_stage: next.stage,
            destination_access: next.access,
        })
    }

    /// Readers since the last writer.
    pub fn active_readers(&self) -> &[AccessScope] {
        &self.readers
    }

    /// Last writer.
    pub fn last_writer(&self) -> Option<AccessScope> {
        self.writer
    }
}

/// Records `plan` as a global memory barrier, using synchronization2 when enabled.
///
/// # Safety
/// `command_buffer` must be recording, owned by the calling thread, and allocated from `device`.
pub unsafe fn record_barrier(device: &Device, command_buffer: vk::CommandBuffer, plan: &BarrierPlan) {
    if device.enabled_features().synchronization2 {
        let barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::from_raw(u64::from(
                plan.source_stage.as_raw(),
            )))
            .src_access_mask(vk::AccessFlags2::from_raw(u64::from(plan.source_access.as_raw())))
            .dst_stage_mask(vk::PipelineStageFlags2::from_raw(u64::from(
                plan.destination_stage.as_raw(),
            )))
            .dst_access_mask(vk::AccessFlags2::from_raw(u64::from(
                plan.destination_access.as_raw(),
            )))];
        let dependency = vk::DependencyInfo::default().memory_barriers(&barrier);
        // SAFETY: forwarded from the caller's contract; the dependency info outlives the call.
        if unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency) } {
            return;
        }
    }
    let barrier = [vk::MemoryBarrier::default()
        .src_access_mask(plan.source_access)
        .dst_access_mask(plan.destination_access)];
    // SAFETY: forwarded from the caller's contract.
    unsafe {
        device.raw().cmd_pipeline_barrier(
            command_buffer,
            plan.source_stage,
            plan.destination_stage,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
    }
}

/// Records the barrier (if any) that orders `next` after `previous` on a fresh tracker.
///
/// # Safety
/// As for [`record_barrier`].
pub unsafe fn record_transition(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    tracker: &mut HazardTracker,
    next: AccessKind,
) {
    if let Some(plan) = tracker.access(next) {
        // SAFETY: forwarded from the caller's contract.
        unsafe { record_barrier(device, command_buffer, &plan) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task_graph::tests::SplitMix;
    use vk::{AccessFlags as A, PipelineStageFlags as S};

    #[test]
    fn golden_table_covers_required_transitions() {
        let cases = [
            (
                AccessKind::TransferWrite,
                AccessKind::ComputeRead,
                Some((S::TRANSFER, A::TRANSFER_WRITE, S::COMPUTE_SHADER, A::SHADER_READ)),
            ),
            (
                AccessKind::ComputeWrite,
                AccessKind::ComputeRead,
                Some((
                    S::COMPUTE_SHADER,
                    A::SHADER_WRITE,
                    S::COMPUTE_SHADER,
                    A::SHADER_READ,
                )),
            ),
            (
                AccessKind::ComputeWrite,
                AccessKind::TransferRead,
                Some((S::COMPUTE_SHADER, A::SHADER_WRITE, S::TRANSFER, A::TRANSFER_READ)),
            ),
            (
                AccessKind::ComputeWrite,
                AccessKind::HostRead,
                Some((S::COMPUTE_SHADER, A::SHADER_WRITE, S::HOST, A::HOST_READ)),
            ),
            (
                AccessKind::HostWrite,
                AccessKind::ComputeRead,
                Some((S::HOST, A::HOST_WRITE, S::COMPUTE_SHADER, A::SHADER_READ)),
            ),
            (AccessKind::ComputeRead, AccessKind::ComputeRead, None),
        ];
        for (previous, next, expected) in cases {
            let mut tracker = HazardTracker::new();
            tracker.access(previous);
            let plan = tracker.access(next);
            let expected = expected.map(|(ss, sa, ds, da)| BarrierPlan {
                source_stage: ss,
                source_access: sa,
                destination_stage: ds,
                destination_access: da,
            });
            assert_eq!(plan, expected, "{previous:?} -> {next:?}");
            if let Some(plan) = plan {
                assert_ne!(plan.source_stage, S::ALL_COMMANDS);
                assert_ne!(plan.destination_stage, S::ALL_COMMANDS);
            }
        }
    }

    #[test]
    fn writer_orders_every_reader_and_readers_do_not_order_each_other() {
        let mut tracker = HazardTracker::new();
        assert!(tracker.access(AccessKind::ComputeWrite).is_none());
        assert!(tracker.access(AccessKind::ComputeRead).is_some());
        assert!(tracker.access(AccessKind::TransferRead).is_some());
        assert_eq!(tracker.active_readers().len(), 2);
        let barrier = tracker.access(AccessKind::HostWrite).unwrap();
        assert_eq!(
            barrier.source_access,
            A::SHADER_WRITE | A::SHADER_READ | A::TRANSFER_READ
        );
        assert!(tracker.active_readers().is_empty());
        assert_eq!(tracker.last_writer(), Some(access_scope(AccessKind::HostWrite)));
    }

    #[test]
    fn random_sequences_preserve_brute_force_orderings() {
        let kinds = [
            AccessKind::HostRead,
            AccessKind::HostWrite,
            AccessKind::TransferRead,
            AccessKind::TransferWrite,
            AccessKind::ComputeRead,
            AccessKind::ComputeWrite,
        ];
        let mut random = SplitMix(0x4b4f_5241);
        for _ in 0..256 {
            let mut tracker = HazardTracker::new();
            let mut writer: Option<AccessScope> = None;
            let mut readers: Vec<AccessScope> = Vec::new();
            for _ in 0..128 {
                let kind = kinds[random.below(6) as usize];
                let scope = access_scope(kind);
                let (mut stages, mut access) = (S::empty(), A::empty());
                if let Some(w) = writer.filter(|w| accesses_conflict(*w, scope)) {
                    stages |= w.stage;
                    access |= w.access;
                }
                if scope.writes {
                    for r in readers.iter().filter(|r| accesses_conflict(**r, scope)) {
                        stages |= r.stage;
                        access |= r.access;
                    }
                }
                match tracker.access(kind) {
                    Some(plan) => {
                        assert_eq!((plan.source_stage, plan.source_access), (stages, access));
                        assert_eq!(
                            (plan.destination_stage, plan.destination_access),
                            (scope.stage, scope.access)
                        );
                    }
                    None => assert!(stages.is_empty()),
                }
                if scope.writes {
                    writer = Some(scope);
                    readers.clear();
                } else {
                    readers.push(scope);
                }
            }
        }
    }
}
