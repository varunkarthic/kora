//! Queue-family ownership transfer barriers for exclusive buffers.

use ash::vk;

use super::hazard::AccessScope;

/// Cached queue-family strategy of a logical device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueFamilyPolicy {
    /// Compute family.
    pub compute_family: u32,
    /// Transfer family.
    pub transfer_family: u32,
    /// Buffers use concurrent sharing between the two families.
    pub concurrent_sharing: bool,
}

impl QueueFamilyPolicy {
    /// A policy (concurrent sharing by default).
    pub const fn new(compute_family: u32, transfer_family: u32) -> Self {
        Self {
            compute_family,
            transfer_family,
            concurrent_sharing: true,
        }
    }
    fn distinct(&self) -> bool {
        self.compute_family != vk::QUEUE_FAMILY_IGNORED
            && self.transfer_family != vk::QUEUE_FAMILY_IGNORED
            && self.compute_family != self.transfer_family
    }
    /// Exclusive buffers must be released and acquired between families.
    pub fn requires_ownership_transfer(&self) -> bool {
        self.distinct() && !self.concurrent_sharing
    }
    /// Buffers are created with concurrent sharing.
    pub fn uses_concurrent_sharing(&self) -> bool {
        self.distinct() && self.concurrent_sharing
    }
}

/// Matching release (source queue) and acquire (destination queue) barriers.
#[derive(Debug, Clone, Copy)]
pub struct OwnershipBarrierPair {
    /// synchronization2 release barrier.
    pub release: vk::BufferMemoryBarrier2<'static>,
    /// synchronization2 acquire barrier.
    pub acquire: vk::BufferMemoryBarrier2<'static>,
    /// Legacy release barrier.
    pub release_legacy: vk::BufferMemoryBarrier<'static>,
    /// Legacy acquire barrier.
    pub acquire_legacy: vk::BufferMemoryBarrier<'static>,
    /// Legacy release source stage.
    pub release_legacy_source_stage: vk::PipelineStageFlags,
    /// Legacy acquire destination stage.
    pub acquire_legacy_destination_stage: vk::PipelineStageFlags,
    /// Releasing family.
    pub source_family: u32,
    /// Acquiring family.
    pub destination_family: u32,
    /// The acquire must wait on a semaphore signalled after the release.
    pub requires_semaphore_dependency: bool,
}

fn stage2(stage: vk::PipelineStageFlags) -> vk::PipelineStageFlags2 {
    vk::PipelineStageFlags2::from_raw(u64::from(stage.as_raw()))
}

fn access2(access: vk::AccessFlags) -> vk::AccessFlags2 {
    vk::AccessFlags2::from_raw(u64::from(access.as_raw()))
}

/// Builds a release/acquire pair, or `None` for same-family use and invalid ranges.
pub fn make_ownership_barrier_pair(
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
    source: AccessScope,
    destination: AccessScope,
    source_family: u32,
    destination_family: u32,
) -> Option<OwnershipBarrierPair> {
    if buffer == vk::Buffer::null()
        || size == 0
        || source_family == destination_family
        || source_family == vk::QUEUE_FAMILY_IGNORED
        || destination_family == vk::QUEUE_FAMILY_IGNORED
    {
        return None;
    }
    let base2 = vk::BufferMemoryBarrier2::default()
        .src_queue_family_index(source_family)
        .dst_queue_family_index(destination_family)
        .buffer(buffer)
        .offset(offset)
        .size(size);
    let base = vk::BufferMemoryBarrier::default()
        .src_queue_family_index(source_family)
        .dst_queue_family_index(destination_family)
        .buffer(buffer)
        .offset(offset)
        .size(size);
    Some(OwnershipBarrierPair {
        release: base2
            .src_stage_mask(stage2(source.stage))
            .src_access_mask(access2(source.access)),
        acquire: base2
            .dst_stage_mask(stage2(destination.stage))
            .dst_access_mask(access2(destination.access)),
        release_legacy: base.src_access_mask(source.access),
        acquire_legacy: base.dst_access_mask(destination.access),
        release_legacy_source_stage: source.stage,
        acquire_legacy_destination_stage: destination.stage,
        source_family,
        destination_family,
        requires_semaphore_dependency: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use vk::{AccessFlags as A, PipelineStageFlags as S};

    fn scope(stage: S, access: A, writes: bool) -> AccessScope {
        AccessScope {
            stage,
            access,
            writes,
        }
    }

    #[test]
    fn same_family_use_requires_no_barriers() {
        let buffer = vk::Buffer::from_raw(7);
        assert!(
            make_ownership_barrier_pair(
                buffer,
                4,
                64,
                scope(S::TRANSFER, A::TRANSFER_WRITE, true),
                scope(S::COMPUTE_SHADER, A::SHADER_READ, false),
                3,
                3
            )
            .is_none()
        );
    }

    #[test]
    fn different_families_generate_matching_release_acquire_pair() {
        let buffer = vk::Buffer::from_raw(7);
        let pair = make_ownership_barrier_pair(
            buffer,
            16,
            128,
            scope(S::TRANSFER, A::TRANSFER_WRITE, true),
            scope(S::COMPUTE_SHADER, A::SHADER_READ, false),
            2,
            5,
        )
        .unwrap();
        assert!(pair.requires_semaphore_dependency);
        assert_eq!((pair.source_family, pair.destination_family), (2, 5));
        for b in [&pair.release, &pair.acquire] {
            assert_eq!(
                (
                    b.buffer,
                    b.offset,
                    b.size,
                    b.src_queue_family_index,
                    b.dst_queue_family_index
                ),
                (buffer, 16, 128, 2, 5)
            );
        }
        assert_eq!(pair.release.src_stage_mask, vk::PipelineStageFlags2::TRANSFER);
        assert_eq!(pair.release.src_access_mask, vk::AccessFlags2::TRANSFER_WRITE);
        assert_eq!(pair.release.dst_stage_mask, vk::PipelineStageFlags2::NONE);
        assert_eq!(pair.release.dst_access_mask, vk::AccessFlags2::NONE);
        assert_eq!(pair.acquire.src_stage_mask, vk::PipelineStageFlags2::NONE);
        assert_eq!(
            pair.acquire.dst_stage_mask,
            vk::PipelineStageFlags2::COMPUTE_SHADER
        );
        assert_eq!(pair.acquire.dst_access_mask, vk::AccessFlags2::SHADER_READ);
        assert_eq!(
            (
                pair.release_legacy.src_access_mask,
                pair.release_legacy.dst_access_mask
            ),
            (A::TRANSFER_WRITE, A::empty())
        );
        assert_eq!(
            (
                pair.acquire_legacy.src_access_mask,
                pair.acquire_legacy.dst_access_mask
            ),
            (A::empty(), A::SHADER_READ)
        );
        assert_eq!(pair.release_legacy_source_stage, S::TRANSFER);
        assert_eq!(pair.acquire_legacy_destination_stage, S::COMPUTE_SHADER);
    }

    #[test]
    fn invalid_or_empty_ranges_do_not_produce_barriers() {
        let s = scope(S::TRANSFER, A::TRANSFER_WRITE, true);
        assert!(make_ownership_barrier_pair(vk::Buffer::null(), 0, 4, s, s, 1, 2).is_none());
        assert!(make_ownership_barrier_pair(vk::Buffer::from_raw(7), 0, 0, s, s, 1, 2).is_none());
        assert!(
            make_ownership_barrier_pair(vk::Buffer::from_raw(7), 0, 4, s, s, vk::QUEUE_FAMILY_IGNORED, 2)
                .is_none()
        );
    }

    #[test]
    fn cached_policy_only_requires_transfer_for_distinct_exclusive_families() {
        assert!(!QueueFamilyPolicy::new(4, 4).requires_ownership_transfer());
        assert!(!QueueFamilyPolicy::new(1, 6).requires_ownership_transfer());
        assert!(
            QueueFamilyPolicy {
                concurrent_sharing: false,
                ..QueueFamilyPolicy::new(1, 6)
            }
            .requires_ownership_transfer()
        );
        assert!(QueueFamilyPolicy::new(1, 6).uses_concurrent_sharing());
        assert!(!QueueFamilyPolicy::new(vk::QUEUE_FAMILY_IGNORED, 6).requires_ownership_transfer());
    }
}
