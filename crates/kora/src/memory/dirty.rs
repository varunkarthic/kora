//! Dirty byte ranges and non-coherent mapped-memory maintenance.

use ash::vk;

use crate::checked;
use crate::error::{Error, ErrorCode, Result, Status};
use crate::vulkan::Device;

/// A byte range `[offset, offset + size)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DirtyRange {
    /// Start.
    pub offset: u64,
    /// Length.
    pub size: u64,
}

impl DirtyRange {
    /// Creates a range.
    pub const fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }
    fn end(self) -> u64 {
        self.offset + self.size
    }
}

/// Sorted, non-empty, non-overlapping, non-adjacent dirty intervals.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DirtyRangeSet {
    ranges: Vec<DirtyRange>,
}

impl DirtyRangeSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a range, merging with overlapping or adjacent ranges. Empty ranges are ignored.
    pub fn add(&mut self, range: DirtyRange) -> Status {
        if range.size == 0 {
            return Ok(());
        }
        let end = range
            .offset
            .checked_add(range.size)
            .ok_or_else(|| Error::new(ErrorCode::Overflow, "dirty range endpoint overflows u64"))?;
        let first = self.ranges.partition_point(|r| r.end() < range.offset);
        let mut last = first;
        let (mut start, mut stop) = (range.offset, end);
        while last < self.ranges.len() && self.ranges[last].offset <= stop {
            start = start.min(self.ranges[last].offset);
            stop = stop.max(self.ranges[last].end());
            last += 1;
        }
        self.ranges
            .splice(first..last, [DirtyRange::new(start, stop - start)]);
        Ok(())
    }

    /// Removes every range.
    pub fn clear(&mut self) {
        self.ranges.clear();
    }

    /// The merged ranges.
    pub fn ranges(&self) -> &[DirtyRange] {
        &self.ranges
    }

    /// `true` when nothing is dirty.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Expands every range to atom boundaries, clamping only at the allocation end, and merges
    /// ranges that touch after alignment.
    pub fn aligned(&self, allocation_size: u64, atom_size: u64) -> Result<Vec<DirtyRange>> {
        if !checked::is_pow2(atom_size) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "non-coherent atom size must be a non-zero power of two",
            ));
        }
        let mut result: Vec<DirtyRange> = Vec::with_capacity(self.ranges.len());
        for range in &self.ranges {
            if !checked::in_range(range.offset, range.size, allocation_size) {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    "dirty range exceeds mapped allocation",
                ));
            }
            let start = checked::align_down(range.offset, atom_size).unwrap_or(range.offset);
            let end = checked::align_up(range.end(), atom_size)
                .unwrap_or(allocation_size)
                .min(allocation_size);
            match result.last_mut() {
                Some(previous) if start <= previous.end() => {
                    previous.size = previous.end().max(end) - previous.offset
                }
                _ => result.push(DirtyRange::new(start, end - start)),
            }
        }
        Ok(result)
    }
}

/// Host-visible memory maintenance operations (implemented by [`Device`]; faked in tests).
pub trait MappedMemoryOps {
    /// `vkFlushMappedMemoryRanges`.
    fn flush_ranges(&self, ranges: &[vk::MappedMemoryRange<'_>]) -> std::result::Result<(), vk::Result>;
    /// `vkInvalidateMappedMemoryRanges`.
    fn invalidate_ranges(&self, ranges: &[vk::MappedMemoryRange<'_>]) -> std::result::Result<(), vk::Result>;
}

impl MappedMemoryOps for Device {
    fn flush_ranges(&self, ranges: &[vk::MappedMemoryRange<'_>]) -> std::result::Result<(), vk::Result> {
        // SAFETY: every range references mapped memory of this device (built by
        // `mapped_memory_operation` from a live allocation) with atom-aligned bounds.
        unsafe { self.raw().flush_mapped_memory_ranges(ranges) }
    }
    fn invalidate_ranges(&self, ranges: &[vk::MappedMemoryRange<'_>]) -> std::result::Result<(), vk::Result> {
        // SAFETY: as for `flush_ranges`.
        unsafe { self.raw().invalidate_mapped_memory_ranges(ranges) }
    }
}

fn mapped_memory_operation(
    ops: &dyn MappedMemoryOps,
    memory: vk::DeviceMemory,
    allocation_size: u64,
    atom_size: u64,
    ranges: &[DirtyRange],
    host_coherent: bool,
    flush: bool,
) -> Status {
    if host_coherent || ranges.is_empty() {
        return Ok(());
    }
    if memory == vk::DeviceMemory::null() {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "mapped-memory operation requires a live memory handle",
        ));
    }
    let mut dirty = DirtyRangeSet::new();
    for &range in ranges {
        dirty.add(range)?;
    }
    let mapped: Vec<vk::MappedMemoryRange<'_>> = dirty
        .aligned(allocation_size, atom_size)?
        .into_iter()
        .map(|range| {
            let size = if range.end() == allocation_size {
                vk::WHOLE_SIZE
            } else {
                range.size
            };
            vk::MappedMemoryRange::default()
                .memory(memory)
                .offset(range.offset)
                .size(size)
        })
        .collect();
    if mapped.is_empty() {
        return Ok(());
    }
    let result = if flush {
        ops.flush_ranges(&mapped)
    } else {
        ops.invalidate_ranges(&mapped)
    };
    result.map_err(|r| {
        let name = if flush {
            "vkFlushMappedMemoryRanges"
        } else {
            "vkInvalidateMappedMemoryRanges"
        };
        crate::vulkan::vk_error(ErrorCode::TransferFailed, format!("{name} failed"), r)
    })
}

/// Makes host writes in `ranges` visible to the device (no-op for coherent memory).
pub fn flush_mapped_ranges(
    ops: &dyn MappedMemoryOps,
    memory: vk::DeviceMemory,
    allocation_size: u64,
    atom_size: u64,
    ranges: &[DirtyRange],
    host_coherent: bool,
) -> Status {
    mapped_memory_operation(
        ops,
        memory,
        allocation_size,
        atom_size,
        ranges,
        host_coherent,
        true,
    )
}

/// Makes device writes in `ranges` visible to the host (no-op for coherent memory).
pub fn invalidate_mapped_ranges(
    ops: &dyn MappedMemoryOps,
    memory: vk::DeviceMemory,
    allocation_size: u64,
    atom_size: u64,
    ranges: &[DirtyRange],
    host_coherent: bool,
) -> Status {
    mapped_memory_operation(
        ops,
        memory,
        allocation_size,
        atom_size,
        ranges,
        host_coherent,
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task_graph::tests::SplitMix;
    use std::cell::RefCell;

    #[derive(Default)]
    struct FakeMemory {
        flushed: RefCell<Vec<(u64, u64)>>,
        invalidated: RefCell<Vec<(u64, u64)>>,
        calls: RefCell<(u32, u32)>,
        flush_result: Option<vk::Result>,
    }

    impl MappedMemoryOps for FakeMemory {
        fn flush_ranges(&self, ranges: &[vk::MappedMemoryRange<'_>]) -> std::result::Result<(), vk::Result> {
            self.calls.borrow_mut().0 += 1;
            *self.flushed.borrow_mut() = ranges.iter().map(|r| (r.offset, r.size)).collect();
            self.flush_result.map_or(Ok(()), Err)
        }
        fn invalidate_ranges(
            &self,
            ranges: &[vk::MappedMemoryRange<'_>],
        ) -> std::result::Result<(), vk::Result> {
            self.calls.borrow_mut().1 += 1;
            *self.invalidated.borrow_mut() = ranges.iter().map(|r| (r.offset, r.size)).collect();
            Ok(())
        }
    }

    fn memory() -> vk::DeviceMemory {
        vk::Handle::from_raw(2)
    }

    #[test]
    fn merges_overlapping_and_adjacent_ranges_in_sorted_order() {
        let mut dirty = DirtyRangeSet::new();
        for (offset, size) in [(24, 8), (0, 8), (8, 16), (64, 4), (66, 8)] {
            dirty.add(DirtyRange::new(offset, size)).unwrap();
        }
        assert_eq!(dirty.ranges(), [DirtyRange::new(0, 32), DirtyRange::new(64, 10)]);
        dirty.add(DirtyRange::new(100, 0)).unwrap();
        assert_eq!(dirty.ranges().len(), 2);
    }

    #[test]
    fn aligns_to_atoms_and_clamps_allocation_end() {
        let mut dirty = DirtyRangeSet::new();
        dirty.add(DirtyRange::new(3, 5)).unwrap();
        dirty.add(DirtyRange::new(70, 10)).unwrap();
        assert_eq!(dirty.aligned(100, 64).unwrap(), [DirtyRange::new(0, 100)]);
        let mut tail = DirtyRangeSet::new();
        tail.add(DirtyRange::new(96, 4)).unwrap();
        assert_eq!(tail.aligned(100, 64).unwrap(), [DirtyRange::new(64, 36)]);
    }

    #[test]
    fn rejects_invalid_alignment_out_of_bounds_and_overflow() {
        let mut dirty = DirtyRangeSet::new();
        assert_eq!(
            dirty.add(DirtyRange::new(u64::MAX, 1)).unwrap_err().code(),
            ErrorCode::Overflow
        );
        dirty.add(DirtyRange::new(10, 2)).unwrap();
        assert_eq!(
            dirty.aligned(11, 64).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            dirty.aligned(12, 0).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            dirty.aligned(12, 3).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn random_intervals_match_byte_oracle() {
        const CAPACITY: usize = 2048;
        let mut touched = [false; CAPACITY];
        let mut dirty = DirtyRangeSet::new();
        let mut random = SplitMix(0x25D17);
        for _ in 0..1500 {
            let offset = random.below(CAPACITY as u64) as usize;
            let size = 1 + random.below((CAPACITY - offset) as u64) as usize;
            dirty.add(DirtyRange::new(offset as u64, size as u64)).unwrap();
            touched[offset..offset + size].iter_mut().for_each(|t| *t = true);
        }
        let mut expected = Vec::new();
        let mut index = 0;
        while index < CAPACITY {
            if !touched[index] {
                index += 1;
                continue;
            }
            let start = index;
            while index < CAPACITY && touched[index] {
                index += 1;
            }
            expected.push(DirtyRange::new(start as u64, (index - start) as u64));
        }
        assert_eq!(dirty.ranges(), expected.as_slice());
    }

    #[test]
    fn random_aligned_ranges_cover_every_dirty_byte() {
        const CAPACITY: u64 = 4099;
        const ATOM: u64 = 64;
        let mut touched = vec![false; CAPACITY as usize];
        let mut dirty = DirtyRangeSet::new();
        let mut random = SplitMix(0xA11E25);
        for _ in 0..700 {
            let offset = random.below(CAPACITY);
            let size = 1 + random.below(CAPACITY - offset);
            dirty.add(DirtyRange::new(offset, size)).unwrap();
            touched[offset as usize..(offset + size) as usize]
                .iter_mut()
                .for_each(|t| *t = true);
        }
        let aligned = dirty.aligned(CAPACITY, ATOM).unwrap();
        let mut prior_end = 0;
        for range in &aligned {
            assert_eq!(range.offset % ATOM, 0);
            assert!(range.offset >= prior_end);
            assert!(range.end() <= CAPACITY);
            assert!(range.end() == CAPACITY || range.size % ATOM == 0);
            prior_end = range.end();
        }
        for (byte, _) in touched.iter().enumerate().filter(|(_, t)| **t) {
            assert!(
                aligned
                    .iter()
                    .any(|r| (byte as u64) >= r.offset && (byte as u64) < r.end())
            );
        }
    }

    #[test]
    fn flushes_and_invalidates_merged_ranges_and_skips_coherent_memory() {
        let fake = FakeMemory::default();
        let ranges = [DirtyRange::new(3, 5), DirtyRange::new(70, 10)];
        flush_mapped_ranges(&fake, memory(), 100, 64, &ranges, false).unwrap();
        assert_eq!(*fake.flushed.borrow(), [(0, vk::WHOLE_SIZE)]);
        invalidate_mapped_ranges(&fake, memory(), 100, 64, &ranges, false).unwrap();
        assert_eq!(*fake.invalidated.borrow(), [(0, vk::WHOLE_SIZE)]);
        flush_mapped_ranges(&fake, vk::DeviceMemory::null(), 0, 0, &ranges, true).unwrap();
        invalidate_mapped_ranges(&fake, vk::DeviceMemory::null(), 0, 0, &ranges, true).unwrap();
        assert_eq!(*fake.calls.borrow(), (1, 1));
        let partial = [DirtyRange::new(0, 10)];
        flush_mapped_ranges(&fake, memory(), 1000, 64, &partial, false).unwrap();
        assert_eq!(*fake.flushed.borrow(), [(0, 64)]);
    }

    #[test]
    fn propagates_vulkan_failures() {
        let fake = FakeMemory {
            flush_result: Some(vk::Result::ERROR_DEVICE_LOST),
            ..Default::default()
        };
        let error =
            flush_mapped_ranges(&fake, memory(), 128, 64, &[DirtyRange::new(16, 8)], false).unwrap_err();
        assert_eq!(error.code(), ErrorCode::TransferFailed);
        assert_eq!(error.vk_result(), vk::Result::ERROR_DEVICE_LOST.as_raw());
        assert!(
            flush_mapped_ranges(
                &fake,
                vk::DeviceMemory::null(),
                128,
                64,
                &[DirtyRange::new(0, 1)],
                false
            )
            .is_err()
        );
    }
}
