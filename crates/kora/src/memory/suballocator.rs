//! Two-level segregated-fit (TLSF-style) suballocator for offsets inside one memory block.
//!
//! Free regions live in 64 power-of-two classes with four linear subdivisions each; two bitmaps
//! find the first non-empty class in O(1). Occupied spans are rounded to 16 bytes, alignment prefixes
//! and suffixes are split off, and freed regions coalesce with both neighbours immediately.
//! Regions are stored in an index arena (no pointers), so the structure is entirely safe Rust.
//! Statistics are maintained incrementally, so `stats()` is O(classes) instead of O(regions).

use std::collections::HashMap;

use crate::checked;
use crate::error::{Error, ErrorCode, Result, Status};

const FIRST_LEVELS: usize = 64;
const SECOND_LEVELS: usize = 4;
const BINS: usize = FIRST_LEVELS * SECOND_LEVELS;
const QUANTUM: u64 = 16;
const NONE: usize = usize::MAX;

/// A byte range returned by [`Suballocator::allocate`]. `allocated_size` includes rounding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Suballocation {
    /// Offset within the block.
    pub offset: u64,
    /// Size the caller asked for.
    pub requested_size: u64,
    /// Size actually reserved (rounded to 16 bytes).
    pub allocated_size: u64,
}

/// Allocator statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SuballocatorStats {
    /// Capacity.
    pub total_bytes: u64,
    /// Bytes in live allocations (rounded).
    pub allocated_bytes: u64,
    /// Bytes requested by live allocations.
    pub requested_bytes: u64,
    /// Free bytes.
    pub free_bytes: u64,
    /// `allocated - requested`.
    pub internal_fragmentation_bytes: u64,
    /// `free - largest_free_block`.
    pub external_fragmentation_bytes: u64,
    /// Largest free region.
    pub largest_free_block: u64,
    /// Peak `allocated_bytes`.
    pub high_water_mark: u64,
    /// Live allocation count.
    pub live_allocations: usize,
}

#[derive(Debug, Clone, Copy)]
struct Region {
    offset: u64,
    size: u64,
    requested: u64,
    previous: usize,
    next: usize,
    previous_free: usize,
    next_free: usize,
    free: bool,
    active: bool,
}

impl Region {
    const fn new(offset: u64, size: u64) -> Self {
        Self {
            offset,
            size,
            requested: 0,
            previous: NONE,
            next: NONE,
            previous_free: NONE,
            next_free: NONE,
            free: false,
            active: true,
        }
    }
}

/// Offset allocator for one block. Not internally synchronized.
#[derive(Debug, Clone)]
pub struct Suballocator {
    capacity: u64,
    regions: Vec<Region>,
    recycled: Vec<usize>,
    first: usize,
    heads: [usize; BINS],
    first_bitmap: u64,
    second_bitmaps: [u8; FIRST_LEVELS],
    live: HashMap<u64, usize>,
    allocated_bytes: u64,
    requested_bytes: u64,
    high_water_mark: u64,
}

fn size_class(size: u64) -> (usize, usize) {
    debug_assert!(size != 0);
    let first = 63 - size.leading_zeros() as usize;
    let base = 1u64 << first;
    let step = (base / SECOND_LEVELS as u64).max(1);
    let second = ((size - base) / step).min(SECOND_LEVELS as u64 - 1) as usize;
    (first, second)
}

const fn bin(first: usize, second: usize) -> usize {
    first * SECOND_LEVELS + second
}

impl Suballocator {
    /// Creates an allocator over `[0, capacity)`.
    pub fn new(capacity: u64) -> Result<Suballocator> {
        if capacity == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "suballocator capacity must be greater than zero",
            ));
        }
        let mut allocator = Suballocator {
            capacity,
            regions: vec![Region {
                free: true,
                ..Region::new(0, capacity)
            }],
            recycled: Vec::new(),
            first: 0,
            heads: [NONE; BINS],
            first_bitmap: 0,
            second_bitmaps: [0; FIRST_LEVELS],
            live: HashMap::new(),
            allocated_bytes: 0,
            requested_bytes: 0,
            high_water_mark: 0,
        };
        allocator.insert_free(0);
        Ok(allocator)
    }

    /// Capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// `true` when nothing is allocated.
    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    fn create_region(&mut self, region: Region) -> usize {
        match self.recycled.pop() {
            Some(index) => {
                self.regions[index] = region;
                index
            }
            None => {
                self.regions.push(region);
                self.regions.len() - 1
            }
        }
    }

    fn retire_region(&mut self, index: usize) {
        self.regions[index] = Region {
            active: false,
            ..Region::new(0, 0)
        };
        self.recycled.push(index);
    }

    fn insert_free(&mut self, index: usize) {
        let (first, second) = size_class(self.regions[index].size);
        let b = bin(first, second);
        let head = self.heads[b];
        let region = &mut self.regions[index];
        region.previous_free = NONE;
        region.next_free = head;
        if head != NONE {
            self.regions[head].previous_free = index;
        }
        self.heads[b] = index;
        self.second_bitmaps[first] |= 1 << second;
        self.first_bitmap |= 1 << first;
    }

    fn remove_free(&mut self, index: usize) {
        let region = self.regions[index];
        let (first, second) = size_class(region.size);
        let b = bin(first, second);
        if region.previous_free == NONE {
            self.heads[b] = region.next_free;
        } else {
            self.regions[region.previous_free].next_free = region.next_free;
        }
        if region.next_free != NONE {
            self.regions[region.next_free].previous_free = region.previous_free;
        }
        self.regions[index].previous_free = NONE;
        self.regions[index].next_free = NONE;
        if self.heads[b] == NONE {
            self.second_bitmaps[first] &= !(1 << second);
            if self.second_bitmaps[first] == 0 {
                self.first_bitmap &= !(1 << first);
            }
        }
    }

    fn next_nonempty_bin(&self, first: usize, second: usize) -> Option<(usize, usize)> {
        let mask = if second >= SECOND_LEVELS {
            0
        } else {
            0xFu8 << second
        };
        let remaining = self.second_bitmaps[first] & mask;
        if remaining != 0 {
            return Some((first, remaining.trailing_zeros() as usize));
        }
        if first == FIRST_LEVELS - 1 {
            return None;
        }
        let remaining_first = self.first_bitmap & (!0u64 << (first + 1));
        if remaining_first == 0 {
            return None;
        }
        let next = remaining_first.trailing_zeros() as usize;
        Some((next, self.second_bitmaps[next].trailing_zeros() as usize))
    }

    fn find_fit(&self, size: u64, alignment: u64) -> Option<(usize, u64)> {
        let (mut first, mut second) = size_class(size);
        loop {
            let mut index = self.heads[bin(first, second)];
            while index != NONE {
                let region = &self.regions[index];
                if let Some(aligned) = checked::align_up(region.offset, alignment)
                    && let (Some(end), Some(region_end)) =
                        (aligned.checked_add(size), region.offset.checked_add(region.size))
                    && end <= region_end
                {
                    return Some((index, aligned));
                }
                index = region.next_free;
            }
            (first, second) = self.next_nonempty_bin(first, second + 1)?;
        }
    }

    /// Allocates `size` bytes at a multiple of the power-of-two `alignment`.
    pub fn allocate(&mut self, size: u64, alignment: u64) -> Result<Suballocation> {
        if size == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "suballocation size must be greater than zero",
            ));
        }
        if !checked::is_pow2(alignment) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "suballocation alignment must be a non-zero power of two",
            ));
        }
        let rounded = checked::align_up(size, QUANTUM)
            .ok_or_else(|| Error::new(ErrorCode::Overflow, "suballocation size rounding overflow"))?;
        let (free_index, offset) = self.find_fit(rounded, alignment).ok_or_else(|| {
            Error::new(
                ErrorCode::AllocationFailed,
                "no free suballocation region satisfies the requested size and alignment",
            )
        })?;
        let original = self.regions[free_index];
        let prefix = offset - original.offset;
        let end = offset + rounded;
        let suffix = original.offset + original.size - end;
        self.remove_free(free_index);

        let allocation_index = if prefix != 0 {
            self.regions[free_index] = Region {
                previous: original.previous,
                free: true,
                ..Region::new(original.offset, prefix)
            };
            let index = self.create_region(Region {
                requested: size,
                previous: free_index,
                next: original.next,
                ..Region::new(offset, rounded)
            });
            self.regions[free_index].next = index;
            self.insert_free(free_index);
            index
        } else {
            self.regions[free_index] = Region {
                requested: size,
                previous: original.previous,
                next: original.next,
                ..Region::new(offset, rounded)
            };
            free_index
        };
        if suffix != 0 {
            let suffix_index = self.create_region(Region {
                previous: allocation_index,
                next: original.next,
                free: true,
                ..Region::new(end, suffix)
            });
            self.regions[allocation_index].next = suffix_index;
            if original.next != NONE {
                self.regions[original.next].previous = suffix_index;
            }
            self.insert_free(suffix_index);
        } else if original.next != NONE {
            self.regions[original.next].previous = allocation_index;
        }
        self.live.insert(offset, allocation_index);
        self.allocated_bytes += rounded;
        self.requested_bytes += size;
        self.high_water_mark = self.high_water_mark.max(self.allocated_bytes);
        self.debug_validate();
        Ok(Suballocation {
            offset,
            requested_size: size,
            allocated_size: rounded,
        })
    }

    /// Frees a live allocation; unknown, mismatched, or repeated frees are errors.
    pub fn deallocate(&mut self, allocation: &Suballocation) -> Status {
        let Some(&index) = self.live.get(&allocation.offset) else {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "suballocation is unknown or has already been freed",
            ));
        };
        let region = self.regions[index];
        if region.requested != allocation.requested_size || region.size != allocation.allocated_size {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "suballocation handle does not match the live allocation",
            ));
        }
        self.live.remove(&allocation.offset);
        self.allocated_bytes -= region.size;
        self.requested_bytes -= region.requested;
        self.regions[index].requested = 0;
        self.regions[index].free = true;
        let mut index = index;
        let previous = self.regions[index].previous;
        if previous != NONE && self.regions[previous].free {
            self.remove_free(previous);
            let current = self.regions[index];
            self.regions[previous].size += current.size;
            self.regions[previous].next = current.next;
            if current.next != NONE {
                self.regions[current.next].previous = previous;
            }
            self.retire_region(index);
            index = previous;
        }
        let next = self.regions[index].next;
        if next != NONE && self.regions[next].free {
            self.remove_free(next);
            let following = self.regions[next];
            self.regions[index].size += following.size;
            self.regions[index].next = following.next;
            if following.next != NONE {
                self.regions[following.next].previous = index;
            }
            self.retire_region(next);
        }
        self.insert_free(index);
        self.debug_validate();
        Ok(())
    }

    /// Largest free region, found from the highest non-empty class.
    pub fn largest_free_block(&self) -> u64 {
        if self.first_bitmap == 0 {
            return 0;
        }
        let first = 63 - self.first_bitmap.leading_zeros() as usize;
        let second = 7 - self.second_bitmaps[first].leading_zeros() as usize;
        let mut largest = 0;
        let mut index = self.heads[bin(first, second)];
        while index != NONE {
            largest = largest.max(self.regions[index].size);
            index = self.regions[index].next_free;
        }
        largest
    }

    /// Statistics (computed from running counters and the class bitmaps).
    pub fn stats(&self) -> SuballocatorStats {
        let free_bytes = self.capacity - self.allocated_bytes;
        let largest = self.largest_free_block();
        SuballocatorStats {
            total_bytes: self.capacity,
            allocated_bytes: self.allocated_bytes,
            requested_bytes: self.requested_bytes,
            free_bytes,
            internal_fragmentation_bytes: self.allocated_bytes - self.requested_bytes,
            external_fragmentation_bytes: free_bytes - largest,
            largest_free_block: largest,
            high_water_mark: self.high_water_mark,
            live_allocations: self.live.len(),
        }
    }

    /// Exhaustively checks every structural invariant (O(regions)); used by tests and debug builds.
    pub fn validate_invariants(&self) -> bool {
        if self.capacity == 0 || self.first == NONE || self.regions.is_empty() {
            return false;
        }
        let mut membership = vec![0u32; self.regions.len()];
        for first in 0..FIRST_LEVELS {
            if (self.first_bitmap & (1 << first) != 0) != (self.second_bitmaps[first] != 0) {
                return false;
            }
            for second in 0..SECOND_LEVELS {
                let head = self.heads[bin(first, second)];
                if (self.second_bitmaps[first] & (1 << second) != 0) != (head != NONE) {
                    return false;
                }
                let (mut previous, mut visited, mut index) = (NONE, 0, head);
                while index != NONE {
                    visited += 1;
                    if visited > self.regions.len() || index >= self.regions.len() {
                        return false;
                    }
                    let region = &self.regions[index];
                    if !region.active
                        || !region.free
                        || region.previous_free != previous
                        || size_class(region.size) != (first, second)
                    {
                        return false;
                    }
                    membership[index] += 1;
                    if membership[index] != 1 {
                        return false;
                    }
                    previous = index;
                    index = region.next_free;
                }
            }
        }
        let (mut expected_offset, mut previous, mut live, mut visited, mut index) =
            (0u64, NONE, 0usize, 0usize, self.first);
        let (mut allocated, mut requested) = (0u64, 0u64);
        while index != NONE {
            visited += 1;
            if visited > self.regions.len() || index >= self.regions.len() {
                return false;
            }
            let region = &self.regions[index];
            if !region.active
                || region.size == 0
                || region.previous != previous
                || region.offset != expected_offset
            {
                return false;
            }
            if region.free != (membership[index] == 1) {
                return false;
            }
            if region.free && region.next != NONE && self.regions[region.next].free {
                return false;
            }
            let Some(end) = region.offset.checked_add(region.size) else {
                return false;
            };
            expected_offset = end;
            if !region.free {
                if self.live.get(&region.offset) != Some(&index)
                    || region.requested == 0
                    || region.requested > region.size
                {
                    return false;
                }
                live += 1;
                allocated += region.size;
                requested += region.requested;
            }
            previous = index;
            index = region.next;
        }
        expected_offset == self.capacity
            && live == self.live.len()
            && allocated == self.allocated_bytes
            && requested == self.requested_bytes
    }

    fn debug_validate(&self) {
        #[cfg(debug_assertions)]
        if self.regions.len() <= 1024 {
            debug_assert!(self.validate_invariants());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task_graph::tests::SplitMix;

    fn alloc(allocator: &mut Suballocator, size: u64, alignment: u64) -> Suballocation {
        allocator.allocate(size, alignment).unwrap()
    }

    #[test]
    fn splits_and_tracks_fragmentation_statistics() {
        let mut allocator = Suballocator::new(256).unwrap();
        let first = alloc(&mut allocator, 17, 1);
        let second = alloc(&mut allocator, 32, 64);
        assert_eq!((first.offset, first.allocated_size), (0, 32));
        assert_eq!((second.offset, second.allocated_size), (64, 32));
        let stats = allocator.stats();
        assert_eq!(stats.total_bytes, 256);
        assert_eq!(stats.allocated_bytes, 64);
        assert_eq!(stats.requested_bytes, 49);
        assert_eq!(stats.internal_fragmentation_bytes, 15);
        assert_eq!(stats.free_bytes, 192);
        assert_eq!(stats.largest_free_block, 160);
        assert_eq!(stats.external_fragmentation_bytes, 32);
        assert_eq!(stats.live_allocations, 2);
        assert_eq!(stats.high_water_mark, 64);
        assert!(allocator.validate_invariants());
    }

    #[test]
    fn coalesces_no_left_right_and_both_neighbours() {
        let mut a = Suballocator::new(96).unwrap();
        let (l, m, r) = (alloc(&mut a, 32, 1), alloc(&mut a, 32, 1), alloc(&mut a, 32, 1));
        a.deallocate(&m).unwrap();
        assert_eq!(a.stats().largest_free_block, 32);
        assert_eq!(a.stats().live_allocations, 2);
        a.deallocate(&l).unwrap();
        a.deallocate(&r).unwrap();
        assert_eq!(a.stats().largest_free_block, 96);

        let mut a = Suballocator::new(96).unwrap();
        let (l, m, r) = (alloc(&mut a, 32, 1), alloc(&mut a, 32, 1), alloc(&mut a, 32, 1));
        a.deallocate(&l).unwrap();
        a.deallocate(&m).unwrap();
        assert_eq!(a.stats().largest_free_block, 64);
        a.deallocate(&r).unwrap();
        assert_eq!(a.stats().largest_free_block, 96);

        let mut a = Suballocator::new(96).unwrap();
        let (l, m, r) = (alloc(&mut a, 32, 1), alloc(&mut a, 32, 1), alloc(&mut a, 32, 1));
        a.deallocate(&r).unwrap();
        a.deallocate(&m).unwrap();
        assert_eq!(a.stats().largest_free_block, 64);
        a.deallocate(&l).unwrap();
        assert_eq!(a.stats().largest_free_block, 96);

        let mut a = Suballocator::new(128).unwrap();
        let blocks: Vec<_> = (0..4).map(|_| alloc(&mut a, 32, 1)).collect();
        a.deallocate(&blocks[0]).unwrap();
        a.deallocate(&blocks[2]).unwrap();
        a.deallocate(&blocks[1]).unwrap();
        assert_eq!(a.stats().largest_free_block, 96);
        a.deallocate(&blocks[3]).unwrap();
        assert_eq!(a.stats().largest_free_block, 128);
        assert!(a.validate_invariants());
    }

    #[test]
    fn honors_power_of_two_alignment_and_reports_bad_requests() {
        assert_eq!(
            Suballocator::new(0).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        let mut a = Suballocator::new(1024).unwrap();
        for alignment in [1, 2, 8, 64, 256] {
            assert_eq!(alloc(&mut a, 1, alignment).offset % alignment, 0);
        }
        assert_eq!(a.allocate(0, 1).unwrap_err().code(), ErrorCode::InvalidArgument);
        assert_eq!(a.allocate(1, 0).unwrap_err().code(), ErrorCode::InvalidArgument);
        assert_eq!(a.allocate(1, 3).unwrap_err().code(), ErrorCode::InvalidArgument);
        assert_eq!(a.allocate(u64::MAX, 1).unwrap_err().code(), ErrorCode::Overflow);
    }

    #[test]
    fn exhaustion_and_invalid_free_are_recoverable() {
        let mut a = Suballocator::new(64).unwrap();
        let all = alloc(&mut a, 64, 1);
        assert_eq!(a.allocate(1, 1).unwrap_err().code(), ErrorCode::AllocationFailed);
        let wrong = Suballocation {
            requested_size: 63,
            ..all
        };
        assert_eq!(
            a.deallocate(&wrong).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        a.deallocate(&all).unwrap();
        assert_eq!(a.deallocate(&all).unwrap_err().code(), ErrorCode::InvalidArgument);
        assert_eq!(a.stats().largest_free_block, 64);
        assert!(a.is_empty());
    }

    #[test]
    fn randomized_operations_match_interval_oracle() {
        let capacity = 8192;
        let mut a = Suballocator::new(capacity).unwrap();
        let mut live: Vec<Suballocation> = Vec::new();
        let mut random = SplitMix(0x4b4f_5241);
        for _ in 0..5000 {
            let allocate = live.is_empty() || (live.len() < 80 && random.below(100) < 65);
            if allocate {
                let size = 1 + random.below(257);
                let alignment = 1 << random.below(9);
                match a.allocate(size, alignment) {
                    Ok(candidate) => {
                        assert_eq!(candidate.offset % alignment, 0);
                        assert!(candidate.offset + candidate.allocated_size <= capacity);
                        for existing in &live {
                            assert!(
                                candidate.offset + candidate.allocated_size <= existing.offset
                                    || existing.offset + existing.allocated_size <= candidate.offset
                            );
                        }
                        live.push(candidate);
                    }
                    Err(error) => assert_eq!(error.code(), ErrorCode::AllocationFailed),
                }
            } else {
                let position = random.below(live.len() as u64) as usize;
                a.deallocate(&live.swap_remove(position)).unwrap();
            }
            assert!(a.validate_invariants());
            let stats = a.stats();
            assert_eq!(stats.allocated_bytes + stats.free_bytes, capacity);
            assert_eq!(stats.live_allocations, live.len());
        }
        for allocation in &live {
            a.deallocate(allocation).unwrap();
        }
        let stats = a.stats();
        assert_eq!(
            (
                stats.free_bytes,
                stats.largest_free_block,
                stats.external_fragmentation_bytes,
                stats.live_allocations
            ),
            (capacity, capacity, 0, 0)
        );
        assert!(a.validate_invariants());
    }

    #[test]
    fn largest_free_block_matches_a_full_scan() {
        let mut a = Suballocator::new(1 << 20).unwrap();
        let mut random = SplitMix(7);
        let mut live = Vec::new();
        for _ in 0..2000 {
            if live.len() < 200 && random.below(3) != 0 {
                if let Ok(x) = a.allocate(1 + random.below(4096), 1 << random.below(6)) {
                    live.push(x);
                }
            } else if !live.is_empty() {
                let index = random.below(live.len() as u64) as usize;
                a.deallocate(&live.swap_remove(index)).unwrap();
            }
            let mut scan = 0;
            let mut index = a.first;
            while index != NONE {
                if a.regions[index].free {
                    scan = scan.max(a.regions[index].size);
                }
                index = a.regions[index].next;
            }
            assert_eq!(a.largest_free_block(), scan);
        }
    }

    #[test]
    fn size_classes_are_monotone() {
        let mut previous = (0, 0);
        for size in 1..4096u64 {
            let class = size_class(size);
            assert!(class >= previous);
            previous = class;
        }
        assert_eq!(size_class(u64::MAX).0, 63);
    }
}
