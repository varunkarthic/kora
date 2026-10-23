//! Memory-type selection and per-heap budget accounting.

use ash::vk;

use super::pool::MemoryPoolStats;
use crate::device::{DeviceInfo, MemoryTopology, MemoryTypeInfo, MemoryTypePreferences};
use crate::error::{Error, ErrorCode, Result, Status};
use crate::vulkan::MemoryBudget;

/// Access pattern used to rank memory types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryPolicy {
    /// Topology-aware default.
    #[default]
    Auto,
    /// Device-only data.
    DevicePreferred,
    /// Must be mappable.
    HostVisible,
    /// Mostly read by the host.
    ReadMostly,
    /// Mostly written by the host.
    WriteMostly,
    /// Streamed every use.
    Streaming,
    /// Read back to the host.
    Readback,
}

/// Hard requirements a memory type must meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemoryTypeRequirements {
    /// Device local.
    pub device_local: bool,
    /// Host visible.
    pub host_visible: bool,
    /// Host coherent.
    pub host_coherent: bool,
    /// Host cached.
    pub host_cached: bool,
}

/// Budget inputs for one heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HeapBudget {
    /// Usable bytes for KORA.
    pub limit_bytes: u64,
    /// Bytes used by others.
    pub externally_used_bytes: u64,
    /// Came from `VK_EXT_memory_budget`.
    pub queried: bool,
    /// Physical heap size (0 = unknown).
    pub heap_size_bytes: u64,
}

/// Budget state for one heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HeapBudgetStats {
    /// Effective limit.
    pub limit_bytes: u64,
    /// Reserved by KORA.
    pub reserved_bytes: u64,
    /// Still available.
    pub available_bytes: u64,
    /// Used by others.
    pub externally_used_bytes: u64,
    /// Came from `VK_EXT_memory_budget`.
    pub queried: bool,
}

/// Per-heap telemetry returned by `Runtime::memory_stats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemoryHeapStats {
    /// Heap index.
    pub heap_index: u32,
    /// Effective budget.
    pub budget_bytes: u64,
    /// Reserved by KORA.
    pub reserved_bytes: u64,
    /// Still available.
    pub available_bytes: u64,
    /// Used by others.
    pub externally_used_bytes: u64,
    /// Came from `VK_EXT_memory_budget`.
    pub budget_queried: bool,
}

/// Point-in-time sum of a device's memory pools plus per-heap budgets.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MemoryStats {
    /// Device memory owned by pools.
    pub reserved_bytes: u64,
    /// Bytes handed out.
    pub allocated_bytes: u64,
    /// Bytes requested.
    pub requested_bytes: u64,
    /// Free bytes inside blocks.
    pub free_bytes: u64,
    /// Rounding waste.
    pub internal_fragmentation_bytes: u64,
    /// Free bytes outside each block's largest free region.
    pub external_fragmentation_bytes: u64,
    /// Largest free region.
    pub largest_free_block: u64,
    /// Sum of pool peaks.
    pub high_water_mark: u64,
    /// Pools.
    pub pools: usize,
    /// Blocks.
    pub blocks: usize,
    /// Dedicated allocations.
    pub dedicated_allocations: usize,
    /// Live allocations.
    pub live_allocations: usize,
    /// `vkAllocateMemory` calls.
    pub device_allocation_calls: u64,
    /// Per-heap budgets.
    pub heaps: Vec<MemoryHeapStats>,
}

fn scale(value: u64, factor: f64) -> u64 {
    (value as f64 * factor) as u64
}

/// Underflow-safe accounting of KORA reservations against each heap's usable budget.
#[derive(Debug, Clone)]
pub struct BudgetLedger {
    heaps: Vec<HeapBudget>,
    reserved: Vec<u64>,
    shrink: Vec<f64>,
}

impl BudgetLedger {
    /// Creates a ledger; limits are clamped to known heap sizes.
    pub fn new(mut heaps: Vec<HeapBudget>) -> Self {
        for heap in &mut heaps {
            if heap.heap_size_bytes != 0 {
                heap.limit_bytes = heap.limit_bytes.min(heap.heap_size_bytes);
            }
        }
        let count = heaps.len();
        Self {
            heaps,
            reserved: vec![0; count],
            shrink: vec![1.0; count],
        }
    }

    fn index(&self, heap: u32) -> Result<usize> {
        let index = heap as usize;
        if index >= self.heaps.len() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "memory budget heap index is out of range",
            ));
        }
        Ok(index)
    }

    /// Number of heaps.
    pub fn heap_count(&self) -> usize {
        self.heaps.len()
    }

    /// Reserves bytes, failing with `OutOfMemoryBudget` above the limit.
    pub fn reserve(&mut self, heap: u32, bytes: u64) -> Status {
        let index = self.index(heap)?;
        let updated = self.reserved[index].checked_add(bytes).ok_or_else(|| {
            Error::new(
                ErrorCode::Overflow,
                "memory budget reservation arithmetic overflow",
            )
        })?;
        if updated > self.heaps[index].limit_bytes {
            return Err(Error::new(
                ErrorCode::OutOfMemoryBudget,
                "memory budget reservation exceeds usable heap budget",
            ));
        }
        self.reserved[index] = updated;
        Ok(())
    }

    /// Releases bytes; releasing more than reserved is an invariant violation.
    pub fn release(&mut self, heap: u32, bytes: u64) -> Status {
        let index = self.index(heap)?;
        if bytes > self.reserved[index] {
            return Err(Error::new(
                ErrorCode::InternalInvariantViolation,
                "memory budget release would underflow the heap reservation",
            ));
        }
        self.reserved[index] -= bytes;
        Ok(())
    }

    /// Budget state of a heap.
    pub fn stats(&self, heap: u32) -> Result<HeapBudgetStats> {
        let index = self.index(heap)?;
        let budget = &self.heaps[index];
        let reserved = self.reserved[index];
        let limit = budget.limit_bytes.max(reserved);
        Ok(HeapBudgetStats {
            limit_bytes: limit,
            reserved_bytes: reserved,
            available_bytes: limit - reserved,
            externally_used_bytes: budget.externally_used_bytes,
            queried: budget.queried,
        })
    }

    /// Replaces the budget inputs, preserving allocation-failure feedback and never letting a
    /// refresh make existing reservations appear over budget.
    pub fn refresh(&mut self, mut heaps: Vec<HeapBudget>) -> Status {
        if heaps.len() != self.heaps.len() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "refreshed memory budget heap count changed",
            ));
        }
        for (index, heap) in heaps.iter().enumerate() {
            let size = if heap.heap_size_bytes == 0 {
                self.heaps[index].heap_size_bytes
            } else {
                heap.heap_size_bytes
            };
            if size != 0 && self.reserved[index] > size {
                return Err(Error::new(
                    ErrorCode::InternalInvariantViolation,
                    "existing memory reservations exceed the physical heap size",
                ));
            }
        }
        for (index, heap) in heaps.iter_mut().enumerate() {
            let prior = self.heaps[index];
            if heap.heap_size_bytes == 0 {
                heap.heap_size_bytes = prior.heap_size_bytes;
            }
            heap.limit_bytes = heap.limit_bytes.min(scale(heap.limit_bytes, self.shrink[index]));
            if !heap.queried && !prior.queried {
                heap.limit_bytes = heap.limit_bytes.min(prior.limit_bytes);
            }
            if heap.heap_size_bytes != 0 {
                heap.limit_bytes = heap.limit_bytes.min(heap.heap_size_bytes);
            }
            heap.limit_bytes = heap.limit_bytes.max(self.reserved[index]);
            self.heaps[index] = *heap;
        }
        Ok(())
    }

    /// Lowers the limit after the driver refused an allocation.
    pub fn record_allocation_failure(&mut self, heap: u32, shrink_factor: f64) -> Status {
        let index = self.index(heap)?;
        if !(0.0..=1.0).contains(&shrink_factor) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "memory budget shrink factor must be between zero and one",
            ));
        }
        self.shrink[index] *= shrink_factor;
        let shrunk = scale(self.heaps[index].limit_bytes, shrink_factor);
        self.heaps[index].limit_bytes = shrunk.max(self.reserved[index]);
        Ok(())
    }
}

fn preference_order(preferences: &MemoryTypePreferences, policy: MemoryPolicy) -> &[u32] {
    match policy {
        MemoryPolicy::Auto => &preferences.automatic,
        MemoryPolicy::DevicePreferred => &preferences.device_preferred,
        MemoryPolicy::HostVisible => &preferences.host_visible,
        MemoryPolicy::ReadMostly => &preferences.read_mostly,
        MemoryPolicy::WriteMostly => &preferences.write_mostly,
        MemoryPolicy::Streaming => &preferences.streaming,
        MemoryPolicy::Readback => &preferences.readback,
    }
}

fn satisfies(memory_type: &MemoryTypeInfo, required: &MemoryTypeRequirements) -> bool {
    (!required.device_local || memory_type.device_local)
        && (!required.host_visible || memory_type.host_visible)
        && (!required.host_coherent || memory_type.host_coherent)
        && (!required.host_cached || memory_type.host_cached)
}

/// Selects memory types in policy order and tracks per-heap reservations.
#[derive(Debug, Clone)]
pub struct MemoryManager {
    memory_types: Vec<MemoryTypeInfo>,
    preferences: MemoryTypePreferences,
    topology: MemoryTopology,
    ledger: BudgetLedger,
}

fn heap_budgets(
    memory: &vk::PhysicalDeviceMemoryProperties,
    budget: Option<&MemoryBudget>,
    reserved: &dyn Fn(usize) -> u64,
    fraction: f64,
) -> Vec<HeapBudget> {
    (0..(memory.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS))
        .map(|index| {
            let heap_size = memory.memory_heaps[index].size;
            match budget {
                Some(budget) => {
                    let reported = budget.heap_budget[index].min(heap_size);
                    let usage = budget.heap_usage[index];
                    let external = usage.saturating_sub(reserved(index));
                    let available = reported.saturating_sub(external);
                    HeapBudget {
                        limit_bytes: scale(available, fraction).min(heap_size),
                        externally_used_bytes: external,
                        queried: true,
                        heap_size_bytes: heap_size,
                    }
                }
                None => HeapBudget {
                    limit_bytes: scale(heap_size, fraction).min(heap_size),
                    externally_used_bytes: 0,
                    queried: false,
                    heap_size_bytes: heap_size,
                },
            }
        })
        .collect()
}

fn check_fraction(fraction: f64) -> Status {
    if !(0.0..=1.0).contains(&fraction) {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "fallback heap fraction must be between zero and one",
        ));
    }
    Ok(())
}

impl MemoryManager {
    /// Builds the manager from a device's memory properties and optional `VK_EXT_memory_budget`
    /// snapshot. Without a budget, `fallback_heap_fraction` of each heap is usable.
    pub fn new(
        info: &DeviceInfo,
        memory: &vk::PhysicalDeviceMemoryProperties,
        budget: Option<&MemoryBudget>,
        fallback_heap_fraction: f64,
    ) -> Result<MemoryManager> {
        check_fraction(fallback_heap_fraction)?;
        Ok(MemoryManager {
            memory_types: info.memory_types.clone(),
            preferences: info.memory_type_preferences.clone(),
            topology: info.memory_topology,
            ledger: BudgetLedger::new(heap_budgets(memory, budget, &|_| 0, fallback_heap_fraction)),
        })
    }

    /// Selects the first compatible type in `policy` order that meets `required`.
    pub fn select_memory_type_from(
        memory_types: &[MemoryTypeInfo],
        preferences: &MemoryTypePreferences,
        policy: MemoryPolicy,
        compatible_type_bits: u32,
        required: &MemoryTypeRequirements,
    ) -> Result<u32> {
        if compatible_type_bits == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "Vulkan memory type compatibility mask is empty",
            ));
        }
        preference_order(preferences, policy)
            .iter()
            .copied()
            .find(|&index| {
                (index as usize) < memory_types.len()
                    && index < 32
                    && compatible_type_bits & (1 << index) != 0
                    && satisfies(&memory_types[index as usize], required)
            })
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::UnsupportedFeature,
                    "no compatible Vulkan memory type satisfies the selected memory policy and requirements",
                )
            })
    }

    /// [`select_memory_type_from`](Self::select_memory_type_from) with this device's types.
    pub fn select_memory_type(
        &self,
        policy: MemoryPolicy,
        compatible_type_bits: u32,
        required: &MemoryTypeRequirements,
    ) -> Result<u32> {
        Self::select_memory_type_from(
            &self.memory_types,
            &self.preferences,
            policy,
            compatible_type_bits,
            required,
        )
    }

    /// Memory type facts.
    pub fn memory_type(&self, index: u32) -> Option<&MemoryTypeInfo> {
        self.memory_types.get(index as usize)
    }
    /// Reserves heap budget.
    pub fn reserve(&mut self, heap: u32, bytes: u64) -> Status {
        self.ledger.reserve(heap, bytes)
    }
    /// Releases heap budget.
    pub fn release(&mut self, heap: u32, bytes: u64) -> Status {
        self.ledger.release(heap, bytes)
    }
    /// Re-derives limits from a fresh budget snapshot; external usage excludes KORA's own
    /// reservations.
    pub fn refresh_budget(
        &mut self,
        memory: &vk::PhysicalDeviceMemoryProperties,
        budget: Option<&MemoryBudget>,
        fallback_heap_fraction: f64,
    ) -> Status {
        check_fraction(fallback_heap_fraction)?;
        let reserved: Vec<u64> = (0..self.ledger.heap_count())
            .map(|i| self.ledger.reserved[i])
            .collect();
        let heaps = heap_budgets(
            memory,
            budget,
            &|i| reserved.get(i).copied().unwrap_or(0),
            fallback_heap_fraction,
        );
        self.ledger.refresh(heaps)
    }
    /// Shrinks a heap's limit after an allocation failure.
    pub fn record_allocation_failure(&mut self, heap: u32, shrink_factor: f64) -> Status {
        self.ledger.record_allocation_failure(heap, shrink_factor)
    }
    /// Budget state of a heap.
    pub fn budget(&self, heap: u32) -> Result<HeapBudgetStats> {
        self.ledger.stats(heap)
    }
    /// Number of heaps.
    pub fn heap_count(&self) -> usize {
        self.ledger.heap_count()
    }
    /// Type rankings.
    pub fn preferences(&self) -> &MemoryTypePreferences {
        &self.preferences
    }
    /// Topology.
    pub fn topology(&self) -> MemoryTopology {
        self.topology
    }

    /// Sums pool statistics with overflow checks.
    pub fn aggregate_stats(pools: &[MemoryPoolStats]) -> Result<MemoryStats> {
        let overflow = || Error::new(ErrorCode::Overflow, "memory pool statistics aggregation overflow");
        let mut total = MemoryStats::default();
        for pool in pools {
            let add = |a: u64, b: u64| a.checked_add(b).ok_or_else(overflow);
            total.reserved_bytes = add(total.reserved_bytes, pool.reserved_bytes)?;
            total.allocated_bytes = add(total.allocated_bytes, pool.allocated_bytes)?;
            total.requested_bytes = add(total.requested_bytes, pool.requested_bytes)?;
            total.free_bytes = add(total.free_bytes, pool.free_bytes)?;
            total.internal_fragmentation_bytes = add(
                total.internal_fragmentation_bytes,
                pool.internal_fragmentation_bytes,
            )?;
            total.external_fragmentation_bytes = add(
                total.external_fragmentation_bytes,
                pool.external_fragmentation_bytes,
            )?;
            total.high_water_mark = add(total.high_water_mark, pool.high_water_mark)?;
            total.device_allocation_calls = add(total.device_allocation_calls, pool.device_allocation_calls)?;
            total.largest_free_block = total.largest_free_block.max(pool.largest_free_block);
            total.blocks = total.blocks.checked_add(pool.blocks).ok_or_else(overflow)?;
            total.dedicated_allocations = total
                .dedicated_allocations
                .checked_add(pool.dedicated_allocations)
                .ok_or_else(overflow)?;
            total.live_allocations = total
                .live_allocations
                .checked_add(pool.live_allocations)
                .ok_or_else(overflow)?;
            total.pools += 1;
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::device_info;
    use vk::MemoryPropertyFlags as P;

    fn info_for(types: &[(P, u32)], heap_size: u64) -> (DeviceInfo, vk::PhysicalDeviceMemoryProperties) {
        let mut raw = crate::vulkan::PhysicalDeviceInfo::default();
        raw.memory_properties.memory_heap_count = 1;
        raw.memory_properties.memory_heaps[0].size = heap_size;
        raw.memory_properties.memory_type_count = types.len() as u32;
        for (index, &(flags, heap)) in types.iter().enumerate() {
            raw.memory_properties.memory_types[index] = vk::MemoryType {
                property_flags: flags,
                heap_index: heap,
            };
        }
        (device_info(&raw), raw.memory_properties)
    }

    fn select(
        info: &DeviceInfo,
        policy: MemoryPolicy,
        bits: u32,
        required: MemoryTypeRequirements,
    ) -> Result<u32> {
        MemoryManager::select_memory_type_from(
            &info.memory_types,
            &info.memory_type_preferences,
            policy,
            bits,
            &required,
        )
    }

    #[test]
    fn applies_policy_order_and_compatibility_mask() {
        let (info, _) = info_for(
            &[
                (P::DEVICE_LOCAL, 0),
                (P::HOST_VISIBLE | P::HOST_COHERENT, 0),
                (P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT, 0),
            ],
            1000,
        );
        assert_eq!(
            select(&info, MemoryPolicy::DevicePreferred, 0b110, Default::default()).unwrap(),
            2
        );
        let host = MemoryTypeRequirements {
            host_visible: true,
            host_coherent: true,
            ..Default::default()
        };
        assert_eq!(select(&info, MemoryPolicy::HostVisible, 0b111, host).unwrap(), 1);
    }

    #[test]
    fn uma_and_discrete_topologies_choose_different_automatic_types() {
        let (discrete, _) = info_for(
            &[(P::DEVICE_LOCAL, 0), (P::HOST_VISIBLE | P::HOST_COHERENT, 0)],
            1000,
        );
        assert_eq!(
            select(&discrete, MemoryPolicy::Auto, 0b11, Default::default()).unwrap(),
            0
        );
        assert_eq!(discrete.memory_topology, MemoryTopology::Discrete);
        let (uma, _) = info_for(
            &[
                (P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT, 0),
                (P::HOST_VISIBLE | P::HOST_CACHED, 0),
            ],
            1000,
        );
        assert_eq!(
            select(&uma, MemoryPolicy::Auto, 0b11, Default::default()).unwrap(),
            0
        );
        assert_eq!(uma.memory_topology, MemoryTopology::Uma);
    }

    #[test]
    fn rejects_empty_masks_and_unsatisfied_requirements() {
        let (info, _) = info_for(&[(P::DEVICE_LOCAL, 0)], 1000);
        assert_eq!(
            select(&info, MemoryPolicy::Auto, 0, Default::default())
                .unwrap_err()
                .code(),
            ErrorCode::InvalidArgument
        );
        let host = MemoryTypeRequirements {
            host_visible: true,
            ..Default::default()
        };
        assert_eq!(
            select(&info, MemoryPolicy::HostVisible, 1, host)
                .unwrap_err()
                .code(),
            ErrorCode::UnsupportedFeature
        );
    }

    #[test]
    fn ledger_enforces_limit_and_recovers_after_release() {
        let mut ledger = BudgetLedger::new(vec![HeapBudget {
            limit_bytes: 100,
            externally_used_bytes: 25,
            queried: true,
            heap_size_bytes: 0,
        }]);
        ledger.reserve(0, 80).unwrap();
        assert_eq!(
            ledger.reserve(0, 21).unwrap_err().code(),
            ErrorCode::OutOfMemoryBudget
        );
        let s = ledger.stats(0).unwrap();
        assert_eq!(
            (
                s.reserved_bytes,
                s.available_bytes,
                s.externally_used_bytes,
                s.queried
            ),
            (80, 20, 25, true)
        );
        ledger.release(0, 30).unwrap();
        ledger.reserve(0, 20).unwrap();
        assert_eq!(ledger.stats(0).unwrap().reserved_bytes, 70);
    }

    #[test]
    fn ledger_checked_arithmetic_and_invalid_operations_do_not_mutate() {
        let mut ledger = BudgetLedger::new(vec![HeapBudget {
            limit_bytes: u64::MAX,
            ..Default::default()
        }]);
        ledger.reserve(0, u64::MAX - 1).unwrap();
        assert_eq!(ledger.reserve(0, 2).unwrap_err().code(), ErrorCode::Overflow);
        assert_eq!(
            ledger.release(0, u64::MAX).unwrap_err().code(),
            ErrorCode::InternalInvariantViolation
        );
        assert_eq!(ledger.stats(0).unwrap().reserved_bytes, u64::MAX - 1);
        assert_eq!(
            ledger.reserve(1, 1).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            ledger.release(1, 1).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(ledger.stats(1).unwrap_err().code(), ErrorCode::InvalidArgument);
    }

    #[test]
    fn refresh_clamps_to_heap_and_preserves_reservations() {
        let mut ledger = BudgetLedger::new(vec![HeapBudget {
            limit_bytes: 800,
            externally_used_bytes: 0,
            queried: false,
            heap_size_bytes: 1000,
        }]);
        ledger.reserve(0, 600).unwrap();
        ledger
            .refresh(vec![HeapBudget {
                limit_bytes: 100,
                externally_used_bytes: 50,
                queried: true,
                heap_size_bytes: 1000,
            }])
            .unwrap();
        let s = ledger.stats(0).unwrap();
        assert_eq!(
            (s.limit_bytes, s.reserved_bytes, s.available_bytes),
            (600, 600, 0)
        );
        assert_eq!(
            ledger.reserve(0, 1).unwrap_err().code(),
            ErrorCode::OutOfMemoryBudget
        );
        ledger
            .refresh(vec![HeapBudget {
                limit_bytes: 2000,
                externally_used_bytes: 0,
                queried: true,
                heap_size_bytes: 1000,
            }])
            .unwrap();
        assert_eq!(ledger.stats(0).unwrap().limit_bytes, 1000);
        let oversized = BudgetLedger::new(vec![HeapBudget {
            limit_bytes: 2000,
            externally_used_bytes: 0,
            queried: true,
            heap_size_bytes: 1000,
        }]);
        assert_eq!(oversized.stats(0).unwrap().limit_bytes, 1000);
        assert!(ledger.refresh(vec![]).is_err());
    }

    #[test]
    fn failed_allocation_shrinks_unreserved_budget_safely() {
        let mut ledger = BudgetLedger::new(vec![HeapBudget {
            limit_bytes: 1000,
            externally_used_bytes: 0,
            queried: false,
            heap_size_bytes: 1200,
        }]);
        ledger.reserve(0, 200).unwrap();
        ledger.record_allocation_failure(0, 0.8).unwrap();
        let s = ledger.stats(0).unwrap();
        assert_eq!((s.limit_bytes, s.available_bytes), (800, 600));
        ledger.record_allocation_failure(0, 0.5).unwrap();
        assert_eq!(ledger.stats(0).unwrap().limit_bytes, 400);
        ledger
            .refresh(vec![HeapBudget {
                limit_bytes: 1000,
                externally_used_bytes: 0,
                queried: true,
                heap_size_bytes: 1200,
            }])
            .unwrap();
        assert_eq!(ledger.stats(0).unwrap().limit_bytes, 400);
        assert_eq!(
            ledger.record_allocation_failure(0, 1.1).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn aggregates_pool_telemetry_and_rejects_overflow() {
        let pools = [
            MemoryPoolStats {
                reserved_bytes: 1024,
                allocated_bytes: 256,
                requested_bytes: 240,
                free_bytes: 768,
                internal_fragmentation_bytes: 16,
                external_fragmentation_bytes: 128,
                largest_free_block: 640,
                high_water_mark: 512,
                blocks: 1,
                dedicated_allocations: 0,
                live_allocations: 2,
                device_allocation_calls: 1,
            },
            MemoryPoolStats {
                reserved_bytes: 2048,
                allocated_bytes: 512,
                requested_bytes: 500,
                free_bytes: 1536,
                internal_fragmentation_bytes: 12,
                external_fragmentation_bytes: 256,
                largest_free_block: 1280,
                high_water_mark: 1024,
                blocks: 1,
                dedicated_allocations: 1,
                live_allocations: 3,
                device_allocation_calls: 2,
            },
        ];
        let t = MemoryManager::aggregate_stats(&pools).unwrap();
        assert_eq!(
            (
                t.reserved_bytes,
                t.allocated_bytes,
                t.requested_bytes,
                t.free_bytes
            ),
            (3072, 768, 740, 2304)
        );
        assert_eq!(
            (
                t.internal_fragmentation_bytes,
                t.external_fragmentation_bytes,
                t.largest_free_block
            ),
            (28, 384, 1280)
        );
        assert_eq!(
            (
                t.high_water_mark,
                t.blocks,
                t.dedicated_allocations,
                t.live_allocations
            ),
            (1536, 2, 1, 5)
        );
        assert_eq!((t.device_allocation_calls, t.pools), (3, 2));
        let overflowing = [
            MemoryPoolStats {
                reserved_bytes: u64::MAX,
                ..Default::default()
            },
            MemoryPoolStats {
                reserved_bytes: 1,
                ..Default::default()
            },
        ];
        assert_eq!(
            MemoryManager::aggregate_stats(&overflowing).unwrap_err().code(),
            ErrorCode::Overflow
        );
    }

    #[test]
    fn uses_budget_extension_or_conservative_heap_fraction() {
        let (info, memory) = info_for(&[(P::DEVICE_LOCAL, 0)], 1000);
        let fallback = MemoryManager::new(&info, &memory, None, 0.5).unwrap();
        let b = fallback.budget(0).unwrap();
        assert_eq!((b.limit_bytes, b.queried), (500, false));

        let mut budget = MemoryBudget::default();
        budget.heap_budget[0] = 700;
        budget.heap_usage[0] = 250;
        let mut queried = MemoryManager::new(&info, &memory, Some(&budget), 0.8).unwrap();
        let b = queried.budget(0).unwrap();
        assert_eq!(
            (b.limit_bytes, b.externally_used_bytes, b.queried),
            (360, 250, true)
        );
        queried.reserve(0, 100).unwrap();
        budget.heap_usage[0] = 350;
        queried.refresh_budget(&memory, Some(&budget), 0.8).unwrap();
        let r = queried.budget(0).unwrap();
        assert_eq!(
            (r.limit_bytes, r.reserved_bytes, r.externally_used_bytes),
            (360, 100, 250)
        );
        queried.record_allocation_failure(0, 0.8).unwrap();
        queried.refresh_budget(&memory, Some(&budget), 0.8).unwrap();
        assert_eq!(queried.budget(0).unwrap().limit_bytes, 288);
        assert_eq!(
            MemoryManager::new(&info, &memory, Some(&budget), 1.1)
                .unwrap_err()
                .code(),
            ErrorCode::InvalidArgument
        );
    }
}
