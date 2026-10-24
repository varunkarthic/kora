//! Device-memory blocks with suballocation, dedicated allocations, and budget hooks.

use std::collections::HashMap;
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;

use super::suballocator::{Suballocation, Suballocator};
use crate::checked;
use crate::error::{Error, ErrorCode, Result, Status};
use crate::vulkan::Device;

const MIB: u64 = 1024 * 1024;

/// Intended use of a pool's memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryUsageClass {
    /// Kernel buffers.
    #[default]
    General,
    /// Host-to-device staging.
    Upload,
    /// Device-to-host staging.
    Readback,
}

/// Budget integration for a pool. Every method has a permissive default.
pub trait PoolBudget: Send + Sync {
    /// Reserves `bytes` before a device allocation; an error prevents the allocation.
    fn reserve(&self, _bytes: u64) -> Status {
        Ok(())
    }
    /// Returns `bytes` after a failed or freed allocation.
    fn release(&self, _bytes: u64) {}
    /// Called when the driver refused an allocation (used to shrink future admission).
    fn allocation_failed(&self) {}
    /// Recovery stage: refresh the device budget.
    fn refresh_budget(&self) -> Status {
        Ok(())
    }
    /// Recovery stage: free deferred releases whose GPU work completed.
    fn reclaim_deferred(&self) -> Status {
        Ok(())
    }
    /// Recovery stage: evict eligible device replicas.
    fn evict_eligible(&self) -> Status {
        Ok(())
    }
}

/// Pool configuration.
#[derive(Clone, Default)]
pub struct MemoryPoolConfig {
    /// Vulkan memory type.
    pub memory_type_index: u32,
    /// Heap of that type.
    pub heap_index: u32,
    /// Usage class (part of the pool key).
    pub usage: MemoryUsageClass,
    /// Map blocks persistently.
    pub host_visible: bool,
    /// Coherent memory (no flush/invalidate needed).
    pub host_coherent: bool,
    /// Allocate with `DEVICE_ADDRESS`.
    pub device_address: bool,
    /// Heap size.
    pub heap_size: u64,
    /// Usable budget (0 = heap size).
    pub budget: u64,
    /// Block size override (0 = heuristic).
    pub block_size_hint: u64,
    /// Budget integration.
    pub budget_hooks: Option<Arc<dyn PoolBudget>>,
}

impl fmt::Debug for MemoryPoolConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryPoolConfig")
            .field("memory_type_index", &self.memory_type_index)
            .field("heap_index", &self.heap_index)
            .field("usage", &self.usage)
            .field("host_visible", &self.host_visible)
            .field("host_coherent", &self.host_coherent)
            .field("heap_size", &self.heap_size)
            .field("budget", &self.budget)
            .finish()
    }
}

/// One allocation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolAllocationRequest {
    /// Size in bytes.
    pub size: u64,
    /// Power-of-two alignment.
    pub alignment: u64,
    /// The resource requires its own `VkDeviceMemory`.
    pub dedicated_required: bool,
    /// The resource prefers its own `VkDeviceMemory`.
    pub dedicated_preferred: bool,
}

impl PoolAllocationRequest {
    /// A pooled request.
    pub const fn new(size: u64, alignment: u64) -> Self {
        Self {
            size,
            alignment,
            dedicated_required: false,
            dedicated_preferred: false,
        }
    }
}

/// Pointer into persistently mapped memory. It is a plain address: dereferencing it is only valid
/// while the owning pool allocation is live and no device work writes the same bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedPtr(NonNull<u8>);

// SAFETY: a mapped pointer is an address into driver-owned host-visible memory that is valid from
// any thread; synchronization of the bytes themselves is the responsibility of the code that
// dereferences it (documented on every unsafe accessor).
unsafe impl Send for MappedPtr {}
// SAFETY: as above; sharing the address does not access the memory.
unsafe impl Sync for MappedPtr {}

impl MappedPtr {
    /// Wraps the address of mapped memory that KORA did not map itself (for example, a buffer an
    /// application or benchmark allocated directly).
    ///
    /// # Safety
    /// `ptr` must point into host-visible memory that stays mapped for as long as the result (or
    /// any [`TransferBuffer`](super::TransferBuffer) holding it) is used.
    pub unsafe fn from_raw(ptr: NonNull<u8>) -> MappedPtr {
        MappedPtr(ptr)
    }

    /// Raw pointer.
    pub fn as_ptr(self) -> *mut u8 {
        self.0.as_ptr()
    }

    /// Pointer `offset` bytes further.
    ///
    /// # Safety
    /// `offset` must stay within the mapped allocation.
    pub unsafe fn add(self, offset: u64) -> MappedPtr {
        // SAFETY: the caller keeps the result inside the same mapping, so it is non-null.
        MappedPtr(unsafe { NonNull::new_unchecked(self.0.as_ptr().add(offset as usize)) })
    }

    /// A shared byte slice.
    ///
    /// # Safety
    /// `len` bytes from this address must be mapped for the lifetime `'a`, and no device or host
    /// writer may access them concurrently.
    pub unsafe fn slice<'a>(self, len: usize) -> &'a [u8] {
        // SAFETY: forwarded from the caller.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr(), len) }
    }

    /// A mutable byte slice.
    ///
    /// # Safety
    /// `len` bytes from this address must be mapped for `'a`, and nothing else (host or device) may
    /// read or write them concurrently.
    pub unsafe fn slice_mut<'a>(self, len: usize) -> &'a mut [u8] {
        // SAFETY: forwarded from the caller.
        unsafe { std::slice::from_raw_parts_mut(self.0.as_ptr(), len) }
    }
}

/// A pooled or dedicated allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolAllocation {
    /// Backing memory.
    pub memory: vk::DeviceMemory,
    /// Offset in `memory`.
    pub offset: u64,
    /// Requested size.
    pub requested_size: u64,
    /// Reserved size.
    pub allocated_size: u64,
    /// Mapped address of `offset` for host-visible pools.
    pub mapped: Option<MappedPtr>,
    /// Block slot (pooled allocations).
    pub block_index: usize,
    /// Dedicated-allocation token.
    pub token: u64,
    /// Owns its own `VkDeviceMemory`.
    pub dedicated: bool,
}

/// Pool statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemoryPoolStats {
    /// Bytes of device memory owned.
    pub reserved_bytes: u64,
    /// Bytes handed out (rounded).
    pub allocated_bytes: u64,
    /// Bytes requested.
    pub requested_bytes: u64,
    /// Free bytes inside blocks.
    pub free_bytes: u64,
    /// `allocated - requested`.
    pub internal_fragmentation_bytes: u64,
    /// Free bytes outside the largest free region of each block.
    pub external_fragmentation_bytes: u64,
    /// Largest free region in any block.
    pub largest_free_block: u64,
    /// Peak allocated bytes.
    pub high_water_mark: u64,
    /// Blocks.
    pub blocks: usize,
    /// Dedicated allocations.
    pub dedicated_allocations: usize,
    /// Live allocations.
    pub live_allocations: usize,
    /// `vkAllocateMemory` calls made.
    pub device_allocation_calls: u64,
}

struct Block {
    memory: vk::DeviceMemory,
    mapped: Option<MappedPtr>,
    allocator: Suballocator,
}

struct Dedicated {
    memory: vk::DeviceMemory,
    size: u64,
}

/// Owns `VkDeviceMemory` blocks and hands out buffer-compatible ranges. Not internally
/// synchronized; owners wrap it in a lock. Surplus empty blocks are released (one spare is kept
/// to avoid allocation churn).
pub struct MemoryPool {
    device: Device,
    config: MemoryPoolConfig,
    block_size: u64,
    blocks: Vec<Option<Block>>,
    dedicated: HashMap<u64, Dedicated>,
    next_token: u64,
    allocated_bytes: u64,
    high_water_mark: u64,
    device_allocation_calls: u64,
}

impl fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryPool")
            .field("config", &self.config)
            .field("stats", &self.stats())
            .finish()
    }
}

fn choose_block_size(config: &MemoryPoolConfig) -> u64 {
    let available = if config.budget == 0 {
        config.heap_size
    } else {
        config.heap_size.min(config.budget)
    };
    if available == 0 {
        return 0;
    }
    if config.block_size_hint != 0 {
        return config.block_size_hint.min(available);
    }
    let (divisor, minimum, maximum) = if config.host_visible {
        (128, 8 * MIB, 64 * MIB)
    } else {
        (64, 16 * MIB, 256 * MIB)
    };
    (available / divisor)
        .min(maximum)
        .max(minimum.min(available))
        .min(available)
}

impl MemoryPool {
    /// Creates an empty pool; blocks are allocated lazily.
    pub fn new(device: Device, config: MemoryPoolConfig) -> Result<MemoryPool> {
        if config.heap_size == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "memory pool heap size must be greater than zero",
            ));
        }
        let block_size = choose_block_size(&config);
        if block_size == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "memory pool block size must be nonzero",
            ));
        }
        Ok(MemoryPool {
            device,
            config,
            block_size,
            blocks: Vec::new(),
            dedicated: HashMap::new(),
            next_token: 1,
            allocated_bytes: 0,
            high_water_mark: 0,
            device_allocation_calls: 0,
        })
    }

    /// Current block size.
    pub fn block_size(&self) -> u64 {
        self.block_size
    }
    /// Memory type index.
    pub fn memory_type_index(&self) -> u32 {
        self.config.memory_type_index
    }
    /// Heap index.
    pub fn heap_index(&self) -> u32 {
        self.config.heap_index
    }
    /// Host-visible pool.
    pub fn host_visible(&self) -> bool {
        self.config.host_visible
    }
    /// Host-coherent pool.
    pub fn host_coherent(&self) -> bool {
        self.config.host_coherent
    }
    /// The device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn allocate_memory_once(&mut self, size: u64) -> Result<vk::DeviceMemory> {
        if let Some(budget) = &self.config.budget_hooks {
            budget.reserve(size)?;
        }
        let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(self.config.memory_type_index);
        if self.config.device_address {
            info = info.push_next(&mut flags);
        }
        self.device_allocation_calls += 1;
        self.device.allocate_memory(&info).map_err(|result| {
            if let Some(budget) = &self.config.budget_hooks {
                budget.release(size);
                budget.allocation_failed();
            }
            crate::vulkan::vk_error(
                ErrorCode::AllocationFailed,
                "vkAllocateMemory failed for memory-pool block",
                result,
            )
        })
    }

    /// Allocates device memory: one attempt, then the bounded recovery stages and one retry.
    fn allocate_memory(&mut self, size: u64) -> Result<vk::DeviceMemory> {
        let first = match self.allocate_memory_once(size) {
            Ok(memory) => return Ok(memory),
            Err(error) => error,
        };
        let mut stages = Vec::new();
        if let Some(budget) = self.config.budget_hooks.clone() {
            stages.push("refresh_budget");
            let _ = budget.refresh_budget();
            stages.push("reclaim_deferred");
            let _ = budget.reclaim_deferred();
            stages.push("evict_eligible");
            let _ = budget.evict_eligible();
        }
        stages.push("retry_once");
        self.allocate_memory_once(size).map_err(|retry| {
            let base = if retry.code() == first.code() {
                retry
            } else {
                first
            };
            base.with_recovery(stages.join(", "))
        })
    }

    fn map(&self, memory: vk::DeviceMemory, size: u64) -> Result<Option<MappedPtr>> {
        if !self.config.host_visible {
            return Ok(None);
        }
        // SAFETY: `memory` was just allocated from a host-visible type on this device, is unmapped,
        // and `size` is its full allocation size.
        let pointer = unsafe {
            self.device
                .raw()
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .map_err(|r| {
            crate::vulkan::vk_error(
                ErrorCode::AllocationFailed,
                "vkMapMemory failed for pool memory",
                r,
            )
        })?;
        Ok(NonNull::new(pointer.cast::<u8>()).map(MappedPtr))
    }

    fn free_memory(&self, memory: vk::DeviceMemory, mapped: bool, size: u64) {
        // SAFETY: `memory` belongs to this device and no live allocation (and therefore no bound
        // resource) uses it any more; it is unmapped before being freed.
        unsafe {
            if mapped {
                self.device.raw().unmap_memory(memory);
            }
            self.device.raw().free_memory(memory, None);
        }
        if let Some(budget) = &self.config.budget_hooks {
            budget.release(size);
        }
    }

    fn create_block(&mut self, size: u64) -> Result<Block> {
        let memory = self.allocate_memory(size)?;
        let mapped = match self.map(memory, size) {
            Ok(mapped) => mapped,
            Err(error) => {
                self.free_memory(memory, false, size);
                return Err(error);
            }
        };
        Ok(Block {
            memory,
            mapped,
            allocator: Suballocator::new(size)?,
        })
    }

    fn note_allocated(&mut self, bytes: u64) {
        self.allocated_bytes += bytes;
        self.high_water_mark = self.high_water_mark.max(self.allocated_bytes);
    }

    fn allocate_dedicated(&mut self, request: &PoolAllocationRequest) -> Result<PoolAllocation> {
        let memory = self.allocate_memory(request.size)?;
        let mapped = match self.map(memory, request.size) {
            Ok(mapped) => mapped,
            Err(error) => {
                self.free_memory(memory, false, request.size);
                return Err(error);
            }
        };
        let token = self.next_token;
        self.next_token += 1;
        self.dedicated.insert(
            token,
            Dedicated {
                memory,
                size: request.size,
            },
        );
        self.note_allocated(request.size);
        Ok(PoolAllocation {
            memory,
            offset: 0,
            requested_size: request.size,
            allocated_size: request.size,
            mapped,
            block_index: 0,
            token,
            dedicated: true,
        })
    }

    fn block_allocation(&mut self, index: usize, sub: Suballocation) -> PoolAllocation {
        self.note_allocated(sub.allocated_size);
        let block = self.blocks[index].as_ref().expect("allocated from a live block");
        PoolAllocation {
            memory: block.memory,
            offset: sub.offset,
            requested_size: sub.requested_size,
            allocated_size: sub.allocated_size,
            // SAFETY: `sub.offset + allocated_size` lies inside the block's mapping.
            mapped: block.mapped.map(|base| unsafe { base.add(sub.offset) }),
            block_index: index,
            token: 0,
            dedicated: false,
        }
    }

    /// Allocates a range. Large requests (more than half a block) and dedicated requests get
    /// their own `VkDeviceMemory`; others are suballocated, creating (or, after an allocation
    /// failure, shrinking) blocks as needed.
    pub fn allocate(&mut self, request: &PoolAllocationRequest) -> Result<PoolAllocation> {
        if request.size == 0 || !checked::is_pow2(request.alignment) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "pool allocation requires nonzero size and power-of-two alignment",
            ));
        }
        if request.dedicated_required || request.dedicated_preferred || request.size > self.block_size / 2 {
            return self.allocate_dedicated(request);
        }
        for index in 0..self.blocks.len() {
            let Some(block) = self.blocks[index].as_mut() else {
                continue;
            };
            match block.allocator.allocate(request.size, request.alignment) {
                Ok(sub) => return Ok(self.block_allocation(index, sub)),
                Err(error) if error.code() == ErrorCode::AllocationFailed => {}
                Err(error) => return Err(error),
            }
        }
        let block = match self.create_block(self.block_size) {
            Ok(block) => block,
            Err(error) => {
                let smaller = (self.block_size / 2).max(MIB);
                if smaller >= self.block_size || request.size > smaller {
                    let recovery = join_recovery(&error, "smaller_block");
                    return Err(error.with_recovery(recovery));
                }
                match self.create_block(smaller) {
                    Ok(block) => {
                        self.block_size = smaller;
                        block
                    }
                    Err(retry) => {
                        let recovery = join_recovery(&retry, "smaller_block");
                        return Err(retry.with_recovery(recovery));
                    }
                }
            }
        };
        let index = match self.blocks.iter().position(Option::is_none) {
            Some(free_slot) => {
                self.blocks[free_slot] = Some(block);
                free_slot
            }
            None => {
                self.blocks.push(Some(block));
                self.blocks.len() - 1
            }
        };
        let sub = self.blocks[index]
            .as_mut()
            .expect("just inserted")
            .allocator
            .allocate(request.size, request.alignment)?;
        Ok(self.block_allocation(index, sub))
    }

    /// Frees an allocation. Stale or foreign handles are rejected.
    pub fn deallocate(&mut self, allocation: &PoolAllocation) -> Status {
        if allocation.dedicated {
            match self.dedicated.get(&allocation.token) {
                Some(d) if d.memory == allocation.memory && d.size == allocation.requested_size => {}
                _ => {
                    return Err(Error::new(
                        ErrorCode::InvalidArgument,
                        "dedicated pool allocation is unknown or invalid",
                    ));
                }
            }
            let entry = self.dedicated.remove(&allocation.token).expect("checked above");
            self.allocated_bytes -= entry.size;
            self.free_memory(entry.memory, self.config.host_visible, entry.size);
            return Ok(());
        }
        let block = self
            .blocks
            .get_mut(allocation.block_index)
            .and_then(Option::as_mut)
            .filter(|b| b.memory == allocation.memory)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidArgument,
                    "pooled allocation references an invalid block",
                )
            })?;
        block.allocator.deallocate(&Suballocation {
            offset: allocation.offset,
            requested_size: allocation.requested_size,
            allocated_size: allocation.allocated_size,
        })?;
        self.allocated_bytes -= allocation.allocated_size;
        let now_empty = block.allocator.is_empty();
        if now_empty {
            let empty_blocks = self
                .blocks
                .iter()
                .flatten()
                .filter(|b| b.allocator.is_empty())
                .count();
            if empty_blocks > 1 {
                let block = self.blocks[allocation.block_index].take().expect("checked above");
                self.free_memory(block.memory, block.mapped.is_some(), block.allocator.capacity());
            }
        }
        Ok(())
    }

    /// Statistics.
    pub fn stats(&self) -> MemoryPoolStats {
        let mut stats = MemoryPoolStats {
            dedicated_allocations: self.dedicated.len(),
            device_allocation_calls: self.device_allocation_calls,
            high_water_mark: self.high_water_mark,
            ..Default::default()
        };
        for block in self.blocks.iter().flatten() {
            let s = block.allocator.stats();
            stats.blocks += 1;
            stats.reserved_bytes += s.total_bytes;
            stats.allocated_bytes += s.allocated_bytes;
            stats.requested_bytes += s.requested_bytes;
            stats.free_bytes += s.free_bytes;
            stats.internal_fragmentation_bytes += s.internal_fragmentation_bytes;
            stats.external_fragmentation_bytes += s.external_fragmentation_bytes;
            stats.largest_free_block = stats.largest_free_block.max(s.largest_free_block);
            stats.live_allocations += s.live_allocations;
        }
        for dedicated in self.dedicated.values() {
            stats.reserved_bytes += dedicated.size;
            stats.allocated_bytes += dedicated.size;
            stats.requested_bytes += dedicated.size;
            stats.live_allocations += 1;
        }
        stats
    }
}

fn join_recovery(error: &Error, stage: &str) -> String {
    if error.recovery_action().is_empty() {
        stage.to_owned()
    } else {
        format!("{}, {stage}", error.recovery_action())
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        let host_visible = self.config.host_visible;
        for (_, dedicated) in std::mem::take(&mut self.dedicated) {
            self.free_memory(dedicated.memory, host_visible, dedicated.size);
        }
        for block in std::mem::take(&mut self.blocks).into_iter().flatten() {
            self.free_memory(block.memory, block.mapped.is_some(), block.allocator.capacity());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_heuristic_scales_with_heap_and_respects_hints() {
        let config = |heap, visible, hint| MemoryPoolConfig {
            heap_size: heap,
            host_visible: visible,
            block_size_hint: hint,
            ..Default::default()
        };
        assert_eq!(choose_block_size(&config(8 << 30, false, 0)), 128 * MIB);
        assert_eq!(choose_block_size(&config(64 << 30, false, 0)), 256 * MIB);
        assert_eq!(choose_block_size(&config(256 * MIB, false, 0)), 16 * MIB);
        assert_eq!(choose_block_size(&config(8 * MIB, false, 0)), 8 * MIB);
        assert_eq!(choose_block_size(&config(8 << 30, true, 0)), 64 * MIB);
        assert_eq!(choose_block_size(&config(1 << 30, true, 0)), 8 * MIB);
        assert_eq!(choose_block_size(&config(1 << 30, true, 3 * MIB)), 3 * MIB);
        assert_eq!(choose_block_size(&config(0, true, 0)), 0);
        let budgeted = MemoryPoolConfig {
            heap_size: 8 << 30,
            budget: 1 << 30,
            ..Default::default()
        };
        assert_eq!(choose_block_size(&budgeted), 16 * MIB);
    }
}
