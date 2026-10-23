//! Memory subsystem: suballocation, device-memory pools, budgets, staging rings, dirty ranges,
//! residency tracking, eviction planning, and out-of-memory recovery.
//!
//! These are the building blocks the [`Runtime`](crate) uses to manage device memory; most
//! applications do not use them directly.

mod dirty;
mod eviction;
mod manager;
mod pool;
mod recovery;
mod residency;
mod staging;
mod suballocator;
mod transfer;

pub use dirty::{DirtyRange, DirtyRangeSet, MappedMemoryOps, flush_mapped_ranges, invalidate_mapped_ranges};
pub use eviction::{
    AccessHistory, EvictionAction, EvictionCandidate, EvictionDecision, EvictionPlan, EvictionPolicy,
    EvictionPolicyConfig,
};
pub use manager::{
    BudgetLedger, HeapBudget, HeapBudgetStats, MemoryHeapStats, MemoryManager, MemoryPolicy, MemoryStats,
    MemoryTypeRequirements,
};
pub use pool::{
    MappedPtr, MemoryPool, MemoryPoolConfig, MemoryPoolStats, MemoryUsageClass, PoolAllocation,
    PoolAllocationRequest, PoolBudget,
};
pub use recovery::{
    AdmissionEstimate, OomRecoveryHooks, RecoveryAction, RecoveryTrace, estimate_admission,
    recover_allocation_failure, recover_task_allocation_failure,
};
pub use residency::{
    ResidencyDeviceId, ResidencyEvent, ResidencyLocation, ResidencySnapshot, ResidencyState,
    ResidencyTracker, TransferPlan, transition_residency,
};
pub use staging::{SharedPool, StagingDirection, StagingOptions, StagingRing, StagingSlice};
pub use suballocator::{Suballocation, Suballocator, SuballocatorStats};
pub use transfer::{
    PendingUses, QueueTimelineUse, TransferBuffer, TransferEngine, TransferOptions, maintain_mapped_range,
};
