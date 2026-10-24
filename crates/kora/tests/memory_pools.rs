//! Memory pools and staging rings against real Vulkan devices.

mod common;

use std::sync::{Arc, Mutex};

use kora::memory::{
    MemoryManager, MemoryPolicy, MemoryPool, MemoryPoolConfig, MemoryTypeRequirements, MemoryUsageClass,
    PoolAllocationRequest, PoolBudget, StagingDirection, StagingOptions, StagingRing, StagingSlice,
};
use kora::vulkan::ash::vk;
use kora::vulkan::{Device, DeviceOptions, FaultFunction, InstanceOptions, device_info};
use kora::{DeviceInfo, ErrorCode};

struct Setup {
    device: Device,
    info: DeviceInfo,
}

fn setup() -> Option<Setup> {
    let (_, instance) = common::vulkan(&InstanceOptions::default())?;
    let physical = instance.physical_devices().ok()?;
    let physical = physical.into_iter().find(|p| {
        p.queue_families
            .iter()
            .any(|f| f.queue_count > 0 && f.queue_flags.contains(vk::QueueFlags::COMPUTE))
    })?;
    let info = device_info(&physical);
    let device = instance
        .create_device(&physical, &DeviceOptions::default())
        .expect("device");
    Some(Setup { device, info })
}

fn host_pool_config(setup: &Setup, block: u64) -> Option<MemoryPoolConfig> {
    let required = MemoryTypeRequirements {
        host_visible: true,
        ..Default::default()
    };
    let index = MemoryManager::select_memory_type_from(
        &setup.info.memory_types,
        &setup.info.memory_type_preferences,
        MemoryPolicy::HostVisible,
        u32::MAX,
        &required,
    )
    .ok()?;
    let memory_type = setup.info.memory_types[index as usize];
    let heap = setup.info.memory_heaps[memory_type.heap_index as usize].size_bytes;
    Some(MemoryPoolConfig {
        memory_type_index: index,
        heap_index: memory_type.heap_index,
        usage: MemoryUsageClass::Upload,
        host_visible: true,
        host_coherent: memory_type.host_coherent,
        heap_size: heap,
        budget: heap,
        block_size_hint: block,
        ..Default::default()
    })
}

fn shared_pool(setup: &Setup, block: u64) -> Option<Arc<Mutex<MemoryPool>>> {
    let config = host_pool_config(setup, block)?;
    Some(Arc::new(Mutex::new(
        MemoryPool::new(setup.device.clone(), config).expect("pool"),
    )))
}

#[test]
fn repeated_small_allocations_report_consistent_fragmentation() {
    let Some(setup) = setup() else { return };
    let Some(config) = host_pool_config(&setup, 8 << 20) else {
        return;
    };
    let mut pool = MemoryPool::new(setup.device.clone(), config).unwrap();
    let mut random = 0x004b_4f52_415f_3237_u64;
    let mut next = || {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        random
    };
    for _ in 0..3 {
        let mut live = Vec::with_capacity(3000);
        for _ in 0..3000 {
            let size = 16 + next() % 1009;
            let alignment = 1 << (next() % 8);
            let allocation = pool
                .allocate(&PoolAllocationRequest::new(size, alignment))
                .unwrap();
            assert_eq!(allocation.offset % alignment, 0);
            assert!(allocation.mapped.is_some());
            live.push(allocation);
        }
        let s = pool.stats();
        assert_eq!(s.live_allocations, live.len());
        assert_eq!(s.allocated_bytes + s.free_bytes, s.reserved_bytes);
        assert_eq!(
            s.internal_fragmentation_bytes,
            s.allocated_bytes - s.requested_bytes
        );
        assert_eq!(
            s.external_fragmentation_bytes,
            s.free_bytes - s.largest_free_block
        );
        assert!(s.high_water_mark >= s.allocated_bytes);
        let total = MemoryManager::aggregate_stats(&[s]).unwrap();
        assert_eq!(
            (
                total.reserved_bytes,
                total.allocated_bytes,
                total.live_allocations
            ),
            (s.reserved_bytes, s.allocated_bytes, s.live_allocations)
        );
        for index in (1..live.len()).rev() {
            live.swap(index, (next() % (index as u64 + 1)) as usize);
        }
        for allocation in &live {
            pool.deallocate(allocation).unwrap();
        }
        let empty = pool.stats();
        assert_eq!((empty.live_allocations, empty.allocated_bytes), (0, 0));
        assert_eq!(empty.free_bytes, empty.reserved_bytes);
        assert_eq!(empty.largest_free_block, empty.free_bytes);
        assert_eq!(empty.external_fragmentation_bytes, 0);
        assert_eq!(
            empty.blocks, 1,
            "surplus empty blocks are released, one spare is kept"
        );
    }
}

#[test]
fn mapped_allocations_hold_written_bytes_and_stale_handles_are_rejected() {
    let Some(setup) = setup() else { return };
    let Some(config) = host_pool_config(&setup, 1 << 20) else {
        return;
    };
    let mut pool = MemoryPool::new(setup.device.clone(), config).unwrap();
    let small = pool.allocate(&PoolAllocationRequest::new(256, 64)).unwrap();
    let large = pool
        .allocate(&PoolAllocationRequest::new(900 << 10, 256))
        .unwrap();
    assert!(
        !small.dedicated && large.dedicated,
        "requests above half a block are dedicated"
    );
    // SAFETY: both allocations are live, host-visible, and unused by the device.
    unsafe {
        small.mapped.unwrap().slice_mut(256).fill(0x5a);
        large.mapped.unwrap().slice_mut(900 << 10).fill(0xa5);
        assert!(small.mapped.unwrap().slice(256).iter().all(|&b| b == 0x5a));
        assert!(large.mapped.unwrap().slice(900 << 10).iter().all(|&b| b == 0xa5));
    }
    pool.deallocate(&small).unwrap();
    assert_eq!(
        pool.deallocate(&small).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
    pool.deallocate(&large).unwrap();
    assert_eq!(
        pool.deallocate(&large).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
    assert!(pool.allocate(&PoolAllocationRequest::new(0, 1)).is_err());
    assert!(pool.allocate(&PoolAllocationRequest::new(1, 3)).is_err());
}

#[derive(Default)]
struct CountingBudget {
    events: Mutex<Vec<&'static str>>,
}

impl PoolBudget for CountingBudget {
    fn reserve(&self, _bytes: u64) -> kora::Status {
        self.events.lock().unwrap().push("reserve");
        Ok(())
    }
    fn release(&self, _bytes: u64) {
        self.events.lock().unwrap().push("release");
    }
    fn allocation_failed(&self) {
        self.events.lock().unwrap().push("failed");
    }
    fn refresh_budget(&self) -> kora::Status {
        self.events.lock().unwrap().push("refresh");
        Ok(())
    }
}

#[test]
fn injected_allocation_failure_runs_recovery_and_retries_once() {
    let Some(setup) = setup() else { return };
    let Some(mut config) = host_pool_config(&setup, 1 << 20) else {
        return;
    };
    let budget = Arc::new(CountingBudget::default());
    config.budget_hooks = Some(budget.clone());
    let mut pool = MemoryPool::new(setup.device.clone(), config).unwrap();
    setup
        .device
        .faults()
        .fail_on_nth(
            FaultFunction::AllocateMemory,
            1,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        )
        .unwrap();
    let allocation = pool
        .allocate(&PoolAllocationRequest::new(64, 16))
        .expect("the retry succeeds");
    assert_eq!(
        *budget.events.lock().unwrap(),
        ["reserve", "release", "failed", "refresh", "reserve"]
    );
    pool.deallocate(&allocation).unwrap();

    // Both the first attempt and the retry fail: the error carries the stage trace.
    let dedicated = PoolAllocationRequest {
        dedicated_required: true,
        ..PoolAllocationRequest::new(4096, 16)
    };
    setup
        .device
        .faults()
        .fail_calls(
            FaultFunction::AllocateMemory,
            1,
            2,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        )
        .unwrap();
    let error = pool.allocate(&dedicated).unwrap_err();
    assert_eq!(error.code(), ErrorCode::AllocationFailed);
    assert_eq!(error.vk_result(), vk::Result::ERROR_OUT_OF_DEVICE_MEMORY.as_raw());
    assert!(
        error.recovery_action().contains("refresh_budget") && error.recovery_action().contains("retry_once"),
        "{error}"
    );
    setup.device.faults().clear();
    let recovered = pool.allocate(&dedicated).unwrap();
    pool.deallocate(&recovered).unwrap();
}

fn reserve(ring: &StagingRing, size: u64, completed: u64) -> StagingSlice {
    ring.reserve(size, 1, completed, None).expect("reservation")
}

#[test]
fn staging_ring_waits_for_oldest_timeline_and_never_overlaps_in_flight() {
    let Some(setup) = setup() else { return };
    let Some(pool) = shared_pool(&setup, 1 << 20) else {
        return;
    };
    let options = StagingOptions {
        initial_capacity: 4096,
        maximum_capacity: 4096,
        topology: setup.info.memory_topology,
        measured_staging_benefit: true,
        ..Default::default()
    };
    let ring = StagingRing::new(setup.device.clone(), pool.clone(), options).unwrap();
    let completed = std::cell::Cell::new(0u64);
    let waited = std::cell::Cell::new(false);
    let mut live: Vec<(StagingSlice, u64)> = Vec::new();
    let mut random = 0x5354_4147u64;
    for timeline in 1..=1000u64 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let size = 64 + random % 384;
        let mut wait = |value: u64| {
            waited.set(true);
            completed.set(completed.get().max(value));
            Ok(completed.get())
        };
        let mut slice = ring.reserve(size, 16, completed.get(), Some(&mut wait)).unwrap();
        live.retain(|(_, release)| *release > completed.get());
        for (other, _) in &live {
            if other.buffer == slice.buffer {
                assert!(
                    slice.offset + slice.size <= other.offset || other.offset + other.size <= slice.offset
                );
            }
        }
        // SAFETY: the slice is unsubmitted, so no device work touches it.
        unsafe { slice.bytes_mut().fill(0x6d) };
        ring.flush(&slice).unwrap();
        ring.retire(&slice, timeline).unwrap();
        live.push((slice, timeline));
    }
    assert!(waited.get());
    assert_eq!(ring.capacity(), 4096);

    let next = 1001;
    let full = ring.reserve(4096, 1, next, None).unwrap();
    let mut waited_for_unsubmitted = false;
    let mut wait = |_| {
        waited_for_unsubmitted = true;
        Ok(next)
    };
    let exhausted = ring.reserve(1, 1, next, Some(&mut wait)).unwrap_err();
    assert_eq!(exhausted.code(), ErrorCode::AllocationFailed);
    assert!(!waited_for_unsubmitted);
    ring.cancel(full).unwrap();
    let reusable = reserve(&ring, 1, next);
    ring.cancel(reusable).unwrap();
    ring.reclaim_completed(u64::MAX);
    drop(live);
    drop(ring);

    // Offset-ordered insertion after wrapping back to a freed low-address gap.
    let ring = StagingRing::new(setup.device.clone(), pool.clone(), options).unwrap();
    let mut ordered: Vec<StagingSlice> = (0..4).map(|_| reserve(&ring, 256, 0)).collect();
    ring.cancel(ordered.remove(0)).unwrap();
    let tail = reserve(&ring, 4096 - 4 * 256, 0);
    assert_eq!(tail.offset, 4 * 256);
    let wrapped = reserve(&ring, 128, 0);
    assert_eq!(wrapped.offset, 0);
    for other in &ordered {
        assert!(
            wrapped.offset + wrapped.reserved_size <= other.offset
                || other.offset + other.reserved_size <= wrapped.offset
        );
    }
    for slice in ordered.into_iter().chain([tail, wrapped]) {
        ring.cancel(slice).unwrap();
    }
    assert_eq!(ring.live_reservations(), 0);

    // A completed readback slice survives reclamation until its bytes were consumed.
    let readback = StagingRing::new(
        setup.device.clone(),
        pool,
        StagingOptions {
            direction: StagingDirection::Readback,
            ..options
        },
    )
    .unwrap();
    let slice = readback.reserve(128, 16, next, None).unwrap();
    readback.retire(&slice, next).unwrap();
    readback.invalidate(&slice).unwrap();
    readback.reclaim_completed(next + 100);
    readback.invalidate(&slice).unwrap();
    readback.consumed(&slice).unwrap();
    readback.reclaim_completed(next + 100);
    assert!(readback.invalidate(&slice).is_err());
    assert!(readback.cancel(slice).is_err());
}

#[test]
fn staging_ring_grows_to_its_maximum_and_rejects_foreign_slices() {
    let Some(setup) = setup() else { return };
    let Some(pool) = shared_pool(&setup, 1 << 20) else {
        return;
    };
    let options = StagingOptions {
        initial_capacity: 1024,
        maximum_capacity: 4096,
        measured_staging_benefit: true,
        ..Default::default()
    };
    let ring = StagingRing::new(setup.device.clone(), pool.clone(), options).unwrap();
    let slices: Vec<_> = (0..4).map(|_| reserve(&ring, 1024, 0)).collect();
    assert_eq!(ring.capacity(), 4096);
    assert_eq!(
        ring.reserve(1, 1, 0, None).unwrap_err().code(),
        ErrorCode::AllocationFailed
    );
    let other = StagingRing::new(setup.device.clone(), pool.clone(), options).unwrap();
    assert!(other.retire(&slices[0], 1).is_err());
    assert!(ring.retire(&slices[0], 0).is_err());
    ring.retire(&slices[0], 5).unwrap();
    assert!(ring.retire(&slices[0], 6).is_err());
    let mut waits = 0;
    let mut wait = |value| {
        waits += 1;
        Ok(value)
    };
    let after_wait = ring.reserve(512, 1, 0, Some(&mut wait)).unwrap();
    assert_eq!(waits, 1);
    assert_eq!(after_wait.buffer, slices[0].buffer);
    let invalid = StagingOptions {
        measured_staging_benefit: false,
        topology: kora::MemoryTopology::Uma,
        ..options
    };
    assert_eq!(
        StagingRing::new(setup.device.clone(), pool, invalid)
            .unwrap_err()
            .code(),
        ErrorCode::UnsupportedFeature
    );
}
