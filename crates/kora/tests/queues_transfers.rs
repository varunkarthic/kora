//! Queue submission, command pools, and transfers against real Vulkan devices.

mod common;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use kora::ShutdownMode;
use kora::device::MemoryTopology;
use kora::memory::{
    MemoryManager, MemoryPolicy, MemoryPool, MemoryPoolConfig, MemoryTypeRequirements, TransferBuffer,
    TransferEngine, TransferOptions,
};
use kora::vulkan::ash::vk;
use kora::vulkan::{
    CoalescingOptions, CommandPoolSet, Device, DeviceOptions, InstanceOptions, PhysicalDeviceInfo,
    QueueContext, QueueContextOptions, QueueWait, ResourceAccess, Submission, device_info,
};

fn devices(options: &DeviceOptions) -> Vec<(PhysicalDeviceInfo, Device)> {
    let Some((_, instance)) = common::vulkan(&InstanceOptions::default()) else {
        return Vec::new();
    };
    instance
        .physical_devices()
        .unwrap()
        .into_iter()
        .filter(|p| {
            p.queue_families
                .iter()
                .any(|f| f.queue_count > 0 && f.queue_flags.contains(vk::QueueFlags::COMPUTE))
        })
        .filter_map(|p| instance.create_device(&p, options).ok().map(|d| (p, d)))
        .collect()
}

fn first_device() -> Option<Device> {
    devices(&DeviceOptions::default())
        .into_iter()
        .next()
        .map(|(_, d)| d)
}

fn context(device: &Device, options: QueueContextOptions) -> QueueContext<Device> {
    QueueContext::new(device.clone(), device.compute_queue(), options).expect("queue context")
}

#[test]
fn synchronization2_is_used_only_when_enabled_on_the_device() {
    for (physical, device) in devices(&DeviceOptions::default()) {
        let ctx = context(&device, QueueContextOptions::default());
        assert_eq!(
            ctx.uses_synchronization2(),
            device.enabled_features().synchronization2
        );
        let legacy = device
            .instance()
            .create_device(
                &physical,
                &DeviceOptions {
                    enable_synchronization2: false,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(!context(&legacy, QueueContextOptions::default()).uses_synchronization2());
    }
}

#[test]
fn completes_concurrent_real_submissions() {
    let Some(device) = first_device() else { return };
    let ctx = context(&device, QueueContextOptions::default());
    let completed = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));
    std::thread::scope(|scope| {
        for _ in 0..16 {
            scope.spawn(|| {
                for _ in 0..64 {
                    let (c, e) = (Arc::clone(&completed), Arc::clone(&errors));
                    let submission = Submission {
                        on_complete: Some(Box::new(move |_, status| {
                            if status.is_err() {
                                e.fetch_add(1, Ordering::SeqCst);
                            }
                            c.fetch_add(1, Ordering::SeqCst);
                        })),
                        ..Default::default()
                    };
                    if ctx.submit(submission).is_err() {
                        errors.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });
        }
    });
    ctx.shutdown(ShutdownMode::Graceful).unwrap();
    assert_eq!(completed.load(Ordering::SeqCst), 16 * 64);
    assert_eq!(errors.load(Ordering::SeqCst), 0);
}

#[test]
fn coalesces_independent_records_with_per_submission_ids() {
    let Some(device) = first_device() else { return };
    if !device.enabled_features().timeline_semaphore {
        return;
    }
    let coalescing = CoalescingOptions {
        enabled: true,
        maximum_batch_size: 8,
        maximum_wait_ns: 2_000_000,
    };
    let ctx = context(
        &device,
        QueueContextOptions {
            coalescing,
            ..Default::default()
        },
    );
    let ids = Arc::new(Mutex::new(Vec::new()));
    let values = Arc::new(Mutex::new(Vec::new()));
    for index in 0..24u32 {
        let (ids, values) = (Arc::clone(&ids), Arc::clone(&values));
        ctx.submit(Submission {
            coalescible: true,
            dependency_free: true,
            coalescing_key: 42,
            resources: vec![ResourceAccess {
                slot: index,
                generation: 1,
                write: false,
            }],
            on_submitted: Some(Box::new(move |value, status| {
                assert!(status.is_ok());
                values.lock().unwrap().push(value);
            })),
            on_complete: Some(Box::new(move |id, status| {
                assert!(status.is_ok());
                ids.lock().unwrap().push(id);
            })),
            ..Default::default()
        })
        .unwrap();
    }
    ctx.shutdown(ShutdownMode::Graceful).unwrap();
    let mut ids = ids.lock().unwrap().clone();
    ids.sort_unstable();
    assert_eq!(ids, (1..=24).collect::<Vec<u64>>());
    assert!(values.lock().unwrap().iter().all(|&v| v != 0));
}

#[test]
fn repeated_create_submit_shutdown_drains_completion_before_destruction() {
    let Some(device) = first_device() else { return };
    let completed = Arc::new(AtomicU64::new(0));
    for iteration in 0..32u64 {
        let ctx = context(&device, QueueContextOptions::default());
        let token = Arc::new(iteration);
        let counter = Arc::clone(&completed);
        ctx.submit(Submission {
            on_complete: Some(Box::new(move |_, status| {
                assert!(status.is_ok());
                assert_eq!(*token, iteration);
                counter.fetch_add(1, Ordering::SeqCst);
            })),
            ..Default::default()
        })
        .unwrap();
        ctx.shutdown(ShutdownMode::Graceful).unwrap();
        assert_eq!(completed.load(Ordering::SeqCst), iteration + 1);
    }
}

#[test]
fn waits_for_timeline_value_and_orders_dependent_submissions() {
    let Some(device) = first_device() else { return };
    if !device.enabled_features().timeline_semaphore {
        return;
    }
    let mut kind = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
    // SAFETY: valid create info on a live device.
    let dependency = unsafe { device.raw().create_semaphore(&info, None) }.unwrap();
    let ctx = context(&device, QueueContextOptions::default());
    let callbacks = Arc::new(AtomicU64::new(0));
    let counter = |c: &Arc<AtomicU64>| -> kora::vulkan::OnComplete {
        let c = Arc::clone(c);
        Box::new(move |_, status| {
            assert!(status.is_ok());
            c.fetch_add(1, Ordering::SeqCst);
        })
    };
    ctx.submit(Submission {
        signals: vec![(dependency, 1)],
        on_complete: Some(counter(&callbacks)),
        ..Default::default()
    })
    .unwrap();
    let second = ctx
        .submit(Submission {
            waits: vec![QueueWait {
                semaphore: dependency,
                value: 1,
                stage: vk::PipelineStageFlags::ALL_COMMANDS,
            }],
            signals: vec![(dependency, 2)],
            on_complete: Some(counter(&callbacks)),
            ..Default::default()
        })
        .unwrap();
    ctx.wait_for_value(second, 5_000_000_000).unwrap();
    assert!(ctx.completed_value().unwrap() >= second);
    ctx.shutdown(ShutdownMode::Graceful).unwrap();
    assert_eq!(callbacks.load(Ordering::SeqCst), 2);
    assert_eq!(device.semaphore_counter_value(dependency).unwrap(), 2);
    // SAFETY: all work signalling the semaphore has completed.
    unsafe { device.raw().destroy_semaphore(dependency, None) };
}

#[test]
fn command_pool_set_repeats_ten_thousand_record_submit_reuse_cycles() {
    let Some(device) = first_device() else { return };
    let ctx = Arc::new(context(&device, QueueContextOptions::default()));
    let pool = Arc::new(Mutex::new(
        CommandPoolSet::new(device.clone(), device.compute_queue_family(), 8).unwrap(),
    ));
    let current = Arc::new(Mutex::new(vk::CommandBuffer::null()));
    let mut last = 0;
    for _ in 0..10_000 {
        let (recorder, tracker) = (Arc::clone(&pool), Arc::clone(&pool));
        let (recorded, submitted) = (Arc::clone(&current), Arc::clone(&current));
        let queue = Arc::clone(&ctx);
        last = ctx
            .submit(Submission {
                record: Some(Box::new(move |_, commands| {
                    let mut pool = recorder.lock().unwrap();
                    let command = loop {
                        let completed = queue.completed_value()?;
                        match pool.acquire(completed) {
                            Ok(command) => break command,
                            Err(error) if error.code() == kora::ErrorCode::OutOfMemoryBudget => {
                                queue.wait_for_value(pool.oldest_pending_value(), u64::MAX)?;
                            }
                            Err(error) => return Err(error),
                        }
                    };
                    pool.end_recording(command)?;
                    *recorded.lock().unwrap() = command;
                    commands.push(command);
                    Ok(())
                })),
                on_submitted: Some(Box::new(move |value, status| {
                    assert!(status.is_ok());
                    let command = *submitted.lock().unwrap();
                    tracker.lock().unwrap().submitted(command, value).unwrap();
                })),
                ..Default::default()
            })
            .unwrap();
    }
    ctx.wait_for_value(last, u64::MAX).unwrap();
    ctx.shutdown(ShutdownMode::Graceful).unwrap();
    let pool = pool.lock().unwrap();
    assert_eq!(pool.capacity(), 8, "the ring never grows");
    assert!(pool.oldest_pending_value() <= last);
}

struct DeviceBuffer {
    device: Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: transfers hold this object as a keep-alive until completion.
        unsafe {
            self.device.raw().destroy_buffer(self.buffer, None);
            self.device.raw().free_memory(self.memory, None);
        }
    }
}

fn device_buffer(device: &Device, size: u64) -> Option<(Arc<DeviceBuffer>, TransferBuffer)> {
    let info = device_info(device.physical());
    let create = vk::BufferCreateInfo::default().size(size).usage(
        vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER,
    );
    // SAFETY: valid create info on a live device.
    let buffer = unsafe { device.raw().create_buffer(&create, None) }.ok()?;
    // SAFETY: `buffer` is live.
    let requirements = unsafe { device.raw().get_buffer_memory_requirements(buffer) };
    let required = MemoryTypeRequirements {
        device_local: true,
        ..Default::default()
    };
    let index = MemoryManager::select_memory_type_from(
        &info.memory_types,
        &info.memory_type_preferences,
        MemoryPolicy::DevicePreferred,
        requirements.memory_type_bits,
        &required,
    )
    .ok()?;
    let memory = device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(index),
        )
        .ok()?;
    // SAFETY: memory satisfies the requirements.
    unsafe { device.raw().bind_buffer_memory(buffer, memory, 0) }.ok()?;
    let owner = Arc::new(DeviceBuffer {
        device: device.clone(),
        buffer,
        memory,
    });
    let view = TransferBuffer {
        buffer,
        size,
        keepalive: Some(owner.clone()),
        ..Default::default()
    };
    Some((owner, view))
}

fn host_pool(device: &Device) -> Option<Arc<Mutex<MemoryPool>>> {
    let info = device_info(device.physical());
    let required = MemoryTypeRequirements {
        host_visible: true,
        ..Default::default()
    };
    let index = MemoryManager::select_memory_type_from(
        &info.memory_types,
        &info.memory_type_preferences,
        MemoryPolicy::HostVisible,
        u32::MAX,
        &required,
    )
    .ok()?;
    let memory_type = info.memory_types[index as usize];
    let heap = info.memory_heaps[memory_type.heap_index as usize].size_bytes;
    let config = MemoryPoolConfig {
        memory_type_index: index,
        heap_index: memory_type.heap_index,
        host_visible: true,
        host_coherent: memory_type.host_coherent,
        heap_size: heap,
        budget: heap,
        block_size_hint: 8 << 20,
        ..Default::default()
    };
    Some(Arc::new(Mutex::new(
        MemoryPool::new(device.clone(), config).ok()?,
    )))
}

fn engine(device: &Device, initial: u64, maximum: u64) -> Option<TransferEngine> {
    let queue = Arc::new(
        QueueContext::new(
            device.clone(),
            device.transfer_queue(),
            QueueContextOptions::default(),
        )
        .ok()?,
    );
    let options = TransferOptions {
        queue_family: device.transfer_queue_family(),
        command_buffer_count: 8,
        staging_initial_capacity: initial,
        staging_maximum_capacity: maximum,
        // Exercise the staged path on unified-memory devices too.
        topology: MemoryTopology::Discrete,
        non_coherent_atom_size: device.physical().properties.limits.non_coherent_atom_size,
        ..Default::default()
    };
    TransferEngine::new(
        device.clone(),
        queue,
        host_pool(device)?,
        host_pool(device)?,
        options,
    )
    .ok()
}

fn read_back(engine: &TransferEngine, view: &TransferBuffer, offset: u64, size: u64) -> Vec<u8> {
    let output = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&output);
    engine
        .download(view, offset, size, None, move |bytes| {
            *sink.lock().unwrap() = bytes.to_vec();
            Ok(())
        })
        .unwrap()
        .wait()
        .unwrap();
    Arc::try_unwrap(output).unwrap().into_inner().unwrap()
}

#[test]
fn round_trips_odd_offsets_and_sizes_through_every_local_device() {
    let mut tested = 0;
    for (_, device) in devices(&DeviceOptions::default()) {
        let Some(engine) = engine(&device, 64 << 20, 128 << 20) else {
            continue;
        };
        const LARGEST: u64 = 64 << 20;
        let (Some((_first_owner, first)), Some((_second_owner, second))) = (
            device_buffer(&device, LARGEST + 32),
            device_buffer(&device, LARGEST + 32),
        ) else {
            continue;
        };
        let mut seed = 0x1234_5678u64;
        for size in [
            1u64,
            2,
            3,
            5,
            17,
            4093,
            65539,
            (1 << 20) + 3,
            (8 << 20) + 1,
            LARGEST,
        ] {
            let input: Vec<u8> = (0..size)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed as u8
                })
                .collect();
            engine.upload(&first, 3, &input).unwrap().wait().unwrap();
            assert!(read_back(&engine, &first, 3, size) == input, "size {size}");
            engine.copy(&first, 3, &second, 5, size).unwrap().wait().unwrap();
            assert!(
                read_back(&engine, &second, 5, size) == input,
                "copied size {size}"
            );
        }
        engine.fill(&second, 0, 16, 0x1234_5678).unwrap().wait().unwrap();
        let filled = read_back(&engine, &second, 0, 16);
        assert_eq!(filled, 0x1234_5678u32.to_ne_bytes().repeat(4));
        engine.queue().shutdown(ShutdownMode::Graceful).unwrap();
        tested += 1;
    }
    if tested == 0 {
        eprintln!("skipping: no usable device");
    }
}

#[test]
fn concurrent_staged_uploads_pipeline_through_a_small_ring() {
    let Some(device) = first_device() else { return };
    // A 256 KiB ring forces producers to reuse ranges while other uploads are still unsubmitted.
    let Some(engine) = engine(&device, 256 << 10, 256 << 10) else {
        return;
    };
    let Some((_owner, target)) = device_buffer(&device, 8 << 20) else {
        return;
    };
    std::thread::scope(|scope| {
        for thread in 0..8u8 {
            let (engine, target) = (&engine, &target);
            scope.spawn(move || {
                let mut events = Vec::new();
                for chunk in 0..16u64 {
                    let offset = (u64::from(thread) * 16 + chunk) * (64 << 10);
                    let data = vec![thread.wrapping_mul(31).wrapping_add(chunk as u8); 64 << 10];
                    events.push(engine.upload(target, offset, &data).unwrap());
                }
                for event in events {
                    event.wait().unwrap();
                }
            });
        }
    });
    let all = read_back(&engine, &target, 0, 8 << 20);
    for thread in 0..8u8 {
        for chunk in 0..16u64 {
            let offset = ((u64::from(thread) * 16 + chunk) * (64 << 10)) as usize;
            let expected = thread.wrapping_mul(31).wrapping_add(chunk as u8);
            assert!(
                all[offset..offset + (64 << 10)].iter().all(|&b| b == expected),
                "thread {thread} chunk {chunk}"
            );
        }
    }
}

#[test]
fn transfers_validate_ranges_and_hold_buffers_alive_until_completion() {
    let Some(device) = first_device() else { return };
    let Some(engine) = engine(&device, 1 << 20, 4 << 20) else {
        return;
    };
    let Some((owner, view)) = device_buffer(&device, 4096) else {
        return;
    };
    assert!(engine.upload(&view, 4090, &[0u8; 16]).is_err());
    assert!(engine.download(&view, 0, 8192, None, |_| Ok(())).is_err());
    assert!(engine.fill(&view, 1, 4, 0).is_err());
    assert!(
        engine.copy(&view, 0, &view, 8, 16).is_err(),
        "overlapping same-buffer copy"
    );
    assert!(engine.upload(&view, 0, &[]).unwrap().wait().is_ok());
    let event = engine.fill(&view, 0, 4096, 7).unwrap();
    drop(owner);
    event.wait().unwrap();
    let bytes = read_back(&engine, &view, 0, 8);
    assert_eq!(bytes, 7u32.to_ne_bytes().repeat(2));
}
