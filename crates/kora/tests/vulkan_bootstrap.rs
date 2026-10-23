//! Vulkan loader, instance, physical-device, and logical-device lifecycle against real drivers.

mod common;

use std::sync::{Arc, Mutex};

use kora::log::{self, Category, Level};
use kora::vulkan::ash::vk;
use kora::vulkan::{DeviceOptions, FaultFunction, InstanceOptions, device_info};

/// Captures validation errors routed to the KORA log for the duration of a test.
struct ValidationErrors {
    messages: Arc<Mutex<Vec<String>>>,
}

static LOG_LOCK: Mutex<()> = Mutex::new(());

impl ValidationErrors {
    fn capture() -> (Self, std::sync::MutexGuard<'static, ()>) {
        let guard = LOG_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let messages = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&messages);
        log::set_level(Level::Warn);
        log::set_sink(Some(Box::new(move |level, category, message| {
            if category == Category::Device && level == Level::Error {
                sink.lock().unwrap().push(message.to_string());
            }
        })));
        (Self { messages }, guard)
    }

    fn assert_none(&self) {
        let messages = self.messages.lock().unwrap();
        assert!(messages.is_empty(), "validation errors: {messages:#?}");
    }
}

impl Drop for ValidationErrors {
    fn drop(&mut self) {
        log::set_sink(None);
    }
}

fn validated() -> InstanceOptions {
    InstanceOptions {
        enable_validation: true,
        ..Default::default()
    }
}

#[test]
fn repeated_instance_creation_and_destruction() {
    let Some((loader, _)) = common::vulkan(&InstanceOptions::default()) else {
        return;
    };
    for _ in 0..100 {
        let instance = loader.create_instance(&validated()).expect("instance");
        assert_ne!(instance.handle(), vk::Instance::null());
        assert!(instance.api_version() <= vk::API_VERSION_1_3);
    }
}

#[test]
fn requested_api_version_is_clamped() {
    let Some((loader, _)) = common::vulkan(&InstanceOptions::default()) else {
        return;
    };
    assert!(loader.supported_api_version() >= vk::API_VERSION_1_0);
    let options = InstanceOptions {
        requested_api_version: vk::API_VERSION_1_1,
        ..Default::default()
    };
    let instance = loader.create_instance(&options).expect("instance");
    assert!(instance.api_version() <= vk::API_VERSION_1_1);
    assert!(instance.api_version() <= loader.supported_api_version());
}

#[test]
fn enumerates_and_queries_every_device() {
    let (errors, _guard) = ValidationErrors::capture();
    let Some((_, instance)) = common::vulkan(&validated()) else {
        return;
    };
    let devices = instance.physical_devices().expect("enumerate");
    assert!(!devices.is_empty());
    for device in &devices {
        assert_ne!(device.handle, vk::PhysicalDevice::null());
        assert_ne!(device.properties.api_version, 0);
        assert!(!device.name().is_empty());
        assert!(device.api_version <= instance.api_version());
        assert!(device.memory_properties.memory_type_count >= 1);
        assert!(device.memory_properties.memory_heap_count >= 1);
        assert!(!device.queue_families.is_empty());
        if device.api_version >= vk::API_VERSION_1_1 {
            assert!(device.device_uuid.is_some());
            assert!(device.subgroup.is_some());
        }
        let info = device_info(device);
        assert_eq!(info.name, device.name());
        assert_eq!(info.queue_families.len(), device.queue_families.len());
        assert_eq!(
            info.memory_heaps.len(),
            device.memory_properties.memory_heap_count as usize
        );
        assert_eq!(
            info.memory_types.len(),
            device.memory_properties.memory_type_count as usize
        );
        assert_eq!(
            info.memory_type_preferences.automatic.len(),
            info.memory_types.len()
        );
        eprintln!(
            "device: {} ({}, Vulkan {}, driver {:?}, topology {}, tier {:?})",
            info.name,
            info.kind.name(),
            info.api_version,
            info.driver_name,
            info.memory_topology.name(),
            info.tier
        );
    }
    errors.assert_none();
}

#[test]
fn creates_a_logical_device_for_every_compute_capable_device() {
    let (errors, _guard) = ValidationErrors::capture();
    let Some((_, instance)) = common::vulkan(&validated()) else {
        return;
    };
    let mut eligible = 0;
    for physical in instance.physical_devices().expect("enumerate") {
        if !physical
            .queue_families
            .iter()
            .any(|f| f.queue_count != 0 && f.queue_flags.contains(vk::QueueFlags::COMPUTE))
        {
            continue;
        }
        eligible += 1;
        let device = instance
            .create_device(&physical, &DeviceOptions::default())
            .expect("logical device");
        assert_ne!(device.handle(), vk::Device::null());
        assert_ne!(device.compute_queue(), vk::Queue::null());
        assert_ne!(device.transfer_queue(), vk::Queue::null());
        for extension in device.enabled_extensions() {
            assert!(
                physical.extensions.contains(extension),
                "{extension} enabled but not advertised"
            );
        }
        let portability = physical.has_extension(kora::vulkan::ash::khr::portability_subset::NAME);
        assert_eq!(
            device
                .enabled_extensions()
                .iter()
                .any(|e| e == "VK_KHR_portability_subset"),
            portability
        );
        let enabled = device.enabled_features();
        assert!(!enabled.timeline_semaphore || physical.support.timeline_semaphore);
        assert!(!enabled.synchronization2 || physical.support.synchronization2);
        if enabled.timeline_semaphore {
            let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(7);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
            // SAFETY: the device is live and the create info is valid for the call.
            let semaphore = unsafe { device.raw().create_semaphore(&info, None) }.expect("semaphore");
            assert_eq!(device.semaphore_counter_value(semaphore).unwrap(), 7);
            let semaphores = [semaphore];
            let values = [7];
            let wait = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            device.wait_semaphores(&wait, 0).expect("already signaled");
            // SAFETY: no pending work uses the semaphore.
            unsafe { device.raw().destroy_semaphore(semaphore, None) };
        }
    }
    if eligible == 0 {
        eprintln!("skipping: no compute-capable device");
    }
    errors.assert_none();
}

#[test]
fn injected_allocation_failure_is_returned_without_reaching_the_driver() {
    let Some((_, instance)) = common::vulkan(&InstanceOptions::default()) else {
        return;
    };
    let physical = instance.physical_devices().expect("enumerate");
    let Some(physical) = physical.iter().find(|p| {
        p.queue_families
            .iter()
            .any(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
    }) else {
        return;
    };
    let device = instance
        .create_device(physical, &DeviceOptions::default())
        .expect("device");
    device
        .faults()
        .fail_on_nth(
            FaultFunction::AllocateMemory,
            2,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        )
        .unwrap();
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(4096)
        .memory_type_index(0);
    let first = device
        .allocate_memory(&info)
        .expect("first allocation is forwarded");
    assert_eq!(
        device.allocate_memory(&info),
        Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
    );
    let third = device
        .allocate_memory(&info)
        .expect("third allocation is forwarded");
    // SAFETY: both allocations belong to this device and are unused.
    unsafe {
        device.raw().free_memory(first, None);
        device.raw().free_memory(third, None);
    }
}

#[test]
fn devices_and_instances_can_outlive_their_creators() {
    let Some((_, instance)) = common::vulkan(&InstanceOptions::default()) else {
        return;
    };
    let physical = instance.physical_devices().expect("enumerate");
    let Some(physical) = physical.iter().find(|p| {
        p.queue_families
            .iter()
            .any(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
    }) else {
        return;
    };
    let device = instance
        .create_device(physical, &DeviceOptions::default())
        .expect("device");
    drop(instance);
    // The device keeps its instance (and loader) alive.
    assert_ne!(device.instance().handle(), vk::Instance::null());
    let clone = device.clone();
    drop(device);
    assert!(clone.wait_idle().is_ok());
}
