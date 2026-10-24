//! Physical-device capability queries.
//!
//! Queries walk version- and extension-gated `pNext` chains, then copy every result into plain
//! owned data, so no chain pointer ever escapes the query.

use ash::vk;

use super::loader::InstanceInner;
use super::vk_error;
use crate::error::{ErrorCode, Result};

/// Support for the optional features KORA can enable, merged from core and extension structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureSupport {
    /// Timeline semaphores (core 1.2 or `VK_KHR_timeline_semaphore`).
    pub timeline_semaphore: bool,
    /// Buffer device addresses (core 1.2, KHR, or EXT).
    pub buffer_device_address: bool,
    /// Buffer device address is available only through `VK_EXT_buffer_device_address`.
    pub buffer_device_address_ext_only: bool,
    /// Any descriptor-indexing capability.
    pub descriptor_indexing: bool,
    /// Core 1.2 `descriptorIndexing` (all of the mandatory subset).
    pub descriptor_indexing_core: bool,
    /// Runtime descriptor arrays.
    pub runtime_descriptor_array: bool,
    /// Partially bound descriptors.
    pub descriptor_binding_partially_bound: bool,
    /// Variable descriptor counts.
    pub descriptor_binding_variable_descriptor_count: bool,
    /// Synchronization2 (core 1.3 or `VK_KHR_synchronization2`).
    pub synchronization2: bool,
    /// Subgroup size control.
    pub subgroup_size_control: bool,
    /// `VK_EXT_memory_priority`.
    pub memory_priority: bool,
    /// `VK_EXT_device_fault`.
    pub device_fault: bool,
    /// `VK_EXT_robustness2` robust buffer access 2.
    pub robust_buffer_access2: bool,
    /// `maintenance4` (core 1.3 or `VK_KHR_maintenance4`), which allows `LocalSizeId`.
    pub maintenance4: bool,
}

/// Feature support on a pre-1.2 device, merged from the promoted extensions' query structures.
fn extension_feature_support(
    timeline: &vk::PhysicalDeviceTimelineSemaphoreFeatures<'_>,
    bda: &vk::PhysicalDeviceBufferDeviceAddressFeatures<'_>,
    bda_ext: &vk::PhysicalDeviceBufferDeviceAddressFeaturesEXT<'_>,
    ext_bda_khr: bool,
    indexing: &vk::PhysicalDeviceDescriptorIndexingFeatures<'_>,
) -> FeatureSupport {
    let on = |b: vk::Bool32| b == vk::TRUE;
    let f = indexing;
    let any_indexing = [
        f.shader_input_attachment_array_dynamic_indexing,
        f.shader_uniform_texel_buffer_array_dynamic_indexing,
        f.shader_storage_texel_buffer_array_dynamic_indexing,
        f.shader_uniform_buffer_array_non_uniform_indexing,
        f.shader_sampled_image_array_non_uniform_indexing,
        f.shader_storage_buffer_array_non_uniform_indexing,
        f.shader_storage_image_array_non_uniform_indexing,
        f.shader_input_attachment_array_non_uniform_indexing,
        f.shader_uniform_texel_buffer_array_non_uniform_indexing,
        f.shader_storage_texel_buffer_array_non_uniform_indexing,
        f.descriptor_binding_uniform_buffer_update_after_bind,
        f.descriptor_binding_sampled_image_update_after_bind,
        f.descriptor_binding_storage_image_update_after_bind,
        f.descriptor_binding_storage_buffer_update_after_bind,
        f.descriptor_binding_uniform_texel_buffer_update_after_bind,
        f.descriptor_binding_storage_texel_buffer_update_after_bind,
        f.descriptor_binding_update_unused_while_pending,
        f.descriptor_binding_partially_bound,
        f.descriptor_binding_variable_descriptor_count,
        f.runtime_descriptor_array,
    ]
    .contains(&vk::TRUE);
    FeatureSupport {
        timeline_semaphore: on(timeline.timeline_semaphore),
        buffer_device_address: on(bda.buffer_device_address) || on(bda_ext.buffer_device_address),
        buffer_device_address_ext_only: !ext_bda_khr && on(bda_ext.buffer_device_address),
        descriptor_indexing: any_indexing,
        runtime_descriptor_array: on(f.runtime_descriptor_array),
        descriptor_binding_partially_bound: on(f.descriptor_binding_partially_bound),
        descriptor_binding_variable_descriptor_count: on(f.descriptor_binding_variable_descriptor_count),
        ..Default::default()
    }
}

/// `VK_EXT_memory_budget` snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemoryBudget {
    /// Usable bytes per heap.
    pub heap_budget: [u64; vk::MAX_MEMORY_HEAPS],
    /// Bytes in use per heap (all processes).
    pub heap_usage: [u64; vk::MAX_MEMORY_HEAPS],
}

/// Raw, owned capability snapshot of one physical device.
#[derive(Debug, Clone, Default)]
pub struct PhysicalDeviceInfo {
    /// Handle, owned by its instance.
    pub handle: vk::PhysicalDevice,
    /// `min(instance version, device version)`.
    pub api_version: u32,
    /// Core properties (limits, name, IDs).
    pub properties: vk::PhysicalDeviceProperties,
    /// Device UUID (Vulkan 1.1+).
    pub device_uuid: Option<[u8; vk::UUID_SIZE]>,
    /// Driver UUID (Vulkan 1.1+).
    pub driver_uuid: Option<[u8; vk::UUID_SIZE]>,
    /// Driver name (Vulkan 1.2+), or empty.
    pub driver_name: String,
    /// Driver info string (Vulkan 1.2+), or empty.
    pub driver_info: String,
    /// Core features.
    pub features: vk::PhysicalDeviceFeatures,
    /// Optional feature support.
    pub support: FeatureSupport,
    /// Subgroup size and operations (Vulkan 1.1+).
    pub subgroup: Option<(u32, vk::SubgroupFeatureFlags)>,
    /// Memory heaps and types.
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Memory budget when `VK_EXT_memory_budget` is advertised.
    pub memory_budget: Option<MemoryBudget>,
    /// Queue families.
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    /// Advertised device extensions.
    pub extensions: Vec<String>,
    /// Calibrateable time domains (empty without calibrated timestamps).
    pub calibrated_time_domains: Vec<vk::TimeDomainKHR>,
}

impl PhysicalDeviceInfo {
    /// `true` when the device advertises `name`.
    pub fn has_extension(&self, name: &std::ffi::CStr) -> bool {
        let name = name.to_bytes();
        self.extensions.iter().any(|e| e.as_bytes() == name)
    }

    /// Device name.
    pub fn name(&self) -> String {
        super::fixed_c_string(&self.properties.device_name)
    }
}

enum Properties2<'a> {
    Core(&'a ash::Instance),
    Khr(ash::khr::get_physical_device_properties2::Instance),
}

pub(crate) fn query_physical_device(
    instance: &InstanceInner,
    handle: vk::PhysicalDevice,
) -> Result<PhysicalDeviceInfo> {
    let raw = &instance.raw;
    let mut info = PhysicalDeviceInfo {
        handle,
        ..Default::default()
    };
    // SAFETY: `handle` was enumerated from this live instance.
    let extensions = unsafe { raw.enumerate_device_extension_properties(handle) }.map_err(|r| {
        vk_error(
            ErrorCode::NoVulkanImplementation,
            "Failed to enumerate device extensions",
            r,
        )
    })?;
    info.extensions = extensions
        .iter()
        .filter_map(|e| {
            e.extension_name_as_c_str()
                .ok()
                .map(|n| n.to_string_lossy().into_owned())
        })
        .collect();
    // SAFETY: core 1.0 query on a valid physical device.
    info.properties = unsafe { raw.get_physical_device_properties(handle) };
    info.api_version = instance.api_version.min(info.properties.api_version);

    let has = |name: &std::ffi::CStr| info.has_extension(name);
    let properties2 = if instance.api_version >= vk::API_VERSION_1_1 {
        Some(Properties2::Core(raw))
    } else if instance.properties2_extension {
        Some(Properties2::Khr(
            ash::khr::get_physical_device_properties2::Instance::new(&instance.entry, raw),
        ))
    } else {
        None
    };
    let core11 = info.api_version >= vk::API_VERSION_1_1;
    let core12 = info.api_version >= vk::API_VERSION_1_2;
    let core13 = info.api_version >= vk::API_VERSION_1_3;
    let ext_timeline = has(ash::khr::timeline_semaphore::NAME);
    let ext_bda_khr = has(ash::khr::buffer_device_address::NAME);
    let ext_bda_ext = has(ash::ext::buffer_device_address::NAME);
    let ext_descriptor_indexing = has(ash::ext::descriptor_indexing::NAME);
    let ext_sync2 = has(ash::khr::synchronization2::NAME);
    let ext_subgroup_size = has(ash::ext::subgroup_size_control::NAME);
    let ext_memory_budget = has(ash::ext::memory_budget::NAME);
    let ext_memory_priority = has(ash::ext::memory_priority::NAME);
    let ext_device_fault = has(ash::ext::device_fault::NAME);
    let ext_robustness2 = has(ash::ext::robustness2::NAME);
    let ext_maintenance4 = has(ash::khr::maintenance4::NAME);
    let ext_calibrated_khr = has(ash::khr::calibrated_timestamps::NAME);
    let ext_calibrated_ext = has(ash::ext::calibrated_timestamps::NAME);

    let Some(properties2) = properties2 else {
        // Vulkan 1.0 without properties2: core queries only.
        // SAFETY: core 1.0 queries on a valid physical device.
        unsafe {
            info.features = raw.get_physical_device_features(handle);
            info.memory_properties = raw.get_physical_device_memory_properties(handle);
            info.queue_families = raw.get_physical_device_queue_family_properties(handle);
        }
        return Ok(info);
    };

    // Features.
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
    let mut bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
    let mut bda_ext = vk::PhysicalDeviceBufferDeviceAddressFeaturesEXT::default();
    let mut indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
    let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default();
    let mut subgroup_size = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
    let mut priority = vk::PhysicalDeviceMemoryPriorityFeaturesEXT::default();
    let mut fault = vk::PhysicalDeviceFaultFeaturesEXT::default();
    let mut robustness2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
    let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default();
    info.features = {
        let mut features = vk::PhysicalDeviceFeatures2::default();
        if core12 {
            features = features.push_next(&mut f12);
        } else {
            if ext_timeline {
                features = features.push_next(&mut timeline);
            }
            if ext_bda_khr {
                features = features.push_next(&mut bda);
            } else if ext_bda_ext {
                features = features.push_next(&mut bda_ext);
            }
            if ext_descriptor_indexing {
                features = features.push_next(&mut indexing);
            }
        }
        if core13 {
            features = features.push_next(&mut f13);
        } else {
            if ext_sync2 {
                features = features.push_next(&mut sync2);
            }
            if ext_subgroup_size {
                features = features.push_next(&mut subgroup_size);
            }
            if ext_maintenance4 {
                features = features.push_next(&mut maintenance4);
            }
        }
        if ext_memory_priority {
            features = features.push_next(&mut priority);
        }
        if ext_device_fault {
            features = features.push_next(&mut fault);
        }
        if ext_robustness2 {
            features = features.push_next(&mut robustness2);
        }
        // SAFETY: every chained structure is advertised by the device (extension or core version)
        // and lives until the end of this block.
        unsafe {
            match &properties2 {
                Properties2::Core(raw) => raw.get_physical_device_features2(handle, &mut features),
                Properties2::Khr(khr) => khr.get_physical_device_features2(handle, &mut features),
            }
        }
        features.features
    };
    let on = |b: vk::Bool32| b == vk::TRUE;
    info.support = if core12 {
        FeatureSupport {
            timeline_semaphore: on(f12.timeline_semaphore),
            buffer_device_address: on(f12.buffer_device_address),
            descriptor_indexing: on(f12.descriptor_indexing),
            descriptor_indexing_core: on(f12.descriptor_indexing),
            runtime_descriptor_array: on(f12.runtime_descriptor_array),
            descriptor_binding_partially_bound: on(f12.descriptor_binding_partially_bound),
            descriptor_binding_variable_descriptor_count: on(f12.descriptor_binding_variable_descriptor_count),
            ..Default::default()
        }
    } else {
        extension_feature_support(&timeline, &bda, &bda_ext, ext_bda_khr, &indexing)
    };
    info.support.synchronization2 = if core13 {
        on(f13.synchronization2)
    } else {
        on(sync2.synchronization2)
    };
    info.support.subgroup_size_control = if core13 {
        on(f13.subgroup_size_control)
    } else {
        on(subgroup_size.subgroup_size_control)
    };
    info.support.maintenance4 = if core13 {
        on(f13.maintenance4)
    } else {
        on(maintenance4.maintenance4)
    };
    info.support.memory_priority = on(priority.memory_priority);
    info.support.device_fault = on(fault.device_fault);
    info.support.robust_buffer_access2 = on(robustness2.robust_buffer_access2);

    // Properties.
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut p12 = vk::PhysicalDeviceVulkan12Properties::default();
    {
        let mut properties = vk::PhysicalDeviceProperties2::default();
        if core11 {
            properties = properties.push_next(&mut id).push_next(&mut subgroup);
        }
        if core12 {
            properties = properties.push_next(&mut p12);
        }
        // SAFETY: the chained structures are core in the device's API version and outlive the call.
        unsafe {
            match &properties2 {
                Properties2::Core(raw) => raw.get_physical_device_properties2(handle, &mut properties),
                Properties2::Khr(khr) => khr.get_physical_device_properties2(handle, &mut properties),
            }
        }
        info.properties = properties.properties;
    }
    if core11 {
        info.device_uuid = Some(id.device_uuid);
        info.driver_uuid = Some(id.driver_uuid);
        info.subgroup =
            (subgroup.subgroup_size != 0).then_some((subgroup.subgroup_size, subgroup.supported_operations));
    }
    if core12 {
        info.driver_name = super::fixed_c_string(&p12.driver_name);
        info.driver_info = super::fixed_c_string(&p12.driver_info);
    }

    // Memory.
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    {
        let mut memory = vk::PhysicalDeviceMemoryProperties2::default();
        if ext_memory_budget {
            memory = memory.push_next(&mut budget);
        }
        // SAFETY: the budget structure is chained only when the extension is advertised.
        unsafe {
            match &properties2 {
                Properties2::Core(raw) => raw.get_physical_device_memory_properties2(handle, &mut memory),
                Properties2::Khr(khr) => khr.get_physical_device_memory_properties2(handle, &mut memory),
            }
        }
        info.memory_properties = memory.memory_properties;
    }
    if ext_memory_budget {
        info.memory_budget = Some(MemoryBudget {
            heap_budget: budget.heap_budget,
            heap_usage: budget.heap_usage,
        });
    }

    // SAFETY: core 1.0 query on a valid physical device.
    info.queue_families = unsafe { raw.get_physical_device_queue_family_properties(handle) };

    if ext_calibrated_khr {
        let calibrated = ash::khr::calibrated_timestamps::Instance::new(&instance.entry, raw);
        // SAFETY: the extension is advertised by this device.
        info.calibrated_time_domains =
            unsafe { calibrated.get_physical_device_calibrateable_time_domains(handle) }.map_err(|r| {
                vk_error(
                    ErrorCode::UnsupportedFeature,
                    "Failed to query calibrated time domains",
                    r,
                )
            })?;
    } else if ext_calibrated_ext {
        let calibrated = ash::ext::calibrated_timestamps::Instance::new(&instance.entry, raw);
        // SAFETY: the extension is advertised by this device.
        info.calibrated_time_domains =
            unsafe { calibrated.get_physical_device_calibrateable_time_domains(handle) }.map_err(|r| {
                vk_error(
                    ErrorCode::UnsupportedFeature,
                    "Failed to query calibrated time domains",
                    r,
                )
            })?;
    }
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_promoted_features_from_extension_query_structures() {
        let timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
        let bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let bda_ext = vk::PhysicalDeviceBufferDeviceAddressFeaturesEXT::default();
        let indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .descriptor_binding_partially_bound(true)
            .runtime_descriptor_array(true);
        let support = extension_feature_support(&timeline, &bda, &bda_ext, true, &indexing);
        assert!(support.timeline_semaphore);
        assert!(support.buffer_device_address);
        assert!(!support.buffer_device_address_ext_only);
        assert!(support.descriptor_indexing);
        assert!(!support.descriptor_indexing_core);
        assert!(support.runtime_descriptor_array);
        assert!(support.descriptor_binding_partially_bound);
        assert!(!support.descriptor_binding_variable_descriptor_count);

        let bda_ext = vk::PhysicalDeviceBufferDeviceAddressFeaturesEXT::default().buffer_device_address(true);
        let none = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        let ext_only = extension_feature_support(
            &vk::PhysicalDeviceTimelineSemaphoreFeatures::default(),
            &vk::PhysicalDeviceBufferDeviceAddressFeatures::default(),
            &bda_ext,
            false,
            &none,
        );
        assert!(!ext_only.timeline_semaphore);
        assert!(ext_only.buffer_device_address && ext_only.buffer_device_address_ext_only);
        assert!(!ext_only.descriptor_indexing);
    }
}
