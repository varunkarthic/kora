//! Logical devices and the wrapped device-level calls used by the runtime.

use std::ffi::{CStr, c_char};
use std::fmt;
use std::sync::Arc;

use ash::vk;

use super::fault::{FaultFunction, FaultInjector};
use super::loader::Instance;
use super::physical::PhysicalDeviceInfo;
use super::vk_error;
use crate::error::{Error, ErrorCode, Result};

/// Logical-device creation options. Features are enabled only when the device reports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceOptions {
    /// Use a transfer-only queue family for transfers when one exists.
    pub enable_dedicated_transfer_queue: bool,
    /// Enable timeline semaphores.
    pub enable_timeline_semaphore: bool,
    /// Enable synchronization2.
    pub enable_synchronization2: bool,
    /// Enable `VK_EXT_device_fault` diagnostics.
    pub enable_device_fault_diagnostics: bool,
}

impl Default for DeviceOptions {
    fn default() -> Self {
        Self {
            enable_dedicated_transfer_queue: true,
            enable_timeline_semaphore: true,
            enable_synchronization2: true,
            enable_device_fault_diagnostics: true,
        }
    }
}

/// Optional features that were actually enabled on a logical device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EnabledFeatures {
    /// Timeline semaphores.
    pub timeline_semaphore: bool,
    /// Buffer device addresses.
    pub buffer_device_address: bool,
    /// Descriptor indexing.
    pub descriptor_indexing: bool,
    /// Synchronization2.
    pub synchronization2: bool,
    /// `VK_EXT_device_fault`.
    pub device_fault_diagnostics: bool,
    /// `maintenance4` (allows `LocalSizeId` workgroup sizes).
    pub maintenance4: bool,
    /// Calibrated timestamps (KHR or EXT).
    pub calibrated_timestamps: bool,
}

enum Timeline {
    Core,
    Khr(ash::khr::timeline_semaphore::Device),
    Unavailable,
}

enum Sync2 {
    Core,
    Khr(ash::khr::synchronization2::Device),
    Unavailable,
}

enum Calibrated {
    Khr(ash::khr::calibrated_timestamps::Device),
    Ext(ash::ext::calibrated_timestamps::Device),
}

pub(crate) struct DeviceInner {
    instance: Instance,
    raw: ash::Device,
    physical: PhysicalDeviceInfo,
    compute_family: u32,
    transfer_family: u32,
    compute_queue: vk::Queue,
    transfer_queue: vk::Queue,
    enabled: EnabledFeatures,
    extensions: Vec<String>,
    timeline: Timeline,
    sync2: Sync2,
    calibrated: Option<Calibrated>,
    #[expect(dead_code, reason = "read by Runtime device-loss reports (#258)")]
    device_fault: Option<ash::ext::device_fault::Device>,
    faults: FaultInjector,
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        // SAFETY: owners of device children (pools, pipelines, buffers, queues) hold a clone of this
        // `Device`, so the last reference drops only after every child is destroyed.
        unsafe { self.raw.destroy_device(None) };
    }
}

/// A logical device with one compute queue and one transfer queue (possibly the same queue).
/// Cloning shares it.
#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.inner.physical.name())
            .field("compute_family", &self.inner.compute_family)
            .field("transfer_family", &self.inner.transfer_family)
            .field("enabled", &self.inner.enabled)
            .finish()
    }
}

fn select_families(families: &[vk::QueueFamilyProperties]) -> (Option<u32>, Option<u32>) {
    let mut compute = None;
    let mut dedicated_transfer = None;
    for (index, family) in families.iter().enumerate() {
        if family.queue_count == 0 {
            continue;
        }
        let flags = family.queue_flags;
        if compute.is_none() && flags.contains(vk::QueueFlags::COMPUTE) {
            compute = Some(index as u32);
        }
        if dedicated_transfer.is_none()
            && flags.contains(vk::QueueFlags::TRANSFER)
            && !flags.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        {
            dedicated_transfer = Some(index as u32);
        }
    }
    (compute, dedicated_transfer)
}

impl Instance {
    /// Creates a logical device enabling only queried optional features.
    pub fn create_device(&self, physical: &PhysicalDeviceInfo, options: &DeviceOptions) -> Result<Device> {
        let name = physical.name();
        let (compute_family, dedicated_transfer) = select_families(&physical.queue_families);
        let compute_family = compute_family.ok_or_else(|| {
            Error::new(
                ErrorCode::NoComputeQueue,
                "Physical device exposes no compute-capable queue",
            )
            .with_device(&name)
        })?;
        let transfer_family = dedicated_transfer
            .filter(|_| options.enable_dedicated_transfer_queue)
            .unwrap_or(compute_family);
        let priorities = [1.0f32];
        let mut queue_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(compute_family)
                .queue_priorities(&priorities),
        ];
        if transfer_family != compute_family {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(transfer_family)
                    .queue_priorities(&priorities),
            );
        }

        let support = physical.support;
        let core11_or_properties2 =
            physical.api_version >= vk::API_VERSION_1_1 || self.inner.properties2_extension;
        let core12 = physical.api_version >= vk::API_VERSION_1_2;
        let core13 = physical.api_version >= vk::API_VERSION_1_3;
        // Promoted features need the features2 chain; on a bare 1.0 instance only core features apply.
        let chain = core11_or_properties2;
        let timeline = chain && options.enable_timeline_semaphore && support.timeline_semaphore;
        let bda = chain && support.buffer_device_address;
        let indexing = chain
            && if core12 {
                support.descriptor_indexing_core
            } else {
                support.runtime_descriptor_array
            };
        let sync2 = chain && options.enable_synchronization2 && support.synchronization2;
        // maintenance4 lets kernels declare their workgroup size with LocalSizeId (autotuning).
        let maintenance4 = chain && support.maintenance4;
        let fault = chain
            && options.enable_device_fault_diagnostics
            && support.device_fault
            && physical.has_extension(ash::ext::device_fault::NAME);

        let mut extensions: Vec<&'static CStr> = Vec::new();
        let mut enable = |name: &'static CStr| {
            if physical.has_extension(name) {
                extensions.push(name);
            }
        };
        if !core12 && timeline {
            enable(ash::khr::timeline_semaphore::NAME);
        }
        if !core12 && bda {
            enable(if support.buffer_device_address_ext_only {
                ash::ext::buffer_device_address::NAME
            } else {
                ash::khr::buffer_device_address::NAME
            });
        }
        if !core12 && indexing {
            enable(ash::ext::descriptor_indexing::NAME);
        }
        if !core13 && sync2 {
            enable(ash::khr::synchronization2::NAME);
        }
        if !core13 && maintenance4 {
            enable(ash::khr::maintenance4::NAME);
        }
        if fault {
            enable(ash::ext::device_fault::NAME);
        }
        enable(ash::khr::portability_subset::NAME);
        enable(ash::khr::calibrated_timestamps::NAME);
        if !physical.has_extension(ash::khr::calibrated_timestamps::NAME) {
            enable(ash::ext::calibrated_timestamps::NAME);
        }
        let extension_pointers: Vec<*const c_char> = extensions.iter().map(|n| n.as_ptr()).collect();

        // Shader arithmetic features are enabled whenever supported so that modules declaring the
        // matching capabilities (which reflection accepts only when supported) are valid.
        let base = vk::PhysicalDeviceFeatures {
            robust_buffer_access: physical.features.robust_buffer_access,
            shader_int16: physical.features.shader_int16,
            shader_int64: physical.features.shader_int64,
            shader_float64: physical.features.shader_float64,
            ..Default::default()
        };
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(timeline)
            .buffer_device_address(bda)
            .descriptor_indexing(indexing)
            .runtime_descriptor_array(indexing && support.runtime_descriptor_array)
            .descriptor_binding_partially_bound(indexing && support.descriptor_binding_partially_bound)
            .descriptor_binding_variable_descriptor_count(
                indexing && support.descriptor_binding_variable_descriptor_count,
            );
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(sync2)
            .maintenance4(maintenance4);
        let mut maintenance4_features = vk::PhysicalDeviceMaintenance4Features::default().maintenance4(true);
        let mut timeline_features =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
        let mut bda_features =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut bda_ext_features =
            vk::PhysicalDeviceBufferDeviceAddressFeaturesEXT::default().buffer_device_address(true);
        let mut indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .runtime_descriptor_array(support.runtime_descriptor_array)
            .descriptor_binding_partially_bound(support.descriptor_binding_partially_bound)
            .descriptor_binding_variable_descriptor_count(
                support.descriptor_binding_variable_descriptor_count,
            );
        let mut sync2_features = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut fault_features = vk::PhysicalDeviceFaultFeaturesEXT::default().device_fault(true);

        let raw = {
            let mut features2 = vk::PhysicalDeviceFeatures2::default().features(base);
            if core12 {
                features2 = features2.push_next(&mut f12);
            } else {
                if timeline {
                    features2 = features2.push_next(&mut timeline_features);
                }
                if bda && support.buffer_device_address_ext_only {
                    features2 = features2.push_next(&mut bda_ext_features);
                } else if bda {
                    features2 = features2.push_next(&mut bda_features);
                }
                if indexing {
                    features2 = features2.push_next(&mut indexing_features);
                }
            }
            if core13 {
                features2 = features2.push_next(&mut f13);
            } else {
                if sync2 {
                    features2 = features2.push_next(&mut sync2_features);
                }
                if maintenance4 {
                    features2 = features2.push_next(&mut maintenance4_features);
                }
            }
            if fault {
                features2 = features2.push_next(&mut fault_features);
            }
            let mut create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&extension_pointers);
            if chain {
                create_info = create_info.push_next(&mut features2);
            } else {
                create_info = create_info.enabled_features(&base);
            }
            // SAFETY: `physical.handle` belongs to this live instance; every chained structure and
            // name pointer refers to locals that outlive the call; each enabled extension and
            // feature was queried as supported.
            unsafe { self.inner.raw.create_device(physical.handle, &create_info, None) }.map_err(|r| {
                vk_error(
                    ErrorCode::NoEligibleDevice,
                    "Failed to create Vulkan logical device",
                    r,
                )
                .with_device(&name)
            })?
        };

        // SAFETY: both families were requested with one queue each in `queue_infos`.
        let (compute_queue, transfer_queue) = unsafe {
            (
                raw.get_device_queue(compute_family, 0),
                raw.get_device_queue(transfer_family, 0),
            )
        };
        let instance = &self.inner.raw;
        let calibrated = if extensions.contains(&ash::khr::calibrated_timestamps::NAME) {
            Some(Calibrated::Khr(ash::khr::calibrated_timestamps::Device::new(
                instance, &raw,
            )))
        } else if extensions.contains(&ash::ext::calibrated_timestamps::NAME) {
            Some(Calibrated::Ext(ash::ext::calibrated_timestamps::Device::new(
                instance, &raw,
            )))
        } else {
            None
        };
        let enabled = EnabledFeatures {
            timeline_semaphore: timeline,
            buffer_device_address: bda,
            descriptor_indexing: indexing,
            synchronization2: sync2,
            device_fault_diagnostics: fault,
            maintenance4,
            calibrated_timestamps: calibrated.is_some(),
        };
        let inner = DeviceInner {
            timeline: match (timeline, core12) {
                (false, _) => Timeline::Unavailable,
                (true, true) => Timeline::Core,
                (true, false) => Timeline::Khr(ash::khr::timeline_semaphore::Device::new(instance, &raw)),
            },
            sync2: match (sync2, core13) {
                (false, _) => Sync2::Unavailable,
                (true, true) => Sync2::Core,
                (true, false) => Sync2::Khr(ash::khr::synchronization2::Device::new(instance, &raw)),
            },
            calibrated,
            device_fault: fault.then(|| ash::ext::device_fault::Device::new(instance, &raw)),
            instance: self.clone(),
            raw,
            physical: physical.clone(),
            compute_family,
            transfer_family,
            compute_queue,
            transfer_queue,
            enabled,
            extensions: extensions
                .iter()
                .map(|n| n.to_string_lossy().into_owned())
                .collect(),
            faults: FaultInjector::default(),
        };
        Ok(Device {
            inner: Arc::new(inner),
        })
    }
}

impl Device {
    /// The ash device (for interop and the runtime's own recording code).
    pub fn raw(&self) -> &ash::Device {
        &self.inner.raw
    }
    /// Raw handle.
    pub fn handle(&self) -> vk::Device {
        self.inner.raw.handle()
    }
    /// The instance this device belongs to.
    pub fn instance(&self) -> &Instance {
        &self.inner.instance
    }
    /// The physical-device snapshot used to create this device.
    pub fn physical(&self) -> &PhysicalDeviceInfo {
        &self.inner.physical
    }
    /// Compute queue (externally synchronized: owned by one submission thread).
    pub fn compute_queue(&self) -> vk::Queue {
        self.inner.compute_queue
    }
    /// Transfer queue; equal to the compute queue when no dedicated family is used.
    pub fn transfer_queue(&self) -> vk::Queue {
        self.inner.transfer_queue
    }
    /// Compute queue family.
    pub fn compute_queue_family(&self) -> u32 {
        self.inner.compute_family
    }
    /// Transfer queue family.
    pub fn transfer_queue_family(&self) -> u32 {
        self.inner.transfer_family
    }
    /// Enabled optional features.
    pub fn enabled_features(&self) -> EnabledFeatures {
        self.inner.enabled
    }
    /// Enabled device extensions.
    pub fn enabled_extensions(&self) -> &[String] {
        &self.inner.extensions
    }
    /// Fault injection controls for this device's wrapped calls.
    pub fn faults(&self) -> &FaultInjector {
        &self.inner.faults
    }
    /// `true` when both handles refer to the same logical device.
    pub fn same_device(&self, other: &Device) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// `vkAllocateMemory`, subject to fault injection.
    pub fn allocate_memory(
        &self,
        info: &vk::MemoryAllocateInfo<'_>,
    ) -> std::result::Result<vk::DeviceMemory, vk::Result> {
        if let Some(injected) = self.inner.faults.check(FaultFunction::AllocateMemory) {
            return Err(injected);
        }
        // SAFETY: `info` is a valid allocate-info structure borrowed for the call; the device is live.
        unsafe { self.inner.raw.allocate_memory(info, None) }
    }

    /// `vkQueueSubmit`, subject to fault injection. The caller owns external queue synchronization.
    pub fn queue_submit(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
    ) -> std::result::Result<(), vk::Result> {
        if let Some(injected) = self.inner.faults.check(FaultFunction::QueueSubmit) {
            return Err(injected);
        }
        // SAFETY: the caller guarantees exclusive access to `queue` and that every handle referenced
        // by `submits` stays alive until the submission completes.
        unsafe { self.inner.raw.queue_submit(queue, submits, fence) }
    }

    /// `vkQueueSubmit2` (core 1.3 or KHR), subject to fault injection. Requires synchronization2.
    pub fn queue_submit2(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> std::result::Result<(), vk::Result> {
        if let Some(injected) = self.inner.faults.check(FaultFunction::QueueSubmit) {
            return Err(injected);
        }
        // SAFETY: as for `queue_submit`; the function is dispatched through the table matching how
        // synchronization2 was enabled.
        unsafe {
            match &self.inner.sync2 {
                Sync2::Core => self.inner.raw.queue_submit2(queue, submits, fence),
                Sync2::Khr(khr) => khr.queue_submit2(queue, submits, fence),
                Sync2::Unavailable => Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
            }
        }
    }

    /// `vkCmdPipelineBarrier2` (core 1.3 or KHR). Returns `false` when synchronization2 is disabled.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state, owned by the calling thread, and allocated
    /// from this device; `dependency` must describe resources of this device.
    pub unsafe fn cmd_pipeline_barrier2(
        &self,
        command_buffer: vk::CommandBuffer,
        dependency: &vk::DependencyInfo<'_>,
    ) -> bool {
        // SAFETY: forwarded from the caller's contract.
        unsafe {
            match &self.inner.sync2 {
                Sync2::Core => self.inner.raw.cmd_pipeline_barrier2(command_buffer, dependency),
                Sync2::Khr(khr) => khr.cmd_pipeline_barrier2(command_buffer, dependency),
                Sync2::Unavailable => return false,
            }
        }
        true
    }

    /// Current value of a timeline semaphore.
    pub fn semaphore_counter_value(&self, semaphore: vk::Semaphore) -> std::result::Result<u64, vk::Result> {
        // SAFETY: `semaphore` is a timeline semaphore of this device (the caller created it).
        unsafe {
            match &self.inner.timeline {
                Timeline::Core => self.inner.raw.get_semaphore_counter_value(semaphore),
                Timeline::Khr(khr) => khr.get_semaphore_counter_value(semaphore),
                Timeline::Unavailable => Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
            }
        }
    }

    /// Waits on timeline semaphores.
    pub fn wait_semaphores(
        &self,
        info: &vk::SemaphoreWaitInfo<'_>,
        timeout_ns: u64,
    ) -> std::result::Result<(), vk::Result> {
        // SAFETY: `info` references timeline semaphores of this device that outlive the call.
        unsafe {
            match &self.inner.timeline {
                Timeline::Core => self.inner.raw.wait_semaphores(info, timeout_ns),
                Timeline::Khr(khr) => khr.wait_semaphores(info, timeout_ns),
                Timeline::Unavailable => Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
            }
        }
    }

    /// `vkGetQueryPoolResults` for 64-bit results, subject to fault injection.
    pub fn get_query_pool_results(
        &self,
        pool: vk::QueryPool,
        first: u32,
        data: &mut [u64],
        flags: vk::QueryResultFlags,
    ) -> std::result::Result<(), vk::Result> {
        if let Some(injected) = self.inner.faults.check(FaultFunction::GetQueryPoolResults) {
            return Err(injected);
        }
        // SAFETY: `pool` belongs to this device and contains `first..first + data.len()`; `data`
        // is sized for one 64-bit result per query (flags must include TYPE_64).
        unsafe {
            self.inner
                .raw
                .get_query_pool_results(pool, first, data, flags | vk::QueryResultFlags::TYPE_64)
        }
    }

    /// Calibrated `(device ticks, host monotonic ns)` pair and max deviation, when supported.
    pub fn calibrated_timestamps(&self, host_domain: vk::TimeDomainKHR) -> Option<(u64, u64, u64)> {
        let infos = [
            vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::DEVICE),
            vk::CalibratedTimestampInfoKHR::default().time_domain(host_domain),
        ];
        // SAFETY: both domains are standard; the extension functions were loaded for this device.
        let result = unsafe {
            match self.inner.calibrated.as_ref()? {
                Calibrated::Khr(khr) => khr.get_calibrated_timestamps(&infos),
                Calibrated::Ext(ext) => ext.get_calibrated_timestamps(&infos),
            }
        };
        result
            .ok()
            .map(|(values, deviation)| (values[0], values[1], deviation))
    }

    /// Re-queries `VK_EXT_memory_budget` (None when the extension is not advertised or the
    /// instance lacks properties2).
    pub fn memory_budget(&self) -> Option<super::MemoryBudget> {
        if !self.inner.physical.has_extension(ash::ext::memory_budget::NAME)
            || self.inner.instance.api_version() < vk::API_VERSION_1_1
        {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut memory = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: the instance is Vulkan 1.1+, the physical device advertises the budget extension,
        // and both structures outlive the call.
        unsafe {
            self.inner
                .instance
                .raw()
                .get_physical_device_memory_properties2(self.inner.physical.handle, &mut memory)
        };
        Some(super::MemoryBudget {
            heap_budget: budget.heap_budget,
            heap_usage: budget.heap_usage,
        })
    }

    /// Blocks until the device is idle (teardown and recovery only).
    pub fn wait_idle(&self) -> std::result::Result<(), vk::Result> {
        // SAFETY: requires external synchronization of all queues; callers use it only after their
        // queue owners have stopped submitting.
        unsafe { self.inner.raw.device_wait_idle() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn family(flags: vk::QueueFlags, count: u32) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: count,
            ..Default::default()
        }
    }

    #[test]
    fn selects_first_compute_family_and_first_transfer_only_family() {
        let families = [
            family(
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                1,
            ),
            family(vk::QueueFlags::TRANSFER, 0),
            family(vk::QueueFlags::COMPUTE, 2),
            family(vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING, 1),
        ];
        assert_eq!(select_families(&families), (Some(0), Some(3)));
        assert_eq!(
            select_families(&[family(vk::QueueFlags::GRAPHICS, 1)]),
            (None, None)
        );
        assert_eq!(select_families(&[]), (None, None));
    }
}
