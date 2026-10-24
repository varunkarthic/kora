//! Converts raw Vulkan queries into the vendor-neutral [`DeviceInfo`] snapshot.

use ash::vk;

use super::physical::PhysicalDeviceInfo;
use crate::device::{
    ApiVersion, CapabilityTier, DeviceFeatures, DeviceInfo, DeviceKind, DeviceLimits, MemoryHeapInfo,
    MemoryTopology, MemoryTypeInfo, MemoryTypePreferences, QueueFamilyInfo, SubgroupInfo,
};

/// Memory topology and per-policy type rankings.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MemoryClassification {
    /// Classified topology.
    pub topology: MemoryTopology,
    /// Placement should be measured before relying on the topology.
    pub needs_benchmark: bool,
    /// Type rankings.
    pub type_preferences: MemoryTypePreferences,
}

#[derive(Clone, Copy)]
enum MemoryPolicy {
    Auto,
    DevicePreferred,
    HostVisible,
    ReadMostly,
    WriteMostly,
    Streaming,
    Readback,
}

fn types(memory: &vk::PhysicalDeviceMemoryProperties) -> &[vk::MemoryType] {
    &memory.memory_types[..(memory.memory_type_count as usize).min(vk::MAX_MEMORY_TYPES)]
}

fn heaps(memory: &vk::PhysicalDeviceMemoryProperties) -> &[vk::MemoryHeap] {
    &memory.memory_heaps[..(memory.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS)]
}

fn type_is_device_local(memory: &vk::PhysicalDeviceMemoryProperties, memory_type: &vk::MemoryType) -> bool {
    memory_type
        .property_flags
        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        || heaps(memory)
            .get(memory_type.heap_index as usize)
            .is_some_and(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
}

fn preference_score(
    policy: MemoryPolicy,
    topology: MemoryTopology,
    local: bool,
    visible: bool,
    coherent: bool,
    cached: bool,
) -> u8 {
    let score = |a: bool, b: bool, c: bool| u8::from(a) * 4 + u8::from(b) * 2 + u8::from(c);
    match policy {
        MemoryPolicy::Auto => match topology {
            MemoryTopology::Uma => score(local && visible, visible && coherent, local),
            MemoryTopology::Discrete | MemoryTopology::DiscreteWithBar => {
                score(local && !visible, local && visible, visible && cached)
            }
            MemoryTopology::Unknown => score(local && !visible, local && visible, visible),
        },
        MemoryPolicy::DevicePreferred => score(local && !visible, local && visible, local),
        MemoryPolicy::HostVisible => score(visible, coherent, cached),
        MemoryPolicy::ReadMostly => score(local, cached, visible),
        MemoryPolicy::WriteMostly => score(local, coherent, visible),
        MemoryPolicy::Streaming => score(visible, coherent, local),
        MemoryPolicy::Readback => score(visible, cached, coherent),
    }
}

fn order_memory_types(
    memory: &vk::PhysicalDeviceMemoryProperties,
    policy: MemoryPolicy,
    topology: MemoryTopology,
) -> Vec<u32> {
    let scores: Vec<u8> = types(memory)
        .iter()
        .map(|t| {
            let flags = t.property_flags;
            preference_score(
                policy,
                topology,
                type_is_device_local(memory, t),
                flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
                flags.contains(vk::MemoryPropertyFlags::HOST_CACHED),
            )
        })
        .collect();
    let mut indices: Vec<u32> = (0..scores.len() as u32).collect();
    // Stable: equal scores keep enumeration order.
    indices.sort_by(|&l, &r| scores[r as usize].cmp(&scores[l as usize]));
    indices
}

/// Classifies memory topology from type and heap flags (never from vendor identity).
pub fn classify_memory_properties(memory: &vk::PhysicalDeviceMemoryProperties) -> MemoryClassification {
    let (mut local_visible, mut local_invisible) = (false, false);
    for memory_type in types(memory) {
        let local = type_is_device_local(memory, memory_type);
        let visible = memory_type
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        local_visible |= local && visible;
        local_invisible |= local && !visible;
    }
    // A discrete GPU (even with a host-visible BAR window) also exposes a separate system-memory
    // heap. When every heap is device-local, host-visible and host-invisible types share one
    // physical pool, so the device is unified however many types it advertises.
    let host_only_heap = heaps(memory)
        .iter()
        .any(|h| !h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL));
    let (topology, needs_benchmark) = if local_visible && (!local_invisible || !host_only_heap) {
        (MemoryTopology::Uma, false)
    } else if local_visible {
        (MemoryTopology::DiscreteWithBar, true)
    } else if local_invisible {
        (MemoryTopology::Discrete, false)
    } else {
        (MemoryTopology::Unknown, false)
    };
    let order = |policy| order_memory_types(memory, policy, topology);
    MemoryClassification {
        topology,
        needs_benchmark,
        type_preferences: MemoryTypePreferences {
            automatic: order(MemoryPolicy::Auto),
            device_preferred: order(MemoryPolicy::DevicePreferred),
            host_visible: order(MemoryPolicy::HostVisible),
            read_mostly: order(MemoryPolicy::ReadMostly),
            write_mostly: order(MemoryPolicy::WriteMostly),
            streaming: order(MemoryPolicy::Streaming),
            readback: order(MemoryPolicy::Readback),
        },
    }
}

/// Classifies queue families from their flags only.
pub fn classify_queue_families(families: &[vk::QueueFamilyProperties]) -> Vec<QueueFamilyInfo> {
    families
        .iter()
        .map(|family| {
            let flags = family.queue_flags;
            let graphics = flags.contains(vk::QueueFlags::GRAPHICS);
            let compute = flags.contains(vk::QueueFlags::COMPUTE);
            let transfer = flags.contains(vk::QueueFlags::TRANSFER) || graphics || compute;
            let granularity = family.min_image_transfer_granularity;
            QueueFamilyInfo {
                queue_count: family.queue_count,
                timestamp_valid_bits: family.timestamp_valid_bits,
                min_image_transfer_granularity: [granularity.width, granularity.height, granularity.depth],
                graphics,
                compute,
                transfer,
                sparse_binding: flags.contains(vk::QueueFlags::SPARSE_BINDING),
                dedicated_compute: compute && !graphics,
                dedicated_transfer: transfer && !graphics && !compute,
            }
        })
        .collect()
}

fn classify_tier(features: &DeviceFeatures, subgroup: &SubgroupInfo) -> CapabilityTier {
    if !features.timeline_semaphore || !subgroup.supported || !subgroup.basic {
        CapabilityTier::A
    } else if !features.buffer_device_address {
        CapabilityTier::B
    } else if !features.runtime_descriptor_array || !features.synchronization2 {
        CapabilityTier::C
    } else {
        CapabilityTier::D
    }
}

/// Builds the public snapshot for one physical device.
pub fn device_info(raw: &PhysicalDeviceInfo) -> DeviceInfo {
    let properties = &raw.properties;
    let limits = &properties.limits;
    let on = |b: vk::Bool32| b == vk::TRUE;
    let features = DeviceFeatures {
        robust_buffer_access: on(raw.features.robust_buffer_access),
        shader_int16: on(raw.features.shader_int16),
        shader_int64: on(raw.features.shader_int64),
        shader_float64: on(raw.features.shader_float64),
        timeline_semaphore: raw.support.timeline_semaphore,
        buffer_device_address: raw.support.buffer_device_address,
        descriptor_indexing: raw.support.descriptor_indexing,
        runtime_descriptor_array: raw.support.runtime_descriptor_array,
        synchronization2: raw.support.synchronization2,
        portability_subset: raw.has_extension(ash::khr::portability_subset::NAME),
        maintenance4: raw.support.maintenance4,
    };
    let subgroup = match raw.subgroup {
        Some((size, ops)) => SubgroupInfo {
            size,
            supported: true,
            basic: ops.contains(vk::SubgroupFeatureFlags::BASIC),
            vote: ops.contains(vk::SubgroupFeatureFlags::VOTE),
            arithmetic: ops.contains(vk::SubgroupFeatureFlags::ARITHMETIC),
            ballot: ops.contains(vk::SubgroupFeatureFlags::BALLOT),
            shuffle: ops.contains(vk::SubgroupFeatureFlags::SHUFFLE),
            shuffle_relative: ops.contains(vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE),
            clustered: ops.contains(vk::SubgroupFeatureFlags::CLUSTERED),
            quad: ops.contains(vk::SubgroupFeatureFlags::QUAD),
        },
        None => SubgroupInfo::default(),
    };
    let memory = &raw.memory_properties;
    let classification = classify_memory_properties(memory);
    let kind = match properties.device_type {
        vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceKind::Integrated,
        vk::PhysicalDeviceType::DISCRETE_GPU => DeviceKind::Discrete,
        vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceKind::Virtual,
        vk::PhysicalDeviceType::CPU => DeviceKind::Cpu,
        _ => DeviceKind::Other,
    };
    DeviceInfo {
        name: raw.name(),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        kind,
        api_version: ApiVersion::from_packed(properties.api_version),
        driver_name: raw.driver_name.clone(),
        driver_version: properties.driver_version,
        uuid: raw.device_uuid.map(|u| super::hex(&u)).unwrap_or_default(),
        limits: DeviceLimits {
            max_compute_work_group_count: limits.max_compute_work_group_count,
            max_compute_work_group_size: limits.max_compute_work_group_size,
            max_compute_work_group_invocations: limits.max_compute_work_group_invocations,
            max_compute_shared_memory_size: limits.max_compute_shared_memory_size,
            max_storage_buffer_range: u64::from(limits.max_storage_buffer_range),
            max_push_constants_size: limits.max_push_constants_size,
        },
        tier: classify_tier(&features, &subgroup),
        features,
        subgroup,
        timestamp_period_nanoseconds: limits.timestamp_period,
        queue_families: classify_queue_families(&raw.queue_families),
        memory_heaps: heaps(memory)
            .iter()
            .map(|h| MemoryHeapInfo {
                size_bytes: h.size,
                device_local: h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                multi_instance: h.flags.contains(vk::MemoryHeapFlags::MULTI_INSTANCE),
            })
            .collect(),
        memory_types: types(memory)
            .iter()
            .map(|t| {
                let f = t.property_flags;
                MemoryTypeInfo {
                    heap_index: t.heap_index,
                    device_local: f.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
                    host_visible: f.contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                    host_coherent: f.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
                    host_cached: f.contains(vk::MemoryPropertyFlags::HOST_CACHED),
                    lazily_allocated: f.contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED),
                    protected_memory: f.contains(vk::MemoryPropertyFlags::PROTECTED),
                }
            })
            .collect(),
        memory_topology: classification.topology,
        memory_topology_needs_benchmark: classification.needs_benchmark,
        memory_type_preferences: classification.type_preferences,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::vulkan::FeatureSupport;

    pub(crate) fn synthetic_device() -> PhysicalDeviceInfo {
        let mut device = PhysicalDeviceInfo::default();
        for (slot, byte) in device
            .properties
            .device_name
            .iter_mut()
            .zip(b"Synthetic Vulkan device\0")
        {
            *slot = *byte as std::ffi::c_char;
        }
        device.properties.vendor_id = 0x1234;
        device.properties.device_id = 0x5678;
        device.properties.api_version = vk::API_VERSION_1_3;
        device.api_version = vk::API_VERSION_1_3;
        let limits = &mut device.properties.limits;
        limits.max_compute_work_group_count = [65535; 3];
        limits.max_compute_work_group_size = [1024, 1024, 64];
        limits.max_compute_work_group_invocations = 1024;
        limits.max_compute_shared_memory_size = 32768;
        limits.max_storage_buffer_range = 1 << 30;
        limits.max_push_constants_size = 256;
        limits.timestamp_period = 2.5;
        device.features.robust_buffer_access = vk::TRUE;
        device.features.shader_int16 = vk::TRUE;
        device.features.shader_int64 = vk::TRUE;
        device.features.shader_float64 = vk::TRUE;
        device.subgroup = Some((
            32,
            vk::SubgroupFeatureFlags::BASIC
                | vk::SubgroupFeatureFlags::ARITHMETIC
                | vk::SubgroupFeatureFlags::BALLOT,
        ));
        device.memory_properties = memory(
            &[vk::MemoryHeapFlags::DEVICE_LOCAL],
            &[(
                0,
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            )],
        );
        device.memory_properties.memory_heaps[0].size = 8 << 30;
        device.queue_families = vec![vk::QueueFamilyProperties {
            queue_flags: vk::QueueFlags::COMPUTE,
            queue_count: 2,
            timestamp_valid_bits: 64,
            min_image_transfer_granularity: vk::Extent3D {
                width: 1,
                height: 2,
                depth: 3,
            },
        }];
        device
    }

    pub(crate) fn memory(
        heap_flags: &[vk::MemoryHeapFlags],
        types: &[(u32, vk::MemoryPropertyFlags)],
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut properties = vk::PhysicalDeviceMemoryProperties {
            memory_heap_count: heap_flags.len() as u32,
            memory_type_count: types.len() as u32,
            ..Default::default()
        };
        for (index, &flags) in heap_flags.iter().enumerate() {
            properties.memory_heaps[index] = vk::MemoryHeap {
                size: (index as u64 + 1) * 1024,
                flags,
            };
        }
        for (index, &(heap_index, property_flags)) in types.iter().enumerate() {
            properties.memory_types[index] = vk::MemoryType {
                property_flags,
                heap_index,
            };
        }
        properties
    }

    #[test]
    fn starts_at_baseline_and_does_not_require_higher_tier_features() {
        let info = device_info(&synthetic_device());
        assert_eq!(info.tier, CapabilityTier::A);
        assert!(!info.features.timeline_semaphore);
        assert!(!info.features.buffer_device_address);
    }

    #[test]
    fn assigns_additive_capability_tiers() {
        let mut raw = synthetic_device();
        raw.support.timeline_semaphore = true;
        assert_eq!(device_info(&raw).tier, CapabilityTier::B);
        raw.support.buffer_device_address = true;
        assert_eq!(device_info(&raw).tier, CapabilityTier::C);
        raw.support.runtime_descriptor_array = true;
        raw.support.synchronization2 = true;
        assert_eq!(device_info(&raw).tier, CapabilityTier::D);
        raw.subgroup = None;
        assert_eq!(device_info(&raw).tier, CapabilityTier::A);
    }

    #[test]
    fn maps_limits_features_queues_and_memory() {
        let info = device_info(&synthetic_device());
        assert_eq!(info.name, "Synthetic Vulkan device");
        assert_eq!((info.vendor_id, info.device_id), (0x1234, 0x5678));
        assert_eq!((info.api_version.major, info.api_version.minor), (1, 3));
        assert_eq!(info.limits.max_compute_work_group_count[0], 65535);
        assert_eq!(info.limits.max_compute_work_group_size[2], 64);
        assert_eq!(info.limits.max_compute_work_group_invocations, 1024);
        assert_eq!(info.limits.max_compute_shared_memory_size, 32768);
        assert_eq!(info.limits.max_storage_buffer_range, 1 << 30);
        assert_eq!(info.limits.max_push_constants_size, 256);
        assert_eq!(info.timestamp_period_nanoseconds, 2.5);
        assert!(
            info.features.robust_buffer_access
                && info.features.shader_int16
                && info.features.shader_int64
                && info.features.shader_float64
        );
        assert!(
            info.subgroup.supported
                && info.subgroup.basic
                && info.subgroup.arithmetic
                && info.subgroup.ballot
        );
        assert!(!info.subgroup.vote);
        assert_eq!(info.subgroup.size, 32);
        let queue = &info.queue_families[0];
        assert_eq!(
            (
                queue.queue_count,
                queue.timestamp_valid_bits,
                queue.min_image_transfer_granularity[2]
            ),
            (2, 64, 3)
        );
        assert!(queue.compute && queue.transfer && queue.dedicated_compute && !queue.dedicated_transfer);
        assert_eq!(info.memory_heaps[0].size_bytes, 8 << 30);
        assert!(info.memory_heaps[0].device_local);
        assert!(
            info.memory_types[0].device_local
                && info.memory_types[0].host_visible
                && info.memory_types[0].host_coherent
        );
        assert_eq!(info.memory_topology, MemoryTopology::Uma);
        assert!(!info.memory_topology_needs_benchmark);
        assert_eq!(info.memory_type_preferences.automatic, [0]);
        assert_eq!(info.memory_type_preferences.readback, [0]);
        assert!(info.uuid.is_empty());
    }

    #[test]
    fn classifies_synthetic_memory_topologies_and_type_preferences() {
        use vk::MemoryHeapFlags as H;
        use vk::MemoryPropertyFlags as P;
        let discrete = memory(
            &[H::DEVICE_LOCAL, H::empty()],
            &[(0, P::empty()), (1, P::HOST_VISIBLE | P::HOST_COHERENT)],
        );
        let c = classify_memory_properties(&discrete);
        assert_eq!((c.topology, c.needs_benchmark), (MemoryTopology::Discrete, false));
        assert_eq!(c.type_preferences.automatic, [0, 1]);
        assert_eq!(c.type_preferences.host_visible, [1, 0]);

        let uma = memory(
            &[H::DEVICE_LOCAL],
            &[(0, P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT)],
        );
        let c = classify_memory_properties(&uma);
        assert_eq!((c.topology, c.needs_benchmark), (MemoryTopology::Uma, false));
        assert_eq!(c.type_preferences.automatic, [0]);

        let bar = memory(
            &[H::DEVICE_LOCAL, H::empty()],
            &[
                (0, P::DEVICE_LOCAL),
                (0, P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT),
                (1, P::HOST_VISIBLE | P::HOST_CACHED),
            ],
        );
        let c = classify_memory_properties(&bar);
        assert_eq!(
            (c.topology, c.needs_benchmark),
            (MemoryTopology::DiscreteWithBar, true)
        );
        let p = &c.type_preferences;
        assert_eq!(p.automatic, [0, 1, 2]);
        assert_eq!(p.device_preferred, [0, 1, 2]);
        assert_eq!(p.host_visible, [1, 2, 0]);
        assert_eq!(p.read_mostly, [1, 0, 2]);
        assert_eq!(p.write_mostly, [1, 0, 2]);
        assert_eq!(p.streaming, [1, 2, 0]);
        assert_eq!(p.readback, [2, 1, 0]);

        let single_heap = memory(
            &[H::DEVICE_LOCAL],
            &[
                (0, P::DEVICE_LOCAL),
                (
                    0,
                    P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT | P::HOST_CACHED,
                ),
                (0, P::DEVICE_LOCAL | P::LAZILY_ALLOCATED),
            ],
        );
        let c = classify_memory_properties(&single_heap);
        assert_eq!((c.topology, c.needs_benchmark), (MemoryTopology::Uma, false));
        assert_eq!(c.type_preferences.automatic[0], 1);

        let unknown = memory(&[H::empty()], &[(0, P::HOST_VISIBLE | P::HOST_CACHED)]);
        let c = classify_memory_properties(&unknown);
        assert_eq!((c.topology, c.needs_benchmark), (MemoryTopology::Unknown, false));
        assert_eq!(c.type_preferences.automatic, [0]);
    }

    #[test]
    fn reports_individual_descriptor_indexing_support() {
        let mut raw = synthetic_device();
        raw.support = FeatureSupport {
            descriptor_indexing: true,
            ..Default::default()
        };
        let info = device_info(&raw);
        assert!(info.features.descriptor_indexing);
        assert!(!info.features.runtime_descriptor_array);
        assert_eq!(info.tier, CapabilityTier::A);
    }

    #[test]
    fn classifies_synthetic_queue_families() {
        use vk::QueueFlags as Q;
        let family = |flags, count, bits, (w, h, d)| vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: count,
            timestamp_valid_bits: bits,
            min_image_transfer_granularity: vk::Extent3D {
                width: w,
                height: h,
                depth: d,
            },
        };
        type Expect = (u32, u32, [u32; 3], bool, bool, bool, bool, bool, bool);
        let layouts: [(&[vk::QueueFamilyProperties], &[Expect]); 3] = [
            (
                &[
                    family(Q::COMPUTE | Q::SPARSE_BINDING, 2, 64, (4, 2, 1)),
                    family(Q::TRANSFER, 1, 32, (8, 4, 1)),
                ],
                &[
                    (2, 64, [4, 2, 1], false, true, true, true, true, false),
                    (1, 32, [8, 4, 1], false, false, true, false, false, true),
                ],
            ),
            (
                &[
                    family(Q::GRAPHICS | Q::COMPUTE, 4, 0, (1, 1, 1)),
                    family(Q::COMPUTE, 2, 32, (2, 2, 1)),
                ],
                &[
                    (4, 0, [1, 1, 1], true, true, true, false, false, false),
                    (2, 32, [2, 2, 1], false, true, true, false, true, false),
                ],
            ),
            (
                &[family(Q::GRAPHICS | Q::COMPUTE, 1, 64, (0, 0, 0))],
                &[(1, 64, [0, 0, 0], true, true, true, false, false, false)],
            ),
        ];
        for (families, expected) in layouts {
            let classified = classify_queue_families(families);
            let actual: Vec<Expect> = classified
                .iter()
                .map(|q| {
                    (
                        q.queue_count,
                        q.timestamp_valid_bits,
                        q.min_image_transfer_granularity,
                        q.graphics,
                        q.compute,
                        q.transfer,
                        q.sparse_binding,
                        q.dedicated_compute,
                        q.dedicated_transfer,
                    )
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn device_type_maps_to_kind_and_uuid_is_hex() {
        let mut raw = synthetic_device();
        for (ty, kind) in [
            (vk::PhysicalDeviceType::INTEGRATED_GPU, DeviceKind::Integrated),
            (vk::PhysicalDeviceType::DISCRETE_GPU, DeviceKind::Discrete),
            (vk::PhysicalDeviceType::VIRTUAL_GPU, DeviceKind::Virtual),
            (vk::PhysicalDeviceType::CPU, DeviceKind::Cpu),
            (vk::PhysicalDeviceType::OTHER, DeviceKind::Other),
        ] {
            raw.properties.device_type = ty;
            assert_eq!(device_info(&raw).kind, kind);
        }
        raw.device_uuid = Some([0xab; 16]);
        assert_eq!(device_info(&raw).uuid, "ab".repeat(16));
    }
}
