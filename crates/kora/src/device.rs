//! Vendor-neutral device facts and device selection.
//!
//! [`DeviceInfo`] is an immutable snapshot of what a Vulkan physical device reports. Every runtime
//! decision derives from these queried capabilities (and measured behavior), never from vendor
//! identity.

use crate::error::{Error, ErrorCode, Result, Status};
use crate::ids::DeviceId;
use crate::options::{DevicePolicy, ResolvedOptions};

/// Additive convenience tiers. Each tier implies the previous ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum CapabilityTier {
    /// Baseline Vulkan compute.
    #[default]
    A,
    /// Plus timeline semaphores and basic subgroup operations.
    B,
    /// Plus buffer device addresses.
    C,
    /// Plus runtime descriptor arrays and synchronization2.
    D,
}

/// Physical memory organization, classified from memory-type and heap flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryTopology {
    /// Separate device memory and host-visible system memory.
    Discrete,
    /// Host and device share one physical pool.
    Uma,
    /// Discrete memory plus a host-visible device-local window (needs measurement).
    DiscreteWithBar,
    /// Could not be classified.
    #[default]
    Unknown,
}

impl MemoryTopology {
    /// Stable lower-case name used in reports.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Discrete => "discrete",
            Self::Uma => "uma",
            Self::DiscreteWithBar => "discrete_with_bar",
            Self::Unknown => "unknown",
        }
    }
}

/// Kind of physical device as reported by the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeviceKind {
    /// Unclassified.
    #[default]
    Other,
    /// Integrated GPU.
    Integrated,
    /// Discrete GPU.
    Discrete,
    /// Virtual GPU.
    Virtual,
    /// CPU (software rasterizer or compute).
    Cpu,
}

impl DeviceKind {
    /// Stable lower-case name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Other => "other",
            Self::Integrated => "integrated",
            Self::Discrete => "discrete",
            Self::Virtual => "virtual",
            Self::Cpu => "cpu",
        }
    }
}

/// Vulkan API version components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ApiVersion {
    /// Major.
    pub major: u32,
    /// Minor.
    pub minor: u32,
    /// Patch.
    pub patch: u32,
}

impl ApiVersion {
    /// Decodes a packed Vulkan version.
    pub const fn from_packed(version: u32) -> Self {
        Self {
            major: (version >> 22) & 0x7f,
            minor: (version >> 12) & 0x3ff,
            patch: version & 0xfff,
        }
    }
}

impl std::fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Compute-relevant device limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceLimits {
    /// Maximum workgroup counts per dimension.
    pub max_compute_work_group_count: [u32; 3],
    /// Maximum workgroup size per dimension.
    pub max_compute_work_group_size: [u32; 3],
    /// Maximum invocations per workgroup.
    pub max_compute_work_group_invocations: u32,
    /// Maximum shared memory per workgroup.
    pub max_compute_shared_memory_size: u32,
    /// Maximum storage-buffer range.
    pub max_storage_buffer_range: u64,
    /// Maximum push-constant block size.
    pub max_push_constants_size: u32,
}

/// Optional features KORA may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceFeatures {
    /// Robust buffer access.
    pub robust_buffer_access: bool,
    /// 16-bit integers in shaders.
    pub shader_int16: bool,
    /// 64-bit integers in shaders.
    pub shader_int64: bool,
    /// 64-bit floats in shaders.
    pub shader_float64: bool,
    /// Timeline semaphores.
    pub timeline_semaphore: bool,
    /// Buffer device addresses.
    pub buffer_device_address: bool,
    /// Any descriptor-indexing capability.
    pub descriptor_indexing: bool,
    /// Runtime-sized descriptor arrays.
    pub runtime_descriptor_array: bool,
    /// Synchronization2.
    pub synchronization2: bool,
    /// The device is a portability-subset implementation (for example a translation layer).
    pub portability_subset: bool,
}

/// Subgroup support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SubgroupInfo {
    /// Subgroup size (0 when unsupported).
    pub size: u32,
    /// Subgroup properties were reported.
    pub supported: bool,
    /// Basic operations.
    pub basic: bool,
    /// Vote operations.
    pub vote: bool,
    /// Arithmetic operations.
    pub arithmetic: bool,
    /// Ballot operations.
    pub ballot: bool,
    /// Shuffle operations.
    pub shuffle: bool,
    /// Relative shuffle operations.
    pub shuffle_relative: bool,
    /// Clustered operations.
    pub clustered: bool,
    /// Quad operations.
    pub quad: bool,
}

/// One queue family, classified from its flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QueueFamilyInfo {
    /// Queues in the family.
    pub queue_count: u32,
    /// Valid timestamp bits (0 = no timestamps).
    pub timestamp_valid_bits: u32,
    /// Image transfer granularity as reported (not used for buffers).
    pub min_image_transfer_granularity: [u32; 3],
    /// Graphics capable.
    pub graphics: bool,
    /// Compute capable.
    pub compute: bool,
    /// Transfer capable (explicitly, or implied by graphics/compute).
    pub transfer: bool,
    /// Sparse binding.
    pub sparse_binding: bool,
    /// Compute without graphics.
    pub dedicated_compute: bool,
    /// Transfer without graphics or compute.
    pub dedicated_transfer: bool,
}

/// One memory heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemoryHeapInfo {
    /// Size in bytes.
    pub size_bytes: u64,
    /// Device-local heap.
    pub device_local: bool,
    /// Multi-instance heap.
    pub multi_instance: bool,
}

/// One memory type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemoryTypeInfo {
    /// Owning heap.
    pub heap_index: u32,
    /// Device local.
    pub device_local: bool,
    /// Host visible (mappable).
    pub host_visible: bool,
    /// Host coherent.
    pub host_coherent: bool,
    /// Host cached.
    pub host_cached: bool,
    /// Lazily allocated.
    pub lazily_allocated: bool,
    /// Protected.
    pub protected_memory: bool,
}

/// Memory-type rankings per access pattern. These are preferences, not allocation decisions:
/// callers still filter by each resource's `memoryTypeBits`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MemoryTypePreferences {
    /// Topology-aware default.
    pub automatic: Vec<u32>,
    /// Device-only data.
    pub device_preferred: Vec<u32>,
    /// Must be mappable.
    pub host_visible: Vec<u32>,
    /// Mostly read by the host.
    pub read_mostly: Vec<u32>,
    /// Mostly written by the host.
    pub write_mostly: Vec<u32>,
    /// Streamed every use.
    pub streaming: Vec<u32>,
    /// Read back to the host.
    pub readback: Vec<u32>,
}

/// Immutable, vendor-neutral snapshot of a physical device.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeviceInfo {
    /// Device name.
    pub name: String,
    /// PCI vendor ID (informational only; never used for decisions).
    pub vendor_id: u32,
    /// Vendor device ID (informational only).
    pub device_id: u32,
    /// Device kind.
    pub kind: DeviceKind,
    /// Supported Vulkan API version.
    pub api_version: ApiVersion,
    /// Driver name, or empty when unreported.
    pub driver_name: String,
    /// Driver-defined version number.
    pub driver_version: u32,
    /// Physical-device UUID (lower-case hex), or empty when unreported.
    pub uuid: String,
    /// Compute limits.
    pub limits: DeviceLimits,
    /// Optional features.
    pub features: DeviceFeatures,
    /// Subgroup support.
    pub subgroup: SubgroupInfo,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period_nanoseconds: f32,
    /// Queue families.
    pub queue_families: Vec<QueueFamilyInfo>,
    /// Memory heaps.
    pub memory_heaps: Vec<MemoryHeapInfo>,
    /// Memory types.
    pub memory_types: Vec<MemoryTypeInfo>,
    /// Memory topology.
    pub memory_topology: MemoryTopology,
    /// The topology is ambiguous enough that placement should be measured.
    pub memory_topology_needs_benchmark: bool,
    /// Memory-type rankings.
    pub memory_type_preferences: MemoryTypePreferences,
    /// Convenience tier.
    pub tier: CapabilityTier,
}

/// Selection-only facts about one device.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeviceCandidate {
    /// Identity.
    pub id: DeviceId,
    /// Device kind.
    pub kind: DeviceKind,
    /// Capability tier.
    pub tier: CapabilityTier,
    /// Device-local memory.
    pub device_local_memory_bytes: u64,
    /// Compute-capable queues.
    pub compute_queue_count: u32,
    /// Caller workload requirements are met.
    pub satisfies_requirements: bool,
}

fn eligible(device: &DeviceCandidate) -> bool {
    device.satisfies_requirements && device.compute_queue_count != 0
}

fn capability_score(device: &DeviceCandidate) -> u64 {
    let tier = device.tier as u64;
    let memory_gib = (device.device_local_memory_bytes >> 30).min(4095);
    let queues = u64::from(device.compute_queue_count.min(255));
    (tier << 48) | (memory_gib << 16) | queues
}

fn kind_preference(policy: DevicePolicy, kind: DeviceKind) -> u8 {
    match (policy, kind) {
        (DevicePolicy::PreferDiscrete, DeviceKind::Discrete)
        | (DevicePolicy::PreferIntegrated, DeviceKind::Integrated) => 1,
        _ => 0,
    }
}

fn valid_uuid(uuid: &str) -> bool {
    uuid.len() == 32 && uuid.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Selects eligible devices in deterministic preference order: one device for every policy except
/// [`DevicePolicy::AllEligible`]. Scoring uses capability tier, local memory, and compute queues; it
/// never prefers a vendor.
pub fn select_devices(options: &ResolvedOptions, devices: &[DeviceCandidate]) -> Result<Vec<DeviceId>> {
    let exact = |found: Option<&DeviceCandidate>, what: &str| -> Result<Vec<DeviceId>> {
        let device = found.ok_or_else(|| {
            Error::new(
                ErrorCode::NoEligibleDevice,
                format!("Requested device {what} was not found"),
            )
        })?;
        if !eligible(device) {
            return Err(Error::new(
                ErrorCode::NoEligibleDevice,
                format!("Requested device {what} is not eligible"),
            ));
        }
        Ok(vec![device.id.clone()])
    };
    match options.device_policy {
        DevicePolicy::ByIndex => {
            return exact(
                devices.iter().find(|d| d.id.index == options.device_index),
                "index",
            );
        }
        DevicePolicy::ByUuid => {
            if !valid_uuid(&options.device_uuid) {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    "Device UUID must contain exactly 32 hexadecimal characters",
                ));
            }
            let found = devices
                .iter()
                .find(|d| valid_uuid(&d.id.uuid) && d.id.uuid.eq_ignore_ascii_case(&options.device_uuid));
            return exact(found, "UUID");
        }
        _ => {}
    }
    let mut ranked: Vec<&DeviceCandidate> = devices.iter().filter(|d| eligible(d)).collect();
    if ranked.is_empty() {
        return Err(Error::new(
            ErrorCode::NoEligibleDevice,
            "No eligible compute device was found",
        ));
    }
    ranked.sort_by(|l, r| {
        let key = |d: &DeviceCandidate| {
            (
                kind_preference(options.device_policy, d.kind),
                capability_score(d),
            )
        };
        key(r)
            .cmp(&key(l))
            .then(l.id.index.cmp(&r.id.index))
            .then_with(|| {
                l.id.uuid
                    .to_ascii_lowercase()
                    .cmp(&r.id.uuid.to_ascii_lowercase())
            })
    });
    let count = if options.device_policy == DevicePolicy::AllEligible {
        ranked.len()
    } else {
        1
    };
    Ok(ranked.into_iter().take(count).map(|d| d.id.clone()).collect())
}

/// Initializes selected devices. With `isolate_failures` (the `AllEligible` policy), failing
/// devices are quarantined and reported while healthy ones continue; otherwise the first failure
/// is returned immediately. Errors carry the device identity.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by Runtime initialization (#258)")
)]
pub(crate) fn initialize_selected_devices(
    selected: &[DeviceId],
    isolate_failures: bool,
    mut initialize: impl FnMut(&DeviceId) -> Status,
    mut quarantine: impl FnMut(&DeviceId),
    mut report_failure: impl FnMut(&DeviceId, &Error),
) -> Status {
    let mut first_error: Option<Error> = None;
    let mut initialized = 0usize;
    for id in selected {
        match initialize(id) {
            Ok(()) => initialized += 1,
            Err(error) => {
                let identity = if id.uuid.is_empty() {
                    format!("index {}", id.index)
                } else {
                    format!("index {} uuid {}", id.index, id.uuid)
                };
                let error = error.with_device(identity);
                if !isolate_failures {
                    return Err(error);
                }
                quarantine(id);
                report_failure(id, &error);
                first_error.get_or_insert(error);
            }
        }
    }
    if initialized != 0 {
        return Ok(());
    }
    Err(first_error.unwrap_or_else(|| {
        Error::new(
            ErrorCode::NoEligibleDevice,
            "No selected device could be initialized",
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        index: u32,
        kind: DeviceKind,
        tier: CapabilityTier,
        memory_gib: u64,
        queues: u32,
        ok: bool,
    ) -> DeviceCandidate {
        DeviceCandidate {
            id: DeviceId::new(index, char::from(b'0' + index as u8).to_string().repeat(32)),
            kind,
            tier,
            device_local_memory_bytes: memory_gib << 30,
            compute_queue_count: queues,
            satisfies_requirements: ok,
        }
    }

    fn options(policy: DevicePolicy) -> ResolvedOptions {
        ResolvedOptions {
            device_policy: policy,
            ..Default::default()
        }
    }

    #[test]
    fn auto_can_select_a_device_after_the_first_enumeration_entry() {
        let devices = [
            candidate(0, DeviceKind::Discrete, CapabilityTier::A, 8, 1, true),
            candidate(1, DeviceKind::Integrated, CapabilityTier::C, 4, 1, true),
        ];
        assert_eq!(
            select_devices(&options(DevicePolicy::Auto), &devices).unwrap()[0].index,
            1
        );
    }

    #[test]
    fn explicit_preferences_win_and_fall_back() {
        let devices = [
            candidate(0, DeviceKind::Integrated, CapabilityTier::D, 32, 1, true),
            candidate(1, DeviceKind::Discrete, CapabilityTier::A, 2, 1, true),
        ];
        assert_eq!(
            select_devices(&options(DevicePolicy::PreferDiscrete), &devices).unwrap()[0].index,
            1
        );
        assert_eq!(
            select_devices(&options(DevicePolicy::PreferIntegrated), &devices).unwrap()[0].index,
            0
        );
        let only_cpu = [candidate(2, DeviceKind::Cpu, CapabilityTier::B, 1, 1, true)];
        assert_eq!(
            select_devices(&options(DevicePolicy::PreferDiscrete), &only_cpu).unwrap()[0].index,
            2
        );
    }

    #[test]
    fn index_and_uuid_are_exact_and_report_errors() {
        let devices = [
            candidate(0, DeviceKind::Integrated, CapabilityTier::A, 1, 1, true),
            candidate(1, DeviceKind::Discrete, CapabilityTier::A, 1, 0, true),
        ];
        let mut by_index = options(DevicePolicy::ByIndex);
        assert_eq!(select_devices(&by_index, &devices).unwrap()[0].index, 0);
        by_index.device_index = 7;
        assert_eq!(
            select_devices(&by_index, &devices).unwrap_err().code(),
            ErrorCode::NoEligibleDevice
        );
        by_index.device_index = 1;
        assert_eq!(
            select_devices(&by_index, &devices).unwrap_err().code(),
            ErrorCode::NoEligibleDevice
        );
        let mut by_uuid = options(DevicePolicy::ByUuid);
        by_uuid.device_uuid = "0".repeat(32);
        assert_eq!(select_devices(&by_uuid, &devices).unwrap()[0].index, 0);
        by_uuid.device_uuid = "bad".into();
        assert_eq!(
            select_devices(&by_uuid, &devices).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn all_eligible_is_ranked_and_ties_use_stable_index() {
        let devices = [
            candidate(4, DeviceKind::Discrete, CapabilityTier::B, 4, 1, true),
            candidate(2, DeviceKind::Integrated, CapabilityTier::B, 4, 1, true),
            candidate(1, DeviceKind::Integrated, CapabilityTier::D, 8, 1, false),
        ];
        let selected = select_devices(&options(DevicePolicy::AllEligible), &devices).unwrap();
        assert_eq!(selected.iter().map(|d| d.index).collect::<Vec<_>>(), [2, 4]);
    }

    #[test]
    fn empty_or_ineligible_set_returns_no_eligible_device() {
        assert_eq!(
            select_devices(&options(DevicePolicy::Auto), &[])
                .unwrap_err()
                .code(),
            ErrorCode::NoEligibleDevice
        );
    }

    #[test]
    fn all_eligible_isolates_failures_and_keeps_healthy_devices() {
        let selected = [
            DeviceId::new(0, "device-zero"),
            DeviceId::new(1, "device-one"),
            DeviceId::new(2, "device-two"),
        ];
        let (mut initialized, mut quarantined, mut failures) = (Vec::new(), Vec::new(), Vec::new());
        initialize_selected_devices(
            &selected,
            true,
            |id| {
                if id.index == 1 {
                    return Err(Error::new(ErrorCode::UnsupportedFeature, "injected"));
                }
                initialized.push(id.clone());
                Ok(())
            },
            |id| quarantined.push(id.clone()),
            |_, error| failures.push(error.clone()),
        )
        .unwrap();
        assert_eq!(initialized, [selected[0].clone(), selected[2].clone()]);
        assert_eq!(quarantined, [selected[1].clone()]);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].code(), ErrorCode::UnsupportedFeature);
        assert_eq!(failures[0].device(), "index 1 uuid device-one");
    }

    #[test]
    fn explicit_selection_fails_fast_without_quarantining() {
        let selected = [DeviceId::new(4, "device-four"), DeviceId::new(5, "device-five")];
        let (mut attempted, mut quarantined) = (Vec::new(), Vec::new());
        let error = initialize_selected_devices(
            &selected,
            false,
            |id| {
                attempted.push(id.clone());
                Err(Error::new(ErrorCode::AllocationFailed, "injected"))
            },
            |id| quarantined.push(id.clone()),
            |_, _| {},
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::AllocationFailed);
        assert_eq!(error.device(), "index 4 uuid device-four");
        assert_eq!(attempted, [selected[0].clone()]);
        assert!(quarantined.is_empty());
    }

    #[test]
    fn all_failures_return_first_error_with_identity() {
        let selected = [DeviceId::new(2, "device-two"), DeviceId::new(3, "device-three")];
        let mut quarantined = Vec::new();
        let error = initialize_selected_devices(
            &selected,
            true,
            |id| {
                Err(Error::new(
                    if id.index == 2 {
                        ErrorCode::UnsupportedFeature
                    } else {
                        ErrorCode::AllocationFailed
                    },
                    "x",
                ))
            },
            |id| quarantined.push(id.clone()),
            |_, _| {},
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::UnsupportedFeature);
        assert_eq!(error.device(), "index 2 uuid device-two");
        assert_eq!(quarantined.len(), 2);
        assert_eq!(
            initialize_selected_devices(&[], true, |_| Ok(()), |_| {}, |_, _| {})
                .unwrap_err()
                .code(),
            ErrorCode::NoEligibleDevice
        );
    }

    #[test]
    fn api_version_decodes_packed_values() {
        let packed = (1 << 22) | (3 << 12) | 281;
        assert_eq!(ApiVersion::from_packed(packed).to_string(), "1.3.281");
    }
}
