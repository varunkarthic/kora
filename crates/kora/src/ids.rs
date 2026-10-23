//! Identifiers shared across subsystems.

use std::fmt;
use std::sync::Arc;

/// Identity of a logical buffer: a slot plus a generation that changes whenever the slot is
/// reused, so stale handles are detected instead of aliasing new allocations. Generation `0` is
/// never issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct BufferId {
    /// Registry slot.
    pub slot: u32,
    /// Slot generation (non-zero for live buffers).
    pub generation: u64,
}

impl BufferId {
    /// Creates an identifier.
    pub const fn new(slot: u32, generation: u64) -> Self {
        Self { slot, generation }
    }
}

/// Stable identity of a device selected by a runtime: its Vulkan enumeration index plus its
/// physical-device UUID (lower-case hex). Cloning is cheap.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct DeviceId {
    /// Vulkan enumeration index.
    pub index: u32,
    /// Physical-device UUID as lower-case hex (or an opaque test label).
    pub uuid: Arc<str>,
}

impl DeviceId {
    /// Creates an identifier.
    pub fn new(index: u32, uuid: impl Into<Arc<str>>) -> Self {
        Self {
            index,
            uuid: uuid.into(),
        }
    }
}

impl fmt::Debug for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DeviceId({}, {})", self.index, self.uuid)
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "device {} ({})", self.index, self.uuid)
    }
}
