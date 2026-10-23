//! Low-level Vulkan layer built on [`ash`].
//!
//! The Vulkan loader is resolved at run time, so building KORA needs no Vulkan SDK and a missing
//! loader is the recoverable [`ErrorCode::NoVulkanImplementation`](crate::ErrorCode). Every `unsafe`
//! Vulkan call lives in this module behind a documented safety boundary; the higher layers use the
//! safe wrappers here.
//!
//! Most applications use [`Runtime`](crate) instead. This module is public for tools (such as
//! `kora-info`) and interop with other Vulkan code.

mod caps;
mod device;
mod fault;
mod loader;
mod physical;

pub use ash;
pub use caps::{MemoryClassification, classify_memory_properties, classify_queue_families, device_info};
pub use device::{Device, DeviceOptions, EnabledFeatures};
pub use fault::{FaultFunction, FaultInjector};
pub use loader::{Instance, InstanceOptions, Loader};
pub use physical::{FeatureSupport, MemoryBudget, PhysicalDeviceInfo};

use ash::vk;

use crate::error::{Error, ErrorCode};

/// Builds an error carrying a `VkResult`.
pub(crate) fn vk_error(code: ErrorCode, message: impl Into<String>, result: vk::Result) -> Error {
    Error::new(code, message).with_vk(result.as_raw())
}

/// Packs `major.minor.patch` the way Vulkan does.
pub const fn make_api_version(major: u32, minor: u32, patch: u32) -> u32 {
    vk::make_api_version(0, major, minor, patch)
}

/// Formats a packed Vulkan version as `major.minor.patch`.
pub fn version_string(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}

/// Converts a fixed-size, NUL-terminated Vulkan string to UTF-8 (lossily).
pub(crate) fn fixed_c_string(bytes: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = bytes.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Lower-case hex of a UUID.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_format_versions_strings_and_uuids() {
        assert_eq!(version_string(make_api_version(1, 3, 281)), "1.3.281");
        let name = [
            b'a' as std::ffi::c_char,
            b'b' as std::ffi::c_char,
            0,
            b'c' as std::ffi::c_char,
        ];
        assert_eq!(fixed_c_string(&name), "ab");
        assert_eq!(hex(&[0x00, 0xab, 0x10]), "00ab10");
        let error = vk_error(ErrorCode::DeviceLost, "lost", vk::Result::ERROR_DEVICE_LOST);
        assert_eq!(error.vk_result(), vk::Result::ERROR_DEVICE_LOST.as_raw());
    }
}
