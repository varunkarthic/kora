//! Shared helpers for Vulkan-dependent integration tests.

#![allow(dead_code)]

use kora::ErrorCode;
use kora::vulkan::{Instance, InstanceOptions, Loader};

/// Returns a loader and instance, or `None` (after printing why) when no Vulkan implementation is
/// usable, so tests skip instead of failing on machines without a GPU driver.
pub fn vulkan(options: &InstanceOptions) -> Option<(Loader, Instance)> {
    let loader = match Loader::load() {
        Ok(loader) => loader,
        Err(error) => {
            eprintln!("skipping: {error}");
            return None;
        }
    };
    match loader.create_instance(options) {
        Ok(instance) => Some((loader, instance)),
        Err(error) if error.code() == ErrorCode::NoVulkanImplementation => {
            eprintln!("skipping: {error}");
            None
        }
        Err(error) => panic!("instance creation failed: {error}"),
    }
}
