//! With no installable client driver, instance creation reports `NoVulkanImplementation`.
//! This file holds a single test because it changes process-wide environment variables.

use kora::ErrorCode;
use kora::vulkan::{InstanceOptions, Loader};

#[test]
fn missing_icd_returns_no_vulkan_implementation() {
    // SAFETY: this test binary runs exactly one test, so no other thread reads the environment
    // while it is modified.
    unsafe {
        std::env::set_var("VK_ICD_FILENAMES", "kora-no-such-vulkan-icd.json");
        std::env::set_var("VK_DRIVER_FILES", "kora-no-such-vulkan-icd.json");
    }
    let loader = match Loader::load() {
        Ok(loader) => loader,
        Err(error) => {
            assert_eq!(error.code(), ErrorCode::NoVulkanImplementation);
            return;
        }
    };
    let error = loader.create_instance(&InstanceOptions::default()).unwrap_err();
    assert_eq!(error.code(), ErrorCode::NoVulkanImplementation);
}
