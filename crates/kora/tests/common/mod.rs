//! Shared helpers for Vulkan-dependent integration tests.

#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use kora::ErrorCode;
use kora::log::{self, Category, Level};
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

/// Captures validation errors routed to the KORA log for the duration of a test.
pub struct ValidationErrors {
    messages: Arc<Mutex<Vec<String>>>,
}

static LOG_LOCK: Mutex<()> = Mutex::new(());

impl ValidationErrors {
    pub fn capture() -> (Self, std::sync::MutexGuard<'static, ()>) {
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

    pub fn assert_none(&self) {
        let messages = self.messages.lock().unwrap();
        assert!(messages.is_empty(), "validation errors: {messages:#?}");
    }
}

impl Drop for ValidationErrors {
    fn drop(&mut self) {
        log::set_sink(None);
    }
}
