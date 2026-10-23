//! Deterministic fault injection for the Vulkan wrappers.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use ash::vk;

use crate::error::{Error, ErrorCode, Status};

/// Wrapped Vulkan entry points that can be made to fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaultFunction {
    /// `vkAllocateMemory`.
    AllocateMemory,
    /// `vkQueueSubmit` / `vkQueueSubmit2`.
    QueueSubmit,
    /// `vkGetQueryPoolResults`.
    GetQueryPoolResults,
    /// `vkCreateComputePipelines`.
    CreateComputePipelines,
    /// `vkAllocateDescriptorSets`.
    AllocateDescriptorSets,
}

const FUNCTIONS: usize = 5;

/// Fails the Nth call of a function with a chosen `VkResult` without forwarding it to the driver;
/// every other call is forwarded unchanged. Unarmed injectors cost one relaxed atomic load.
#[derive(Debug, Default)]
pub struct FaultInjector {
    armed: AtomicBool,
    calls: [AtomicU64; FUNCTIONS],
    rules: Mutex<[Option<(u64, u64, vk::Result)>; FUNCTIONS]>,
}

impl FaultInjector {
    /// Arms a failure of call number `call` (1-based, counted from now on) with `result`, which must
    /// be a Vulkan error code.
    pub fn fail_on_nth(&self, function: FaultFunction, call: u64, result: vk::Result) -> Status {
        self.fail_calls(function, call, 1, result)
    }

    /// Fails `count` consecutive calls starting at call number `first` (1-based, from now on).
    pub fn fail_calls(&self, function: FaultFunction, first: u64, count: u64, result: vk::Result) -> Status {
        if first == 0 || count == 0 || result.as_raw() >= 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "fault injection needs a 1-based call and a Vulkan error",
            ));
        }
        let index = function as usize;
        let mut rules = self.rules.lock().unwrap_or_else(|p| p.into_inner());
        self.calls[index].store(0, Ordering::SeqCst);
        rules[index] = Some((first, count, result));
        self.armed.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Removes every armed rule.
    pub fn clear(&self) {
        *self.rules.lock().unwrap_or_else(|p| p.into_inner()) = [None; FUNCTIONS];
        self.armed.store(false, Ordering::SeqCst);
    }

    /// Records a call. Returns the injected error when this call must fail.
    pub fn check(&self, function: FaultFunction) -> Option<vk::Result> {
        if !self.armed.load(Ordering::Relaxed) {
            return None;
        }
        let index = function as usize;
        let rules = self.rules.lock().unwrap_or_else(|p| p.into_inner());
        let (first, count, result) = rules[index]?;
        let call = self.calls[index].fetch_add(1, Ordering::SeqCst) + 1;
        (call >= first && call - first < count).then_some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fails_nth_call_and_then_forwards() {
        let injector = FaultInjector::default();
        assert_eq!(injector.check(FaultFunction::AllocateMemory), None);
        injector
            .fail_on_nth(
                FaultFunction::AllocateMemory,
                2,
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            )
            .unwrap();
        assert_eq!(injector.check(FaultFunction::AllocateMemory), None);
        assert_eq!(
            injector.check(FaultFunction::AllocateMemory),
            Some(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
        );
        assert_eq!(injector.check(FaultFunction::AllocateMemory), None);
        assert_eq!(injector.check(FaultFunction::QueueSubmit), None);
    }

    #[test]
    fn rules_are_per_function_and_clearable() {
        let injector = FaultInjector::default();
        injector
            .fail_on_nth(
                FaultFunction::QueueSubmit,
                1,
                vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            )
            .unwrap();
        assert_eq!(injector.check(FaultFunction::AllocateMemory), None);
        assert_eq!(
            injector.check(FaultFunction::QueueSubmit),
            Some(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        );
        injector
            .fail_on_nth(
                FaultFunction::GetQueryPoolResults,
                1,
                vk::Result::ERROR_DEVICE_LOST,
            )
            .unwrap();
        injector.clear();
        assert_eq!(injector.check(FaultFunction::GetQueryPoolResults), None);
    }

    #[test]
    fn consecutive_failures_cover_a_window() {
        let injector = FaultInjector::default();
        injector
            .fail_calls(
                FaultFunction::AllocateMemory,
                2,
                2,
                vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            )
            .unwrap();
        let results: Vec<_> = (0..5)
            .map(|_| injector.check(FaultFunction::AllocateMemory))
            .collect();
        let failed = Some(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
        assert_eq!(results, [None, failed, failed, None, None]);
        assert!(
            injector
                .fail_calls(FaultFunction::AllocateMemory, 1, 0, vk::Result::ERROR_DEVICE_LOST)
                .is_err()
        );
    }

    #[test]
    fn rejects_invalid_rules() {
        let injector = FaultInjector::default();
        assert!(
            injector
                .fail_on_nth(FaultFunction::QueueSubmit, 0, vk::Result::ERROR_DEVICE_LOST)
                .is_err()
        );
        assert!(
            injector
                .fail_on_nth(FaultFunction::QueueSubmit, 1, vk::Result::SUCCESS)
                .is_err()
        );
        assert!(
            injector
                .fail_on_nth(FaultFunction::QueueSubmit, 1, vk::Result::INCOMPLETE)
                .is_err()
        );
    }
}
