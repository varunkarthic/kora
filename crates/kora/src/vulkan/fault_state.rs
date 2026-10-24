//! Device-loss state shared by every queue of one logical device.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ash::vk;

/// One `VK_EXT_device_fault` address record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceFaultAddressRecord {
    /// Address type.
    pub address_type: u32,
    /// Reported address (returned to the caller only; never logged or persisted).
    pub reported_address: u64,
    /// Precision.
    pub precision: u64,
}

/// One vendor-specific fault record.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceFaultVendorRecord {
    /// Description.
    pub description: String,
    /// Vendor fault code.
    pub code: u64,
    /// Vendor fault data.
    pub data: u64,
}

/// Fault details captured after device loss.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceFaultSnapshot {
    /// The device supports fault queries.
    pub supported: bool,
    /// A query was made.
    pub queried: bool,
    /// Records were truncated to the bound.
    pub truncated: bool,
    /// `VkResult` of the query.
    pub query_result: i32,
    /// Description.
    pub description: String,
    /// Address records.
    pub addresses: Vec<DeviceFaultAddressRecord>,
    /// Vendor records.
    pub vendor_records: Vec<DeviceFaultVendorRecord>,
}

/// Fault query installed by the runtime.
pub type DiagnosticsQuery = Arc<dyn Fn() -> DeviceFaultSnapshot + Send + Sync>;

#[derive(Default)]
struct Inner {
    query: Option<DiagnosticsQuery>,
    diagnostics: DeviceFaultSnapshot,
}

/// Shared by all queues of a logical device: the first `VK_ERROR_DEVICE_LOST` is retained,
/// later submissions are refused, and optional diagnostics are captured exactly once.
#[derive(Default)]
pub struct DeviceFaultState {
    lost: AtomicBool,
    last_result: AtomicI32,
    inner: Mutex<Inner>,
    submission: Mutex<()>,
}

impl std::fmt::Debug for DeviceFaultState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceFaultState")
            .field("healthy", &self.healthy())
            .field("last_result", &self.last_result())
            .finish()
    }
}

impl DeviceFaultState {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `true` until device loss is observed.
    pub fn healthy(&self) -> bool {
        !self.lost.load(Ordering::Acquire)
    }

    /// Installs the diagnostics query used on first loss.
    pub fn set_diagnostics_query(&self, query: Option<DiagnosticsQuery>) {
        let mut inner = self.lock();
        inner.diagnostics.supported = query.is_some();
        inner.query = query;
    }

    /// Records device loss once and captures diagnostics outside the lock.
    pub fn mark_lost(&self, result: vk::Result, capture_diagnostics: bool) {
        let query = {
            let inner = self.lock();
            if self.lost.load(Ordering::Relaxed) {
                return;
            }
            self.last_result.store(result.as_raw(), Ordering::Relaxed);
            self.lost.store(true, Ordering::Release);
            if capture_diagnostics {
                inner.query.clone()
            } else {
                None
            }
        };
        let Some(query) = query else { return };
        let snapshot = catch_unwind(AssertUnwindSafe(|| query()));
        let mut inner = self.lock();
        inner.diagnostics = snapshot.unwrap_or_else(|_| DeviceFaultSnapshot {
            query_result: vk::Result::ERROR_UNKNOWN.as_raw(),
            ..Default::default()
        });
        inner.diagnostics.supported = true;
        inner.diagnostics.queried = true;
    }

    /// The loss result (`VK_SUCCESS` while healthy).
    pub fn last_result(&self) -> vk::Result {
        vk::Result::from_raw(self.last_result.load(Ordering::Relaxed))
    }

    /// Captured diagnostics.
    pub fn diagnostics(&self) -> DeviceFaultSnapshot {
        self.lock().diagnostics.clone()
    }

    /// Serializes queue submission across the device's queues.
    pub(crate) fn submission_lock(&self) -> MutexGuard<'_, ()> {
        self.submission.lock().unwrap_or_else(|p| p.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn captures_diagnostics_once_across_concurrent_loss_reports() {
        let state = DeviceFaultState::default();
        let queries = Arc::new(AtomicU32::new(0));
        let counted = Arc::clone(&queries);
        state.set_diagnostics_query(Some(Arc::new(move || {
            counted.fetch_add(1, Ordering::SeqCst);
            DeviceFaultSnapshot {
                description: "injected diagnostic".into(),
                addresses: vec![DeviceFaultAddressRecord {
                    address_type: 3,
                    reported_address: 0x1234,
                    precision: 16,
                }],
                vendor_records: vec![DeviceFaultVendorRecord {
                    description: "injected vendor record".into(),
                    code: 7,
                    data: 9,
                }],
                ..Default::default()
            }
        })));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| state.mark_lost(vk::Result::ERROR_DEVICE_LOST, true));
            }
        });
        let report = state.diagnostics();
        assert!(!state.healthy());
        assert_eq!(state.last_result(), vk::Result::ERROR_DEVICE_LOST);
        assert_eq!(queries.load(Ordering::SeqCst), 1);
        assert!(report.supported && report.queried && !report.truncated);
        assert_eq!(report.query_result, 0);
        assert_eq!(report.description, "injected diagnostic");
        assert_eq!(report.addresses[0].reported_address, 0x1234);
        assert_eq!(report.vendor_records[0].code, 7);
    }

    #[test]
    fn panicking_query_is_recorded_as_unknown() {
        let state = DeviceFaultState::default();
        state.set_diagnostics_query(Some(Arc::new(|| panic!("query failure"))));
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        state.mark_lost(vk::Result::ERROR_DEVICE_LOST, true);
        std::panic::set_hook(hook);
        let report = state.diagnostics();
        assert!(report.queried);
        assert_eq!(report.query_result, vk::Result::ERROR_UNKNOWN.as_raw());
        assert!(DeviceFaultState::default().healthy());
    }
}
