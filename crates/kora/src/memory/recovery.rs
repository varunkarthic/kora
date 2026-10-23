//! Memory admission estimates and the bounded out-of-memory recovery coordinator.

use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::error::{Error, ErrorCode, Result, Status};

/// Advisory admission estimate.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AdmissionEstimate {
    /// New bytes the operation needs.
    pub required_new_bytes: u64,
    /// Bytes eviction would likely free.
    pub likely_eviction_bytes: u64,
    /// Estimated eviction cost.
    pub eviction_cost_ms: f64,
    /// Budget left after admission (0 when not admitted).
    pub budget_after_admission_bytes: u64,
    /// Fragmentation risk in `[0, 1]`.
    pub fragmentation_risk: f64,
    /// The operation fits.
    pub admitted: bool,
}

/// Checked admission arithmetic.
pub fn estimate_admission(
    available_bytes: u64,
    required_new_bytes: u64,
    likely_eviction_bytes: u64,
    eviction_cost_ms: f64,
    fragmentation_risk: f64,
) -> Result<AdmissionEstimate> {
    if !(eviction_cost_ms.is_finite() && eviction_cost_ms >= 0.0)
        || !(0.0..=1.0).contains(&fragmentation_risk)
    {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "admission eviction cost must be nonnegative and fragmentation risk must be in [0,1]",
        ));
    }
    let capacity = available_bytes
        .checked_add(likely_eviction_bytes)
        .ok_or_else(|| Error::new(ErrorCode::Overflow, "admission capacity arithmetic overflow"))?;
    let admitted = capacity >= required_new_bytes;
    Ok(AdmissionEstimate {
        required_new_bytes,
        likely_eviction_bytes,
        eviction_cost_ms,
        budget_after_admission_bytes: if admitted {
            capacity - required_new_bytes
        } else {
            0
        },
        fragmentation_risk,
        admitted,
    })
}

/// The stage that recovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RecoveryAction {
    /// Plain retry.
    #[default]
    Retry,
    /// A smaller backing block was created.
    SmallerBlock,
    /// The retried allocation after a smaller block.
    RetrySmallerAllocation,
    /// The task was split into chunks.
    Chunking,
    /// The task ran on the CPU.
    CpuFallback,
}

/// What the coordinator did.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RecoveryTrace {
    /// The recovering stage.
    pub action: RecoveryAction,
    /// Whether any stage recovered.
    pub recovered: bool,
    /// Every stage invoked, in order.
    pub attempted_stages: Vec<&'static str>,
}

type Prepare<'a> = Option<&'a mut dyn FnMut() -> Status>;
type Attempt<'a> = Option<&'a mut dyn FnMut() -> Result<bool>>;

/// Recovery stages. Preparation stages may fail without preventing later alternatives; each
/// callback runs at most once.
#[derive(Default)]
pub struct OomRecoveryHooks<'a> {
    /// Re-query the device budget.
    pub refresh_budget: Prepare<'a>,
    /// Free completed deferred releases.
    pub reclaim_deferred: Prepare<'a>,
    /// Evict eligible replicas.
    pub evict_eligible: Prepare<'a>,
    /// One more allocation attempt.
    pub retry_once: Attempt<'a>,
    /// Create a smaller backing block (not itself an allocation success).
    pub smaller_block: Attempt<'a>,
    /// Allocate from the smaller block.
    pub retry_smaller_allocation: Attempt<'a>,
    /// Split the task.
    pub chunking: Attempt<'a>,
    /// Run the task on the CPU.
    pub cpu_fallback: Attempt<'a>,
    /// Stop at the first successful allocation stage (allocator policy). Task policy (`false`)
    /// continues into chunking and CPU fallback.
    pub stop_after_allocation: bool,
    /// Error returned when every stage fails (default `OutOfMemoryBudget`).
    pub terminal_error: Option<Error>,
}

impl std::fmt::Debug for OomRecoveryHooks<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OomRecoveryHooks")
            .field("stop_after_allocation", &self.stop_after_allocation)
            .finish()
    }
}

fn guarded<T>(f: impl FnOnce() -> Result<T>, what: &str) -> Result<T> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        Err(Error::new(
            ErrorCode::InternalInvariantViolation,
            format!("allocation recovery {what} callback panicked"),
        ))
    })
}

/// Runs the fixed preparation sequence, then each allocation alternative at most once.
pub fn recover_allocation_failure(mut hooks: OomRecoveryHooks<'_>) -> Result<RecoveryTrace> {
    let mut trace = RecoveryTrace::default();
    for (callback, name) in [
        (hooks.refresh_budget.take(), "refresh_budget"),
        (hooks.reclaim_deferred.take(), "reclaim_deferred"),
        (hooks.evict_eligible.take(), "evict_eligible"),
    ] {
        if let Some(callback) = callback {
            trace.attempted_stages.push(name);
            let _ = guarded(callback, "preparation");
        }
    }
    let stages = [
        (hooks.retry_once.take(), "retry_once", RecoveryAction::Retry),
        (
            hooks.smaller_block.take(),
            "smaller_block",
            RecoveryAction::SmallerBlock,
        ),
        (
            hooks.retry_smaller_allocation.take(),
            "retry_smaller_allocation",
            RecoveryAction::RetrySmallerAllocation,
        ),
        (hooks.chunking.take(), "chunking", RecoveryAction::Chunking),
        (
            hooks.cpu_fallback.take(),
            "cpu_fallback",
            RecoveryAction::CpuFallback,
        ),
    ];
    for (callback, name, action) in stages {
        let Some(callback) = callback else { continue };
        trace.attempted_stages.push(name);
        let succeeded = guarded(callback, "attempt").map_err(|cause| {
            if cause.code() == ErrorCode::InternalInvariantViolation && cause.message().contains("panicked") {
                cause
            } else {
                Error::new(
                    ErrorCode::InternalInvariantViolation,
                    "allocation recovery attempt callback returned an error",
                )
                .with_cause(cause)
            }
        })?;
        if !succeeded {
            continue;
        }
        trace.action = action;
        trace.recovered = true;
        if hooks.stop_after_allocation {
            if action != RecoveryAction::SmallerBlock {
                return Ok(trace);
            }
        } else if matches!(action, RecoveryAction::Chunking | RecoveryAction::CpuFallback) {
            return Ok(trace);
        }
    }
    let mut error = hooks.terminal_error.unwrap_or_else(|| {
        Error::new(
            ErrorCode::OutOfMemoryBudget,
            "allocation recovery exhausted its bounded stages",
        )
    });
    if !error.message().contains("bounded recovery stages were exhausted") {
        let message = format!("{}; bounded recovery stages were exhausted", error.message());
        error = error.with_message(message);
    }
    if !trace.attempted_stages.is_empty() {
        error = error.with_recovery(trace.attempted_stages.join(", "));
    }
    Err(error)
}

/// Task-level policy: allocation retries may lead into chunking and CPU fallback.
pub fn recover_task_allocation_failure(hooks: OomRecoveryHooks<'_>) -> Result<RecoveryTrace> {
    recover_allocation_failure(OomRecoveryHooks {
        stop_after_allocation: false,
        ..hooks
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn admission_accounts_for_eviction_and_reports_remaining_budget() {
        let estimate = estimate_admission(40, 70, 50, 2.5, 0.2).unwrap();
        assert!(estimate.admitted);
        assert_eq!(estimate.budget_after_admission_bytes, 20);
        assert_eq!(estimate.eviction_cost_ms, 2.5);
        assert!(estimate_admission(10, 1, 1, 0.0, 1.1).is_err());
        assert_eq!(
            estimate_admission(u64::MAX, 1, 1, 0.0, 0.0).unwrap_err().code(),
            ErrorCode::Overflow
        );
        assert!(!estimate_admission(10, 12, 1, 0.0, 0.0).unwrap().admitted);
    }

    #[test]
    fn executes_preparation_and_alternatives_once_in_order() {
        let order = RefCell::new(Vec::new());
        let push = |n| order.borrow_mut().push(n);
        let (mut a, mut b, mut c) = (
            || {
                push(1);
                Ok(())
            },
            || {
                push(2);
                Ok(())
            },
            || {
                push(3);
                Ok(())
            },
        );
        let mut d = || {
            push(4);
            Ok(false)
        };
        let mut e = || {
            push(5);
            Ok(true)
        };
        let mut f = || {
            push(6);
            Ok(false)
        };
        let mut g = || {
            push(7);
            Ok(true)
        };
        let result = recover_allocation_failure(OomRecoveryHooks {
            refresh_budget: Some(&mut a),
            reclaim_deferred: Some(&mut b),
            evict_eligible: Some(&mut c),
            retry_once: Some(&mut d),
            smaller_block: Some(&mut e),
            retry_smaller_allocation: Some(&mut f),
            chunking: Some(&mut g),
            stop_after_allocation: false,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(result.action, RecoveryAction::Chunking);
        assert_eq!(*order.borrow(), [1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn smaller_block_is_not_allocation_success_and_retry_runs_once() {
        let (smaller, retry) = (RefCell::new(0), RefCell::new(0));
        let mut r1 = || {
            *retry.borrow_mut() += 1;
            Ok(false)
        };
        let mut s = || {
            *smaller.borrow_mut() += 1;
            Ok(true)
        };
        let mut r2 = || {
            *retry.borrow_mut() += 1;
            Ok(true)
        };
        let result = recover_allocation_failure(OomRecoveryHooks {
            retry_once: Some(&mut r1),
            smaller_block: Some(&mut s),
            retry_smaller_allocation: Some(&mut r2),
            stop_after_allocation: true,
            ..Default::default()
        })
        .unwrap();
        assert!(result.recovered);
        assert_eq!(result.action, RecoveryAction::RetrySmallerAllocation);
        assert_eq!((*smaller.borrow(), *retry.borrow()), (1, 2));
    }

    #[test]
    fn preparation_failures_do_not_suppress_later_options() {
        let counts = RefCell::new([0; 4]);
        let bump = |i: usize| counts.borrow_mut()[i] += 1;
        let mut refresh = || {
            bump(0);
            Err(Error::new(ErrorCode::TransferFailed, "refresh"))
        };
        let mut reclaim = || {
            bump(1);
            Ok(())
        };
        let mut evict = || {
            bump(2);
            Err(Error::new(ErrorCode::OutOfMemoryBudget, "evict"))
        };
        let mut retry = || {
            bump(3);
            Ok(true)
        };
        let result = recover_allocation_failure(OomRecoveryHooks {
            refresh_budget: Some(&mut refresh),
            reclaim_deferred: Some(&mut reclaim),
            evict_eligible: Some(&mut evict),
            retry_once: Some(&mut retry),
            stop_after_allocation: true,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(result.action, RecoveryAction::Retry);
        assert_eq!(*counts.borrow(), [1; 4]);
    }

    #[test]
    fn tries_each_fallback_once_and_reports_exhaustion() {
        let count = RefCell::new(0);
        let mut f = [(); 5].map(|_| {
            || {
                *count.borrow_mut() += 1;
                Ok(false)
            }
        });
        let [a, b, c, d, e] = &mut f;
        let error = recover_allocation_failure(OomRecoveryHooks {
            retry_once: Some(a),
            smaller_block: Some(b),
            retry_smaller_allocation: Some(c),
            chunking: Some(d),
            cpu_fallback: Some(e),
            stop_after_allocation: true,
            ..Default::default()
        })
        .unwrap_err();
        assert_eq!(*count.borrow(), 5);
        assert_eq!(error.code(), ErrorCode::OutOfMemoryBudget);
        assert!(error.recovery_action().contains("cpu_fallback"));
        assert!(error.message().contains("bounded recovery stages were exhausted"));
    }

    #[test]
    fn panicking_hooks_become_structured_errors() {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let mut retry = || -> Result<bool> { panic!("injected") };
        let result = recover_allocation_failure(OomRecoveryHooks {
            retry_once: Some(&mut retry),
            ..Default::default()
        });
        std::panic::set_hook(hook);
        assert_eq!(result.unwrap_err().code(), ErrorCode::InternalInvariantViolation);
        let mut failing = || -> Result<bool> { Err(Error::new(ErrorCode::DeviceLost, "lost")) };
        let error = recover_allocation_failure(OomRecoveryHooks {
            retry_once: Some(&mut failing),
            ..Default::default()
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InternalInvariantViolation);
        assert_eq!(error.cause().unwrap().code(), ErrorCode::DeviceLost);
    }

    #[test]
    fn task_policy_continues_to_chunking_and_cpu_fallback() {
        let counts = RefCell::new([0; 4]);
        let bump = |i: usize| counts.borrow_mut()[i] += 1;
        let mut retry = || {
            bump(0);
            Ok(true)
        };
        let mut smaller = || {
            bump(1);
            Ok(false)
        };
        let mut retry_smaller = || {
            bump(1);
            Ok(false)
        };
        let mut chunk = || {
            bump(2);
            Ok(false)
        };
        let mut cpu = || {
            bump(3);
            Ok(true)
        };
        let result = recover_task_allocation_failure(OomRecoveryHooks {
            retry_once: Some(&mut retry),
            smaller_block: Some(&mut smaller),
            retry_smaller_allocation: Some(&mut retry_smaller),
            chunking: Some(&mut chunk),
            cpu_fallback: Some(&mut cpu),
            stop_after_allocation: true,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(result.action, RecoveryAction::CpuFallback);
        assert_eq!(*counts.borrow(), [1, 2, 1, 1]);
    }

    #[test]
    fn preserves_terminal_error_and_adds_stage_trace() {
        let mut retry = || Ok(false);
        let mut smaller = || Ok(false);
        let error = recover_allocation_failure(OomRecoveryHooks {
            retry_once: Some(&mut retry),
            smaller_block: Some(&mut smaller),
            stop_after_allocation: true,
            terminal_error: Some(
                Error::new(ErrorCode::AllocationFailed, "vkAllocateMemory failed").with_vk(-2),
            ),
            ..Default::default()
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::AllocationFailed);
        assert_eq!(error.vk_result(), -2);
        assert!(error.recovery_action().contains("retry_once"));
    }
}
