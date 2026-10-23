//! Copyable completion handles for asynchronous work.
//!
//! An [`Event`] is a shared view of one operation's completion [`Status`]. A timed-out wait does
//! not cancel or change the operation. Completion callbacks run outside the internal lock, so they
//! may register further callbacks or wait on other events; they must not block on work that is
//! scheduled to the same completion thread.

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::error::{Error, ErrorCode, Status};

type Callback = Box<dyn FnOnce(Status) + Send>;

struct Inner {
    status: Option<Status>,
    callbacks: Vec<Callback>,
}

struct State {
    done: AtomicBool,
    inner: Mutex<Inner>,
    changed: Condvar,
}

impl State {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            done: AtomicBool::new(false),
            inner: Mutex::new(Inner {
                status: None,
                callbacks: Vec::new(),
            }),
            changed: Condvar::new(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // A panic while holding this lock cannot leave `Inner` inconsistent: every critical section
        // only moves values in or out.
        self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Shared asynchronous completion state. Cloning is cheap; an `Event::default()` is empty.
#[derive(Clone, Default)]
pub struct Event {
    state: Option<Arc<State>>,
}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = match &self.state {
            None => "empty",
            Some(state) if state.done.load(Ordering::Acquire) => "complete",
            Some(_) => "pending",
        };
        write!(f, "Event({state})")
    }
}

fn empty_event(action: &str) -> Error {
    Error::new(
        ErrorCode::InvalidArgument,
        format!("cannot {action} an empty Event"),
    )
}

fn pending() -> Error {
    Error::new(ErrorCode::Timeout, "Event is pending")
}

impl Event {
    /// An event that is already complete with `status`.
    pub fn ready(status: Status) -> Event {
        let promise = EventPromise::new();
        let event = promise.event();
        promise.complete(status);
        event
    }

    /// `true` unless this is an empty (default) event.
    pub fn is_valid(&self) -> bool {
        self.state.is_some()
    }

    /// `true` once the operation has completed (successfully or not). Lock-free.
    pub fn is_done(&self) -> bool {
        self.state
            .as_ref()
            .is_some_and(|state| state.done.load(Ordering::Acquire))
    }

    /// Blocks until completion and returns the completion status.
    pub fn wait(&self) -> Status {
        let state = self.state.as_ref().ok_or_else(|| empty_event("wait on"))?;
        let mut inner = state.lock();
        loop {
            if let Some(status) = &inner.status {
                return status.clone();
            }
            inner = state
                .changed
                .wait(inner)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// Waits up to `timeout`. Returns [`ErrorCode::Timeout`] if it expires; the work continues.
    pub fn wait_timeout(&self, timeout: Duration) -> Status {
        let state = self.state.as_ref().ok_or_else(|| empty_event("wait on"))?;
        let deadline = Instant::now().checked_add(timeout);
        let mut inner = state.lock();
        loop {
            if let Some(status) = &inner.status {
                return status.clone();
            }
            let remaining = match deadline {
                Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                None => Duration::MAX,
            };
            if remaining.is_zero() {
                return Err(Error::new(
                    ErrorCode::Timeout,
                    "Event wait timed out; the operation continues",
                ));
            }
            inner = state
                .changed
                .wait_timeout(inner, remaining)
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
    }

    /// The completion status, or [`ErrorCode::Timeout`] while pending.
    pub fn status(&self) -> Status {
        let state = self.state.as_ref().ok_or_else(|| empty_event("query"))?;
        state.lock().status.clone().unwrap_or_else(|| Err(pending()))
    }

    /// Registers a completion callback. If the event is already complete, the callback runs
    /// synchronously before this returns; a panic there is reported as
    /// [`ErrorCode::InternalInvariantViolation`].
    pub fn on_complete(&self, callback: impl FnOnce(Status) + Send + 'static) -> Status {
        let state = self.state.as_ref().ok_or_else(|| empty_event("register on"))?;
        let ready = {
            let mut inner = state.lock();
            match &inner.status {
                Some(status) => status.clone(),
                None => {
                    inner.callbacks.push(Box::new(callback));
                    return Ok(());
                }
            }
        };
        catch_unwind(AssertUnwindSafe(|| callback(ready))).map_err(|_| {
            Error::new(
                ErrorCode::InternalInvariantViolation,
                "Event completion callback panicked",
            )
        })
    }

    /// Returns an event completed with `continuation(self's status)` after this one completes.
    /// A panicking continuation completes the chained event with
    /// [`ErrorCode::InternalInvariantViolation`].
    pub fn then(&self, continuation: impl FnOnce(Status) -> Status + Send + 'static) -> crate::Result<Event> {
        if self.state.is_none() {
            return Err(empty_event("chain from"));
        }
        let promise = EventPromise::new();
        let chained = promise.event();
        self.on_complete(move |previous| {
            let status = catch_unwind(AssertUnwindSafe(|| continuation(previous))).unwrap_or_else(|_| {
                Err(Error::new(
                    ErrorCode::InternalInvariantViolation,
                    "Event continuation panicked",
                ))
            });
            promise.complete(status);
        })?;
        Ok(chained)
    }

    /// Completes after every input, with the first failure in input order (or success).
    pub fn when_all(events: &[Event]) -> crate::Result<Event> {
        if events.iter().any(|event| !event.is_valid()) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "when_all received an empty Event",
            ));
        }
        if events.is_empty() {
            return Ok(Event::ready(Ok(())));
        }
        struct Aggregate {
            remaining: usize,
            statuses: Vec<Option<Status>>,
            promise: Option<EventPromise>,
        }
        let promise = EventPromise::new();
        let combined = promise.event();
        let aggregate = Arc::new(Mutex::new(Aggregate {
            remaining: events.len(),
            statuses: vec![None; events.len()],
            promise: Some(promise),
        }));
        for (index, event) in events.iter().enumerate() {
            let aggregate = Arc::clone(&aggregate);
            event.on_complete(move |status| {
                let finished = {
                    let mut state = aggregate.lock().unwrap_or_else(|p| p.into_inner());
                    state.statuses[index] = Some(status);
                    state.remaining -= 1;
                    if state.remaining == 0 {
                        let first_failure = state.statuses.iter().flatten().find(|s| s.is_err()).cloned();
                        state
                            .promise
                            .take()
                            .map(|promise| (promise, first_failure.unwrap_or(Ok(()))))
                    } else {
                        None
                    }
                };
                if let Some((promise, status)) = finished {
                    promise.complete(status);
                }
            })?;
        }
        Ok(combined)
    }

    /// `true` when both handles refer to the same operation.
    pub fn same_as(&self, other: &Event) -> bool {
        match (&self.state, &other.state) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// The producer side of an [`Event`]. Completing it more than once has no effect. Dropping an
/// uncompleted promise completes its event with [`ErrorCode::Cancelled`], so waiters never hang
/// on abandoned work.
pub struct EventPromise {
    state: Arc<State>,
}

impl fmt::Debug for EventPromise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EventPromise({:?})", self.event())
    }
}

impl Default for EventPromise {
    fn default() -> Self {
        Self::new()
    }
}

impl EventPromise {
    /// Creates a pending promise.
    pub fn new() -> Self {
        Self { state: State::new() }
    }

    /// A consumer handle for this promise's event.
    pub fn event(&self) -> Event {
        Event {
            state: Some(Arc::clone(&self.state)),
        }
    }

    /// `true` once completed.
    pub fn is_done(&self) -> bool {
        self.state.done.load(Ordering::Acquire)
    }

    /// Completes the event, wakes waiters, and runs callbacks (a panicking callback is isolated).
    pub fn complete(&self, status: Status) {
        let callbacks = {
            let mut inner = self.state.lock();
            if inner.status.is_some() {
                return;
            }
            inner.status = Some(status.clone());
            self.state.done.store(true, Ordering::Release);
            std::mem::take(&mut inner.callbacks)
        };
        self.state.changed.notify_all();
        for callback in callbacks {
            let status = status.clone();
            let _ = catch_unwind(AssertUnwindSafe(move || callback(status)));
        }
    }
}

impl Drop for EventPromise {
    fn drop(&mut self) {
        if !self.is_done() {
            self.complete(Err(Error::new(
                ErrorCode::Cancelled,
                "operation was abandoned before completion",
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn quiet_panics<R>(f: impl FnOnce() -> R) -> R {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = f();
        std::panic::set_hook(hook);
        result
    }

    #[test]
    fn timeout_does_not_change_pending_state_and_multiple_waiters_complete() {
        let promise = EventPromise::new();
        let event = promise.event();
        assert_eq!(
            event.wait_timeout(Duration::ZERO).unwrap_err().code(),
            ErrorCode::Timeout
        );
        assert!(!event.is_done());
        assert_eq!(event.status().unwrap_err().code(), ErrorCode::Timeout);
        let results: Vec<Status> = std::thread::scope(|scope| {
            let waiters: Vec<_> = (0..8).map(|_| scope.spawn(|| event.wait())).collect();
            std::thread::sleep(Duration::from_millis(5));
            promise.complete(Ok(()));
            waiters.into_iter().map(|w| w.join().unwrap()).collect()
        });
        assert!(event.is_done());
        assert!(results.iter().all(Result::is_ok));
        assert!(event.wait_timeout(Duration::ZERO).is_ok());
    }

    #[test]
    fn then_preserves_failure_and_runs_once() {
        let promise = EventPromise::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&calls);
        let chained = promise
            .event()
            .then(move |status| {
                counted.fetch_add(1, Ordering::SeqCst);
                status
            })
            .unwrap();
        promise.complete(Err(Error::new(ErrorCode::Cancelled, "source failed")));
        promise.complete(Ok(()));
        assert_eq!(chained.wait().unwrap_err().code(), ErrorCode::Cancelled);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn when_all_waits_for_every_input_and_returns_first_input_failure() {
        let first = EventPromise::new();
        let second = EventPromise::new();
        let combined = Event::when_all(&[first.event(), second.event()]).unwrap();
        first.complete(Err(Error::new(ErrorCode::Cancelled, "first failure")));
        assert!(!combined.is_done());
        second.complete(Err(Error::new(ErrorCode::Timeout, "second failure")));
        assert_eq!(combined.wait().unwrap_err().code(), ErrorCode::Cancelled);
    }

    #[test]
    fn when_all_reports_later_failure_after_earlier_success() {
        let first = EventPromise::new();
        let second = EventPromise::new();
        let combined = Event::when_all(&[first.event(), second.event()]).unwrap();
        second.complete(Err(Error::new(ErrorCode::DeviceLost, "lost")));
        first.complete(Ok(()));
        assert_eq!(combined.wait().unwrap_err().code(), ErrorCode::DeviceLost);
    }

    #[test]
    fn when_all_empty_input_completes_immediately_and_rejects_empty_events() {
        let combined = Event::when_all(&[]).unwrap();
        assert!(combined.is_done());
        assert!(combined.wait().is_ok());
        assert_eq!(
            Event::when_all(&[Event::default()]).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn empty_event_operations_fail_cleanly() {
        let empty = Event::default();
        assert!(!empty.is_valid());
        assert!(!empty.is_done());
        for status in [
            empty.wait(),
            empty.status(),
            empty.wait_timeout(Duration::ZERO),
            empty.on_complete(|_| {}),
        ] {
            assert_eq!(status.unwrap_err().code(), ErrorCode::InvalidArgument);
        }
        assert!(empty.then(|s| s).is_err());
    }

    #[test]
    fn on_complete_after_completion_runs_synchronously_and_reports_panics() {
        let event = Event::ready(Ok(()));
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        event
            .on_complete(move |status| flag.store(status.is_ok(), Ordering::SeqCst))
            .unwrap();
        assert!(ran.load(Ordering::SeqCst));
        let failed = quiet_panics(|| event.on_complete(|_| panic!("callback failure")));
        assert_eq!(failed.unwrap_err().code(), ErrorCode::InternalInvariantViolation);
    }

    #[test]
    fn panicking_callbacks_do_not_prevent_other_callbacks_or_waiters() {
        let promise = EventPromise::new();
        let event = promise.event();
        let count = Arc::new(AtomicUsize::new(0));
        event.on_complete(|_| panic!("first")).unwrap();
        let counted = Arc::clone(&count);
        event
            .on_complete(move |_| {
                counted.fetch_add(1, Ordering::SeqCst);
            })
            .unwrap();
        quiet_panics(|| promise.complete(Ok(())));
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(event.wait().is_ok());
    }

    #[test]
    fn panicking_continuation_fails_the_chained_event() {
        let promise = EventPromise::new();
        let chained = promise.event().then(|_| panic!("continuation")).unwrap();
        quiet_panics(|| promise.complete(Ok(())));
        assert_eq!(
            chained.wait().unwrap_err().code(),
            ErrorCode::InternalInvariantViolation
        );
    }

    #[test]
    fn dropping_an_uncompleted_promise_cancels_its_event() {
        let event = {
            let promise = EventPromise::new();
            promise.event()
        };
        assert_eq!(event.wait().unwrap_err().code(), ErrorCode::Cancelled);
        let completed = EventPromise::new();
        let event = completed.event();
        completed.complete(Ok(()));
        drop(completed);
        assert!(event.wait().is_ok());
    }

    #[test]
    fn callbacks_may_register_more_callbacks_without_deadlock() {
        let promise = EventPromise::new();
        let event = promise.event();
        let inner_event = event.clone();
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        event
            .on_complete(move |_| {
                inner_event
                    .on_complete(move |_| flag.store(true, Ordering::SeqCst))
                    .unwrap();
            })
            .unwrap();
        promise.complete(Ok(()));
        assert!(ran.load(Ordering::SeqCst));
    }

    #[test]
    fn same_as_compares_identity() {
        let promise = EventPromise::new();
        assert!(promise.event().same_as(&promise.event()));
        assert!(!promise.event().same_as(&Event::ready(Ok(()))));
        assert!(!Event::default().same_as(&Event::default()));
    }

    #[test]
    fn concurrent_completion_and_registration_runs_every_callback_once() {
        for _ in 0..200 {
            let promise = EventPromise::new();
            let event = promise.event();
            let count = Arc::new(AtomicUsize::new(0));
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    let event = event.clone();
                    let count = Arc::clone(&count);
                    scope.spawn(move || {
                        for _ in 0..8 {
                            let count = Arc::clone(&count);
                            event
                                .on_complete(move |_| {
                                    count.fetch_add(1, Ordering::SeqCst);
                                })
                                .unwrap();
                        }
                    });
                }
                scope.spawn(|| promise.complete(Ok(())));
            });
            assert_eq!(count.load(Ordering::SeqCst), 32);
        }
    }
}
