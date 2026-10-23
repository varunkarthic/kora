//! Fixed-size CPU worker pool with priority lanes and work stealing.
//!
//! Each worker owns four priority lanes. Workers pop their own newest work first (LIFO, cache
//! friendly) and steal the oldest work from other workers (FIFO). Submission never blocks on
//! capacity and takes only the target worker's lane lock; idle workers sleep on a condition
//! variable and are woken only when someone is actually asleep. A worker that waits on an [`Event`]
//! through [`CpuExecutor::wait`] keeps executing queued work, so recursive submit-and-wait cannot
//! deadlock, even with a single worker.

use std::cell::Cell;
use std::collections::VecDeque;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::error::{Error, ErrorCode, Result, Status};
use crate::event::{Event, EventPromise};
use crate::options::ShutdownMode;

/// Task priority lane. Higher lanes are always drained first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum CpuTaskPriority {
    /// Background work.
    Low,
    /// Default.
    #[default]
    Normal,
    /// Latency-sensitive work.
    High,
    /// Run before everything else.
    Critical,
}

const LANES: usize = 4;
const MAXIMUM_WORKERS: usize = 1024;

/// Executor sizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuExecutorOptions {
    /// Zero selects `max(1, available_parallelism - reserved_host_threads)`.
    pub worker_count: usize,
    /// Logical CPUs left for the application when `worker_count` is zero.
    pub reserved_host_threads: usize,
}

impl Default for CpuExecutorOptions {
    fn default() -> Self {
        Self {
            worker_count: 0,
            reserved_host_threads: 1,
        }
    }
}

type Task = Box<dyn FnOnce() -> Status + Send>;

struct WorkItem {
    task: Task,
    promise: EventPromise,
}

type Lanes = [VecDeque<WorkItem>; LANES];

struct Shared {
    lanes: Box<[Mutex<Lanes>]>,
    /// Items sitting in lanes.
    queued: AtomicUsize,
    /// Items admitted but not yet finished (queued + running + being admitted).
    outstanding: AtomicUsize,
    next_worker: AtomicUsize,
    accepting: AtomicBool,
    stopping: AtomicBool,
    cancel_pending: AtomicBool,
    sleepers: AtomicUsize,
    sleep_lock: Mutex<()>,
    wake: Condvar,
}

thread_local! {
    /// `(Shared address, worker index)` of the executor running on this thread, if any.
    static CURRENT: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Shared {
    fn identity(self: &Arc<Self>) -> usize {
        Arc::as_ptr(self) as usize
    }

    fn current_worker(self: &Arc<Self>) -> Option<usize> {
        let identity = self.identity();
        CURRENT
            .with(|current| current.get())
            .and_then(|(id, index)| (id == identity).then_some(index))
    }

    fn wake_one(&self) {
        if self.sleepers.load(SeqCst) > 0 {
            let _guard = lock(&self.sleep_lock);
            self.wake.notify_one();
        }
    }

    fn wake_all(&self) {
        let _guard = lock(&self.sleep_lock);
        self.wake.notify_all();
    }

    fn finish_one(&self) {
        if self.outstanding.fetch_sub(1, SeqCst) == 1 && self.stopping.load(SeqCst) {
            self.wake_all();
        }
    }

    fn try_pop(&self, index: usize, steal: bool) -> Option<WorkItem> {
        let mut lanes = lock(&self.lanes[index]);
        lanes
            .iter_mut()
            .rev()
            .find_map(|lane| if steal { lane.pop_front() } else { lane.pop_back() })
    }

    fn run_one(&self, worker: usize) -> bool {
        if self.queued.load(SeqCst) == 0 {
            return false;
        }
        let count = self.lanes.len();
        let item = self
            .try_pop(worker, false)
            .or_else(|| (1..count).find_map(|offset| self.try_pop((worker + offset) % count, true)));
        let Some(item) = item else {
            return false;
        };
        self.queued.fetch_sub(1, SeqCst);
        if self.cancel_pending.load(SeqCst) {
            // A submission can race past the admission check and land after shutdown drained the
            // lanes; cancel it here so its event completes and shutdown can finish.
            let cancelled = Error::new(
                ErrorCode::Cancelled,
                "CPU task cancelled during executor shutdown",
            );
            item.promise.complete(Err(cancelled));
            self.finish_one();
            return true;
        }
        let status = catch_unwind(AssertUnwindSafe(item.task)).unwrap_or_else(|panic| {
            let detail = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".into());
            Err(Error::new(
                ErrorCode::InternalInvariantViolation,
                format!("CPU task panicked: {detail}"),
            ))
        });
        item.promise.complete(status);
        self.finish_one();
        true
    }

    fn should_exit(&self) -> bool {
        self.stopping.load(SeqCst) && self.outstanding.load(SeqCst) == 0
    }

    fn worker_loop(self: Arc<Self>, index: usize) {
        CURRENT.with(|current| current.set(Some((self.identity(), index))));
        loop {
            if self.run_one(index) {
                continue;
            }
            let mut guard = lock(&self.sleep_lock);
            self.sleepers.fetch_add(1, SeqCst);
            while self.queued.load(SeqCst) == 0 && !self.should_exit() {
                guard = self
                    .wake
                    .wait(guard)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            self.sleepers.fetch_sub(1, SeqCst);
            drop(guard);
            if self.should_exit() {
                break;
            }
        }
        CURRENT.with(|current| current.set(None));
    }
}

/// Fixed-size CPU task pool. See the [module documentation](self).
pub struct CpuExecutor {
    shared: Arc<Shared>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl fmt::Debug for CpuExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuExecutor")
            .field("workers", &self.worker_count())
            .field("queued", &self.shared.queued.load(SeqCst))
            .finish()
    }
}

fn default_worker_count(reserved: usize) -> usize {
    let hardware = std::thread::available_parallelism().map_or(1, |n| n.get());
    hardware.saturating_sub(reserved).max(1)
}

impl CpuExecutor {
    /// Starts the workers.
    pub fn new(options: CpuExecutorOptions) -> Result<CpuExecutor> {
        let count = match options.worker_count {
            0 => default_worker_count(options.reserved_host_threads),
            n => n,
        };
        if count > MAXIMUM_WORKERS {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "CPU executor worker count is outside the supported range",
            ));
        }
        let shared = Arc::new(Shared {
            lanes: (0..count).map(|_| Mutex::new(Default::default())).collect(),
            queued: AtomicUsize::new(0),
            outstanding: AtomicUsize::new(0),
            next_worker: AtomicUsize::new(0),
            accepting: AtomicBool::new(true),
            stopping: AtomicBool::new(false),
            cancel_pending: AtomicBool::new(false),
            sleepers: AtomicUsize::new(0),
            sleep_lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let executor = CpuExecutor {
            shared,
            threads: Mutex::new(Vec::with_capacity(count)),
        };
        for index in 0..count {
            let shared = Arc::clone(&executor.shared);
            let spawned = std::thread::Builder::new()
                .name(format!("kora-cpu-{index}"))
                .spawn(move || shared.worker_loop(index));
            match spawned {
                Ok(handle) => lock(&executor.threads).push(handle),
                Err(error) => {
                    let _ = executor.shutdown(ShutdownMode::Graceful);
                    return Err(Error::new(
                        ErrorCode::AllocationFailed,
                        format!("unable to start CPU executor worker: {error}"),
                    ));
                }
            }
        }
        Ok(executor)
    }

    /// Number of worker threads.
    pub fn worker_count(&self) -> usize {
        self.shared.lanes.len()
    }

    /// Enqueues `task` and returns an event completed with its status (a panic becomes
    /// [`ErrorCode::InternalInvariantViolation`]). After shutdown begins, only this executor's own
    /// tasks may submit (descendants are drained); `CancelPending` rejects everything.
    pub fn submit(
        &self,
        priority: CpuTaskPriority,
        task: impl FnOnce() -> Status + Send + 'static,
    ) -> Result<Event> {
        let shared = &self.shared;
        let nested = shared.current_worker();
        // Count the item before checking admission so shutdown cannot observe zero outstanding
        // work while this submission is still being admitted.
        shared.outstanding.fetch_add(1, SeqCst);
        if shared.cancel_pending.load(SeqCst) || (!shared.accepting.load(SeqCst) && nested.is_none()) {
            shared.finish_one();
            return Err(Error::new(
                ErrorCode::Cancelled,
                "CPU executor is shutting down and rejects new work",
            ));
        }
        let promise = EventPromise::new();
        let event = promise.event();
        let index = nested.unwrap_or_else(|| shared.next_worker.fetch_add(1, SeqCst) % shared.lanes.len());
        lock(&shared.lanes[index])[priority as usize].push_back(WorkItem {
            task: Box::new(task),
            promise,
        });
        shared.queued.fetch_add(1, SeqCst);
        shared.wake_one();
        Ok(event)
    }

    /// Waits for `event`. On one of this executor's workers, keeps running queued work meanwhile.
    pub fn wait(&self, event: &Event) -> Status {
        if !event.is_valid() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "cannot wait on an invalid Event",
            ));
        }
        let Some(worker) = self.shared.current_worker() else {
            return event.wait();
        };
        while !event.is_done() {
            if !self.shared.run_one(worker) {
                let _ = event.wait_timeout(Duration::from_micros(250));
            }
        }
        event.wait()
    }

    /// Stops admission, then drains (`Graceful`) or cancels (`CancelPending`) queued work and joins
    /// the workers. Running tasks are never interrupted. Rejected from inside one of this
    /// executor's own tasks. Repeated calls are no-ops.
    pub fn shutdown(&self, mode: ShutdownMode) -> Status {
        let shared = &self.shared;
        if shared.current_worker().is_some() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "CPU executor cannot be shut down by one of its tasks",
            ));
        }
        let mut threads = lock(&self.threads);
        shared.accepting.store(false, SeqCst);
        if mode == ShutdownMode::CancelPending {
            shared.cancel_pending.store(true, SeqCst);
        }
        shared.stopping.store(true, SeqCst);
        if mode == ShutdownMode::CancelPending {
            for lanes in shared.lanes.iter() {
                let cancelled: Vec<WorkItem> = {
                    let mut lanes = lock(lanes);
                    lanes.iter_mut().flat_map(std::mem::take).collect()
                };
                shared.queued.fetch_sub(cancelled.len(), SeqCst);
                for item in cancelled {
                    item.promise.complete(Err(Error::new(
                        ErrorCode::Cancelled,
                        "CPU task cancelled during executor shutdown",
                    )));
                    shared.finish_one();
                }
            }
        }
        shared.wake_all();
        for handle in threads.drain(..) {
            let _ = handle.join();
        }
        Ok(())
    }
}

impl Drop for CpuExecutor {
    fn drop(&mut self) {
        if self.shared.current_worker().is_some() {
            // Dropped from one of its own tasks: stop admission and let the workers exit on their own.
            self.shared.accepting.store(false, SeqCst);
            self.shared.stopping.store(true, SeqCst);
            self.shared.wake_all();
            return;
        }
        let _ = self.shutdown(ShutdownMode::Graceful);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn pool(workers: usize) -> Arc<CpuExecutor> {
        Arc::new(
            CpuExecutor::new(CpuExecutorOptions {
                worker_count: workers,
                ..Default::default()
            })
            .unwrap(),
        )
    }

    const PRIORITIES: [CpuTaskPriority; 4] = [
        CpuTaskPriority::Low,
        CpuTaskPriority::Normal,
        CpuTaskPriority::High,
        CpuTaskPriority::Critical,
    ];

    #[test]
    fn drains_one_million_tasks_across_stealing_workers() {
        let executor = pool(4);
        assert_eq!(executor.worker_count(), 4);
        let completed = Arc::new(AtomicUsize::new(0));
        for index in 0..1_000_000usize {
            let completed = Arc::clone(&completed);
            executor
                .submit(PRIORITIES[index % 4], move || {
                    completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(())
                })
                .unwrap();
        }
        executor.shutdown(ShutdownMode::Graceful).unwrap();
        assert_eq!(completed.load(SeqCst), 1_000_000);
        assert_eq!(
            executor
                .submit(CpuTaskPriority::Normal, || Ok(()))
                .unwrap_err()
                .code(),
            ErrorCode::Cancelled
        );
    }

    #[test]
    fn repeated_graceful_shutdown_drains_short_task_bursts() {
        for iteration in 0..128 {
            let executor = pool(4);
            let completed = Arc::new(AtomicUsize::new(0));
            for _ in 0..128 {
                let completed = Arc::clone(&completed);
                executor
                    .submit(CpuTaskPriority::Normal, move || {
                        completed.fetch_add(1, SeqCst);
                        Ok(())
                    })
                    .unwrap();
            }
            executor.shutdown(ShutdownMode::Graceful).unwrap();
            assert_eq!(completed.load(SeqCst), 128, "iteration {iteration}");
        }
    }

    struct Gate {
        state: Mutex<(bool, bool)>,
        changed: Condvar,
    }

    impl Gate {
        fn new() -> Arc<Self> {
            Arc::new(Gate {
                state: Mutex::new((false, false)),
                changed: Condvar::new(),
            })
        }
        fn block(&self) {
            let mut state = self.state.lock().unwrap();
            state.0 = true;
            self.changed.notify_all();
            while !state.1 {
                state = self.changed.wait(state).unwrap();
            }
        }
        fn wait_started(&self) {
            let mut state = self.state.lock().unwrap();
            while !state.0 {
                state = self.changed.wait(state).unwrap();
            }
        }
        fn release(&self) {
            self.state.lock().unwrap().1 = true;
            self.changed.notify_all();
        }
    }

    #[test]
    fn higher_priority_lane_runs_before_queued_lower_priority_work() {
        let executor = pool(1);
        let gate = Gate::new();
        let blocker_gate = Arc::clone(&gate);
        let blocker = executor
            .submit(CpuTaskPriority::Normal, move || {
                blocker_gate.block();
                Ok(())
            })
            .unwrap();
        gate.wait_started();
        let order = Arc::new(Mutex::new(Vec::new()));
        let low_order = Arc::clone(&order);
        let low = executor
            .submit(CpuTaskPriority::Low, move || {
                low_order.lock().unwrap().push(0);
                Ok(())
            })
            .unwrap();
        let high_order = Arc::clone(&order);
        let high = executor
            .submit(CpuTaskPriority::High, move || {
                high_order.lock().unwrap().push(1);
                Ok(())
            })
            .unwrap();
        gate.release();
        for event in [&blocker, &low, &high] {
            executor.wait(event).unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![1, 0]);
    }

    fn recurse(executor: Arc<CpuExecutor>, depth: u32) -> Status {
        if depth == 0 {
            return Ok(());
        }
        let child_executor = Arc::clone(&executor);
        let child = executor.submit(CpuTaskPriority::Normal, move || {
            recurse(child_executor, depth - 1)
        })?;
        executor.wait(&child)
    }

    #[test]
    fn cooperative_wait_prevents_single_worker_recursive_deadlock() {
        let executor = pool(1);
        let root_executor = Arc::clone(&executor);
        let root = executor
            .submit(CpuTaskPriority::Normal, move || recurse(root_executor, 96))
            .unwrap();
        executor.wait(&root).unwrap();
        executor.shutdown(ShutdownMode::Graceful).unwrap();
    }

    #[test]
    fn captures_task_panics_into_event_status() {
        let executor = pool(1);
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let task = executor
            .submit(CpuTaskPriority::Normal, || panic!("synthetic CPU failure"))
            .unwrap();
        let status = executor.wait(&task);
        std::panic::set_hook(hook);
        let error = status.unwrap_err();
        assert_eq!(error.code(), ErrorCode::InternalInvariantViolation);
        assert!(error.message().contains("synthetic CPU failure"));
        // The worker survives the panic.
        executor
            .wait(&executor.submit(CpuTaskPriority::Normal, || Ok(())).unwrap())
            .unwrap();
    }

    #[test]
    fn cancel_pending_stops_queued_tasks_and_completes_their_events() {
        let executor = pool(1);
        let gate = Gate::new();
        let active_gate = Arc::clone(&gate);
        let active = executor
            .submit(CpuTaskPriority::Normal, move || {
                active_gate.block();
                Ok(())
            })
            .unwrap();
        gate.wait_started();
        let queued_ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&queued_ran);
        let pending = executor
            .submit(CpuTaskPriority::Normal, move || {
                flag.store(true, SeqCst);
                Ok(())
            })
            .unwrap();
        let shutdown_executor = Arc::clone(&executor);
        let shutdown = std::thread::spawn(move || shutdown_executor.shutdown(ShutdownMode::CancelPending));
        let admission_error = loop {
            match executor.submit(CpuTaskPriority::Normal, || Ok(())) {
                Err(error) => break error.code(),
                Ok(_) => std::thread::yield_now(),
            }
        };
        gate.release();
        shutdown.join().unwrap().unwrap();
        assert_eq!(admission_error, ErrorCode::Cancelled);
        executor.wait(&active).unwrap();
        assert_eq!(executor.wait(&pending).unwrap_err().code(), ErrorCode::Cancelled);
        assert!(!queued_ran.load(SeqCst));
    }

    #[test]
    fn cancel_pending_races_with_concurrent_submitters_without_hanging() {
        for _ in 0..200 {
            let executor = pool(2);
            let submitters: Vec<_> = (0..3)
                .map(|_| {
                    let executor = Arc::clone(&executor);
                    std::thread::spawn(move || {
                        let mut events = Vec::new();
                        while let Ok(event) = executor.submit(CpuTaskPriority::Normal, || Ok(())) {
                            events.push(event);
                        }
                        events
                    })
                })
                .collect();
            std::thread::yield_now();
            executor.shutdown(ShutdownMode::CancelPending).unwrap();
            for submitter in submitters {
                for event in submitter.join().unwrap() {
                    let status = event.wait();
                    assert!(status.is_ok() || status.unwrap_err().code() == ErrorCode::Cancelled);
                }
            }
        }
    }

    #[test]
    fn shutdown_from_a_task_is_rejected_and_drop_from_a_task_does_not_deadlock() {
        let executor = pool(2);
        let inner = Arc::clone(&executor);
        let task = executor
            .submit(CpuTaskPriority::Normal, move || {
                let status = inner.shutdown(ShutdownMode::Graceful);
                assert_eq!(status.unwrap_err().code(), ErrorCode::InvalidArgument);
                Ok(())
            })
            .unwrap();
        executor.wait(&task).unwrap();

        let executor = pool(1);
        let owned = Arc::clone(&executor);
        let event = executor
            .submit(CpuTaskPriority::Normal, move || {
                drop(owned);
                Ok(())
            })
            .unwrap();
        drop(executor);
        event.wait().unwrap();
    }

    #[test]
    fn nested_submissions_during_graceful_shutdown_are_drained() {
        let executor = pool(2);
        let completed = Arc::new(AtomicUsize::new(0));
        let gate = Gate::new();
        let (task_executor, task_gate, task_completed) =
            (Arc::clone(&executor), Arc::clone(&gate), Arc::clone(&completed));
        executor
            .submit(CpuTaskPriority::Normal, move || {
                task_gate.block();
                for _ in 0..10 {
                    let completed = Arc::clone(&task_completed);
                    task_executor.submit(CpuTaskPriority::Low, move || {
                        completed.fetch_add(1, SeqCst);
                        Ok(())
                    })?;
                }
                Ok(())
            })
            .unwrap();
        gate.wait_started();
        let shutdown_executor = Arc::clone(&executor);
        let shutdown = std::thread::spawn(move || shutdown_executor.shutdown(ShutdownMode::Graceful));
        while executor.submit(CpuTaskPriority::Normal, || Ok(())).is_ok() {
            std::thread::yield_now();
        }
        gate.release();
        shutdown.join().unwrap().unwrap();
        assert_eq!(completed.load(SeqCst), 10);
    }

    #[test]
    fn default_worker_count_is_at_least_one() {
        assert!(default_worker_count(usize::MAX) >= 1);
        let executor = CpuExecutor::new(CpuExecutorOptions::default()).unwrap();
        assert!(executor.worker_count() >= 1);
        assert!(
            CpuExecutor::new(CpuExecutorOptions {
                worker_count: 5000,
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    fn external_wait_blocks_on_the_event() {
        let executor = pool(2);
        let event = executor
            .submit(CpuTaskPriority::Normal, || {
                std::thread::sleep(Duration::from_millis(10));
                Err(Error::new(ErrorCode::TransferFailed, "x"))
            })
            .unwrap();
        assert_eq!(
            executor.wait(&event).unwrap_err().code(),
            ErrorCode::TransferFailed
        );
        assert_eq!(
            executor.wait(&Event::default()).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
    }
}
