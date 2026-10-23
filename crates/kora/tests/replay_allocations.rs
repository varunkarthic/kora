//! `ExecutionPlan::replay` must not allocate. A counting global allocator observes only the
//! measuring thread, so the test harness's own threads cannot disturb the count.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use kora::{ExecutionPlan, TaskGraph, TaskMetadata};

struct Counting;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: every call is forwarded unchanged to the system allocator; the thread-local counters are
// const-initialized `Cell`s, so touching them never allocates or recurses into the allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.with(Cell::get) {
            ALLOCATIONS.with(|count| count.set(count.get() + 1));
        }
        // SAFETY: forwarded with the caller's valid layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` was allocated by `System` with `layout`.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn replay_is_allocation_free() {
    let mut graph = TaskGraph::new();
    for index in 0..20 {
        graph
            .add_task(TaskMetadata {
                priority: index % 4,
                ..Default::default()
            })
            .unwrap();
    }
    for id in 2..=20 {
        graph.add_dependency(id - 1, id).unwrap();
    }
    let plan = ExecutionPlan::new(&graph).unwrap();
    let mut calls = 0u64;
    TRACKING.with(|t| t.set(true));
    for _ in 0..1000 {
        plan.replay(|task| {
            calls += task.id;
            Ok(())
        })
        .unwrap();
    }
    TRACKING.with(|t| t.set(false));
    assert_eq!(calls, 1000 * 210);
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}
