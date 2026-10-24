//! Host-only benchmarks: scheduler and task-graph overhead, allocators, the CPU executor, and the
//! serial and four-worker CPU compute baselines that match the GPU workloads.

use kora::memory::{Suballocation, Suballocator};
use kora::scheduler::{SchedulerCostInput, evaluate_scheduler_cost};
use kora::{
    BufferManager, BufferUsage, CpuExecutor, CpuExecutorOptions, CpuTaskPriority, ExecutionPlan, TaskGraph,
    TaskMetadata, TaskType,
};

use crate::report::{Failure, Measurement, Options, Outcome, run, run_checked};

const COMPUTE_SIZES: [usize; 3] = [4096, 65536, 1_048_576];
const VECTOR_SIZES: [usize; 9] = [1, 64, 256, 1024, 4096, 65536, 262_144, 1_048_576, 4_194_240];
const MATRIX_DIMENSIONS: [usize; 3] = [16, 64, 128];

fn ensure(ok: bool, what: &str) -> Outcome {
    if ok {
        Ok(())
    } else {
        Err(Failure::Failed(format!("benchmark workload failed: {what}")))
    }
}

fn scheduler_operation() -> bool {
    let input = SchedulerCostInput {
        dependency_ready_ns: 100.0,
        backend_available_ns: 450.0,
        input_transfer_ns: 120.0,
        execute_ns: 800.0,
        required_output_ns: 90.0,
        synchronization_ns: 30.0,
        predicted_usage_bytes: 128.0,
        effective_budget_bytes: 1024.0,
        memory_penalty_scale_ns: 500.0,
        uncertainty_sigma_ns: 20.0,
        locality_saved_ns: 10.0,
        locality_cap_ns: 50.0,
        ..SchedulerCostInput::default()
    };
    evaluate_scheduler_cost(&input).is_ok()
}

fn task(task_type: TaskType) -> TaskMetadata {
    TaskMetadata {
        task_type,
        ..TaskMetadata::default()
    }
}

fn task_graph_operation() -> bool {
    let mut graph = TaskGraph::new();
    let mut previous = 0;
    for _ in 0..64 {
        let Ok(id) = graph.add_task(task(TaskType::Dispatch)) else {
            return false;
        };
        if previous != 0 && graph.add_dependency(previous, id).is_err() {
            return false;
        }
        previous = id;
    }
    graph.finalize().is_ok()
}

fn callback_graph() -> TaskGraph {
    let mut graph = TaskGraph::new();
    for _ in 0..64 {
        graph
            .add_task(task(TaskType::HostCallback))
            .expect("host callback task");
    }
    graph
}

fn graph_replay_operation() -> bool {
    let Ok(plan) = ExecutionPlan::new(&callback_graph()) else {
        return false;
    };
    let mut invoked = 0;
    plan.replay(|_| {
        invoked += 1;
        Ok(())
    })
    .is_ok()
        && invoked == 64
}

fn allocator_throughput_operation() -> bool {
    let Ok(mut allocator) = Suballocator::new(1 << 20) else {
        return false;
    };
    let mut allocations = Vec::with_capacity(256);
    for index in 0..256_u64 {
        let Ok(allocation) = allocator.allocate(32 + (index % 7) * 16, 16) else {
            return false;
        };
        allocations.push(allocation);
    }
    allocations.iter().all(|a| allocator.deallocate(a).is_ok()) && allocator.validate_invariants()
}

fn allocator_fragmentation_operation() -> bool {
    let Ok(mut allocator) = Suballocator::new(1 << 20) else {
        return false;
    };
    let mut allocations = Vec::with_capacity(128);
    for index in 0..128_u64 {
        let Ok(allocation) = allocator.allocate(64 + (index % 3) * 16, 16) else {
            return false;
        };
        allocations.push(allocation);
    }
    if !allocations
        .iter()
        .step_by(2)
        .all(|a| allocator.deallocate(a).is_ok())
    {
        return false;
    }
    let stats = allocator.stats();
    allocator.validate_invariants() && stats.live_allocations == 64 && stats.external_fragmentation_bytes > 0
}

/// 4096 pseudo-random allocations and frees of 16 B-4 KiB with at most 256 live blocks.
fn allocator_churn_operation() -> bool {
    let Ok(mut allocator) = Suballocator::new(4 << 20) else {
        return false;
    };
    let mut live: Vec<Suballocation> = Vec::with_capacity(256);
    let mut state = 0x2545_F491_u32;
    for _ in 0..4096 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        if live.len() == 256 || (!live.is_empty() && state & 1 == 0) {
            let victim = live.swap_remove(state as usize % live.len());
            if allocator.deallocate(&victim).is_err() {
                return false;
            }
        } else {
            let size = 16 + u64::from(state >> 20) % 4080;
            match allocator.allocate(size, 16) {
                Ok(allocation) => live.push(allocation),
                Err(_) => return false,
            }
        }
    }
    live.iter().all(|a| allocator.deallocate(a).is_ok()) && allocator.validate_invariants()
}

fn allocation_operation() -> bool {
    let bytes = std::hint::black_box(vec![0_u8; 4096]);
    bytes[0] == 0
}

/// Runs `operation(first_item, chunk)` on the executor's workers over `output`, split into one
/// contiguous range of whole items (`item_len` elements each) per worker, and waits for every task.
fn parallel_chunks<T: Send + 'static>(
    executor: &CpuExecutor,
    output: &mut [T],
    item_len: usize,
    operation: &(dyn Fn(usize, &mut [T]) + Sync),
) -> Outcome {
    let items = output.len() / item_len;
    let workers = executor.worker_count();
    // SAFETY: every task submitted below is waited for before this function returns (including
    // on the error paths), so the borrow of `operation` outlives every use.
    let operation: &'static (dyn Fn(usize, &mut [T]) + Sync) = unsafe { std::mem::transmute(operation) };
    let mut rest = output;
    let mut events = Vec::with_capacity(workers);
    let mut failure = None;
    for worker in 0..workers {
        let begin = items * worker / workers;
        let end = items * (worker + 1) / workers;
        let (chunk, tail) = std::mem::take(&mut rest).split_at_mut((end - begin) * item_len);
        rest = tail;
        // SAFETY: as above, the task finishes before `output` is usable again, and the chunks are
        // disjoint, so extending the lifetime cannot create aliasing or dangling access.
        let chunk: &'static mut [T] = unsafe { std::mem::transmute(chunk) };
        match executor.submit(CpuTaskPriority::Normal, move || {
            operation(begin, chunk);
            Ok(())
        }) {
            Ok(event) => events.push(event),
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    let waited: Vec<_> = events.iter().map(|e| executor.wait(e)).collect();
    if let Some(error) = failure {
        return Err(Failure::Failed(format!(
            "CPU executor rejected benchmark work: {error}"
        )));
    }
    if waited.iter().any(Result::is_err) {
        return Err(Failure::Failed("CPU executor benchmark task failed".into()));
    }
    Ok(())
}

/// Times a parallel workload; executor failures abort the benchmark.
fn run_parallel(
    name: String,
    options: &Options,
    threads: usize,
    mut operation: impl FnMut() -> Outcome,
    validate: impl FnMut() -> bool,
) -> Outcome<Measurement> {
    let mut error = None;
    let measured = run_checked(
        name,
        options,
        || {
            if let Err(failure) = operation() {
                error.get_or_insert(failure);
            }
        },
        validate,
    );
    if let Some(failure) = error {
        return Err(failure);
    }
    Ok(measured?.with_threads(threads))
}

fn tiled_matmul(a: &[f32], b: &[f32], rows: std::ops::Range<usize>, dimension: usize, output: &mut [f32]) {
    const TILE: usize = 32;
    output.fill(0.0);
    let first = rows.start;
    for kk in (0..dimension).step_by(TILE) {
        for jj in (0..dimension).step_by(TILE) {
            for i in rows.clone() {
                for k in kk..(kk + TILE).min(dimension) {
                    let aik = a[i * dimension + k];
                    for j in jj..(jj + TILE).min(dimension) {
                        output[(i - first) * dimension + j] += aik * b[k * dimension + j];
                    }
                }
            }
        }
    }
}

fn add_cpu_compute(results: &mut Vec<Measurement>, options: &Options, executor: &CpuExecutor) -> Outcome {
    let threads = executor.worker_count();
    for count in COMPUTE_SIZES {
        let int_left: Vec<u32> = (0..count).map(|i| (i % 17) as u32).collect();
        let int_right: Vec<u32> = (0..count).map(|i| (i % 29) as u32).collect();
        let vector_add = |first: usize, out: &mut [u32]| {
            for (offset, value) in out.iter_mut().enumerate() {
                *value = int_left[first + offset] + int_right[first + offset];
            }
        };
        let output = std::cell::RefCell::new(vec![0_u32; count]);
        results.push(run_parallel(
            format!("cpu_vector_add_parallel_{count}"),
            options,
            threads,
            || parallel_chunks(executor, &mut output.borrow_mut(), 1, &vector_add),
            || {
                output
                    .borrow()
                    .iter()
                    .enumerate()
                    .all(|(i, v)| *v == int_left[i] + int_right[i])
            },
        )?);

        let x: Vec<f32> = (0..count)
            .map(|i| ((i % 97) as i32 - 48) as f32 * 0.125)
            .collect();
        let y: Vec<f32> = (0..count).map(|i| ((i % 53) as i32 - 26) as f32 * 0.25).collect();
        let saxpy = |first: usize, out: &mut [f32]| {
            for (offset, value) in out.iter_mut().enumerate() {
                *value = 1.5 * x[first + offset] + y[first + offset];
            }
        };
        let saxpy_ok = |out: &[f32]| {
            out.iter()
                .enumerate()
                .all(|(i, v)| (v - (1.5 * x[i] + y[i])).abs() <= 1e-6)
        };
        let output = std::cell::RefCell::new(vec![0.0_f32; count]);
        results.push(run_checked(
            format!("cpu_saxpy_{count}"),
            options,
            || saxpy(0, &mut output.borrow_mut()),
            || saxpy_ok(&output.borrow()),
        )?);
        results.push(run_parallel(
            format!("cpu_saxpy_parallel_{count}"),
            options,
            threads,
            || parallel_chunks(executor, &mut output.borrow_mut(), 1, &saxpy),
            || saxpy_ok(&output.borrow()),
        )?);

        const WIDTH: usize = 64;
        let groups = count / WIDTH;
        let reduce = |first: usize, out: &mut [f32]| {
            for (offset, value) in out.iter_mut().enumerate() {
                let group = first + offset;
                *value = x[group * WIDTH..(group + 1) * WIDTH].iter().sum();
            }
        };
        let reduce_ok = |out: &[f32]| {
            out.iter().enumerate().all(|(group, v)| {
                let reference: f64 = x[group * WIDTH..(group + 1) * WIDTH]
                    .iter()
                    .map(|v| f64::from(*v))
                    .sum();
                (f64::from(*v) - reference).abs() <= 1e-3
            })
        };
        let reduced = std::cell::RefCell::new(vec![0.0_f32; groups]);
        results.push(run_checked(
            format!("cpu_reduction_{count}"),
            options,
            || reduce(0, &mut reduced.borrow_mut()),
            || reduce_ok(&reduced.borrow()),
        )?);
        results.push(run_parallel(
            format!("cpu_reduction_parallel_{count}"),
            options,
            threads,
            || parallel_chunks(executor, &mut reduced.borrow_mut(), 1, &reduce),
            || reduce_ok(&reduced.borrow()),
        )?);

        let stencil = |first: usize, out: &mut [f32]| {
            for (offset, value) in out.iter_mut().enumerate() {
                let index = first + offset;
                let mut sum = x[index];
                if index > 0 {
                    sum += x[index - 1];
                }
                if index + 1 < count {
                    sum += x[index + 1];
                }
                *value = sum;
            }
        };
        let stencil_ok = |out: &[f32]| {
            out.iter().enumerate().all(|(index, v)| {
                let mut expected = x[index];
                if index > 0 {
                    expected += x[index - 1];
                }
                if index + 1 < count {
                    expected += x[index + 1];
                }
                (v - expected).abs() <= 1e-6
            })
        };
        let output = std::cell::RefCell::new(vec![0.0_f32; count]);
        results.push(run_checked(
            format!("cpu_stencil_{count}"),
            options,
            || stencil(0, &mut output.borrow_mut()),
            || stencil_ok(&output.borrow()),
        )?);
        results.push(run_parallel(
            format!("cpu_stencil_parallel_{count}"),
            options,
            threads,
            || parallel_chunks(executor, &mut output.borrow_mut(), 1, &stencil),
            || stencil_ok(&output.borrow()),
        )?);

        // Inclusive scan within independent 64-element tiles, matching the GPU scan fixture.
        let scan = |first_tile: usize, out: &mut [u32]| {
            for (tile, values) in out.chunks_mut(WIDTH).enumerate() {
                let base = (first_tile + tile) * WIDTH;
                let mut sum = 0;
                for (lane, value) in values.iter_mut().enumerate() {
                    sum += int_left[base + lane];
                    *value = sum;
                }
            }
        };
        let mut reference = vec![0_u32; count];
        for index in 0..count {
            reference[index] = if index % WIDTH == 0 {
                0
            } else {
                reference[index - 1]
            } + int_left[index];
        }
        let scanned = std::cell::RefCell::new(vec![0_u32; count]);
        results.push(run_checked(
            format!("cpu_tiled_scan_{count}"),
            options,
            || scan(0, &mut scanned.borrow_mut()),
            || *scanned.borrow() == reference,
        )?);
        results.push(run_parallel(
            format!("cpu_tiled_scan_parallel_{count}"),
            options,
            threads,
            || parallel_chunks(executor, &mut scanned.borrow_mut(), WIDTH, &scan),
            || *scanned.borrow() == reference,
        )?);
    }

    for dimension in MATRIX_DIMENSIONS {
        let a = vec![0.5_f32; dimension * dimension];
        let b = vec![0.25_f32; dimension * dimension];
        let expected = 0.125 * dimension as f32;
        let ok = |out: &[f32]| out.iter().all(|v| (v - expected).abs() <= 1e-5);
        let output = std::cell::RefCell::new(vec![0.0_f32; dimension * dimension]);
        results.push(run_checked(
            format!("cpu_tiled_matmul_{dimension}"),
            options,
            || tiled_matmul(&a, &b, 0..dimension, dimension, &mut output.borrow_mut()),
            || ok(&output.borrow()),
        )?);
        let rows = |first: usize, out: &mut [f32]| {
            tiled_matmul(&a, &b, first..first + out.len() / dimension, dimension, out);
        };
        results.push(run_parallel(
            format!("cpu_tiled_matmul_parallel_{dimension}"),
            options,
            threads,
            || parallel_chunks(executor, &mut output.borrow_mut(), dimension, &rows),
            || ok(&output.borrow()),
        )?);
    }
    Ok(())
}

/// Every host benchmark, in report order.
pub fn add_host_benchmarks(results: &mut Vec<Measurement>, options: &Options) -> Outcome {
    let plan = ExecutionPlan::new(&callback_graph()).map_err(crate::report::failed("replay plan"))?;
    let mut callbacks = 0_u64;
    let mut replay_ok = true;
    results.push(run("task_graph_replay_64_reuse_plan", options, || {
        replay_ok &= plan
            .replay(|_| {
                callbacks += 1;
                Ok(())
            })
            .is_ok();
    }));
    ensure(
        replay_ok && callbacks.is_multiple_of(64),
        "task_graph_replay_64_reuse_plan",
    )?;

    type Operation = (&'static str, fn() -> bool);
    let operations: [Operation; 7] = [
        ("scheduler_cost", scheduler_operation),
        ("task_graph_finalize_64", task_graph_operation),
        ("task_graph_replay_64", graph_replay_operation),
        ("allocator_allocate_free_256", allocator_throughput_operation),
        ("allocator_fragmentation_128", allocator_fragmentation_operation),
        ("allocator_churn_4096", allocator_churn_operation),
        ("host_allocation_4k", allocation_operation),
    ];
    for (name, operation) in operations {
        let mut ok = true;
        results.push(run(name, options, || ok &= operation()));
        ensure(ok, name)?;
    }

    let buffers = BufferManager::new();
    let mut created = true;
    results.push(run("buffer_create_drop_4096", options, || {
        created &= buffers.create(4096, BufferUsage::Storage).is_ok();
    }));
    ensure(created && buffers.live_buffers() == 0, "buffer_create_drop_4096")?;

    for count in VECTOR_SIZES {
        let left: Vec<u32> = (0..count).map(|i| (i % 17) as u32).collect();
        let right: Vec<u32> = (0..count).map(|i| (i % 29) as u32).collect();
        let output = std::cell::RefCell::new(vec![0_u32; count]);
        results.push(run_checked(
            format!("cpu_vector_add_{count}"),
            options,
            || {
                for ((out, l), r) in output.borrow_mut().iter_mut().zip(&left).zip(&right) {
                    *out = l + r;
                }
            },
            || {
                output
                    .borrow()
                    .iter()
                    .enumerate()
                    .all(|(i, v)| *v == left[i] + right[i])
            },
        )?);
    }

    let executor = CpuExecutor::new(CpuExecutorOptions {
        worker_count: 4,
        ..CpuExecutorOptions::default()
    })
    .map_err(crate::report::failed("CPU benchmark executor"))?;
    let threads = executor.worker_count();
    let mut executor_ok = true;
    results.push(
        run("cpu_executor_submit_wait", options, || {
            executor_ok &= executor
                .submit(CpuTaskPriority::Normal, || Ok(()))
                .is_ok_and(|event| executor.wait(&event).is_ok());
        })
        .with_threads(threads),
    );
    results.push(
        run("cpu_executor_fanout_1024", options, || {
            let events: Vec<_> = (0..1024)
                .filter_map(|_| executor.submit(CpuTaskPriority::Normal, || Ok(())).ok())
                .collect();
            executor_ok &= events.len() == 1024 && events.iter().all(|e| executor.wait(e).is_ok());
        })
        .with_threads(threads),
    );
    ensure(executor_ok, "cpu_executor")?;
    add_cpu_compute(results, options, &executor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_operations_succeed() {
        assert!(scheduler_operation());
        assert!(task_graph_operation());
        assert!(graph_replay_operation());
        assert!(allocator_throughput_operation());
        assert!(allocator_fragmentation_operation());
        assert!(allocator_churn_operation());
        assert!(allocation_operation());
    }

    #[test]
    fn parallel_chunks_cover_every_item_once() {
        let executor = CpuExecutor::new(CpuExecutorOptions {
            worker_count: 4,
            ..CpuExecutorOptions::default()
        })
        .unwrap();
        let mut output = vec![0_usize; 10 * 3];
        parallel_chunks(&executor, &mut output, 3, &|first, chunk: &mut [usize]| {
            for (offset, value) in chunk.iter_mut().enumerate() {
                *value = first * 3 + offset + 1;
            }
        })
        .unwrap();
        assert_eq!(output, (1..=30).collect::<Vec<_>>());
    }

    #[test]
    fn parallel_matmul_rows_match_the_serial_kernel() {
        let dimension = 64;
        let a: Vec<f32> = (0..dimension * dimension).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..dimension * dimension).map(|i| (i % 5) as f32).collect();
        let mut serial = vec![0.0; dimension * dimension];
        tiled_matmul(&a, &b, 0..dimension, dimension, &mut serial);
        let mut split = vec![0.0; dimension * dimension];
        let (top, bottom) = split.split_at_mut(10 * dimension);
        tiled_matmul(&a, &b, 0..10, dimension, top);
        tiled_matmul(&a, &b, 10..dimension, dimension, bottom);
        assert_eq!(serial, split);
    }
}
