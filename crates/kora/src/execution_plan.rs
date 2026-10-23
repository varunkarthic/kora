//! Immutable, reusable task ordering with allocation-free replay.

use crate::error::{Error, Result, Status};
use crate::task_graph::{TaskGraph, TaskGraphPlan, TaskNode};

/// Snapshot of a finalized [`TaskGraph`]. Dependency analysis happens once at creation; replay
/// walks the stable topological order and calls the caller's executor for each task without
/// allocating.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    plan: TaskGraphPlan,
    /// Topological order as indices into `plan.tasks`.
    order: Vec<usize>,
}

fn attach_task(error: Error, task: &TaskNode) -> Error {
    if error.task_id() == 0 {
        error.with_task(task.id)
    } else {
        error
    }
}

impl ExecutionPlan {
    /// Finalizes `graph` into a plan.
    pub fn new(graph: &TaskGraph) -> Result<ExecutionPlan> {
        let plan = graph.finalize()?;
        let order = plan
            .topological_order
            .iter()
            .map(|&id| (id - 1) as usize)
            .collect();
        Ok(ExecutionPlan { plan, order })
    }

    /// Finalizes `graph` and runs `prepare` once per task in topological order (to resolve and
    /// cache resources). A failure aborts creation and carries the task ID.
    pub fn with_preparation(
        graph: &TaskGraph,
        mut prepare: impl FnMut(&TaskNode) -> Status,
    ) -> Result<ExecutionPlan> {
        let plan = Self::new(graph)?;
        for &index in &plan.order {
            let task = &plan.plan.tasks[index];
            prepare(task).map_err(|error| attach_task(error, task))?;
        }
        Ok(plan)
    }

    /// Calls `executor` for every task in deterministic topological order. Stops at the first
    /// failure, attaching the task ID when the error has none.
    pub fn replay(&self, mut executor: impl FnMut(&TaskNode) -> Status) -> Status {
        for &index in &self.order {
            let task = &self.plan.tasks[index];
            executor(task).map_err(|error| attach_task(error, task))?;
        }
        Ok(())
    }

    /// The finalized graph.
    pub fn graph_plan(&self) -> &TaskGraphPlan {
        &self.plan
    }

    /// Number of tasks.
    pub fn len(&self) -> usize {
        self.plan.tasks.len()
    }

    /// `true` for an empty plan.
    pub fn is_empty(&self) -> bool {
        self.plan.tasks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorCode;
    use crate::task_graph::{TaskMetadata, TaskType};

    fn graph(count: usize) -> TaskGraph {
        let mut graph = TaskGraph::new();
        for index in 0..count {
            graph
                .add_task(TaskMetadata {
                    task_type: TaskType::HostCallback,
                    priority: (index % 4) as i32,
                    ..Default::default()
                })
                .unwrap();
        }
        graph
    }

    #[test]
    fn replays_twenty_node_graph_one_thousand_times_in_order() {
        let mut prepared = 0;
        let plan = ExecutionPlan::with_preparation(&graph(20), |_| {
            prepared += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(plan.len(), 20);
        assert_eq!(prepared, 20);
        let mut calls = 0u64;
        let mut order_valid = true;
        for _ in 0..1000 {
            plan.replay(|task| {
                order_valid &= task.id == calls % 20 + 1;
                calls += 1;
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(calls, 20_000);
        assert!(order_valid);
    }

    #[test]
    fn stops_at_first_failure_and_attaches_task_identity() {
        let plan = ExecutionPlan::new(&graph(5)).unwrap();
        let mut seen = Vec::new();
        let error = plan
            .replay(|task| {
                seen.push(task.id);
                if task.id == 3 {
                    Err(Error::new(ErrorCode::TransferFailed, "synthetic"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::TransferFailed);
        assert_eq!(error.task_id(), 3);
        assert_eq!(seen, [1, 2, 3]);
        let preset = plan
            .replay(|_| Err(Error::new(ErrorCode::Timeout, "x").with_task(99)))
            .unwrap_err();
        assert_eq!(preset.task_id(), 99);
    }

    #[test]
    fn preparation_failure_prevents_plan_creation() {
        let error = ExecutionPlan::with_preparation(&graph(5), |task| {
            if task.id == 3 {
                Err(Error::new(ErrorCode::PipelineCreationFailed, "prepare"))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::PipelineCreationFailed);
        assert_eq!(error.task_id(), 3);
    }

    #[test]
    fn rejects_cyclic_graph_and_accepts_empty_graph() {
        let mut cyclic = graph(2);
        cyclic.add_dependency(1, 2).unwrap();
        cyclic.add_dependency(2, 1).unwrap();
        assert_eq!(
            ExecutionPlan::new(&cyclic).unwrap_err().code(),
            ErrorCode::GraphCycle
        );
        let empty = ExecutionPlan::new(&TaskGraph::new()).unwrap();
        assert!(empty.is_empty());
        empty.replay(|_| unreachable!()).unwrap();
    }
}
