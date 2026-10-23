//! Task metadata and deterministic dependency inference for reusable task graphs.
//!
//! [`TaskGraph`] infers read-after-write, write-after-read, and write-after-write edges from each
//! task's declared buffer accesses, accepts explicit ordering edges, and [`TaskGraph::finalize`]s
//! into an immutable [`TaskGraphPlan`] with a deterministic topological order (Kahn's algorithm,
//! lowest ready ID first). The graph stores metadata only; execution is up to the caller (see
//! [`ExecutionPlan`](crate::execution_plan::ExecutionPlan)).

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};

use crate::error::{Error, ErrorCode, Result, Status};
use crate::ids::BufferId;

/// Task identifier (1-based, dense).
pub type TaskId = u64;

/// Kind of operation a task performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TaskType {
    /// Host to device copy.
    Upload,
    /// Device to host copy.
    Download,
    /// Device to device copy.
    Copy,
    /// Buffer fill.
    Fill,
    /// Kernel dispatch.
    Dispatch,
    /// CPU computation.
    CpuTask,
    /// Arbitrary host callback.
    #[default]
    HostCallback,
}

/// How a task accesses a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AccessMode {
    /// Read only.
    #[default]
    Read,
    /// Write only.
    Write,
    /// Read and write.
    ReadWrite,
}

impl AccessMode {
    /// `true` for `Read` and `ReadWrite`.
    pub const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }
    /// `true` for `Write` and `ReadWrite`.
    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
    /// The union of two access modes.
    pub const fn union(self, other: AccessMode) -> AccessMode {
        match (self.reads() || other.reads(), self.writes() || other.writes()) {
            (true, true) => Self::ReadWrite,
            (false, true) => Self::Write,
            _ => Self::Read,
        }
    }
}

/// One declared buffer access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferAccess {
    /// The buffer.
    pub buffer: BufferId,
    /// Access mode.
    pub mode: AccessMode,
}

/// Scheduler-facing information for one operation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TaskMetadata {
    /// Operation kind.
    pub task_type: TaskType,
    /// Kernel identifier for dispatches.
    pub kernel_id: String,
    /// CPU implementation key.
    pub cpu_function_id: String,
    /// Declared buffer accesses (repeated buffers are merged).
    pub accesses: Vec<BufferAccess>,
    /// Problem size.
    pub problem_dimensions: [u64; 3],
    /// Caller priority.
    pub priority: i32,
    /// Soft deadline in nanoseconds.
    pub deadline_hint_ns: Option<u64>,
    /// Devices this task may run on (empty = any).
    pub allowed_devices: Vec<String>,
    /// Preferred device.
    pub preferred_device: Option<String>,
    /// A CPU implementation exists.
    pub cpu_fallback_available: bool,
    /// Estimated bytes read.
    pub estimated_bytes_read: u64,
    /// Estimated bytes written.
    pub estimated_bytes_written: u64,
    /// Estimated work units.
    pub estimated_work_units: u64,
}

/// A task in a finalized plan.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TaskNode {
    /// Task identifier.
    pub id: TaskId,
    /// Task metadata.
    pub metadata: TaskMetadata,
    /// Prerequisite task IDs, sorted and unique.
    pub dependencies: Vec<TaskId>,
}

/// Immutable graph snapshot.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TaskGraphPlan {
    /// Tasks indexed by `id - 1`.
    pub tasks: Vec<TaskNode>,
    /// Deterministic topological order; ties resolve by task ID.
    pub topological_order: Vec<TaskId>,
}

#[derive(Default)]
struct BufferState {
    last_writer: Option<TaskId>,
    active_readers: Vec<TaskId>,
}

/// Mutable graph builder with automatic RAW/WAR/WAW edges and explicit dependencies.
#[derive(Default)]
pub struct TaskGraph {
    tasks: Vec<TaskNode>,
    explicit: Vec<(TaskId, TaskId)>,
    explicit_seen: HashSet<(TaskId, TaskId)>,
    buffers: HashMap<BufferId, BufferState>,
}

impl std::fmt::Debug for TaskGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskGraph")
            .field("tasks", &self.tasks.len())
            .field("explicit", &self.explicit.len())
            .finish()
    }
}

impl TaskGraph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of tasks.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// `true` when no task was added.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Adds a task and infers its dependencies from its buffer accesses.
    pub fn add_task(&mut self, metadata: TaskMetadata) -> Result<TaskId> {
        // Fold repeated declarations of one buffer into a single access mode.
        let mut accesses: BTreeMap<BufferId, AccessMode> = BTreeMap::new();
        for access in &metadata.accesses {
            if access.buffer.generation == 0 {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    "task access has an invalid buffer generation",
                ));
            }
            accesses
                .entry(access.buffer)
                .and_modify(|mode| *mode = mode.union(access.mode))
                .or_insert(access.mode);
        }
        let id = self.tasks.len() as TaskId + 1;
        let mut dependencies = BTreeSet::new();
        for (buffer, mode) in &accesses {
            if let Some(state) = self.buffers.get(buffer) {
                if let Some(writer) = state.last_writer {
                    dependencies.insert(writer);
                }
                if mode.writes() {
                    dependencies.extend(state.active_readers.iter().copied());
                }
            }
        }
        for (buffer, mode) in accesses {
            let state = self.buffers.entry(buffer).or_default();
            if mode.writes() {
                state.last_writer = Some(id);
                state.active_readers.clear();
            } else {
                state.active_readers.push(id);
            }
        }
        self.tasks.push(TaskNode {
            id,
            metadata,
            dependencies: dependencies.into_iter().collect(),
        });
        Ok(id)
    }

    /// Adds an ordering edge `prerequisite -> dependent`. Cycles are reported by [`finalize`](Self::finalize).
    pub fn add_dependency(&mut self, prerequisite: TaskId, dependent: TaskId) -> Status {
        let count = self.tasks.len() as TaskId;
        if prerequisite == 0 || dependent == 0 || prerequisite > count || dependent > count {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "task dependency references an unknown task ID",
            ));
        }
        if self.explicit_seen.insert((prerequisite, dependent)) {
            self.explicit.push((prerequisite, dependent));
        }
        Ok(())
    }

    /// Validates acyclicity and returns the immutable plan in deterministic topological order.
    /// A cycle yields [`ErrorCode::GraphCycle`] carrying one member's task ID.
    pub fn finalize(&self) -> Result<TaskGraphPlan> {
        let count = self.tasks.len();
        let mut dependencies: Vec<BTreeSet<TaskId>> = self
            .tasks
            .iter()
            .map(|task| task.dependencies.iter().copied().collect())
            .collect();
        for &(prerequisite, dependent) in &self.explicit {
            dependencies[(dependent - 1) as usize].insert(prerequisite);
        }
        let mut successors: Vec<Vec<TaskId>> = vec![Vec::new(); count];
        let mut indegree = vec![0usize; count];
        let mut tasks = self.tasks.clone();
        for (index, deps) in dependencies.into_iter().enumerate() {
            indegree[index] = deps.len();
            for &prerequisite in &deps {
                successors[(prerequisite - 1) as usize].push(index as TaskId + 1);
            }
            tasks[index].dependencies = deps.into_iter().collect();
        }
        let mut ready: BinaryHeap<Reverse<TaskId>> = (1..=count as TaskId)
            .filter(|id| indegree[(*id - 1) as usize] == 0)
            .map(Reverse)
            .collect();
        let mut order = Vec::with_capacity(count);
        while let Some(Reverse(id)) = ready.pop() {
            order.push(id);
            for &successor in &successors[(id - 1) as usize] {
                let degree = &mut indegree[(successor - 1) as usize];
                *degree -= 1;
                if *degree == 0 {
                    ready.push(Reverse(successor));
                }
            }
        }
        if order.len() != count {
            let member = indegree
                .iter()
                .position(|&degree| degree != 0)
                .map_or(0, |i| i as TaskId + 1);
            return Err(
                Error::new(ErrorCode::GraphCycle, "task graph contains a dependency cycle").with_task(member),
            );
        }
        Ok(TaskGraphPlan {
            tasks,
            topological_order: order,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Small deterministic generator for randomized tests (SplitMix64).
    pub(crate) struct SplitMix(pub u64);
    impl SplitMix {
        pub(crate) fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        pub(crate) fn below(&mut self, bound: u64) -> u64 {
            self.next() % bound
        }
    }

    pub(crate) fn task(accesses: &[(BufferId, AccessMode)]) -> TaskMetadata {
        TaskMetadata {
            accesses: accesses
                .iter()
                .map(|&(buffer, mode)| BufferAccess { buffer, mode })
                .collect(),
            ..Default::default()
        }
    }

    fn deps(plan: &TaskGraphPlan, id: TaskId) -> BTreeSet<TaskId> {
        plan.tasks
            .iter()
            .find(|t| t.id == id)
            .map(|t| t.dependencies.iter().copied().collect())
            .unwrap_or_default()
    }

    fn set(ids: &[TaskId]) -> BTreeSet<TaskId> {
        ids.iter().copied().collect()
    }

    #[test]
    fn infers_raw_war_waw_and_read_write_edges() {
        let buffer = BufferId::new(3, 17);
        let mut graph = TaskGraph::new();
        let writer = graph.add_task(task(&[(buffer, AccessMode::Write)])).unwrap();
        let reader_a = graph.add_task(task(&[(buffer, AccessMode::Read)])).unwrap();
        let reader_b = graph.add_task(task(&[(buffer, AccessMode::Read)])).unwrap();
        let writer2 = graph.add_task(task(&[(buffer, AccessMode::Write)])).unwrap();
        let read_writer = graph.add_task(task(&[(buffer, AccessMode::ReadWrite)])).unwrap();
        let reader_c = graph.add_task(task(&[(buffer, AccessMode::Read)])).unwrap();
        let plan = graph.finalize().unwrap();
        assert!(deps(&plan, writer).is_empty());
        assert_eq!(deps(&plan, reader_a), set(&[writer]));
        assert_eq!(deps(&plan, reader_b), set(&[writer]));
        assert_eq!(deps(&plan, writer2), set(&[writer, reader_a, reader_b]));
        assert_eq!(deps(&plan, read_writer), set(&[writer2]));
        assert_eq!(deps(&plan, reader_c), set(&[read_writer]));
        assert_eq!(
            plan.topological_order,
            vec![writer, reader_a, reader_b, writer2, read_writer, reader_c]
        );
    }

    #[test]
    fn combines_duplicate_accesses_and_distinguishes_generations() {
        let mut graph = TaskGraph::new();
        let old = BufferId::new(0, 1);
        let recycled = BufferId::new(0, 2);
        let initial = graph.add_task(task(&[(old, AccessMode::Write)])).unwrap();
        let combined = graph
            .add_task(task(&[(old, AccessMode::Read), (old, AccessMode::Write)]))
            .unwrap();
        let independent = graph.add_task(task(&[(recycled, AccessMode::Read)])).unwrap();
        let after = graph.add_task(task(&[(old, AccessMode::Read)])).unwrap();
        let plan = graph.finalize().unwrap();
        assert_eq!(deps(&plan, combined), set(&[initial]));
        assert!(deps(&plan, independent).is_empty());
        // The merged access was a write, so the next reader depends on it.
        assert_eq!(deps(&plan, after), set(&[combined]));
    }

    #[test]
    fn preserves_metadata_for_every_task_type() {
        let types = [
            TaskType::Upload,
            TaskType::Download,
            TaskType::Copy,
            TaskType::Fill,
            TaskType::Dispatch,
            TaskType::CpuTask,
            TaskType::HostCallback,
        ];
        let mut graph = TaskGraph::new();
        for task_type in types {
            let metadata = TaskMetadata {
                task_type,
                kernel_id: "kernel".into(),
                cpu_function_id: "cpu-function".into(),
                problem_dimensions: [2, 3, 4],
                priority: 9,
                deadline_hint_ns: Some(1234),
                allowed_devices: vec!["device-a".into(), "device-b".into()],
                preferred_device: Some("device-a".into()),
                cpu_fallback_available: true,
                estimated_bytes_read: 64,
                estimated_bytes_written: 32,
                estimated_work_units: 10,
                ..Default::default()
            };
            graph.add_task(metadata).unwrap();
        }
        let plan = graph.finalize().unwrap();
        assert_eq!(plan.tasks.len(), types.len());
        for (node, task_type) in plan.tasks.iter().zip(types) {
            let m = &node.metadata;
            assert_eq!(m.task_type, task_type);
            assert_eq!(m.kernel_id, "kernel");
            assert_eq!(m.problem_dimensions, [2, 3, 4]);
            assert_eq!(m.deadline_hint_ns, Some(1234));
            assert_eq!(m.allowed_devices, ["device-a", "device-b"]);
            assert_eq!(m.preferred_device.as_deref(), Some("device-a"));
            assert!(m.cpu_fallback_available);
            assert_eq!(
                (
                    m.estimated_bytes_read,
                    m.estimated_bytes_written,
                    m.estimated_work_units
                ),
                (64, 32, 10)
            );
        }
    }

    #[test]
    fn explicit_dependencies_are_deduplicated_and_unknown_ids_rejected() {
        let mut graph = TaskGraph::new();
        let first = graph.add_task(task(&[])).unwrap();
        let second = graph.add_task(task(&[])).unwrap();
        graph.add_dependency(first, second).unwrap();
        graph.add_dependency(first, second).unwrap();
        assert!(graph.add_dependency(0, second).is_err());
        assert!(graph.add_dependency(first, second + 1).is_err());
        let plan = graph.finalize().unwrap();
        assert_eq!(deps(&plan, second), set(&[first]));
    }

    #[test]
    fn reports_explicit_self_and_multi_task_cycles() {
        for self_cycle in [true, false] {
            let mut graph = TaskGraph::new();
            let first = graph.add_task(task(&[])).unwrap();
            let second = graph.add_task(task(&[])).unwrap();
            graph.add_dependency(first, second).unwrap();
            graph
                .add_dependency(if self_cycle { first } else { second }, first)
                .unwrap();
            let error = graph.finalize().unwrap_err();
            assert_eq!(error.code(), ErrorCode::GraphCycle);
            assert_ne!(error.task_id(), 0);
        }
    }

    #[test]
    fn random_access_sequences_match_independent_hazard_oracle() {
        let modes = [AccessMode::Read, AccessMode::Write, AccessMode::ReadWrite];
        let mut random = SplitMix(0x4b4f_5241);
        for sequence in 0..80 {
            let mut graph = TaskGraph::new();
            let mut expected: Vec<BTreeSet<TaskId>> = vec![BTreeSet::new(); 96];
            let mut hazards: BTreeMap<u32, (Option<TaskId>, BTreeSet<TaskId>)> = BTreeMap::new();
            for (index, expected) in expected.iter_mut().enumerate() {
                let mut accesses: BTreeMap<u32, AccessMode> = BTreeMap::new();
                for _ in 0..1 + random.below(4) {
                    let buffer = random.below(7) as u32;
                    let mode = modes[random.below(3) as usize];
                    accesses
                        .entry(buffer)
                        .and_modify(|m| *m = m.union(mode))
                        .or_insert(mode);
                }
                let mut metadata = TaskMetadata::default();
                let id = index as TaskId + 1;
                for (&buffer, &mode) in &accesses {
                    metadata.accesses.push(BufferAccess {
                        buffer: BufferId::new(buffer, 1),
                        mode,
                    });
                    let (writer, readers) = hazards.entry(buffer).or_default();
                    if let Some(w) = *writer {
                        expected.insert(w);
                    }
                    if mode.writes() {
                        expected.extend(readers.iter().copied());
                        *writer = Some(id);
                        readers.clear();
                    } else {
                        readers.insert(id);
                    }
                }
                graph.add_task(metadata).unwrap();
            }
            let plan = graph.finalize().unwrap();
            for (index, expected) in expected.iter().enumerate() {
                assert_eq!(
                    &deps(&plan, index as TaskId + 1),
                    expected,
                    "sequence {sequence} task {index}"
                );
            }
        }
    }

    #[test]
    fn random_explicit_dags_respect_every_edge_and_stay_deterministic() {
        let mut random = SplitMix(0x4752_4150);
        for _ in 0..64 {
            let mut graph = TaskGraph::new();
            for _ in 0..72 {
                graph.add_task(task(&[])).unwrap();
            }
            let mut edges = Vec::new();
            for before in 1..=72 {
                for after in before + 1..=72 {
                    if random.below(19) == 0 {
                        graph.add_dependency(before, after).unwrap();
                        edges.push((before, after));
                    }
                }
            }
            let plan = graph.finalize().unwrap();
            let mut position = vec![0usize; 73];
            for (index, &id) in plan.topological_order.iter().enumerate() {
                position[id as usize] = index;
            }
            for (before, after) in edges {
                assert!(position[before as usize] < position[after as usize]);
            }
            assert_eq!(
                graph.finalize().unwrap().topological_order,
                plan.topological_order
            );
        }
    }

    #[test]
    fn rejects_zero_generation_without_mutating_the_graph() {
        let mut graph = TaskGraph::new();
        let error = graph
            .add_task(task(&[(BufferId::new(1, 0), AccessMode::Read)]))
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidArgument);
        assert!(graph.is_empty());
    }

    #[test]
    fn access_mode_union_table() {
        use AccessMode::*;
        assert_eq!(Read.union(Read), Read);
        assert_eq!(Write.union(Write), Write);
        assert_eq!(Read.union(Write), ReadWrite);
        assert_eq!(Write.union(ReadWrite), ReadWrite);
    }
}
