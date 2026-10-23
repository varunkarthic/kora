//! Conservative temporary-buffer lifetime planning for finalized task DAGs.
//!
//! Two graph temporaries may share one allocation only when every use of one happens-before every
//! use of the other in the DAG. Independent branches are treated as concurrent even when a
//! topological order lists them sequentially. Reachability is computed once as a bitset transitive
//! closure, so each lifetime comparison is a handful of word tests rather than a graph search.

use std::collections::{BTreeMap, HashMap};

use crate::error::{Error, ErrorCode, Result};
use crate::ids::BufferId;
use crate::task_graph::TaskGraphPlan;

/// Requirements of one graph temporary, supplied by the resource-creation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporaryBufferRequirements {
    /// The buffer.
    pub buffer: BufferId,
    /// Size in bytes (non-zero).
    pub size_bytes: u64,
    /// Power-of-two alignment.
    pub alignment: u64,
    /// Exact compatibility class (for example a normalized buffer-usage combination).
    pub usage_class: u32,
    /// Vulkan-style memory-type mask; aliases need a non-empty intersection.
    pub memory_type_bits: u32,
    /// Persistent or user-visible resources get a private allocation.
    pub persistent: bool,
}

/// Where a temporary lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporaryBufferPlacement {
    /// The buffer.
    pub buffer: BufferId,
    /// Index into [`TemporaryBufferAliasingPlan::allocations`].
    pub allocation_id: u32,
    /// Byte offset in the allocation (currently always 0).
    pub offset: u64,
}

/// One backing allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporaryAllocationPlan {
    /// Allocation index.
    pub id: u32,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Strictest alignment of its occupants.
    pub alignment: u64,
    /// Compatibility class.
    pub usage_class: u32,
    /// Intersection of its occupants' memory-type masks.
    pub memory_type_bits: u32,
    /// Reserved for one persistent buffer.
    pub persistent_exclusive: bool,
}

/// Result of [`plan_temporary_buffer_aliasing`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TemporaryBufferAliasingPlan {
    /// One placement per requirement, in input order.
    pub placements: Vec<TemporaryBufferPlacement>,
    /// Backing allocations.
    pub allocations: Vec<TemporaryAllocationPlan>,
    /// Sum of requested sizes.
    pub unaliased_bytes: u64,
    /// Sum of allocation sizes.
    pub planned_bytes: u64,
}

impl TemporaryBufferAliasingPlan {
    /// Bytes saved by aliasing.
    pub fn planned_savings_bytes(&self) -> u64 {
        self.unaliased_bytes.saturating_sub(self.planned_bytes)
    }

    /// Allocation used by `buffer`, if it was planned.
    pub fn allocation_for(&self, buffer: BufferId) -> Option<u32> {
        self.placements
            .iter()
            .find(|p| p.buffer == buffer)
            .map(|p| p.allocation_id)
    }
}

/// Transitive-closure bitsets: `reach[a]` has bit `b` when a path `a -> ... -> b` exists.
struct Reachability {
    words: usize,
    bits: Vec<u64>,
}

impl Reachability {
    fn new(order: &[usize], successors: &[Vec<usize>]) -> Self {
        let count = successors.len();
        let words = count.div_ceil(64);
        let mut bits = vec![0u64; count * words];
        // Reverse topological order: successors are complete before their predecessors.
        for &task in order.iter().rev() {
            for &next in &successors[task] {
                bits[task * words + next / 64] |= 1 << (next % 64);
                for word in 0..words {
                    let value = bits[next * words + word];
                    bits[task * words + word] |= value;
                }
            }
        }
        Self { words, bits }
    }

    fn reaches(&self, from: usize, to: usize) -> bool {
        self.bits[from * self.words + to / 64] & (1 << (to % 64)) != 0
    }
}

struct Lifetime {
    requirements: TemporaryBufferRequirements,
    uses: Vec<usize>,
}

fn wholly_before(left: &Lifetime, right: &Lifetime, reach: &Reachability) -> bool {
    !left.uses.is_empty()
        && !right.uses.is_empty()
        && left
            .uses
            .iter()
            .all(|&l| right.uses.iter().all(|&r| reach.reaches(l, r)))
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorCode::InvalidArgument, message)
}

/// Colors graph-temporary lifetimes into compatible allocation slots.
pub fn plan_temporary_buffer_aliasing(
    graph: &TaskGraphPlan,
    buffers: &[TemporaryBufferRequirements],
) -> Result<TemporaryBufferAliasingPlan> {
    let count = graph.tasks.len();
    if graph.topological_order.len() != count {
        return Err(invalid("task graph plan has an incomplete topological order"));
    }
    let mut position = vec![usize::MAX; count];
    let mut order = Vec::with_capacity(count);
    for (index, &id) in graph.topological_order.iter().enumerate() {
        let slot = id.checked_sub(1).map(|i| i as usize).filter(|&i| i < count);
        let Some(task) = slot.filter(|&i| position[i] == usize::MAX) else {
            return Err(invalid("task graph plan has an invalid topological task ID"));
        };
        if graph.tasks[task].id != id {
            return Err(invalid("task graph plan task IDs are not contiguous"));
        }
        position[task] = index;
        order.push(task);
    }
    let mut successors = vec![Vec::new(); count];
    for (task_index, task) in graph.tasks.iter().enumerate() {
        for &dependency in &task.dependencies {
            let dep = dependency
                .checked_sub(1)
                .map(|i| i as usize)
                .filter(|&i| i < count);
            match dep {
                Some(dep) if position[dep] < position[task_index] => successors[dep].push(task_index),
                _ => return Err(invalid("task graph plan has an invalid dependency order")),
            }
        }
    }

    let mut lifetimes: BTreeMap<BufferId, Lifetime> = BTreeMap::new();
    for requirements in buffers {
        let valid = requirements.buffer.generation != 0
            && requirements.size_bytes != 0
            && requirements.alignment.is_power_of_two()
            && requirements.memory_type_bits != 0;
        if !valid {
            return Err(invalid("temporary buffer requirements are invalid"));
        }
        if lifetimes
            .insert(
                requirements.buffer,
                Lifetime {
                    requirements: *requirements,
                    uses: Vec::new(),
                },
            )
            .is_some()
        {
            return Err(invalid("temporary buffer requirements contain duplicate IDs"));
        }
    }
    for (task_index, task) in graph.tasks.iter().enumerate() {
        for access in &task.metadata.accesses {
            if access.buffer.generation == 0 {
                return Err(invalid("task graph plan contains an invalid buffer generation"));
            }
            if let Some(lifetime) = lifetimes.get_mut(&access.buffer) {
                lifetime.uses.push(task_index);
            }
        }
    }
    if lifetimes.values().any(|lifetime| lifetime.uses.is_empty()) {
        return Err(invalid("temporary buffer is not accessed by the task graph"));
    }
    let reach = Reachability::new(&order, &successors);

    let mut ordered: Vec<&Lifetime> = lifetimes.values().collect();
    ordered.sort_by(|l, r| {
        r.requirements
            .persistent
            .cmp(&l.requirements.persistent)
            .then(r.requirements.size_bytes.cmp(&l.requirements.size_bytes))
            .then(l.requirements.buffer.cmp(&r.requirements.buffer))
    });

    let mut plan = TemporaryBufferAliasingPlan::default();
    let mut occupants: Vec<Vec<&Lifetime>> = Vec::new();
    let mut placement: HashMap<BufferId, u32> = HashMap::new();
    for lifetime in ordered {
        let req = lifetime.requirements;
        plan.unaliased_bytes = plan
            .unaliased_bytes
            .checked_add(req.size_bytes)
            .ok_or_else(|| Error::new(ErrorCode::Overflow, "unaliased temporary memory total overflows"))?;
        let selected = if req.persistent {
            None
        } else {
            plan.allocations
                .iter()
                .enumerate()
                .filter(|(_, a)| {
                    !a.persistent_exclusive
                        && a.usage_class == req.usage_class
                        && a.size_bytes >= req.size_bytes
                        && a.alignment >= req.alignment
                        && a.memory_type_bits & req.memory_type_bits != 0
                })
                .filter(|(index, _)| {
                    occupants[*index].iter().all(|other| {
                        wholly_before(lifetime, other, &reach) || wholly_before(other, lifetime, &reach)
                    })
                })
                .min_by_key(|(_, a)| a.size_bytes)
                .map(|(index, _)| index)
        };
        let id = match selected {
            Some(index) => {
                let allocation = &mut plan.allocations[index];
                allocation.alignment = allocation.alignment.max(req.alignment);
                allocation.memory_type_bits &= req.memory_type_bits;
                occupants[index].push(lifetime);
                allocation.id
            }
            None => {
                let id = u32::try_from(plan.allocations.len()).map_err(|_| {
                    Error::new(ErrorCode::Overflow, "temporary allocation ID space is exhausted")
                })?;
                plan.allocations.push(TemporaryAllocationPlan {
                    id,
                    size_bytes: req.size_bytes,
                    alignment: req.alignment,
                    usage_class: req.usage_class,
                    memory_type_bits: req.memory_type_bits,
                    persistent_exclusive: req.persistent,
                });
                occupants.push(vec![lifetime]);
                id
            }
        };
        placement.insert(req.buffer, id);
    }
    plan.planned_bytes = plan
        .allocations
        .iter()
        .try_fold(0u64, |sum, a| sum.checked_add(a.size_bytes))
        .ok_or_else(|| Error::new(ErrorCode::Overflow, "planned temporary memory total overflows"))?;
    plan.placements = buffers
        .iter()
        .map(|r| TemporaryBufferPlacement {
            buffer: r.buffer,
            allocation_id: placement[&r.buffer],
            offset: 0,
        })
        .collect();
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plan::ExecutionPlan;
    use crate::task_graph::tests::task;
    use crate::task_graph::{AccessMode::*, TaskGraph};

    const A: BufferId = BufferId::new(1, 1);
    const B: BufferId = BufferId::new(2, 1);
    const C: BufferId = BufferId::new(3, 1);

    fn requirement(buffer: BufferId, size_bytes: u64) -> TemporaryBufferRequirements {
        TemporaryBufferRequirements {
            buffer,
            size_bytes,
            alignment: 16,
            usage_class: 7,
            memory_type_bits: 0b11,
            persistent: false,
        }
    }

    fn chain_graph() -> TaskGraph {
        let mut graph = TaskGraph::new();
        graph.add_task(task(&[(A, Write)])).unwrap();
        graph.add_task(task(&[(A, Read), (B, Write)])).unwrap();
        graph.add_task(task(&[(B, Read), (C, Write)])).unwrap();
        graph.add_task(task(&[(C, Read)])).unwrap();
        graph
    }

    #[test]
    fn colors_non_overlapping_chain_lifetimes() {
        let graph = chain_graph().finalize().unwrap();
        let plan = plan_temporary_buffer_aliasing(
            &graph,
            &[requirement(A, 64), requirement(B, 64), requirement(C, 64)],
        )
        .unwrap();
        assert_eq!(plan.allocations.len(), 2);
        assert_eq!(
            (
                plan.unaliased_bytes,
                plan.planned_bytes,
                plan.planned_savings_bytes()
            ),
            (192, 128, 64)
        );
        assert_eq!(plan.allocation_for(A), plan.allocation_for(C));
        assert_ne!(plan.allocation_for(A), plan.allocation_for(B));
        assert_ne!(plan.allocation_for(B), plan.allocation_for(C));
    }

    #[test]
    fn keeps_persistent_and_incompatible_requirements_separate() {
        let mut graph = TaskGraph::new();
        for buffer in [A, B, C] {
            graph.add_task(task(&[(buffer, Write)])).unwrap();
            graph.add_task(task(&[(buffer, Read)])).unwrap();
        }
        let finalized = graph.finalize().unwrap();
        let persistent = TemporaryBufferRequirements {
            persistent: true,
            ..requirement(A, 64)
        };
        let different_usage = TemporaryBufferRequirements {
            usage_class: 8,
            ..requirement(B, 64)
        };
        let different_memory = TemporaryBufferRequirements {
            memory_type_bits: 0b100,
            ..requirement(C, 64)
        };
        let plan =
            plan_temporary_buffer_aliasing(&finalized, &[persistent, different_usage, different_memory])
                .unwrap();
        assert_eq!(plan.allocations.len(), 3);
        assert!(plan.allocations[plan.allocation_for(A).unwrap() as usize].persistent_exclusive);
    }

    #[test]
    fn does_not_alias_independent_concurrent_branches() {
        let mut graph = TaskGraph::new();
        graph.add_task(task(&[(A, Write)])).unwrap();
        graph.add_task(task(&[(B, Write)])).unwrap();
        let plan = plan_temporary_buffer_aliasing(
            &graph.finalize().unwrap(),
            &[requirement(A, 64), requirement(B, 64)],
        )
        .unwrap();
        assert_eq!(plan.allocations.len(), 2);
    }

    #[test]
    fn rejects_invalid_and_unreferenced_requirements() {
        let graph = chain_graph().finalize().unwrap();
        for bad in [
            requirement(A, 0),
            TemporaryBufferRequirements {
                alignment: 3,
                ..requirement(A, 64)
            },
            TemporaryBufferRequirements {
                memory_type_bits: 0,
                ..requirement(A, 64)
            },
            requirement(BufferId::new(1, 0), 64),
        ] {
            assert!(plan_temporary_buffer_aliasing(&graph, &[bad]).is_err());
        }
        assert!(
            plan_temporary_buffer_aliasing(&graph, &[])
                .unwrap()
                .allocations
                .is_empty()
        );
        assert!(plan_temporary_buffer_aliasing(&graph, &[requirement(A, 64), requirement(A, 64)]).is_err());
        assert!(plan_temporary_buffer_aliasing(&graph, &[requirement(BufferId::new(99, 1), 64)]).is_err());
    }

    #[test]
    fn rejects_malformed_plans() {
        let mut graph = chain_graph().finalize().unwrap();
        graph.topological_order.pop();
        assert!(plan_temporary_buffer_aliasing(&graph, &[]).is_err());
        let mut graph = chain_graph().finalize().unwrap();
        graph.topological_order.swap(0, 1);
        assert!(plan_temporary_buffer_aliasing(&graph, &[]).is_err());
        let mut graph = chain_graph().finalize().unwrap();
        graph.topological_order[0] = 9;
        assert!(plan_temporary_buffer_aliasing(&graph, &[]).is_err());
    }

    #[test]
    fn simulated_execution_preserves_results_with_reduced_peak_storage() {
        let task_graph = chain_graph();
        let plan = plan_temporary_buffer_aliasing(
            &task_graph.finalize().unwrap(),
            &[requirement(A, 32), requirement(B, 32), requirement(C, 32)],
        )
        .unwrap();
        let execution = ExecutionPlan::new(&task_graph).unwrap();
        let mut backing = vec![vec![0u32; 8]; plan.allocations.len()];
        let slot = |id: BufferId| plan.allocation_for(id).unwrap() as usize;
        let mut matches = false;
        execution
            .replay(|node| {
                match node.id {
                    1 => backing[slot(A)]
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32 * 3 + 1),
                    2 => backing[slot(B)] = backing[slot(A)].clone(),
                    3 => backing[slot(C)] = backing[slot(B)].clone(),
                    _ => {
                        matches = backing[slot(C)]
                            .iter()
                            .enumerate()
                            .all(|(i, &v)| v == i as u32 * 3 + 1)
                    }
                }
                Ok(())
            })
            .unwrap();
        assert!(matches);
        assert_eq!(plan.planned_bytes, 64);
    }

    #[test]
    fn long_chains_alias_into_two_slots() {
        let mut graph = TaskGraph::new();
        let ids: Vec<BufferId> = (1..=200).map(|i| BufferId::new(i, 1)).collect();
        graph.add_task(task(&[(ids[0], Write)])).unwrap();
        for pair in ids.windows(2) {
            graph
                .add_task(task(&[(pair[0], Read), (pair[1], Write)]))
                .unwrap();
        }
        graph.add_task(task(&[(ids[199], Read)])).unwrap();
        let requirements: Vec<_> = ids.iter().map(|&id| requirement(id, 128)).collect();
        let plan = plan_temporary_buffer_aliasing(&graph.finalize().unwrap(), &requirements).unwrap();
        assert_eq!(plan.allocations.len(), 2);
    }
}
