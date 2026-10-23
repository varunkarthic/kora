//! Generation-aware LRU-2 eviction planning. The planner performs no I/O: callers execute
//! writebacks and revalidate buffer lifetime before freeing anything.

use std::collections::{HashMap, HashSet};

use crate::error::{Error, ErrorCode, Result, Status};
use crate::ids::BufferId;

/// Fixed-size LRU-K history (K = 2). One recent access alone earns no recency bonus.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AccessHistory {
    ticks: [u64; AccessHistory::DISTANCE],
    next: usize,
    size: usize,
}

impl AccessHistory {
    /// K.
    pub const DISTANCE: usize = 2;

    /// Records an access; ticks must not go backwards.
    pub fn record(&mut self, tick: u64) -> Status {
        if self.most_recent_tick().is_some_and(|recent| tick < recent) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "buffer access ticks must be monotonic",
            ));
        }
        self.ticks[self.next] = tick;
        self.next = (self.next + 1) % Self::DISTANCE;
        self.size = (self.size + 1).min(Self::DISTANCE);
        Ok(())
    }
    /// Recorded accesses (at most K).
    pub fn len(&self) -> usize {
        self.size
    }
    /// `true` when nothing was recorded.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    /// Latest tick.
    pub fn most_recent_tick(&self) -> Option<u64> {
        (self.size != 0).then(|| self.ticks[(self.next + Self::DISTANCE - 1) % Self::DISTANCE])
    }
    /// Age of the K-th most recent access.
    pub fn kth_recent_age(&self, now: u64) -> Option<u64> {
        if self.size < Self::DISTANCE {
            return None;
        }
        now.checked_sub(self.ticks[self.next])
    }
}

/// Keep-value weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvictionPolicyConfig {
    /// Weight of the LRU-2 recency bonus.
    pub recent_access_weight: f64,
    /// Penalty per idle tick.
    pub idle_age_penalty_per_tick: f64,
}

impl Default for EvictionPolicyConfig {
    fn default() -> Self {
        Self {
            recent_access_weight: 1.0,
            idle_age_penalty_per_tick: 0.01,
        }
    }
}

/// One resident replica that might be evicted.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EvictionCandidate {
    /// Buffer identity (slot + generation).
    pub id: BufferId,
    /// Bytes freed by eviction.
    pub resident_bytes: u64,
    /// Probability of reuse in `[0, 1]`.
    pub reuse_probability: f64,
    /// Cost to reload.
    pub estimated_reload_cost_ms: f64,
    /// Caller priority bonus.
    pub user_priority_bonus: f64,
    /// Bonus for expensive-to-recreate data.
    pub expensive_recreation_bonus: f64,
    /// Cost to write device-newer data back.
    pub estimated_writeback_cost_ms: f64,
    /// Last access tick.
    pub last_access_tick: u64,
    /// Pins (never evicted while pinned).
    pub pin_count: u32,
    /// GPU work in flight.
    pub in_flight: bool,
    /// Being recorded into a submission.
    pub being_submitted: bool,
    /// The device copy is newer than the host.
    pub device_newer: bool,
    /// The data may be discarded.
    pub disposable: bool,
    /// Another current copy exists.
    pub alternate_current_copy: bool,
}

/// What to do with a selected candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictionAction {
    /// Drop the replica.
    Evict,
    /// Copy device-newer data to the host first.
    WritebackThenEvict,
}

/// One planned eviction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvictionDecision {
    /// Buffer identity.
    pub id: BufferId,
    /// Bytes freed.
    pub resident_bytes: u64,
    /// Its keep value.
    pub keep_value: f64,
    /// Action.
    pub action: EvictionAction,
}

/// Evictions that free at least the requested bytes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EvictionPlan {
    /// Total bytes freed.
    pub bytes_to_reclaim: u64,
    /// Decisions, lowest keep value first.
    pub decisions: Vec<EvictionDecision>,
}

/// Deterministic keep-value planner.
#[derive(Debug, Clone, Default)]
pub struct EvictionPolicy {
    config: EvictionPolicyConfig,
    histories: HashMap<BufferId, AccessHistory>,
}

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

impl EvictionPolicy {
    /// Creates a planner.
    pub fn new(config: EvictionPolicyConfig) -> Self {
        Self {
            config,
            histories: HashMap::new(),
        }
    }

    /// Records an access.
    pub fn record_access(&mut self, id: BufferId, tick: u64) -> Status {
        self.histories.entry(id).or_default().record(tick)
    }

    /// Forgets a released buffer.
    pub fn forget(&mut self, id: BufferId) {
        self.histories.remove(&id);
    }

    fn writeback_required(candidate: &EvictionCandidate) -> bool {
        candidate.device_newer && !candidate.disposable && !candidate.alternate_current_copy
    }

    /// `reuse * reload + recency + bonuses + writeback - idle penalty`.
    pub fn keep_value(&self, candidate: &EvictionCandidate, now: u64) -> Result<f64> {
        let c = &self.config;
        let valid = finite_nonnegative(c.recent_access_weight)
            && finite_nonnegative(c.idle_age_penalty_per_tick)
            && (0.0..=1.0).contains(&candidate.reuse_probability)
            && finite_nonnegative(candidate.estimated_reload_cost_ms)
            && finite_nonnegative(candidate.user_priority_bonus)
            && finite_nonnegative(candidate.expensive_recreation_bonus)
            && finite_nonnegative(candidate.estimated_writeback_cost_ms);
        if !valid {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "eviction keep-value inputs must be finite and in range",
            ));
        }
        let mut last_access = candidate.last_access_tick;
        let mut recent_bonus = 0.0;
        if let Some(history) = self.histories.get(&candidate.id) {
            if let Some(recent) = history.most_recent_tick() {
                last_access = recent;
            }
            if let Some(age) = history.kth_recent_age(now) {
                recent_bonus = c.recent_access_weight / (1.0 + age as f64);
            }
        }
        let idle = now.checked_sub(last_access).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidArgument,
                "buffer last-access tick is later than the planning tick",
            )
        })?;
        let writeback = if Self::writeback_required(candidate) {
            candidate.estimated_writeback_cost_ms
        } else {
            0.0
        };
        let value = candidate.reuse_probability * candidate.estimated_reload_cost_ms
            + recent_bonus
            + candidate.user_priority_bonus
            + candidate.expensive_recreation_bonus
            + writeback
            - c.idle_age_penalty_per_tick * idle as f64;
        if !value.is_finite() {
            return Err(Error::new(
                ErrorCode::Overflow,
                "eviction keep-value arithmetic overflow",
            ));
        }
        Ok(value)
    }

    /// Plans the cheapest set of evictions freeing `requested - available` bytes. Pinned,
    /// in-flight, and being-submitted replicas are never selected; no partial plan is returned.
    pub fn plan(
        &self,
        candidates: &[EvictionCandidate],
        available: u64,
        requested: u64,
        now: u64,
    ) -> Result<EvictionPlan> {
        let mut plan = EvictionPlan::default();
        if requested <= available {
            return Ok(plan);
        }
        let needed = requested - available;
        let mut seen = HashSet::with_capacity(candidates.len());
        let mut eligible = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if candidate.resident_bytes == 0 {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    "eviction candidates must own resident bytes",
                ));
            }
            if !seen.insert(candidate.id) {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    "eviction candidate IDs must be unique",
                ));
            }
            if candidate.pin_count != 0 || candidate.in_flight || candidate.being_submitted {
                continue;
            }
            eligible.push(EvictionDecision {
                id: candidate.id,
                resident_bytes: candidate.resident_bytes,
                keep_value: self.keep_value(candidate, now)?,
                action: if Self::writeback_required(candidate) {
                    EvictionAction::WritebackThenEvict
                } else {
                    EvictionAction::Evict
                },
            });
        }
        eligible.sort_by(|l, r| l.keep_value.total_cmp(&r.keep_value).then(l.id.cmp(&r.id)));
        for decision in eligible {
            plan.bytes_to_reclaim = plan
                .bytes_to_reclaim
                .checked_add(decision.resident_bytes)
                .ok_or_else(|| Error::new(ErrorCode::Overflow, "eviction plan byte count overflow"))?;
            plan.decisions.push(decision);
            if plan.bytes_to_reclaim >= needed {
                return Ok(plan);
            }
        }
        Err(Error::new(
            ErrorCode::OutOfMemoryBudget,
            "eligible buffers cannot satisfy the requested memory budget",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(slot: u32, bytes: u64, probability: f64) -> EvictionCandidate {
        EvictionCandidate {
            id: BufferId::new(slot, 1),
            resident_bytes: bytes,
            reuse_probability: probability,
            estimated_reload_cost_ms: 100.0,
            ..Default::default()
        }
    }

    #[test]
    fn history_requires_two_references_and_rejects_backward_time() {
        let mut history = AccessHistory::default();
        history.record(2).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history.kth_recent_age(10), None);
        assert_eq!(history.record(1).unwrap_err().code(), ErrorCode::InvalidArgument);
        history.record(4).unwrap();
        history.record(7).unwrap();
        assert_eq!(history.len(), AccessHistory::DISTANCE);
        assert_eq!(history.most_recent_tick(), Some(7));
        assert_eq!(history.kth_recent_age(10), Some(6));
    }

    #[test]
    fn keep_value_matches_weighted_formula_and_lru_k_recency() {
        let mut policy = EvictionPolicy::new(EvictionPolicyConfig {
            recent_access_weight: 2.0,
            idle_age_penalty_per_tick: 0.5,
        });
        let id = BufferId::new(3, 9);
        policy.record_access(id, 1).unwrap();
        policy.record_access(id, 4).unwrap();
        let value = EvictionCandidate {
            id,
            estimated_reload_cost_ms: 20.0,
            user_priority_bonus: 1.0,
            expensive_recreation_bonus: 1.0,
            device_newer: true,
            estimated_writeback_cost_ms: 3.0,
            ..candidate(3, 64, 0.5)
        };
        let keep = policy.keep_value(&value, 6).unwrap();
        assert!((keep - (10.0 + 2.0 / 6.0 + 1.0 + 1.0 + 3.0 - 1.0)).abs() < 1e-9);
        let mut one_touch = EvictionPolicy::new(EvictionPolicyConfig {
            recent_access_weight: 100.0,
            idle_age_penalty_per_tick: 0.5,
        });
        one_touch.record_access(id, 5).unwrap();
        assert_eq!(
            one_touch.keep_value(&value, 6).unwrap(),
            10.0 + 1.0 + 1.0 + 3.0 - 0.5
        );
        let late = EvictionCandidate {
            last_access_tick: 99,
            ..candidate(4, 1, 0.5)
        };
        assert!(EvictionPolicy::default().keep_value(&late, 6).is_err());
    }

    #[test]
    fn selects_lowest_keep_value_until_capacity_is_met() {
        let candidates = [
            candidate(4, 60, 0.9),
            candidate(2, 40, 0.1),
            candidate(3, 40, 0.2),
        ];
        let plan = EvictionPolicy::default().plan(&candidates, 20, 100, 0).unwrap();
        assert_eq!(
            plan.decisions.iter().map(|d| d.id.slot).collect::<Vec<_>>(),
            [2, 3]
        );
        assert_eq!(plan.bytes_to_reclaim, 80);
        assert!(
            EvictionPolicy::default()
                .plan(&candidates, 200, 100, 0)
                .unwrap()
                .decisions
                .is_empty()
        );
    }

    #[test]
    fn never_selects_pinned_in_flight_or_submitting_buffers() {
        let candidates = [
            EvictionCandidate {
                pin_count: 1,
                ..candidate(1, 128, 0.5)
            },
            EvictionCandidate {
                in_flight: true,
                ..candidate(2, 128, 0.5)
            },
            EvictionCandidate {
                being_submitted: true,
                ..candidate(3, 128, 0.5)
            },
            candidate(4, 32, 0.0),
        ];
        let plan = EvictionPolicy::default().plan(&candidates, 0, 32, 0).unwrap();
        assert_eq!(plan.decisions.len(), 1);
        assert_eq!(plan.decisions[0].id.slot, 4);
    }

    #[test]
    fn device_newer_requires_writeback_unless_disposable_or_replicated() {
        let candidates = [
            EvictionCandidate {
                device_newer: true,
                estimated_writeback_cost_ms: 5.0,
                ..candidate(1, 64, 0.5)
            },
            EvictionCandidate {
                device_newer: true,
                disposable: true,
                ..candidate(2, 64, 0.5)
            },
            EvictionCandidate {
                device_newer: true,
                alternate_current_copy: true,
                ..candidate(3, 64, 0.5)
            },
        ];
        let plan = EvictionPolicy::default().plan(&candidates, 0, 192, 0).unwrap();
        let actions: Vec<_> = plan.decisions.iter().map(|d| d.action).collect();
        assert_eq!(
            actions,
            [
                EvictionAction::Evict,
                EvictionAction::Evict,
                EvictionAction::WritebackThenEvict
            ]
        );
    }

    #[test]
    fn low_budget_stress_never_returns_protected_buffers() {
        let mut policy = EvictionPolicy::default();
        let candidates: Vec<_> = (0..128u32)
            .map(|slot| {
                let mut value = candidate(slot, 16, f64::from(slot % 10) / 10.0);
                value.id.generation = u64::from(slot) + 1;
                value.last_access_tick = u64::from(slot);
                value.pin_count = u32::from(slot % 17 == 0);
                value.in_flight = slot % 19 == 0;
                value.being_submitted = slot % 23 == 0;
                if slot % 7 == 0 {
                    policy.record_access(value.id, u64::from(slot)).unwrap();
                    policy.record_access(value.id, u64::from(slot) + 128).unwrap();
                }
                value
            })
            .collect();
        let plan = policy.plan(&candidates, 64, 512, 256).unwrap();
        assert!(plan.bytes_to_reclaim >= 448);
        for decision in &plan.decisions {
            let found = candidates.iter().find(|c| c.id == decision.id).unwrap();
            assert!(found.pin_count == 0 && !found.in_flight && !found.being_submitted);
        }
    }

    #[test]
    fn insufficient_capacity_and_bad_input_return_errors() {
        assert_eq!(
            EvictionPolicy::default()
                .plan(&[candidate(0, 32, 0.5)], 0, 64, 0)
                .unwrap_err()
                .code(),
            ErrorCode::OutOfMemoryBudget
        );
        assert!(
            EvictionPolicy::default()
                .plan(&[candidate(0, 32, 0.5), candidate(0, 16, 0.5)], 0, 64, 0)
                .is_err()
        );
        assert!(
            EvictionPolicy::default()
                .plan(&[candidate(0, 0, 0.5)], 0, 64, 0)
                .is_err()
        );
        assert!(
            EvictionPolicy::default()
                .plan(&[candidate(0, 32, 1.5)], 0, 64, 0)
                .is_err()
        );
        let mut policy = EvictionPolicy::default();
        policy.record_access(BufferId::new(1, 1), 3).unwrap();
        policy.forget(BufferId::new(1, 1));
        policy.record_access(BufferId::new(1, 1), 1).unwrap();
    }
}
