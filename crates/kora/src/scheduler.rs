//! Scheduling building blocks: an earliest-finish-time cost model with an additive breakdown,
//! queue-availability tracking, and a candidate-selection policy with exploration and hysteresis.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::error::{Error, ErrorCode, Result, Status};

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

/// Inputs to [`evaluate_scheduler_cost`]. All values are nanoseconds or bytes, finite and
/// nonnegative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchedulerCostInput {
    /// When all dependencies are ready.
    pub dependency_ready_ns: f64,
    /// When the backend becomes free.
    pub backend_available_ns: f64,
    /// Moving inputs to the backend.
    pub input_transfer_ns: f64,
    /// Kernel execution.
    pub execute_ns: f64,
    /// Moving required outputs back.
    pub required_output_ns: f64,
    /// Synchronization overhead.
    pub synchronization_ns: f64,
    /// Predicted device memory use.
    pub predicted_usage_bytes: f64,
    /// Effective budget (must exceed usage).
    pub effective_budget_bytes: f64,
    /// Scale of the quadratic penalty above 70% budget use.
    pub memory_penalty_scale_ns: f64,
    /// Standard deviation of the estimate.
    pub uncertainty_sigma_ns: f64,
    /// Weight of the uncertainty penalty.
    pub uncertainty_lambda: f64,
    /// Transfer time saved by input locality.
    pub locality_saved_ns: f64,
    /// Cap on the locality bonus.
    pub locality_cap_ns: f64,
    /// Movement downstream consumers incur if this candidate runs here.
    pub estimated_downstream_movement_ns: f64,
    /// Look-ahead weight in `[0, 1]`.
    pub lookahead_beta: f64,
}

impl Default for SchedulerCostInput {
    fn default() -> Self {
        Self {
            dependency_ready_ns: 0.0,
            backend_available_ns: 0.0,
            input_transfer_ns: 0.0,
            execute_ns: 0.0,
            required_output_ns: 0.0,
            synchronization_ns: 0.0,
            predicted_usage_bytes: 0.0,
            effective_budget_bytes: 0.0,
            memory_penalty_scale_ns: 0.0,
            uncertainty_sigma_ns: 0.0,
            uncertainty_lambda: 1.0,
            locality_saved_ns: 0.0,
            locality_cap_ns: 0.0,
            estimated_downstream_movement_ns: 0.0,
            lookahead_beta: 0.25,
        }
    }
}

/// Additive score: `predicted_finish_ns` is the sum of every term (the locality bonus subtracts).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScoreBreakdown {
    /// Dependency readiness.
    pub ready_delay_ns: f64,
    /// Extra wait for the backend.
    pub queue_delay_ns: f64,
    /// Input movement.
    pub input_transfer_ns: f64,
    /// Execution.
    pub execute_ns: f64,
    /// Output movement.
    pub required_output_ns: f64,
    /// Synchronization.
    pub synchronization_ns: f64,
    /// Memory-pressure penalty.
    pub memory_penalty_ns: f64,
    /// Uncertainty penalty.
    pub uncertainty_penalty_ns: f64,
    /// Locality bonus (subtracted).
    pub locality_bonus_ns: f64,
    /// Weighted downstream movement.
    pub downstream_movement_penalty_ns: f64,
    /// Total predicted finish time.
    pub predicted_finish_ns: f64,
}

/// Scores one candidate.
pub fn evaluate_scheduler_cost(input: &SchedulerCostInput) -> Result<ScoreBreakdown> {
    let values = [
        input.dependency_ready_ns,
        input.backend_available_ns,
        input.input_transfer_ns,
        input.execute_ns,
        input.required_output_ns,
        input.synchronization_ns,
        input.predicted_usage_bytes,
        input.effective_budget_bytes,
        input.memory_penalty_scale_ns,
        input.uncertainty_sigma_ns,
        input.uncertainty_lambda,
        input.locality_saved_ns,
        input.locality_cap_ns,
        input.estimated_downstream_movement_ns,
        input.lookahead_beta,
    ];
    if !values.into_iter().all(finite_nonnegative) {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "scheduler costs and policy inputs must be finite and nonnegative",
        ));
    }
    if input.lookahead_beta > 1.0 {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "scheduler look-ahead beta must be in [0, 1]",
        ));
    }
    if input.effective_budget_bytes == 0.0 && input.predicted_usage_bytes > 0.0 {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "positive predicted usage requires an effective memory budget",
        ));
    }
    let usage_ratio = if input.effective_budget_bytes == 0.0 {
        0.0
    } else {
        input.predicted_usage_bytes / input.effective_budget_bytes
    };
    if !usage_ratio.is_finite() || usage_ratio >= 1.0 {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "candidate exceeds effective memory budget",
        ));
    }
    let mut score = ScoreBreakdown {
        ready_delay_ns: input.dependency_ready_ns,
        queue_delay_ns: (input.backend_available_ns - input.dependency_ready_ns).max(0.0),
        input_transfer_ns: input.input_transfer_ns,
        execute_ns: input.execute_ns,
        required_output_ns: input.required_output_ns,
        synchronization_ns: input.synchronization_ns,
        uncertainty_penalty_ns: input.uncertainty_lambda * input.uncertainty_sigma_ns,
        locality_bonus_ns: input.locality_saved_ns.min(input.locality_cap_ns),
        downstream_movement_penalty_ns: input.lookahead_beta * input.estimated_downstream_movement_ns,
        ..Default::default()
    };
    if usage_ratio > 0.70 {
        let normalized = (usage_ratio - 0.70) / 0.30;
        score.memory_penalty_ns = input.memory_penalty_scale_ns * normalized * normalized;
    }
    score.predicted_finish_ns = score.ready_delay_ns
        + score.queue_delay_ns
        + score.input_transfer_ns
        + score.execute_ns
        + score.required_output_ns
        + score.synchronization_ns
        + score.memory_penalty_ns
        + score.uncertainty_penalty_ns
        - score.locality_bonus_ns
        + score.downstream_movement_penalty_ns;
    if !finite_nonnegative(score.predicted_finish_ns) {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "scheduler look-ahead produces a non-finite or negative predicted finish",
        ));
    }
    Ok(score)
}

/// Tracks estimated pending work per queue and corrects it toward measured completion durations.
#[derive(Debug, Default)]
pub struct QueueAvailability {
    state: Mutex<(HashMap<u64, f64>, HashMap<u64, f64>)>,
}

impl QueueAvailability {
    /// A tracker with no pending work.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, (HashMap<u64, f64>, HashMap<u64, f64>)> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Adds estimated work to `queue`.
    pub fn add_estimated_work(&self, queue: u64, duration_ns: f64) -> Status {
        if !finite_nonnegative(duration_ns) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "queue duration must be finite and nonnegative",
            ));
        }
        let mut state = self.lock();
        let pending = state.0.entry(queue).or_insert(0.0);
        let updated = *pending + duration_ns;
        if !updated.is_finite() {
            return Err(Error::new(ErrorCode::Overflow, "queue estimate overflowed"));
        }
        *pending = updated;
        Ok(())
    }

    /// Retires estimated work and learns an EWMA correction from the measured duration.
    pub fn observe_completion(
        &self,
        queue: u64,
        estimated_ns: f64,
        actual_ns: f64,
        correction_alpha: f64,
    ) -> Status {
        if !finite_nonnegative(estimated_ns)
            || !finite_nonnegative(actual_ns)
            || !(correction_alpha > 0.0 && correction_alpha <= 1.0)
        {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "queue completion sample or correction alpha is invalid",
            ));
        }
        let mut state = self.lock();
        let error = actual_ns - estimated_ns;
        state
            .1
            .entry(queue)
            .and_modify(|c| *c += correction_alpha * (error - *c))
            .or_insert(error);
        if let Some(pending) = state.0.get_mut(&queue) {
            *pending = (*pending - estimated_ns).max(0.0);
        }
        Ok(())
    }

    /// Estimated time until `queue` is free.
    pub fn available_at_ns(&self, queue: u64) -> f64 {
        let state = self.lock();
        let pending = state.0.get(&queue).copied().unwrap_or(0.0);
        let correction = state.1.get(&queue).copied().unwrap_or(0.0);
        (pending + correction).max(0.0)
    }
}

/// Scheduling priority hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum SchedulingPriority {
    /// Background.
    Low,
    /// Default.
    #[default]
    Normal,
    /// Latency sensitive.
    High,
    /// Best effort real time.
    RealtimeHint,
}

impl SchedulingPriority {
    fn aging_factor(self) -> f64 {
        match self {
            Self::Low => 3.0,
            Self::Normal => 2.0,
            Self::High => 1.0,
            Self::RealtimeHint => 0.0,
        }
    }
}

/// One pre-scored candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchedulerCandidate {
    /// Caller identifier.
    pub id: u64,
    /// Predicted finish time.
    pub predicted_finish_ns: f64,
    /// Number of measurements behind the estimate.
    pub observations: u64,
    /// How long the task has waited.
    pub wait_age_ns: f64,
    /// Task priority.
    pub priority: SchedulingPriority,
    /// Capability constraint.
    pub supported: bool,
    /// Health constraint.
    pub healthy: bool,
    /// Budget constraint.
    pub memory_fits: bool,
    /// User policy constraint.
    pub allowed: bool,
    /// Affinity constraint.
    pub affinity_matches: bool,
    /// Preferred on cold-start ties.
    pub known_safe: bool,
}

impl SchedulerCandidate {
    /// A feasible candidate with the given estimate.
    pub fn new(id: u64, predicted_finish_ns: f64, observations: u64) -> Self {
        Self {
            id,
            predicted_finish_ns,
            observations,
            wait_age_ns: 0.0,
            priority: SchedulingPriority::Normal,
            supported: true,
            healthy: true,
            memory_fits: true,
            allowed: true,
            affinity_matches: true,
            known_safe: false,
        }
    }

    fn feasible(&self) -> bool {
        self.supported && self.healthy && self.memory_fits && self.allowed && self.affinity_matches
    }
}

/// Selection policy knobs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchedulerPolicyOptions {
    /// Disable exploration.
    pub deterministic: bool,
    /// Estimate for unmeasured candidates without one.
    pub cold_start_estimate_ns: f64,
    /// Observations at which confidence is full.
    pub stable_observations: u64,
    /// Exploration probability at zero confidence.
    pub low_confidence_exploration: f64,
    /// Exploration probability at full confidence.
    pub stable_exploration: f64,
    /// Absolute advantage required to switch away from the current candidate.
    pub absolute_hysteresis_ns: f64,
    /// Relative advantage required to switch away from the current candidate.
    pub relative_hysteresis: f64,
    /// Aging discount per waited nanosecond.
    pub priority_aging_rate: f64,
    /// Aging discount cap.
    pub maximum_aging_discount_ns: f64,
}

impl Default for SchedulerPolicyOptions {
    fn default() -> Self {
        Self {
            deterministic: false,
            cold_start_estimate_ns: 0.0,
            stable_observations: 20,
            low_confidence_exploration: 0.05,
            stable_exploration: 0.01,
            absolute_hysteresis_ns: 50_000.0,
            relative_hysteresis: 0.10,
            priority_aging_rate: 0.0,
            maximum_aging_discount_ns: 0.0,
        }
    }
}

/// Outcome of [`select_scheduler_candidate`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SchedulerSelection {
    /// Selected candidate.
    pub candidate_id: u64,
    /// Its score after aging.
    pub effective_finish_ns: f64,
    /// Exploration probability used.
    pub exploration_probability: f64,
    /// Selected by exploration.
    pub explored: bool,
    /// The current candidate was kept by hysteresis.
    pub retained_by_hysteresis: bool,
}

/// Selects among feasible candidates. `exploration_draw` is in `[0, 1)` and ignored when
/// deterministic.
pub fn select_scheduler_candidate(
    candidates: &[SchedulerCandidate],
    options: &SchedulerPolicyOptions,
    current: Option<u64>,
    exploration_draw: f64,
) -> Result<SchedulerSelection> {
    let option_values = [
        options.cold_start_estimate_ns,
        options.low_confidence_exploration,
        options.stable_exploration,
        options.absolute_hysteresis_ns,
        options.relative_hysteresis,
        options.priority_aging_rate,
        options.maximum_aging_discount_ns,
        exploration_draw,
    ];
    if !option_values.into_iter().all(finite_nonnegative) {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "scheduler policy values must be finite and nonnegative",
        ));
    }
    if options.low_confidence_exploration > 1.0
        || options.stable_exploration > options.low_confidence_exploration
        || options.relative_hysteresis > 1.0
        || exploration_draw >= 1.0
        || options.stable_observations == 0
    {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "scheduler policy probability, margin, or confidence threshold is invalid",
        ));
    }
    if candidates
        .iter()
        .any(|c| !finite_nonnegative(c.predicted_finish_ns) || !finite_nonnegative(c.wait_age_ns))
    {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "candidate estimates and wait ages must be finite and nonnegative",
        ));
    }
    let score = |c: &SchedulerCandidate| {
        let estimate = if c.observations == 0 && c.predicted_finish_ns <= 0.0 {
            options.cold_start_estimate_ns
        } else {
            c.predicted_finish_ns
        };
        let aging = options
            .maximum_aging_discount_ns
            .min(options.priority_aging_rate * c.wait_age_ns * c.priority.aging_factor());
        (estimate - aging).max(0.0)
    };
    let mut eligible: Vec<(f64, &SchedulerCandidate)> = candidates
        .iter()
        .filter(|c| c.feasible())
        .map(|c| (score(c), c))
        .collect();
    if eligible.is_empty() {
        return Err(Error::new(
            ErrorCode::NoEligibleDevice,
            "no candidate satisfies scheduler constraints",
        ));
    }
    eligible.sort_by(|(ls, l), (rs, r)| {
        ls.total_cmp(rs)
            .then_with(|| {
                if l.observations == 0 && r.observations == 0 {
                    r.known_safe.cmp(&l.known_safe)
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .then(l.id.cmp(&r.id))
    });
    let min_observations = eligible.iter().map(|(_, c)| c.observations).min().unwrap_or(0);
    let confidence = (min_observations as f64 / options.stable_observations as f64).min(1.0);
    let exploration_probability = if options.deterministic {
        0.0
    } else {
        options.low_confidence_exploration
            + confidence * (options.stable_exploration - options.low_confidence_exploration)
    };
    let mut selected = eligible[0];
    let mut explored = false;
    if !options.deterministic && eligible.len() > 1 && exploration_draw < exploration_probability {
        let limit = eligible[0].0 * (1.0 + options.relative_hysteresis) + options.absolute_hysteresis_ns;
        let plausible = eligible.iter().take_while(|(s, _)| *s <= limit).count();
        if plausible > 1 {
            let scaled = exploration_draw / exploration_probability * (plausible - 1) as f64;
            selected = eligible[1 + scaled as usize];
            explored = true;
        }
    }
    let mut retained = false;
    if let Some(current) = current
        && !explored
        && selected.1.id != current
        && let Some(&kept) = eligible.iter().find(|(_, c)| c.id == current)
    {
        let advantage = kept.0 - selected.0;
        if advantage < options.absolute_hysteresis_ns || advantage < kept.0 * options.relative_hysteresis {
            selected = kept;
            retained = true;
        }
    }
    Ok(SchedulerSelection {
        candidate_id: selected.1.id,
        effective_finish_ns: selected.0,
        exploration_probability,
        explored,
        retained_by_hysteresis: retained,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer_model::{TransferModelSet, TransferPath};

    #[test]
    fn hand_computed_breakdown_matches_eft_and_sums_to_total() {
        let input = SchedulerCostInput {
            dependency_ready_ns: 100.0,
            backend_available_ns: 160.0,
            input_transfer_ns: 20.0,
            execute_ns: 30.0,
            required_output_ns: 10.0,
            synchronization_ns: 5.0,
            predicted_usage_bytes: 850.0,
            effective_budget_bytes: 1000.0,
            memory_penalty_scale_ns: 40.0,
            uncertainty_sigma_ns: 8.0,
            uncertainty_lambda: 0.5,
            locality_saved_ns: 50.0,
            locality_cap_ns: 25.0,
            estimated_downstream_movement_ns: 40.0,
            lookahead_beta: 0.25,
        };
        let s = evaluate_scheduler_cost(&input).unwrap();
        assert_eq!(s.ready_delay_ns, 100.0);
        assert_eq!(s.queue_delay_ns, 60.0);
        assert!((s.memory_penalty_ns - 10.0).abs() < 1e-9);
        assert_eq!(s.uncertainty_penalty_ns, 4.0);
        assert_eq!(s.locality_bonus_ns, 25.0);
        assert_eq!(s.downstream_movement_penalty_ns, 10.0);
        assert!((s.predicted_finish_ns - 224.0).abs() < 1e-9);
    }

    #[test]
    fn lookahead_keeps_two_gib_chain_on_downstream_preferred_device() {
        let bytes = 2u64 << 30;
        let transfers = TransferModelSet::default();
        transfers
            .observe(TransferPath::CrossDeviceViaHost, bytes, 200_000_000.0)
            .unwrap();
        let movement = transfers
            .estimate_ns(TransferPath::CrossDeviceViaHost, bytes)
            .unwrap();
        assert_eq!(movement, 200_000_000.0);
        let gpu0 = evaluate_scheduler_cost(&SchedulerCostInput {
            execute_ns: 97_000_000.0,
            estimated_downstream_movement_ns: movement,
            ..Default::default()
        })
        .unwrap();
        let gpu1 = evaluate_scheduler_cost(&SchedulerCostInput {
            execute_ns: 100_000_000.0,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(gpu0.downstream_movement_penalty_ns, 50_000_000.0);
        let options = SchedulerPolicyOptions {
            deterministic: true,
            ..Default::default()
        };
        let selected = select_scheduler_candidate(
            &[
                SchedulerCandidate::new(0, gpu0.predicted_finish_ns, 1),
                SchedulerCandidate::new(1, gpu1.predicted_finish_ns, 1),
            ],
            &options,
            None,
            0.5,
        )
        .unwrap();
        assert_eq!(selected.candidate_id, 1);
    }

    #[test]
    fn uses_budget_knee_and_rejects_infeasible_or_invalid_inputs() {
        let mut input = SchedulerCostInput {
            execute_ns: 100.0,
            effective_budget_bytes: 100.0,
            predicted_usage_bytes: 70.0,
            ..Default::default()
        };
        assert_eq!(evaluate_scheduler_cost(&input).unwrap().memory_penalty_ns, 0.0);
        input.predicted_usage_bytes = 85.0;
        input.memory_penalty_scale_ns = 100.0;
        assert!((evaluate_scheduler_cost(&input).unwrap().memory_penalty_ns - 25.0).abs() < 1e-12);
        input.predicted_usage_bytes = 100.0;
        assert!(evaluate_scheduler_cost(&input).is_err());
        input.predicted_usage_bytes = 0.0;
        input.execute_ns = -1.0;
        assert!(evaluate_scheduler_cost(&input).is_err());
        input.execute_ns = 0.0;
        input.estimated_downstream_movement_ns = -1.0;
        assert!(evaluate_scheduler_cost(&input).is_err());
        input.estimated_downstream_movement_ns = 1.0;
        for beta in [-0.1, 1.1] {
            input.lookahead_beta = beta;
            assert!(evaluate_scheduler_cost(&input).is_err());
        }
        let no_budget = SchedulerCostInput {
            predicted_usage_bytes: 1.0,
            ..Default::default()
        };
        assert!(evaluate_scheduler_cost(&no_budget).is_err());
    }

    #[test]
    fn queue_availability_tracks_work_and_corrects_estimates() {
        let queues = QueueAvailability::new();
        queues.add_estimated_work(7, 100.0).unwrap();
        queues.add_estimated_work(7, 50.0).unwrap();
        assert_eq!(queues.available_at_ns(7), 150.0);
        queues.observe_completion(7, 100.0, 120.0, 0.25).unwrap();
        assert_eq!(queues.available_at_ns(7), 70.0);
        queues.observe_completion(7, 20.0, 0.0, 0.5).unwrap();
        assert_eq!(queues.available_at_ns(7), 30.0);
        assert_eq!(queues.available_at_ns(8), 0.0);
        assert!(queues.observe_completion(7, 10.0, 20.0, 0.0).is_err());
        assert!(queues.add_estimated_work(7, f64::NAN).is_err());
    }

    #[test]
    fn excludes_unsupported_unhealthy_and_constrained_candidates() {
        let unsupported = SchedulerCandidate {
            supported: false,
            ..SchedulerCandidate::new(1, 1.0, 0)
        };
        let unhealthy = SchedulerCandidate {
            healthy: false,
            ..SchedulerCandidate::new(2, 2.0, 0)
        };
        let too_large = SchedulerCandidate {
            memory_fits: false,
            ..SchedulerCandidate::new(3, 3.0, 0)
        };
        let allowed = SchedulerCandidate::new(4, 4.0, 0);
        let options = SchedulerPolicyOptions::default();
        let result =
            select_scheduler_candidate(&[unsupported, unhealthy, too_large, allowed], &options, None, 0.0)
                .unwrap();
        assert_eq!(result.candidate_id, 4);
        let wrong_affinity = SchedulerCandidate {
            affinity_matches: false,
            ..allowed
        };
        assert_eq!(
            select_scheduler_candidate(&[wrong_affinity], &options, None, 0.0)
                .unwrap_err()
                .code(),
            ErrorCode::NoEligibleDevice
        );
    }

    #[test]
    fn cold_start_prefers_known_safe_on_ties_and_uses_coarse_estimates() {
        let options = SchedulerPolicyOptions {
            cold_start_estimate_ns: 100.0,
            deterministic: true,
            ..Default::default()
        };
        let safe = SchedulerCandidate {
            known_safe: true,
            ..SchedulerCandidate::new(9, 100.0, 0)
        };
        let mut unknown = SchedulerCandidate::new(1, 100.0, 0);
        assert_eq!(
            select_scheduler_candidate(&[unknown, safe], &options, None, 0.0)
                .unwrap()
                .candidate_id,
            9
        );
        unknown.predicted_finish_ns = 80.0;
        assert_eq!(
            select_scheduler_candidate(&[unknown, safe], &options, None, 0.0)
                .unwrap()
                .candidate_id,
            1
        );
        unknown.predicted_finish_ns = 0.0;
        let estimate = select_scheduler_candidate(&[unknown], &options, None, 0.0).unwrap();
        assert_eq!(estimate.effective_finish_ns, 100.0);
    }

    #[test]
    fn exploration_is_bounded_confidence_decays_and_determinism_disables_it() {
        let mut options = SchedulerPolicyOptions {
            absolute_hysteresis_ns: 1000.0,
            ..Default::default()
        };
        let mut a = SchedulerCandidate::new(1, 100.0, 0);
        let mut b = SchedulerCandidate::new(2, 101.0, 0);
        let result = select_scheduler_candidate(&[a, b], &options, None, 0.025).unwrap();
        assert_eq!(result.exploration_probability, 0.05);
        assert!(result.explored);
        assert_eq!(result.candidate_id, 2);
        a.observations = 20;
        let result = select_scheduler_candidate(&[a, b], &options, None, 0.02).unwrap();
        assert_eq!(result.exploration_probability, 0.05);
        assert!(result.explored);
        b.observations = 20;
        let result = select_scheduler_candidate(&[a, b], &options, None, 0.02).unwrap();
        assert!((result.exploration_probability - 0.01).abs() < 1e-12);
        assert!(!result.explored);
        options.deterministic = true;
        let result = select_scheduler_candidate(&[a, b], &options, None, 0.0).unwrap();
        assert!(!result.explored);
        assert_eq!(result.exploration_probability, 0.0);
        assert_eq!(result.candidate_id, 1);
    }

    #[test]
    fn hysteresis_requires_absolute_and_relative_advantage() {
        let options = SchedulerPolicyOptions {
            deterministic: true,
            absolute_hysteresis_ns: 10.0,
            relative_hysteresis: 0.10,
            ..Default::default()
        };
        let current = SchedulerCandidate::new(1, 100.0, 10);
        let mut faster = SchedulerCandidate::new(2, 95.0, 10);
        let result = select_scheduler_candidate(&[current, faster], &options, Some(1), 0.0).unwrap();
        assert_eq!(result.candidate_id, 1);
        assert!(result.retained_by_hysteresis);
        faster.predicted_finish_ns = 80.0;
        let result = select_scheduler_candidate(&[current, faster], &options, Some(1), 0.0).unwrap();
        assert_eq!(result.candidate_id, 2);
        assert!(!result.retained_by_hysteresis);
    }

    #[test]
    fn priority_aging_prevents_starvation_and_ties_are_stable() {
        let options = SchedulerPolicyOptions {
            deterministic: true,
            priority_aging_rate: 0.2,
            maximum_aging_discount_ns: 40.0,
            ..Default::default()
        };
        let normal = SchedulerCandidate::new(8, 100.0, 10);
        let aged_low = SchedulerCandidate {
            priority: SchedulingPriority::Low,
            wait_age_ns: 100.0,
            ..SchedulerCandidate::new(9, 130.0, 10)
        };
        let result = select_scheduler_candidate(&[normal, aged_low], &options, None, 0.0).unwrap();
        assert_eq!(result.candidate_id, 9);
        assert_eq!(result.effective_finish_ns, 90.0);
        let result = select_scheduler_candidate(
            &[
                SchedulerCandidate::new(4, 50.0, 2),
                SchedulerCandidate::new(3, 50.0, 2),
            ],
            &options,
            None,
            0.0,
        )
        .unwrap();
        assert_eq!(result.candidate_id, 3);
    }

    #[test]
    fn rejects_invalid_policy_and_candidate_numbers() {
        let invalid = SchedulerPolicyOptions {
            low_confidence_exploration: 1.1,
            ..Default::default()
        };
        assert!(
            select_scheduler_candidate(&[SchedulerCandidate::new(1, 10.0, 0)], &invalid, None, 0.0).is_err()
        );
        let options = SchedulerPolicyOptions::default();
        assert!(
            select_scheduler_candidate(&[SchedulerCandidate::new(1, -1.0, 0)], &options, None, 0.0).is_err()
        );
        assert!(
            select_scheduler_candidate(&[SchedulerCandidate::new(1, 10.0, 0)], &options, None, 1.0).is_err()
        );
        let zero_stable = SchedulerPolicyOptions {
            stable_observations: 0,
            ..Default::default()
        };
        assert!(
            select_scheduler_candidate(&[SchedulerCandidate::new(1, 10.0, 0)], &zero_stable, None, 0.0)
                .is_err()
        );
    }
}
