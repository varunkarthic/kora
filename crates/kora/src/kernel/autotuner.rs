//! Measured workgroup-size selection for `LocalSizeId` kernels.
//!
//! Candidates are generated from the device's queried limits and subgroup width, never from its
//! identity. Each candidate gets warmup dispatches (discarded) and timed measurements; once every
//! candidate is evaluated, the fastest one whose measurements are stable (max − min within a
//! fraction of the median) is selected. Without a stable winner the base size stays in use.

use std::sync::Mutex;

use crate::device::{DeviceLimits, SubgroupInfo};
use crate::error::{Error, ErrorCode, Result, Status};
use crate::performance_db::PerformanceStats;

/// One workgroup size to try.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkgroupCandidate {
    /// `local_size_x/y/z`.
    pub local_size: [u32; 3],
}

impl Default for WorkgroupCandidate {
    fn default() -> Self {
        Self {
            local_size: [1, 1, 1],
        }
    }
}

fn fits(local_size: [u32; 3], limits: &DeviceLimits) -> bool {
    let mut invocations = 1_u64;
    for (size, maximum) in local_size.into_iter().zip(limits.max_compute_work_group_size) {
        if size == 0 || size > maximum {
            return false;
        }
        invocations *= u64::from(size);
    }
    invocations <= u64::from(limits.max_compute_work_group_invocations)
}

/// Candidate workgroup sizes: the base size first, then (for `LocalSizeId` kernels) widths of
/// 1–16 subgroups (or powers of two from 32 without subgroup information) along X that fit the
/// device limits, at most `maximum_candidates` in total.
///
/// # Errors
///
/// [`ErrorCode::InvalidArgument`] when the limit is zero or the base size does not fit the device.
pub fn generate_workgroup_candidates(
    base_local_size: [u32; 3],
    local_size_id_based: bool,
    limits: &DeviceLimits,
    subgroup: &SubgroupInfo,
    maximum_candidates: usize,
) -> Result<Vec<WorkgroupCandidate>> {
    if maximum_candidates == 0 {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "workgroup candidate limit must be nonzero",
        ));
    }
    if !fits(base_local_size, limits) {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "base local size exceeds device workgroup limits",
        ));
    }
    let mut candidates = vec![WorkgroupCandidate {
        local_size: base_local_size,
    }];
    if !local_size_id_based {
        return Ok(candidates);
    }
    let widths: Vec<u64> = if subgroup.supported && subgroup.size != 0 {
        [1, 2, 4, 8, 16].map(|m| u64::from(subgroup.size) * m).to_vec()
    } else {
        (5..32).map(|shift| 1_u64 << shift).collect()
    };
    for width in widths {
        let Ok(width) = u32::try_from(width) else { break };
        let local_size = [width, base_local_size[1], base_local_size[2]];
        let candidate = WorkgroupCandidate { local_size };
        if candidates.len() < maximum_candidates
            && fits(local_size, limits)
            && !candidates.contains(&candidate)
        {
            candidates.push(candidate);
        }
    }
    Ok(candidates)
}

/// Performance-database key of one candidate. It includes the driver version so a driver update
/// re-tunes instead of trusting stale timings.
pub fn workgroup_performance_key(
    device_key: &str,
    driver_version: u32,
    kernel_id: &str,
    entry_point: &str,
    candidate: &WorkgroupCandidate,
) -> String {
    let [x, y, z] = candidate.local_size;
    format!("workgroup/{device_key}/{driver_version}/{kernel_id}/{entry_point}/{x}x{y}x{z}")
}

/// One reserved dispatch of the tuning schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trial {
    /// Candidate to dispatch with.
    pub candidate_index: usize,
    /// The timing is discarded.
    pub warmup: bool,
    /// The trial must be reported with [`WorkgroupAutotuner::observe`] or
    /// [`WorkgroupAutotuner::cancel`]; untracked trials use the current best size.
    pub tracked: bool,
}

/// Tuning parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutotuneOptions {
    /// Discarded dispatches per candidate.
    pub warmup_count: u32,
    /// Timed dispatches per candidate.
    pub measurement_count: u32,
    /// Largest accepted `(max − min) / median`.
    pub maximum_dispersion: f64,
}

impl Default for AutotuneOptions {
    fn default() -> Self {
        Self {
            warmup_count: 2,
            measurement_count: 5,
            maximum_dispersion: 0.5,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct CandidateState {
    warmups: u32,
    pending_warmups: u32,
    pending_measurements: u32,
    measurements: Vec<f64>,
    evaluated: bool,
    stable: bool,
    score: f64,
}

#[derive(Debug)]
struct State {
    candidates: Vec<CandidateState>,
    selected: Option<usize>,
    cursor: usize,
}

impl State {
    fn ready(&self) -> bool {
        self.candidates.iter().all(|c| c.evaluated)
    }

    fn evaluate(&mut self) {
        if !self.ready() {
            return;
        }
        self.selected = self
            .candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.stable)
            .min_by(|(_, a), (_, b)| a.score.total_cmp(&b.score))
            .map(|(index, _)| index);
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

/// Thread-safe round-robin tuning schedule over a fixed candidate list.
#[derive(Debug)]
pub struct WorkgroupAutotuner {
    candidates: Vec<WorkgroupCandidate>,
    base: usize,
    options: AutotuneOptions,
    state: Mutex<State>,
}

impl WorkgroupAutotuner {
    /// Creates a schedule. An empty list becomes `[1, 1, 1]`, an out-of-range base becomes 0, and
    /// invalid options fall back to the defaults.
    pub fn new(
        mut candidates: Vec<WorkgroupCandidate>,
        base_candidate: usize,
        options: AutotuneOptions,
    ) -> Self {
        if candidates.is_empty() {
            candidates.push(WorkgroupCandidate::default());
        }
        let valid = options.warmup_count > 0
            && options.measurement_count > 0
            && options.maximum_dispersion.is_finite()
            && options.maximum_dispersion >= 0.0;
        let state = State {
            candidates: vec![CandidateState::default(); candidates.len()],
            selected: None,
            cursor: 0,
        };
        Self {
            base: if base_candidate < candidates.len() {
                base_candidate
            } else {
                0
            },
            candidates,
            options: if valid {
                options
            } else {
                AutotuneOptions::default()
            },
            state: Mutex::new(state),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Candidate list.
    pub fn candidates(&self) -> &[WorkgroupCandidate] {
        &self.candidates
    }

    /// Reserves the next trial, round-robin over candidates still needing samples. Reservations
    /// count toward a candidate's quota so concurrent callers never overbook it.
    pub fn reserve(&self) -> Trial {
        let mut state = self.lock();
        if let Some(selected) = state.selected {
            return Trial {
                candidate_index: selected,
                warmup: false,
                tracked: false,
            };
        }
        let count = state.candidates.len();
        for offset in 0..count {
            let index = (state.cursor + offset) % count;
            let c = &mut state.candidates[index];
            if c.evaluated {
                continue;
            }
            let warmup = c.warmups + c.pending_warmups < self.options.warmup_count;
            if warmup {
                c.pending_warmups += 1;
            } else if c.measurements.len() + (c.pending_measurements as usize)
                < self.options.measurement_count as usize
            {
                c.pending_measurements += 1;
            } else {
                continue;
            }
            state.cursor = (index + 1) % count;
            return Trial {
                candidate_index: index,
                warmup,
                tracked: true,
            };
        }
        Trial {
            candidate_index: self.base,
            warmup: false,
            tracked: false,
        }
    }

    /// Releases a reserved trial that was not dispatched.
    pub fn cancel(&self, trial: &Trial) {
        if !trial.tracked {
            return;
        }
        let mut state = self.lock();
        if let Some(c) = state.candidates.get_mut(trial.candidate_index) {
            let pending = if trial.warmup {
                &mut c.pending_warmups
            } else {
                &mut c.pending_measurements
            };
            *pending = pending.saturating_sub(1);
        }
    }

    /// Records the measured duration of a reserved trial.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for untracked trials and non-positive or non-finite durations.
    pub fn observe(&self, trial: &Trial, duration_nanoseconds: f64) -> Status {
        if !trial.tracked
            || trial.candidate_index >= self.candidates.len()
            || !duration_nanoseconds.is_finite()
            || duration_nanoseconds <= 0.0
        {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "autotuner observation is not a valid measured trial",
            ));
        }
        let mut state = self.lock();
        let c = &mut state.candidates[trial.candidate_index];
        if trial.warmup {
            c.pending_warmups = c.pending_warmups.saturating_sub(1);
            c.warmups += 1;
            return Ok(());
        }
        c.pending_measurements = c.pending_measurements.saturating_sub(1);
        // A candidate restored from the database meanwhile keeps its persisted verdict.
        if c.evaluated {
            return Ok(());
        }
        c.measurements.push(duration_nanoseconds);
        if c.measurements.len() >= self.options.measurement_count as usize {
            let minimum = c.measurements.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = c.measurements.iter().copied().fold(0.0, f64::max);
            let center = median(&mut c.measurements);
            c.score = center;
            c.stable = (maximum - minimum) / center <= self.options.maximum_dispersion;
            c.evaluated = true;
            state.evaluate();
        }
        Ok(())
    }

    /// Restores a candidate from persisted statistics. Records with fewer samples than one
    /// measurement round are ignored.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for an unknown candidate or inconsistent statistics.
    pub fn restore(&self, candidate_index: usize, stats: &PerformanceStats) -> Status {
        if candidate_index >= self.candidates.len() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "persisted autotune candidate index is out of range",
            ));
        }
        let finite = [stats.median, stats.minimum, stats.maximum]
            .iter()
            .all(|v| v.is_finite());
        if !finite || stats.median <= 0.0 || stats.minimum <= 0.0 || stats.maximum < stats.minimum {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "persisted autotune statistics are invalid",
            ));
        }
        if stats.count < u64::from(self.options.measurement_count) {
            return Ok(());
        }
        let mut state = self.lock();
        let c = &mut state.candidates[candidate_index];
        c.score = stats.median;
        c.stable = (stats.maximum - stats.minimum) / stats.median <= self.options.maximum_dispersion;
        c.evaluated = true;
        c.warmups = self.options.warmup_count;
        state.evaluate();
        Ok(())
    }

    /// The chosen size, or the base size until tuning finishes (or when nothing was stable).
    pub fn selected_local_size(&self) -> [u32; 3] {
        let index = self.lock().selected.unwrap_or(self.base);
        self.candidates[index].local_size
    }

    /// Index of the chosen candidate.
    pub fn selected_index(&self) -> Option<usize> {
        self.lock().selected
    }

    /// Every candidate has been evaluated.
    pub fn ready(&self) -> bool {
        self.lock().ready()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn limits(size: [u32; 3], invocations: u32) -> DeviceLimits {
        DeviceLimits {
            max_compute_work_group_size: size,
            max_compute_work_group_invocations: invocations,
            ..DeviceLimits::default()
        }
    }

    fn subgroup(size: u32) -> SubgroupInfo {
        SubgroupInfo {
            size,
            supported: size != 0,
            ..SubgroupInfo::default()
        }
    }

    fn sizes(candidates: &[WorkgroupCandidate]) -> Vec<[u32; 3]> {
        candidates.iter().map(|c| c.local_size).collect()
    }

    fn tuner(sizes: &[[u32; 3]], base: usize) -> WorkgroupAutotuner {
        let candidates = sizes
            .iter()
            .map(|&local_size| WorkgroupCandidate { local_size })
            .collect();
        WorkgroupAutotuner::new(candidates, base, AutotuneOptions::default())
    }

    #[test]
    fn candidates_respect_dimensions_invocations_and_subgroup_width() {
        let l = limits([128, 4, 4], 128);
        let candidates = generate_workgroup_candidates([64, 1, 1], true, &l, &subgroup(32), 8).unwrap();
        assert_eq!(sizes(&candidates), [[64, 1, 1], [32, 1, 1], [128, 1, 1]]);
        let fixed = generate_workgroup_candidates([64, 1, 1], false, &l, &subgroup(32), 8).unwrap();
        assert_eq!(sizes(&fixed), [[64, 1, 1]]);
        assert!(generate_workgroup_candidates([256, 1, 1], true, &l, &subgroup(32), 8).is_err());
        assert!(generate_workgroup_candidates([0, 1, 1], true, &l, &subgroup(32), 8).is_err());
        assert!(generate_workgroup_candidates([64, 1, 1], true, &l, &subgroup(32), 0).is_err());
        // Y and Z shrink the room for X.
        let tall = generate_workgroup_candidates([16, 4, 1], true, &l, &subgroup(16), 8).unwrap();
        assert_eq!(sizes(&tall), [[16, 4, 1], [32, 4, 1]]);
    }

    #[test]
    fn candidates_without_subgroup_information_use_powers_of_two_and_the_cap() {
        let l = limits([1024, 1024, 64], 1024);
        let all = generate_workgroup_candidates([1, 1, 1], true, &l, &subgroup(0), 16).unwrap();
        assert_eq!(
            sizes(&all),
            [
                [1, 1, 1],
                [32, 1, 1],
                [64, 1, 1],
                [128, 1, 1],
                [256, 1, 1],
                [512, 1, 1],
                [1024, 1, 1]
            ]
        );
        let capped = generate_workgroup_candidates([1, 1, 1], true, &l, &subgroup(0), 3).unwrap();
        assert_eq!(sizes(&capped), [[1, 1, 1], [32, 1, 1], [64, 1, 1]]);
        // A huge subgroup width that overflows u32 multiples stops cleanly.
        let wide = generate_workgroup_candidates(
            [1, 1, 1],
            true,
            &limits([u32::MAX; 3], u32::MAX),
            &subgroup(1 << 31),
            8,
        )
        .unwrap();
        assert_eq!(sizes(&wide), [[1, 1, 1], [1 << 31, 1, 1]]);
    }

    #[test]
    fn performance_keys_separate_drivers_and_candidates() {
        let a = WorkgroupCandidate {
            local_size: [32, 1, 1],
        };
        let b = WorkgroupCandidate {
            local_size: [64, 1, 1],
        };
        assert_eq!(
            workgroup_performance_key("dev", 1, "k", "main", &a),
            "workgroup/dev/1/k/main/32x1x1"
        );
        assert_ne!(
            workgroup_performance_key("dev", 1, "k", "main", &a),
            workgroup_performance_key("dev", 2, "k", "main", &a)
        );
        assert_ne!(
            workgroup_performance_key("dev", 1, "k", "main", &a),
            workgroup_performance_key("dev", 1, "k", "main", &b)
        );
    }

    #[test]
    fn two_warmups_five_measurements_and_the_fastest_stable_candidate_wins() {
        let t = tuner(&[[32, 1, 1], [64, 1, 1], [128, 1, 1]], 1);
        let mut warmups = [0; 3];
        let mut measurements = [0; 3];
        for _ in 0..64 {
            if t.ready() {
                break;
            }
            let trial = t.reserve();
            assert!(trial.tracked);
            if trial.warmup {
                warmups[trial.candidate_index] += 1;
            } else {
                measurements[trial.candidate_index] += 1;
            }
            assert_eq!(t.selected_local_size(), [64, 1, 1], "base until tuning finishes");
            let latency = [300.0, 200.0, 100.0][trial.candidate_index];
            t.observe(&trial, latency).unwrap();
        }
        assert!(t.ready());
        assert_eq!(warmups, [2, 2, 2]);
        assert_eq!(measurements, [5, 5, 5]);
        assert_eq!(t.selected_index(), Some(2));
        assert_eq!(t.selected_local_size(), [128, 1, 1]);
        let after = t.reserve();
        assert_eq!((after.candidate_index, after.tracked), (2, false));
    }

    #[test]
    fn unstable_candidates_lose_and_all_unstable_keeps_the_base() {
        let t = tuner(&[[32, 1, 1], [64, 1, 1]], 1);
        let mut samples = [0_u32; 2];
        while !t.ready() {
            let trial = t.reserve();
            samples[trial.candidate_index] += 1;
            // Candidate 0 is faster on median but swings by 10x.
            let latency = match (trial.candidate_index, samples[0] % 2) {
                (0, 0) => 100.0,
                (0, _) => 10.0,
                _ => 80.0,
            };
            t.observe(&trial, latency).unwrap();
        }
        assert_eq!(t.selected_index(), Some(1));

        let noisy = tuner(&[[32, 1, 1], [64, 1, 1]], 1);
        let mut flips = [false; 2];
        while !noisy.ready() {
            let trial = noisy.reserve();
            let flip = &mut flips[trial.candidate_index];
            *flip = !*flip;
            noisy.observe(&trial, if *flip { 1.0 } else { 50.0 }).unwrap();
        }
        assert_eq!(noisy.selected_index(), None);
        assert_eq!(noisy.selected_local_size(), [64, 1, 1]);
        let fallback = noisy.reserve();
        assert_eq!((fallback.candidate_index, fallback.tracked), (1, false));
    }

    #[test]
    fn restores_only_mature_consistent_records() {
        let t = tuner(&[[32, 1, 1], [64, 1, 1]], 0);
        let stats = |count, minimum, median, maximum| PerformanceStats {
            count,
            minimum,
            median,
            maximum,
            p95: maximum,
            ..PerformanceStats::default()
        };
        t.restore(0, &stats(4, 80.0, 80.0, 80.0)).unwrap();
        assert!(!t.ready(), "four samples are not a full measurement round");
        t.restore(0, &stats(5, 50.0, 80.0, 160.0)).unwrap();
        t.restore(1, &stats(5, 118.0, 120.0, 125.0)).unwrap();
        assert!(t.ready());
        assert_eq!(t.selected_index(), Some(1));
        assert!(t.restore(2, &stats(5, 1.0, 1.0, 1.0)).is_err());
        assert!(t.restore(0, &stats(5, 0.0, 1.0, 1.0)).is_err());
        assert!(t.restore(0, &stats(5, 2.0, 1.0, 1.0)).is_err());
        assert!(t.restore(0, &stats(5, 1.0, f64::NAN, 1.0)).is_err());
    }

    #[test]
    fn rejects_bad_observations_and_cancels_reservations() {
        let t = tuner(&[[32, 1, 1]], 0);
        // Warmups come first; measurements may be reserved while warmups are still in flight
        // because dispatches complete in queue order.
        let reserved: Vec<Trial> = (0..7).map(|_| t.reserve()).collect();
        assert!(reserved[..2].iter().all(|r| r.tracked && r.warmup));
        assert!(reserved[2..].iter().all(|r| r.tracked && !r.warmup));
        let over = t.reserve();
        assert!(!over.tracked, "the schedule is fully booked");
        t.cancel(&reserved[1]);
        let again = t.reserve();
        assert!(
            again.tracked && again.warmup,
            "a cancelled warmup is handed out again"
        );
        t.cancel(&reserved[6]);
        assert!(t.reserve().tracked);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(t.observe(&reserved[0], bad).is_err());
        }
        assert!(t.observe(&over, 10.0).is_err());
        let out_of_range = Trial {
            candidate_index: 9,
            warmup: false,
            tracked: true,
        };
        assert!(t.observe(&out_of_range, 10.0).is_err());
        t.cancel(&out_of_range);
        t.cancel(&over);
    }

    #[test]
    fn construction_normalizes_bad_inputs() {
        let empty = WorkgroupAutotuner::new(Vec::new(), 5, AutotuneOptions::default());
        assert_eq!(empty.candidates().len(), 1);
        assert_eq!(empty.selected_local_size(), [1, 1, 1]);
        let bad = AutotuneOptions {
            warmup_count: 0,
            measurement_count: 0,
            maximum_dispersion: f64::NAN,
        };
        let t = WorkgroupAutotuner::new(vec![WorkgroupCandidate::default()], 0, bad);
        assert_eq!(t.options, AutotuneOptions::default());
    }

    #[test]
    fn concurrent_reservations_never_overbook() {
        let t = Arc::new(tuner(&[[32, 1, 1], [64, 1, 1]], 0));
        let workers: Vec<_> = (0..6)
            .map(|_| {
                let t = Arc::clone(&t);
                std::thread::spawn(move || {
                    let mut observed = 0_u32;
                    while !t.ready() {
                        let trial = t.reserve();
                        if !trial.tracked {
                            std::thread::yield_now();
                            continue;
                        }
                        observed += 1;
                        let latency = if trial.candidate_index == 0 { 200.0 } else { 100.0 };
                        t.observe(&trial, latency).unwrap();
                    }
                    observed
                })
            })
            .collect();
        let total: u32 = workers.into_iter().map(|w| w.join().unwrap()).sum();
        assert_eq!(total, 2 * (2 + 5), "exactly the scheduled trials were handed out");
        assert_eq!(t.selected_index(), Some(1));
    }

    #[test]
    fn even_and_odd_medians() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
