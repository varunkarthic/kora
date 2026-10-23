//! Adaptive device placement from calibration records and online execution evidence.

use crate::calibration::{
    CalibratedKernelKind, CalibrationCostWorkload, CalibrationRecord, estimate_calibration_cost,
};
use crate::error::{Error, ErrorCode, Result};
use crate::ids::DeviceId;
use crate::performance_db::PerformanceStats;
use crate::scheduler::{
    SchedulerCandidate, SchedulerCostInput, SchedulerPolicyOptions, ScoreBreakdown, evaluate_scheduler_cost,
    select_scheduler_candidate,
};

/// How an adaptive dispatch's kernel work is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AdaptiveKernelWorkKind {
    /// `kernel_work` tiny dispatches.
    #[default]
    TinyDispatch,
    /// `kernel_work` bytes streamed.
    StreamingBytes,
    /// `kernel_work` integer operations.
    AluOperations,
}

/// Explicit workload description for [`Runtime::dispatch_adaptive`](crate::Runtime::dispatch_adaptive).
/// Byte and work counts are totals for one dispatch; `required_device_memory_bytes` is a
/// conservative residency requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdaptiveDispatchWorkload {
    /// Host-to-device bytes.
    pub input_bytes: u64,
    /// Device-to-host bytes.
    pub output_bytes: u64,
    /// Device-to-device bytes.
    pub device_copy_bytes: u64,
    /// Kernel work (nonzero).
    pub kernel_work: u64,
    /// Device memory the dispatch needs.
    pub required_device_memory_bytes: u64,
    /// Unit of `kernel_work`.
    pub kernel_kind: AdaptiveKernelWorkKind,
}

/// One device considered by [`select_adaptive_device`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementCandidate {
    /// Device.
    pub id: DeviceId,
    /// Driver version the calibration must match.
    pub driver_version: u32,
    /// Startup calibration.
    pub calibration: CalibrationRecord,
    /// Confident online execution statistics for this kernel and workload class.
    pub online_execution: Option<PerformanceStats>,
    /// Largest device-local heap headroom.
    pub available_device_memory_bytes: u64,
    /// Supports the kernel.
    pub supported: bool,
    /// Not lost or quarantined.
    pub healthy: bool,
    /// Permitted by policy.
    pub allowed: bool,
}

/// Outcome of [`select_adaptive_device`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementDecision {
    /// Selected device.
    pub device: DeviceId,
    /// Score of the selection; `None` when incomplete calibration kept the current device.
    pub score: Option<ScoreBreakdown>,
}

fn fits(candidate: &PlacementCandidate, workload: &AdaptiveDispatchWorkload) -> bool {
    workload.required_device_memory_bytes == 0
        || workload.required_device_memory_bytes < candidate.available_device_memory_bytes
}

fn eligible(candidate: &PlacementCandidate, workload: &AdaptiveDispatchWorkload) -> bool {
    candidate.supported && candidate.healthy && candidate.allowed && fits(candidate, workload)
}

/// Chooses the device with the lowest predicted finish time. Calibrated estimates are blended
/// with confident online execution statistics (at least five samples, weighted by
/// `count / (count + 5)`, using mean plus one standard deviation). If any eligible device lacks a
/// complete calibration the kernel's `current` device is kept. Deterministic: no exploration and no
/// hysteresis.
///
/// # Errors
///
/// [`ErrorCode::InvalidArgument`] for zero kernel work and [`ErrorCode::NoEligibleDevice`] when
/// `current` is missing or ineligible.
pub fn select_adaptive_device(
    candidates: &[PlacementCandidate],
    current: &DeviceId,
    workload: &AdaptiveDispatchWorkload,
) -> Result<PlacementDecision> {
    if workload.kernel_work == 0 {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "adaptive dispatch kernel work must be nonzero",
        ));
    }
    let ineligible_current = || {
        Error::new(
            ErrorCode::NoEligibleDevice,
            "the kernel's current device is not eligible for the adaptive workload",
        )
    };
    if !candidates
        .iter()
        .any(|c| c.id == *current && eligible(c, workload))
    {
        return Err(ineligible_current());
    }
    let kind = match workload.kernel_kind {
        AdaptiveKernelWorkKind::TinyDispatch => CalibratedKernelKind::TinyDispatch,
        AdaptiveKernelWorkKind::StreamingBytes => CalibratedKernelKind::Streaming,
        AdaptiveKernelWorkKind::AluOperations => CalibratedKernelKind::Alu,
    };
    let cost_workload = CalibrationCostWorkload {
        input_bytes: workload.input_bytes,
        output_bytes: workload.output_bytes,
        device_copy_bytes: workload.device_copy_bytes,
        kernel_work: workload.kernel_work,
        kernel_kind: kind,
    };
    let mut scored = Vec::with_capacity(candidates.len());
    let mut breakdowns = Vec::with_capacity(candidates.len());
    for candidate in candidates.iter().filter(|c| eligible(c, workload)) {
        let measured = estimate_calibration_cost(
            &candidate.calibration,
            &candidate.id.uuid,
            candidate.driver_version,
            &cost_workload,
        )?;
        if !measured.complete_record {
            return Ok(PlacementDecision {
                device: current.clone(),
                score: None,
            });
        }
        let mut execute_ns = measured.kernel_ns.unwrap_or(0.0);
        if let Some(online) = candidate.online_execution.as_ref().filter(|o| {
            o.count >= 5
                && o.ewma.is_finite()
                && o.ewma > 0.0
                && o.ew_variance.is_finite()
                && o.ew_variance >= 0.0
        }) {
            let confidence = online.count as f64 / (online.count as f64 + 5.0);
            let conservative = online.ewma + online.ew_variance.sqrt();
            execute_ns = (1.0 - confidence) * execute_ns + confidence * conservative;
        }
        let input_transfer_ns =
            measured.input_transfer_ns.unwrap_or(0.0) + measured.device_copy_ns.unwrap_or(0.0);
        let required_output_ns = measured.output_transfer_ns.unwrap_or(0.0);
        let cost = SchedulerCostInput {
            input_transfer_ns,
            execute_ns,
            required_output_ns,
            synchronization_ns: measured.submit_completion_ns.unwrap_or(0.0),
            predicted_usage_bytes: workload.required_device_memory_bytes as f64,
            effective_budget_bytes: candidate.available_device_memory_bytes as f64,
            memory_penalty_scale_ns: execute_ns + input_transfer_ns + required_output_ns,
            ..SchedulerCostInput::default()
        };
        let Ok(score) = evaluate_scheduler_cost(&cost) else {
            continue;
        };
        breakdowns.push((u64::from(candidate.id.index), score));
        scored.push(SchedulerCandidate {
            known_safe: candidate.id == *current,
            memory_fits: fits(candidate, workload),
            ..SchedulerCandidate::new(u64::from(candidate.id.index), score.predicted_finish_ns, 1)
        });
    }
    let policy = SchedulerPolicyOptions {
        deterministic: true,
        absolute_hysteresis_ns: 0.0,
        relative_hysteresis: 0.0,
        ..SchedulerPolicyOptions::default()
    };
    let selected = select_scheduler_candidate(&scored, &policy, Some(u64::from(current.index)), 0.0)?;
    let device = candidates
        .iter()
        .find(|c| u64::from(c.id.index) == selected.candidate_id)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InternalInvariantViolation,
                "scheduler selected an unknown device candidate",
            )
        })?;
    let score = breakdowns
        .iter()
        .find(|(id, _)| *id == selected.candidate_id)
        .map(|(_, score)| *score);
    Ok(PlacementDecision {
        device: device.id.clone(),
        score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(index: u32, rate: u64) -> PlacementCandidate {
        let uuid = char::from(b'0' + index as u8).to_string().repeat(32);
        let id = DeviceId::new(index, uuid.as_str());
        PlacementCandidate {
            calibration: CalibrationRecord {
                submit_overhead_ns: 100,
                tiny_dispatch_ns: 100,
                h2d_bytes_per_second: rate,
                d2h_bytes_per_second: rate,
                device_copy_bytes_per_second: rate,
                streaming_bytes_per_second: rate,
                alu_operations_per_second: rate,
                ..CalibrationRecord::new(uuid, 3)
            },
            id,
            driver_version: 3,
            online_execution: None,
            available_device_memory_bytes: 1 << 30,
            supported: true,
            healthy: true,
            allowed: true,
        }
    }

    fn workload() -> AdaptiveDispatchWorkload {
        AdaptiveDispatchWorkload {
            input_bytes: 1024,
            output_bytes: 1024,
            kernel_work: 1024,
            required_device_memory_bytes: 4096,
            kernel_kind: AdaptiveKernelWorkKind::StreamingBytes,
            ..AdaptiveDispatchWorkload::default()
        }
    }

    #[test]
    fn chooses_the_lowest_measured_finish_time_deterministically() {
        let slow = candidate(0, 1_000_000);
        let fast = candidate(1, 10_000_000);
        let first = select_adaptive_device(&[slow.clone(), fast.clone()], &slow.id, &workload()).unwrap();
        let second = select_adaptive_device(&[slow.clone(), fast.clone()], &slow.id, &workload()).unwrap();
        assert_eq!(first.device, fast.id);
        assert_eq!(first, second);
        assert!(first.score.unwrap().predicted_finish_ns > 0.0);
    }

    #[test]
    fn incomplete_calibration_keeps_the_bound_device() {
        let current = candidate(0, 1_000_000);
        let mut incomplete = candidate(1, 10_000_000);
        incomplete.calibration.alu_operations_per_second = 0;
        let selected =
            select_adaptive_device(&[current.clone(), incomplete], &current.id, &workload()).unwrap();
        assert_eq!(selected.device, current.id);
        assert!(selected.score.is_none());
    }

    #[test]
    fn filters_unsupported_unhealthy_disallowed_and_over_budget_candidates() {
        let current = candidate(0, 1_000_000);
        let variants: [fn(&mut PlacementCandidate); 5] = [
            |c| c.available_device_memory_bytes = 1024,
            |c| c.healthy = false,
            |c| c.supported = false,
            |c| c.allowed = false,
            |c| c.available_device_memory_bytes = 4096,
        ];
        for mutate in variants {
            let mut faster = candidate(1, 10_000_000);
            mutate(&mut faster);
            let selected =
                select_adaptive_device(&[current.clone(), faster], &current.id, &workload()).unwrap();
            assert_eq!(selected.device, current.id);
        }
    }

    #[test]
    fn rejects_invalid_workloads_and_an_ineligible_current_device() {
        let mut current = candidate(0, 1_000_000);
        let invalid = AdaptiveDispatchWorkload {
            kernel_work: 0,
            ..workload()
        };
        assert_eq!(
            select_adaptive_device(std::slice::from_ref(&current), &current.id, &invalid)
                .unwrap_err()
                .code(),
            ErrorCode::InvalidArgument
        );
        let missing = DeviceId::new(7, "7".repeat(32).as_str());
        assert_eq!(
            select_adaptive_device(std::slice::from_ref(&current), &missing, &workload())
                .unwrap_err()
                .code(),
            ErrorCode::NoEligibleDevice
        );
        current.healthy = false;
        assert_eq!(
            select_adaptive_device(std::slice::from_ref(&current), &current.id, &workload())
                .unwrap_err()
                .code(),
            ErrorCode::NoEligibleDevice
        );
    }

    #[test]
    fn blends_confident_online_evidence_with_calibration() {
        let stats = |count, ewma, ew_variance| PerformanceStats {
            count,
            ewma,
            ew_variance,
            ..PerformanceStats::default()
        };
        let mut current = candidate(0, 1_000_000_000_000);
        let mut alternate = candidate(1, 1_000_000);
        current.online_execution = Some(stats(100, 1_000_000.0, 10_000.0));
        alternate.online_execution = Some(stats(100, 100.0, 25.0));
        let execution_only = AdaptiveDispatchWorkload {
            input_bytes: 0,
            output_bytes: 0,
            required_device_memory_bytes: 0,
            ..workload()
        };
        let selected = select_adaptive_device(
            &[current.clone(), alternate.clone()],
            &current.id,
            &execution_only,
        )
        .unwrap();
        assert_eq!(selected.device, alternate.id);
        assert!(selected.score.is_some());

        current.online_execution = None;
        alternate.online_execution = Some(stats(4, 100.0, 25.0));
        let cold =
            select_adaptive_device(&[current.clone(), alternate], &current.id, &execution_only).unwrap();
        assert_eq!(cold.device, current.id, "four samples are not confident evidence");
        assert!(cold.score.is_some());
    }
}
