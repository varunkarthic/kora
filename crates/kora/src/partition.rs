//! Contiguous work partitioning for caller-partitionable tasks.
//!
//! [`DynamicPartitioner`] keeps a per-device throughput EWMA and splits work proportionally into
//! non-overlapping, granularity-aligned slices. It is advisory: callers must make their task
//! range-aware and merge outputs.

use std::collections::HashMap;

use crate::error::{Error, ErrorCode, Result, Status};
use crate::ids::DeviceId;

/// A device available for one partitioned task.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionDevice {
    /// The device.
    pub id: DeviceId,
    /// Bytes available for this task (normally a fresh heap budget snapshot).
    pub available_memory_bytes: u64,
    /// Cold-start throughput estimate in work units per nanosecond.
    pub initial_throughput: f64,
}

/// What to partition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartitionRequest {
    /// Total work units.
    pub total_work_units: u64,
    /// Device memory per work unit.
    pub bytes_per_work_unit: u64,
    /// Slices are multiples of this (the last slice absorbs any tail).
    pub granularity: u64,
    /// Smallest slice worth running.
    pub minimum_chunk_units: u64,
    /// Measured unsplit duration, or 0 to estimate from throughput.
    pub estimated_unsplit_duration_ns: f64,
    /// Fixed cost of splitting (merging, extra submissions).
    pub partition_overhead_ns: f64,
    /// Decline a split unless it predicts at least this fractional improvement.
    pub minimum_gain_fraction: f64,
}

impl Default for PartitionRequest {
    fn default() -> Self {
        Self {
            total_work_units: 0,
            bytes_per_work_unit: 0,
            granularity: 1,
            minimum_chunk_units: 1,
            estimated_unsplit_duration_ns: 0.0,
            partition_overhead_ns: 0.0,
            minimum_gain_fraction: 0.05,
        }
    }
}

/// One contiguous slice.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionSlice {
    /// Target device.
    pub device: DeviceId,
    /// First work unit.
    pub offset: u64,
    /// Number of work units.
    pub work_units: u64,
    /// Throughput share.
    pub weight: f64,
}

/// Partitioning decision.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PartitionPlan {
    /// Slices covering `[0, total_work_units)` exactly.
    pub slices: Vec<PartitionSlice>,
    /// Predicted completion time of the slowest slice plus overhead.
    pub estimated_makespan_ns: f64,
    /// `false` when the plan is a single-device fallback.
    pub split: bool,
}

/// Throughput-proportional partitioner. See the [module documentation](self).
#[derive(Debug, Clone)]
pub struct DynamicPartitioner {
    alpha: f64,
    throughput: HashMap<DeviceId, f64>,
}

impl Default for DynamicPartitioner {
    fn default() -> Self {
        Self::new(0.25)
    }
}

fn valid_rate(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

impl DynamicPartitioner {
    /// Creates a partitioner with EWMA weight `observation_alpha` in `(0, 1]`.
    pub fn new(observation_alpha: f64) -> Self {
        Self {
            alpha: observation_alpha,
            throughput: HashMap::new(),
        }
    }

    /// Records that `device` completed `work_units` in `duration_ns`.
    pub fn observe(&mut self, device: &DeviceId, work_units: u64, duration_ns: f64) -> Status {
        if work_units == 0 || !valid_rate(duration_ns) || !(self.alpha > 0.0 && self.alpha <= 1.0) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "partition observation must have positive finite values",
            ));
        }
        let sample = work_units as f64 / duration_ns;
        if !valid_rate(sample) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "partition throughput is invalid",
            ));
        }
        let alpha = self.alpha;
        self.throughput
            .entry(device.clone())
            .and_modify(|rate| *rate = alpha * sample + (1.0 - alpha) * *rate)
            .or_insert(sample);
        Ok(())
    }

    /// Current throughput estimate for `device`.
    pub fn throughput(&self, device: &DeviceId) -> Option<f64> {
        self.throughput.get(device).copied()
    }

    /// Plans slices for `request` over `devices`.
    pub fn partition(
        &self,
        request: &PartitionRequest,
        devices: &[PartitionDevice],
    ) -> Result<PartitionPlan> {
        let estimates = [
            request.estimated_unsplit_duration_ns,
            request.partition_overhead_ns,
            request.minimum_gain_fraction,
        ];
        if estimates.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "partition estimates must be finite and nonnegative",
            ));
        }
        if request.total_work_units == 0
            || request.bytes_per_work_unit == 0
            || request.granularity == 0
            || request.minimum_chunk_units == 0
            || request.minimum_gain_fraction >= 1.0
            || devices.is_empty()
        {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "partition request or device set is empty or invalid",
            ));
        }
        let chunks = request.total_work_units / request.granularity;
        let tail = request.total_work_units % request.granularity;
        if chunks == 0 || request.total_work_units < request.minimum_chunk_units {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "work extent is smaller than its partition granularity",
            ));
        }
        struct Candidate<'a> {
            device: &'a PartitionDevice,
            rate: f64,
            max_units: u64,
        }
        let eligible: Vec<Candidate<'_>> = devices
            .iter()
            .filter_map(|device| {
                let rate = self
                    .throughput
                    .get(&device.id)
                    .copied()
                    .unwrap_or(device.initial_throughput);
                let max_units = device.available_memory_bytes / request.bytes_per_work_unit;
                (valid_rate(rate) && max_units >= request.minimum_chunk_units).then_some(Candidate {
                    device,
                    rate,
                    max_units,
                })
            })
            .collect();
        if eligible.is_empty() {
            return Err(Error::new(
                ErrorCode::OutOfMemoryBudget,
                "no device can fit the minimum task chunk",
            ));
        }
        let time = |c: &Candidate<'_>| c.max_units.min(request.total_work_units) as f64 / c.rate;
        let best = eligible
            .iter()
            .min_by(|l, r| time(l).total_cmp(&time(r)))
            .expect("eligible is non-empty");
        let unsplit_time = if request.estimated_unsplit_duration_ns > 0.0 {
            request.estimated_unsplit_duration_ns
        } else {
            request.total_work_units as f64 / best.rate
        };
        let fallback = (best.max_units >= request.total_work_units).then(|| PartitionPlan {
            slices: vec![PartitionSlice {
                device: best.device.id.clone(),
                offset: 0,
                work_units: request.total_work_units,
                weight: 1.0,
            }],
            estimated_makespan_ns: unsplit_time,
            split: false,
        });
        let fallback_or_error = || {
            fallback.clone().ok_or_else(|| {
                Error::new(
                    ErrorCode::OutOfMemoryBudget,
                    "no single device can fit the unsplit task",
                )
            })
        };
        if eligible.len() < 2 || chunks < 2 || request.total_work_units < 2 * request.minimum_chunk_units {
            return fallback_or_error();
        }
        let min_chunks = request.minimum_chunk_units.div_ceil(request.granularity).max(1);
        let assigned = min_chunks.saturating_mul(eligible.len() as u64);
        if chunks < min_chunks * 2 || assigned > chunks {
            return fallback_or_error();
        }
        let total_rate: f64 = eligible.iter().map(|c| c.rate).sum();
        let weights: Vec<f64> = eligible.iter().map(|c| c.rate / total_rate).collect();
        let mut allocations = vec![min_chunks; eligible.len()];
        for _ in 0..chunks - assigned {
            let selected = (0..eligible.len())
                .filter(|&i| allocations[i] < eligible[i].max_units / request.granularity)
                .max_by(|&a, &b| {
                    let deficit = |i: usize| weights[i] * chunks as f64 - allocations[i] as f64;
                    // Earlier candidates win ties, matching a strict greater-than scan.
                    deficit(a).total_cmp(&deficit(b)).then(b.cmp(&a))
                });
            match selected {
                Some(index) => allocations[index] += 1,
                None => return fallback_or_error(),
            }
        }
        let mut plan = PartitionPlan {
            split: true,
            ..Default::default()
        };
        let mut offset = 0;
        let mut slowest: f64 = 0.0;
        for (index, candidate) in eligible.iter().enumerate() {
            let mut units = allocations[index] * request.granularity;
            if index + 1 == eligible.len() {
                units += tail;
            }
            plan.slices.push(PartitionSlice {
                device: candidate.device.id.clone(),
                offset,
                work_units: units,
                weight: weights[index],
            });
            offset += units;
            slowest = slowest.max(units as f64 / candidate.rate);
        }
        plan.estimated_makespan_ns = slowest + request.partition_overhead_ns;
        if offset != request.total_work_units {
            return fallback_or_error();
        }
        if plan.estimated_makespan_ns >= unsplit_time * (1.0 - request.minimum_gain_fraction)
            && let Some(fallback) = fallback
        {
            return Ok(fallback);
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> PartitionRequest {
        PartitionRequest {
            total_work_units: 1000,
            bytes_per_work_unit: 4,
            granularity: 10,
            minimum_chunk_units: 100,
            estimated_unsplit_duration_ns: 1000.0,
            partition_overhead_ns: 10.0,
            minimum_gain_fraction: 0.05,
        }
    }

    fn device(id: &DeviceId, memory: u64) -> PartitionDevice {
        PartitionDevice {
            id: id.clone(),
            available_memory_bytes: memory,
            initial_throughput: 1.0,
        }
    }

    #[test]
    fn proportional_weights_create_exact_contiguous_coverage() {
        let mut partitioner = DynamicPartitioner::default();
        let (fast, slow) = (DeviceId::new(0, "fast"), DeviceId::new(1, "slow"));
        partitioner.observe(&fast, 200, 100.0).unwrap();
        partitioner.observe(&slow, 100, 100.0).unwrap();
        let plan = partitioner
            .partition(&request(), &[device(&fast, 10000), device(&slow, 10000)])
            .unwrap();
        assert!(plan.split);
        assert_eq!(plan.slices.len(), 2);
        assert!((plan.slices[0].weight - 2.0 / 3.0).abs() < 1e-12);
        assert!((plan.slices[1].weight - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(plan.slices[0].offset, 0);
        assert_eq!(plan.slices[1].offset, plan.slices[0].work_units);
        assert_eq!(plan.slices[0].work_units + plan.slices[1].work_units, 1000);
        assert_eq!(plan.slices[0].work_units % 10, 0);
    }

    #[test]
    fn new_observations_rebalance_by_ewma() {
        let mut partitioner = DynamicPartitioner::new(1.0);
        let (a, b) = (DeviceId::new(0, "a"), DeviceId::new(1, "b"));
        partitioner.observe(&a, 100, 100.0).unwrap();
        partitioner.observe(&b, 100, 100.0).unwrap();
        let first = partitioner
            .partition(&request(), &[device(&a, 10000), device(&b, 10000)])
            .unwrap();
        assert_eq!(first.slices[0].work_units, first.slices[1].work_units);
        partitioner.observe(&a, 200, 100.0).unwrap();
        let second = partitioner
            .partition(&request(), &[device(&a, 10000), device(&b, 10000)])
            .unwrap();
        assert!(second.slices[0].work_units > second.slices[1].work_units);
    }

    #[test]
    fn falls_back_when_split_is_unprofitable_or_only_one_device_fits() {
        let partitioner = DynamicPartitioner::default();
        let (a, b) = (DeviceId::new(0, "a"), DeviceId::new(1, "b"));
        let slow = PartitionRequest {
            estimated_unsplit_duration_ns: 2000.0,
            partition_overhead_ns: 5000.0,
            ..request()
        };
        let plan = partitioner
            .partition(&slow, &[device(&a, 10000), device(&b, 10000)])
            .unwrap();
        assert!(!plan.split);
        assert_eq!(plan.slices.len(), 1);
        let plan = partitioner
            .partition(&request(), &[device(&a, 10000), device(&b, 100)])
            .unwrap();
        assert!(!plan.split);
        assert_eq!(plan.slices[0].device, a);
    }

    #[test]
    fn enforces_memory_and_reports_impossible_unsplit_task() {
        let partitioner = DynamicPartitioner::default();
        let (a, b) = (DeviceId::new(0, "a"), DeviceId::new(1, "b"));
        let plan = partitioner
            .partition(&request(), &[device(&a, 2000), device(&b, 2000)])
            .unwrap();
        assert!(plan.split);
        assert!(plan.slices.iter().all(|s| s.work_units * 4 <= 2000));
        assert_eq!(
            partitioner
                .partition(&request(), &[device(&a, 100)])
                .unwrap_err()
                .code(),
            ErrorCode::OutOfMemoryBudget
        );
    }

    #[test]
    fn rejects_invalid_inputs_and_observations() {
        let mut partitioner = DynamicPartitioner::default();
        assert_eq!(
            partitioner
                .observe(&DeviceId::new(0, "a"), 1, 0.0)
                .unwrap_err()
                .code(),
            ErrorCode::InvalidArgument
        );
        let invalid = PartitionRequest {
            granularity: 0,
            ..request()
        };
        assert_eq!(
            partitioner
                .partition(&invalid, &[device(&DeviceId::new(0, "a"), 10000)])
                .unwrap_err()
                .code(),
            ErrorCode::InvalidArgument
        );
        assert!(partitioner.partition(&request(), &[]).is_err());
        let negative = PartitionRequest {
            partition_overhead_ns: -1.0,
            ..request()
        };
        assert!(
            partitioner
                .partition(&negative, &[device(&DeviceId::new(0, "a"), 10000)])
                .is_err()
        );
        let mut bad_alpha = DynamicPartitioner::new(0.0);
        assert!(bad_alpha.observe(&DeviceId::new(0, "a"), 1, 1.0).is_err());
    }

    #[test]
    fn tail_goes_to_the_last_slice() {
        let partitioner = DynamicPartitioner::default();
        let (a, b) = (DeviceId::new(0, "a"), DeviceId::new(1, "b"));
        let odd = PartitionRequest {
            total_work_units: 1007,
            estimated_unsplit_duration_ns: 5000.0,
            ..request()
        };
        let plan = partitioner
            .partition(&odd, &[device(&a, 1 << 20), device(&b, 1 << 20)])
            .unwrap();
        assert!(plan.split);
        assert_eq!(plan.slices.iter().map(|s| s.work_units).sum::<u64>(), 1007);
        assert_eq!(plan.slices[1].work_units % 10, 7);
    }

    proptest::proptest! {
        #[test]
        fn slices_always_cover_the_extent_exactly(
            total in 100u64..100_000,
            granularity in 1u64..64,
            rates in proptest::collection::vec(0.01f64..10.0, 1..5),
        ) {
            let devices: Vec<_> = rates.iter().enumerate().map(|(i, &rate)| PartitionDevice {
                id: DeviceId::new(i as u32, format!("d{i}")),
                available_memory_bytes: u64::MAX / 2,
                initial_throughput: rate,
            }).collect();
            let request = PartitionRequest { total_work_units: total, bytes_per_work_unit: 1, granularity, minimum_chunk_units: 1, estimated_unsplit_duration_ns: 1e12, ..Default::default() };
            if let Ok(plan) = DynamicPartitioner::default().partition(&request, &devices) {
                let mut offset = 0;
                for slice in &plan.slices {
                    proptest::prop_assert_eq!(slice.offset, offset);
                    offset += slice.work_units;
                }
                proptest::prop_assert_eq!(offset, total);
            }
        }
    }
}
