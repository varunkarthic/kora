//! Direction-specific transfer latency and bandwidth estimates.
//!
//! Each path keeps one EWMA sample per logarithmic size bucket; a least-squares fit across the
//! bucket means yields one monotone `latency + ns_per_byte * bytes` curve per path.

use std::sync::Mutex;

use crate::error::{Error, ErrorCode, Result, Status};

/// Number of logarithmic size buckets.
pub const TRANSFER_SIZE_BUCKET_COUNT: usize = 64;

/// A power-of-two size bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferSizeBucket {
    /// `floor(log2(bytes))` (0 and 1 share bucket 0).
    pub index: u8,
    /// Smallest size in the bucket.
    pub lower_bytes: u64,
    /// Largest size in the bucket.
    pub upper_bytes: u64,
}

/// Bucket for a transfer size.
pub fn transfer_size_bucket(bytes: u64) -> TransferSizeBucket {
    let index = if bytes == 0 {
        0
    } else {
        63 - bytes.leading_zeros() as u8
    };
    let lower = if index == 0 { 0 } else { 1u64 << index };
    let upper = if index == 63 {
        u64::MAX
    } else {
        (1u64 << (index + 1)) - 1
    };
    TransferSizeBucket {
        index,
        lower_bytes: lower,
        upper_bytes: upper,
    }
}

/// A transfer route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferPath {
    /// Host to device.
    HostToDevice,
    /// Device to host.
    DeviceToHost,
    /// Within a device.
    DeviceCopy,
    /// Between devices through host memory.
    CrossDeviceViaHost,
    /// Direct host access to unified memory.
    UmaDirect,
}

/// Per-bucket statistics.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TransferBucketStats {
    /// Samples recorded.
    pub observations: u64,
    /// EWMA of sample sizes.
    pub ewma_bytes: f64,
    /// EWMA of sample durations.
    pub ewma_duration_ns: f64,
}

/// Fitted linear model.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TransferCostModel {
    /// Samples recorded on the path.
    pub observations: u64,
    /// Fixed latency.
    pub latency_ns: f64,
    /// Marginal cost.
    pub nanoseconds_per_byte: f64,
}

/// Model options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferModelOptions {
    /// EWMA weight in `(0, 1]`.
    pub ewma_alpha: f64,
    /// Multiplier (>= 1) on the byte cost of [`TransferPath::UmaDirect`].
    pub uma_direct_penalty: f64,
}

impl Default for TransferModelOptions {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.2,
            uma_direct_penalty: 1.0,
        }
    }
}

#[derive(Clone, Copy)]
struct PathState {
    buckets: [TransferBucketStats; TRANSFER_SIZE_BUCKET_COUNT],
    model: TransferCostModel,
    initialized: bool,
}

impl Default for PathState {
    fn default() -> Self {
        Self {
            buckets: [TransferBucketStats::default(); TRANSFER_SIZE_BUCKET_COUNT],
            model: Default::default(),
            initialized: false,
        }
    }
}

impl PathState {
    fn refit(&mut self) {
        let measured: Vec<&TransferBucketStats> =
            self.buckets.iter().filter(|b| b.observations > 0).collect();
        if measured.is_empty() {
            return;
        }
        let n = measured.len() as f64;
        let mean_bytes = measured.iter().map(|b| b.ewma_bytes).sum::<f64>() / n;
        let mean_time = measured.iter().map(|b| b.ewma_duration_ns).sum::<f64>() / n;
        let (mut covariance, mut variance) = (0.0, 0.0);
        for bucket in &measured {
            let x = bucket.ewma_bytes - mean_bytes;
            covariance += x * (bucket.ewma_duration_ns - mean_time);
            variance += x * x;
        }
        let mut slope = if variance > 0.0 {
            covariance / variance
        } else {
            mean_time / mean_bytes.max(1.0)
        };
        if !slope.is_finite() || slope < 0.0 {
            slope = 0.0;
        }
        let mut latency = mean_time - slope * mean_bytes;
        if !latency.is_finite() || latency < 0.0 {
            latency = 0.0;
        }
        self.model.latency_ns = latency;
        self.model.nanoseconds_per_byte = slope;
    }
}

/// Thread-safe set of per-path models.
#[derive(Debug)]
pub struct TransferModelSet {
    state: Mutex<(TransferModelOptions, [PathState; 5])>,
}

impl std::fmt::Debug for PathState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PathState").field("model", &self.model).finish()
    }
}

impl Default for TransferModelSet {
    fn default() -> Self {
        Self::new(TransferModelOptions::default())
    }
}

impl TransferModelSet {
    /// Creates empty models; invalid options fall back to the defaults.
    pub fn new(options: TransferModelOptions) -> Self {
        let valid = options.ewma_alpha > 0.0
            && options.ewma_alpha <= 1.0
            && options.uma_direct_penalty.is_finite()
            && options.uma_direct_penalty >= 1.0;
        let options = if valid {
            options
        } else {
            TransferModelOptions::default()
        };
        Self {
            state: Mutex::new((options, [PathState::default(); 5])),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, (TransferModelOptions, [PathState; 5])> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Records a measured transfer.
    pub fn observe(&self, path: TransferPath, bytes: u64, duration_ns: f64) -> Status {
        if bytes == 0 || !duration_ns.is_finite() || duration_ns < 0.0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "transfer sample needs a valid path, nonzero size, and finite duration",
            ));
        }
        let bucket = transfer_size_bucket(bytes).index as usize;
        let mut guard = self.lock();
        let alpha = guard.0.ewma_alpha;
        let state = &mut guard.1[path as usize];
        let stats = &mut state.buckets[bucket];
        if stats.observations == 0 {
            stats.ewma_bytes = bytes as f64;
            stats.ewma_duration_ns = duration_ns;
        } else {
            stats.ewma_bytes += alpha * (bytes as f64 - stats.ewma_bytes);
            stats.ewma_duration_ns += alpha * (duration_ns - stats.ewma_duration_ns);
        }
        stats.observations += 1;
        state.initialized = true;
        state.model.observations += 1;
        state.refit();
        Ok(())
    }

    /// Estimated duration of a transfer; an error for unmeasured paths.
    pub fn estimate_ns(&self, path: TransferPath, bytes: u64) -> Result<f64> {
        let guard = self.lock();
        let state = &guard.1[path as usize];
        if !state.initialized {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "transfer path has no measured model",
            ));
        }
        let penalty = if path == TransferPath::UmaDirect {
            guard.0.uma_direct_penalty
        } else {
            1.0
        };
        let estimate = state.model.latency_ns + penalty * state.model.nanoseconds_per_byte * bytes as f64;
        if !estimate.is_finite() {
            return Err(Error::new(ErrorCode::Overflow, "transfer estimate overflowed"));
        }
        Ok(estimate)
    }

    /// The fitted model of a measured path.
    pub fn model(&self, path: TransferPath) -> Result<TransferCostModel> {
        let guard = self.lock();
        let state = &guard.1[path as usize];
        if !state.initialized {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "transfer path has no measured model",
            ));
        }
        Ok(state.model)
    }

    /// Statistics of one bucket.
    pub fn bucket_stats(&self, path: TransferPath, bucket: u8) -> Result<TransferBucketStats> {
        if usize::from(bucket) >= TRANSFER_SIZE_BUCKET_COUNT {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "transfer path or size bucket is invalid",
            ));
        }
        Ok(self.lock().1[path as usize].buckets[usize::from(bucket)])
    }

    /// Updates the UMA-direct byte-cost multiplier (>= 1).
    pub fn set_uma_direct_penalty(&self, penalty: f64) -> Status {
        if !penalty.is_finite() || penalty < 1.0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "UMA direct penalty must be finite and at least one",
            ));
        }
        self.lock().0.uma_direct_penalty = penalty;
        Ok(())
    }

    /// Current UMA-direct penalty.
    pub fn uma_direct_penalty(&self) -> f64 {
        self.lock().0.uma_direct_penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATHS: [TransferPath; 5] = [
        TransferPath::HostToDevice,
        TransferPath::DeviceToHost,
        TransferPath::DeviceCopy,
        TransferPath::CrossDeviceViaHost,
        TransferPath::UmaDirect,
    ];

    #[test]
    fn buckets_cover_logarithmic_boundaries_and_full_range() {
        assert_eq!(transfer_size_bucket(0).index, 0);
        let one = transfer_size_bucket(1);
        assert_eq!((one.index, one.lower_bytes, one.upper_bytes), (0, 0, 1));
        assert_eq!(transfer_size_bucket(2).index, 1);
        assert_eq!(transfer_size_bucket(3).index, 1);
        assert_eq!(transfer_size_bucket(4).index, 2);
        assert_eq!(transfer_size_bucket(1024).index, 10);
        assert_eq!(transfer_size_bucket(u64::MAX).index, 63);
        assert_eq!(transfer_size_bucket(u64::MAX).upper_bytes, u64::MAX);
    }

    #[test]
    fn fits_latency_and_bandwidth_separately_for_every_direction() {
        let models = TransferModelSet::default();
        for path in PATHS {
            for bytes in [4096u64, 65536, 1_048_576] {
                models.observe(path, bytes, 500.0 + bytes as f64 * 0.025).unwrap();
            }
            let model = models.model(path).unwrap();
            assert_eq!(model.observations, 3);
            assert!((model.latency_ns - 500.0).abs() < 1e-7);
            assert!((model.nanoseconds_per_byte - 0.025).abs() < 1e-10);
            let mut prior = f64::MIN;
            for bytes in [0, 1, 512, 4096, 16384, 65536, 1_048_576, u64::MAX] {
                let estimate = models.estimate_ns(path, bytes).unwrap();
                assert!(estimate >= prior);
                prior = estimate;
            }
        }
    }

    #[test]
    fn tracks_per_path_bucket_statistics_and_ewma() {
        let models = TransferModelSet::new(TransferModelOptions {
            ewma_alpha: 0.5,
            ..Default::default()
        });
        models.observe(TransferPath::HostToDevice, 1024, 100.0).unwrap();
        models.observe(TransferPath::HostToDevice, 1500, 200.0).unwrap();
        let bucket = transfer_size_bucket(1024).index;
        let stats = models.bucket_stats(TransferPath::HostToDevice, bucket).unwrap();
        assert_eq!(stats.observations, 2);
        assert_eq!(stats.ewma_bytes, 1262.0);
        assert_eq!(stats.ewma_duration_ns, 150.0);
        assert_eq!(
            models
                .bucket_stats(TransferPath::DeviceToHost, bucket)
                .unwrap()
                .observations,
            0
        );
        assert!(models.estimate_ns(TransferPath::DeviceToHost, 1024).is_err());
    }

    #[test]
    fn applies_uma_penalty_only_to_direct_byte_cost() {
        let models = TransferModelSet::new(TransferModelOptions {
            uma_direct_penalty: 2.0,
            ..Default::default()
        });
        for path in [TransferPath::HostToDevice, TransferPath::UmaDirect] {
            models.observe(path, 4096, 1000.0).unwrap();
            models.observe(path, 65536, 2000.0).unwrap();
        }
        let staged = models.estimate_ns(TransferPath::HostToDevice, 32768).unwrap();
        let direct = models.estimate_ns(TransferPath::UmaDirect, 32768).unwrap();
        assert!(direct > staged);
        assert_eq!(models.uma_direct_penalty(), 2.0);
        models.set_uma_direct_penalty(1.25).unwrap();
        assert_eq!(models.uma_direct_penalty(), 1.25);
    }

    #[test]
    fn rejects_invalid_samples_buckets_and_penalties() {
        let models = TransferModelSet::default();
        assert!(models.observe(TransferPath::HostToDevice, 0, 10.0).is_err());
        assert!(models.observe(TransferPath::HostToDevice, 1, -1.0).is_err());
        assert!(models.bucket_stats(TransferPath::HostToDevice, 64).is_err());
        assert!(models.set_uma_direct_penalty(0.9).is_err());
        let fallback = TransferModelSet::new(TransferModelOptions {
            ewma_alpha: 0.0,
            uma_direct_penalty: 0.5,
        });
        assert_eq!(fallback.uma_direct_penalty(), 1.0);
    }

    #[test]
    fn single_bucket_uses_mean_rate() {
        let models = TransferModelSet::default();
        models.observe(TransferPath::DeviceCopy, 1000, 2000.0).unwrap();
        let model = models.model(TransferPath::DeviceCopy).unwrap();
        assert_eq!(model.nanoseconds_per_byte, 2.0);
        assert_eq!(model.latency_ns, 0.0);
    }
}
