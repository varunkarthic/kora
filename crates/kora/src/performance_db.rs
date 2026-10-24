//! Bounded, persistent performance observations.
//!
//! Each key keeps an EWMA and exponentially weighted variance, min/max, and exact nearest-rank
//! median/p95 over a bounded window of recent samples. A median-absolute-deviation filter rejects
//! outliers; stale entries reset; the number of keys is bounded (least recently observed evicted).
//! Snapshots are schema-versioned JSON written atomically.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use crate::error::{Error, ErrorCode, Result, Status};
use crate::fsutil;
use crate::json::{self, Value};

const SCHEMA: u64 = 1;
const MAXIMUM_FILE_BYTES: usize = 4 * 1024 * 1024;

/// Timing domain of an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PerformanceTimingDomain {
    /// GPU timestamp interval.
    DeviceExecution = 0,
    /// Host submission.
    HostSubmission = 1,
    /// Host-visible transfer.
    HostTransfer = 2,
    /// Submission to completion.
    QueueCompletion = 3,
}

/// Structured observation identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerformanceObservationKey {
    /// Lower-case hex device UUID.
    pub device_uuid: String,
    /// Driver version.
    pub driver_version: u32,
    /// Runtime schema (non-zero).
    pub runtime_schema: u32,
    /// Semantic kernel fingerprint (non-zero).
    pub kernel_identity: u64,
    /// Workload class.
    pub workload_class: u8,
    /// Workload size bucket.
    pub workload_bucket: u8,
    /// Timing domain.
    pub timing_domain: PerformanceTimingDomain,
}

/// Encodes a key as `obs-v1/<uuid>/d<driver>/s<schema>/k<identity>/c<class>/b<bucket>/t<domain>`.
pub fn performance_observation_key(key: &PerformanceObservationKey) -> Result<String> {
    if key.device_uuid.is_empty() || key.runtime_schema == 0 || key.kernel_identity == 0 {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "performance observation identity is incomplete",
        ));
    }
    if !key
        .device_uuid
        .bytes()
        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "performance observation device identity is invalid",
        ));
    }
    Ok(format!(
        "obs-v1/{}/d{}/s{}/k{:016x}/c{}/b{}/t{}",
        key.device_uuid,
        key.driver_version,
        key.runtime_schema,
        key.kernel_identity,
        key.workload_class,
        key.workload_bucket,
        key.timing_domain as u8
    ))
}

/// Statistics of one key.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PerformanceStats {
    /// Accepted samples.
    pub count: u64,
    /// Rejected outliers.
    pub rejected_outliers: u64,
    /// Exponentially weighted mean.
    pub ewma: f64,
    /// Exponentially weighted variance.
    pub ew_variance: f64,
    /// Smallest sample.
    pub minimum: f64,
    /// Largest sample.
    pub maximum: f64,
    /// Nearest-rank median of the window.
    pub median: f64,
    /// Nearest-rank 95th percentile of the window.
    pub p95: f64,
    /// Time of the latest accepted sample.
    pub last_observed_unix_ms: u64,
}

/// Database options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerformanceDatabaseOptions {
    /// EWMA weight in `(0, 1]`.
    pub ewma_alpha: f64,
    /// Samples kept per key (non-zero).
    pub sample_window: usize,
    /// Entries older than this reset or are ignored.
    pub maximum_age_ms: u64,
    /// Maximum number of keys (non-zero).
    pub maximum_entries: usize,
}

impl Default for PerformanceDatabaseOptions {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.2,
            sample_window: 256,
            maximum_age_ms: 30 * 24 * 60 * 60 * 1000,
            maximum_entries: 256,
        }
    }
}

#[derive(Default)]
struct Entry {
    stats: PerformanceStats,
    samples: std::collections::VecDeque<f64>,
}

/// Nearest-rank quantile; `scratch` is reused to avoid allocation.
fn quantile(values: impl Iterator<Item = f64>, q: f64, scratch: &mut Vec<f64>) -> f64 {
    scratch.clear();
    scratch.extend(values);
    if scratch.is_empty() {
        return 0.0;
    }
    let rank = ((q * scratch.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(scratch.len() - 1);
    *scratch.select_nth_unstable_by(rank, f64::total_cmp).1
}

impl Entry {
    fn refresh_quantiles(&mut self, scratch: &mut Vec<f64>) {
        self.stats.median = quantile(self.samples.iter().copied(), 0.5, scratch);
        self.stats.p95 = quantile(self.samples.iter().copied(), 0.95, scratch);
    }
}

/// Milliseconds since the Unix epoch (0 if the clock is before it).
pub fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Thread-safe performance database. See the [module documentation](self).
pub struct PerformanceDatabase {
    options: PerformanceDatabaseOptions,
    entries: RwLock<HashMap<String, Arc<Mutex<Entry>>>>,
}

impl std::fmt::Debug for PerformanceDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerformanceDatabase")
            .field("entries", &self.len())
            .finish()
    }
}

impl Default for PerformanceDatabase {
    fn default() -> Self {
        Self::new(PerformanceDatabaseOptions::default())
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|p| p.into_inner())
}

impl PerformanceDatabase {
    /// Creates an empty database; invalid options fall back to the defaults.
    pub fn new(options: PerformanceDatabaseOptions) -> Self {
        let valid = options.ewma_alpha > 0.0
            && options.ewma_alpha <= 1.0
            && options.sample_window > 0
            && options.maximum_entries > 0;
        Self {
            options: if valid {
                options
            } else {
                PerformanceDatabaseOptions::default()
            },
            entries: RwLock::new(HashMap::new()),
        }
    }

    fn entry(&self, key: &str) -> Option<Arc<Mutex<Entry>>> {
        self.entries
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(key)
            .cloned()
    }

    /// Records one sample.
    pub fn observe(&self, key: &str, sample: f64, now_unix_ms: u64) -> Status {
        if key.is_empty() || !sample.is_finite() || sample < 0.0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "performance observation needs a key and finite nonnegative sample",
            ));
        }
        let entry = match self.entry(key) {
            Some(entry) => entry,
            None => {
                let mut entries = self.entries.write().unwrap_or_else(|p| p.into_inner());
                if let Some(existing) = entries.get(key) {
                    Arc::clone(existing)
                } else {
                    if entries.len() >= self.options.maximum_entries {
                        let oldest = entries
                            .iter()
                            .min_by_key(|(_, e)| lock(e).stats.last_observed_unix_ms)
                            .map(|(k, _)| k.clone());
                        if let Some(oldest) = oldest {
                            entries.remove(&oldest);
                        }
                    }
                    Arc::clone(entries.entry(key.to_owned()).or_default())
                }
            }
        };
        let mut entry = lock(&entry);
        let options = &self.options;
        if entry.stats.count != 0
            && now_unix_ms > entry.stats.last_observed_unix_ms
            && now_unix_ms - entry.stats.last_observed_unix_ms > options.maximum_age_ms
        {
            *entry = Entry::default();
        }
        let mut scratch = Vec::with_capacity(entry.samples.len());
        if entry.samples.len() >= 5 {
            let center = quantile(entry.samples.iter().copied(), 0.5, &mut scratch);
            let mad = quantile(
                entry.samples.iter().map(|v| (v - center).abs()),
                0.5,
                &mut scratch,
            );
            let scale = (1.4826 * mad).max(1.0f64.max(center.abs() * 0.01));
            if (sample - center).abs() > 6.0 * scale {
                entry.stats.rejected_outliers += 1;
                return Ok(());
            }
        }
        let stats = &mut entry.stats;
        if stats.count == 0 {
            stats.ewma = sample;
            stats.minimum = sample;
            stats.maximum = sample;
        } else {
            let alpha = options.ewma_alpha;
            let delta = sample - stats.ewma;
            stats.ewma += alpha * delta;
            stats.ew_variance = (1.0 - alpha) * (stats.ew_variance + alpha * delta * delta);
            stats.minimum = stats.minimum.min(sample);
            stats.maximum = stats.maximum.max(sample);
        }
        stats.count += 1;
        stats.last_observed_unix_ms = now_unix_ms;
        if entry.samples.len() == options.sample_window {
            entry.samples.pop_front();
        }
        entry.samples.push_back(sample);
        entry.refresh_quantiles(&mut scratch);
        Ok(())
    }

    /// Statistics of a key.
    pub fn query(&self, key: &str) -> Result<PerformanceStats> {
        self.entry(key)
            .map(|entry| lock(&entry).stats)
            .ok_or_else(|| Error::new(ErrorCode::InvalidArgument, "unknown performance key"))
    }

    /// Statistics only when fresh and backed by at least `minimum_samples` samples.
    pub fn query_confident(
        &self,
        key: &str,
        now_unix_ms: u64,
        minimum_samples: u64,
    ) -> Option<PerformanceStats> {
        let stats = self.query(key).ok()?;
        let fresh = now_unix_ms >= stats.last_observed_unix_ms
            && now_unix_ms - stats.last_observed_unix_ms <= self.options.maximum_age_ms;
        (stats.count >= minimum_samples && fresh).then_some(stats)
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.entries.read().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Writes a snapshot atomically.
    pub fn save(&self, path: &Path) -> Status {
        if path.as_os_str().is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "performance database path is empty",
            ));
        }
        let mut rows: Vec<(String, PerformanceStats, Vec<f64>)> = {
            let entries = self.entries.read().unwrap_or_else(|p| p.into_inner());
            entries
                .iter()
                .map(|(key, entry)| {
                    let entry = lock(entry);
                    (key.clone(), entry.stats, entry.samples.iter().copied().collect())
                })
                .collect()
        };
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        let entries = rows
            .into_iter()
            .map(|(key, s, samples)| {
                Value::Object(vec![
                    ("key".into(), Value::String(key)),
                    ("count".into(), Value::uint(s.count)),
                    ("rejected".into(), Value::uint(s.rejected_outliers)),
                    ("ewma".into(), Value::float(s.ewma)),
                    ("variance".into(), Value::float(s.ew_variance)),
                    ("min".into(), Value::float(s.minimum)),
                    ("max".into(), Value::float(s.maximum)),
                    ("last".into(), Value::uint(s.last_observed_unix_ms)),
                    (
                        "samples".into(),
                        Value::Array(samples.into_iter().map(Value::float).collect()),
                    ),
                ])
            })
            .collect();
        let document = Value::Object(vec![
            ("schema".into(), Value::uint(SCHEMA)),
            ("entries".into(), Value::Array(entries)),
        ])
        .to_string();
        if document.len() > MAXIMUM_FILE_BYTES {
            return Err(Error::new(
                ErrorCode::IoError,
                "performance database exceeds size limit",
            ));
        }
        fsutil::write_atomic(path, document.as_bytes())
    }

    /// Replaces the contents from a snapshot, skipping expired rows. A malformed, incompatible, or
    /// unreadable file is an [`ErrorCode::IoError`] and leaves the current contents untouched.
    pub fn load(&self, path: &Path, now_unix_ms: u64) -> Result<usize> {
        let corrupt = |why: &str| Error::new(ErrorCode::IoError, format!("performance database {why}"));
        let bytes = fsutil::read_bounded(path, MAXIMUM_FILE_BYTES as u64)?;
        let limits = json::Limits {
            maximum_bytes: MAXIMUM_FILE_BYTES,
            ..Default::default()
        };
        let root = json::parse_with(&bytes, limits).map_err(|_| corrupt("JSON is corrupt"))?;
        if root.find("schema").and_then(Value::as_u64) != Some(SCHEMA) {
            return Err(corrupt("schema is incompatible"));
        }
        let rows = root
            .find("entries")
            .and_then(Value::as_array)
            .ok_or_else(|| corrupt("entries are invalid"))?;
        if rows.len() > self.options.maximum_entries {
            return Err(corrupt("contains more entries than the configured bound"));
        }
        let mut loaded: HashMap<String, Arc<Mutex<Entry>>> = HashMap::new();
        let mut scratch = Vec::new();
        for row in rows {
            let u = |name: &str| row.find(name).and_then(Value::as_u64);
            let f = |name: &str| row.find(name).and_then(Value::as_f64);
            let (
                Some(key),
                Some(count),
                Some(rejected),
                Some(last),
                Some(ewma),
                Some(variance),
                Some(minimum),
                Some(maximum),
            ) = (
                row.find("key").and_then(Value::as_str).filter(|k| !k.is_empty()),
                u("count"),
                u("rejected"),
                u("last"),
                f("ewma"),
                f("variance"),
                f("min"),
                f("max"),
            )
            else {
                return Err(corrupt("entry is malformed"));
            };
            if count == 0 || ewma < 0.0 || variance < 0.0 || minimum < 0.0 || maximum < minimum {
                return Err(corrupt("statistics are out of range"));
            }
            if now_unix_ms > last && now_unix_ms - last > self.options.maximum_age_ms {
                continue;
            }
            let samples = row
                .find("samples")
                .and_then(Value::as_array)
                .ok_or_else(|| corrupt("sample window is invalid"))?;
            if samples.len() > self.options.sample_window || samples.len() as u64 > count {
                return Err(corrupt("sample window is invalid"));
            }
            let mut entry = Entry {
                stats: PerformanceStats {
                    count,
                    rejected_outliers: rejected,
                    ewma,
                    ew_variance: variance,
                    minimum,
                    maximum,
                    last_observed_unix_ms: last,
                    ..Default::default()
                },
                samples: Default::default(),
            };
            for sample in samples {
                match sample.as_f64() {
                    Some(value) if value >= 0.0 => entry.samples.push_back(value),
                    _ => return Err(corrupt("contains invalid sample")),
                }
            }
            entry.refresh_quantiles(&mut scratch);
            if loaded
                .insert(key.to_owned(), Arc::new(Mutex::new(entry)))
                .is_some()
            {
                return Err(corrupt("contains duplicate keys"));
            }
        }
        let count = loaded.len();
        *self.entries.write().unwrap_or_else(|p| p.into_inner()) = loaded;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsutil::temp_dir;

    #[test]
    fn computes_ewma_variance_and_window_quantiles() {
        let database = PerformanceDatabase::new(PerformanceDatabaseOptions {
            ewma_alpha: 0.5,
            ..Default::default()
        });
        database.observe("kernel/a", 10.0, 1).unwrap();
        database.observe("kernel/a", 20.0, 2).unwrap();
        let s = database.query("kernel/a").unwrap();
        assert_eq!(
            (
                s.count,
                s.ewma,
                s.ew_variance,
                s.median,
                s.p95,
                s.minimum,
                s.maximum
            ),
            (2, 15.0, 25.0, 10.0, 20.0, 10.0, 20.0)
        );
    }

    #[test]
    fn rejects_invalid_observations_and_uses_safe_defaults() {
        let database = PerformanceDatabase::new(PerformanceDatabaseOptions {
            ewma_alpha: 0.0,
            sample_window: 0,
            ..Default::default()
        });
        assert!(database.observe("", 1.0, 1).is_err());
        assert!(database.observe("valid", -1.0, 1).is_err());
        assert!(database.observe("valid", f64::INFINITY, 1).is_err());
        database.observe("valid", 10.0, 1).unwrap();
        database.observe("valid", 20.0, 2).unwrap();
        assert_eq!(database.query("valid").unwrap().ewma, 12.0);
    }

    #[test]
    fn rejects_mad_outliers_without_poisoning_estimate() {
        let database = PerformanceDatabase::default();
        for index in 0..10 {
            database.observe("copy/64k", 100.0, index + 1).unwrap();
        }
        database.observe("copy/64k", 10000.0, 11).unwrap();
        let s = database.query("copy/64k").unwrap();
        assert_eq!((s.count, s.rejected_outliers, s.ewma), (10, 1, 100.0));
    }

    #[test]
    fn bounded_window_and_aging_reset_stale_history() {
        let database = PerformanceDatabase::new(PerformanceDatabaseOptions {
            sample_window: 3,
            maximum_age_ms: 10,
            ..Default::default()
        });
        for index in 0..4u64 {
            database.observe("x", 10.0 + index as f64, index + 1).unwrap();
        }
        assert_eq!(database.query("x").unwrap().median, 12.0);
        database.observe("x", 50.0, 100).unwrap();
        let s = database.query("x").unwrap();
        assert_eq!((s.count, s.ewma), (1, 50.0));
    }

    #[test]
    fn rejected_outlier_does_not_prevent_aging() {
        let database = PerformanceDatabase::new(PerformanceDatabaseOptions {
            maximum_age_ms: 10,
            ..Default::default()
        });
        for index in 0..5 {
            database.observe("regime", 100.0, index + 1).unwrap();
        }
        database.observe("regime", 10000.0, 6).unwrap();
        assert_eq!(database.query("regime").unwrap().rejected_outliers, 1);
        database.observe("regime", 50.0, 20).unwrap();
        let s = database.query("regime").unwrap();
        assert_eq!((s.count, s.rejected_outliers, s.ewma), (1, 0, 50.0));
    }

    #[test]
    fn persists_and_treats_corruption_as_recoverable() {
        let directory = temp_dir("perfdb");
        let path = directory.join("stats.json");
        let source = PerformanceDatabase::default();
        source
            .observe("kernel/\"quoted", 12.123_456_789_012_3, 100)
            .unwrap();
        source.observe("kernel/\"quoted", 13.5, 101).unwrap();
        source.save(&path).unwrap();
        source.observe("kernel/\"quoted", 12.75, 102).unwrap();
        source.save(&path).unwrap();
        let loaded = PerformanceDatabase::default();
        assert_eq!(loaded.load(&path, 103).unwrap(), 1);
        let (a, b) = (
            loaded.query("kernel/\"quoted").unwrap(),
            source.query("kernel/\"quoted").unwrap(),
        );
        assert_eq!(a, b);
        std::fs::write(&path, "{broken").unwrap();
        assert!(loaded.load(&path, 104).is_err());
        assert_eq!(loaded.len(), 1);
        std::fs::write(&path, r#"{"schema":2,"entries":[]}"#).unwrap();
        assert!(loaded.load(&path, 104).is_err());
        std::fs::write(&path, r#"{"schema":1,"entries":[{"key":"k","count":1,"rejected":0,"ewma":1,"variance":0,"min":2,"max":1,"last":1,"samples":[]}]}"#).unwrap();
        assert!(loaded.load(&path, 104).is_err());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn ignores_expired_persisted_rows() {
        let directory = temp_dir("perfdb-aging");
        let path = directory.join("stats.json");
        let source = PerformanceDatabase::default();
        source.observe("old", 1.0, 100).unwrap();
        source.save(&path).unwrap();
        let loaded = PerformanceDatabase::new(PerformanceDatabaseOptions {
            maximum_age_ms: 10,
            ..Default::default()
        });
        assert_eq!(loaded.load(&path, 1000).unwrap(), 0);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn observation_keys_separate_identity_work_and_timing() {
        let base = PerformanceObservationKey {
            device_uuid: "00112233445566778899aabbccddeeff".into(),
            driver_version: 17,
            runtime_schema: 2,
            kernel_identity: 0x1234,
            workload_class: 1,
            workload_bucket: 10,
            timing_domain: PerformanceTimingDomain::DeviceExecution,
        };
        let encoded = performance_observation_key(&base).unwrap();
        assert_eq!(
            encoded,
            "obs-v1/00112233445566778899aabbccddeeff/d17/s2/k0000000000001234/c1/b10/t0"
        );
        let variants: Vec<PerformanceObservationKey> = vec![
            PerformanceObservationKey {
                driver_version: 18,
                ..base.clone()
            },
            PerformanceObservationKey {
                timing_domain: PerformanceTimingDomain::HostTransfer,
                ..base.clone()
            },
            PerformanceObservationKey {
                workload_bucket: 11,
                ..base.clone()
            },
            PerformanceObservationKey {
                workload_class: 2,
                ..base.clone()
            },
            PerformanceObservationKey {
                runtime_schema: 3,
                ..base.clone()
            },
            PerformanceObservationKey {
                kernel_identity: 0x1235,
                ..base.clone()
            },
            PerformanceObservationKey {
                device_uuid: "10112233445566778899aabbccddeeff".into(),
                ..base.clone()
            },
        ];
        for variant in variants {
            assert_ne!(performance_observation_key(&variant).unwrap(), encoded);
        }
        assert!(
            performance_observation_key(&PerformanceObservationKey {
                device_uuid: "GPU 0".into(),
                ..base.clone()
            })
            .is_err()
        );
        assert!(
            performance_observation_key(&PerformanceObservationKey {
                kernel_identity: 0,
                ..base
            })
            .is_err()
        );
    }

    #[test]
    fn requires_fresh_repeated_observations_and_bounds_key_count() {
        let database = PerformanceDatabase::new(PerformanceDatabaseOptions {
            maximum_age_ms: 10,
            maximum_entries: 2,
            ..Default::default()
        });
        database.observe("old", 100.0, 1).unwrap();
        assert!(database.query_confident("old", 2, 5).is_none());
        for sample in 0..5u64 {
            database
                .observe("old", 100.0 + sample as f64, 2 + sample)
                .unwrap();
        }
        assert!(database.query_confident("old", 7, 5).is_some());
        assert!(database.query_confident("old", 100, 5).is_none());
        database.observe("new", 10.0, 8).unwrap();
        database.observe("newest", 5.0, 9).unwrap();
        assert_eq!(database.len(), 2);
        assert!(database.query("old").is_err());
        assert!(database.query("new").is_ok() && database.query("newest").is_ok());
    }

    #[test]
    fn concurrent_keys_and_updates_remain_consistent() {
        let database = PerformanceDatabase::default();
        std::thread::scope(|scope| {
            for worker in 0..8u64 {
                let database = &database;
                scope.spawn(move || {
                    for sample in 0..500u64 {
                        database
                            .observe(&format!("key/{worker}"), sample as f64 + 1.0, sample + 1)
                            .unwrap();
                    }
                });
                scope.spawn(move || {
                    for sample in 0..500u64 {
                        database
                            .observe("shared", 1.0, worker * 500 + sample + 1)
                            .unwrap();
                    }
                });
            }
        });
        assert_eq!(database.len(), 9);
        for worker in 0..8 {
            assert_eq!(database.query(&format!("key/{worker}")).unwrap().count, 500);
        }
        assert_eq!(database.query("shared").unwrap().count, 4000);
    }

    #[test]
    fn quantile_matches_nearest_rank_definition() {
        let mut scratch = Vec::new();
        let values = [5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(quantile(values.iter().copied(), 0.5, &mut scratch), 3.0);
        assert_eq!(quantile(values.iter().copied(), 0.95, &mut scratch), 5.0);
        assert_eq!(quantile(values.iter().copied(), 0.0, &mut scratch), 1.0);
        assert_eq!(quantile(std::iter::empty(), 0.5, &mut scratch), 0.0);
    }
}
