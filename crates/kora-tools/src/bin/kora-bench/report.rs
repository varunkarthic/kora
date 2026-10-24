//! Timing loops, percentiles, and the JSON report.

use std::time::Instant;

use kora::json::{self, Value};
use kora_tools::json_object;

/// Command-line options.
#[derive(Debug, Clone)]
pub struct Options {
    pub iterations: usize,
    pub warmup: usize,
    pub gpu: bool,
    pub profile_phases: bool,
    pub cluster: bool,
}

/// Why a benchmark run stopped early.
#[derive(Debug)]
pub enum Failure {
    /// The GPU path is unavailable on this machine (exit code 3).
    Unavailable(String),
    /// A workload failed or produced a wrong result (exit code 1).
    Failed(String),
}

pub type Outcome<T = ()> = Result<T, Failure>;

/// Converts any error into a workload failure with context.
pub fn failed(context: &str) -> impl FnOnce(kora::Error) -> Failure + '_ {
    move |error| Failure::Failed(format!("{context}: {error}"))
}

/// Converts any error into an unavailable GPU path with context.
pub fn unavailable(context: &str) -> impl FnOnce(kora::Error) -> Failure + '_ {
    move |error| Failure::Unavailable(format!("{context}: {error}"))
}

/// One measured benchmark.
#[derive(Debug, Clone, Default)]
pub struct Measurement {
    pub name: String,
    pub samples: Vec<u64>,
    pub median: u64,
    pub p95: u64,
    pub bytes_transferred: u64,
    pub throughput_bytes_per_second: u64,
    pub thread_count: u32,
}

/// Nearest-rank percentile, matching the C++ kora-bench.
pub fn percentile(values: &[u64], percent: usize) -> u64 {
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    let rank = (percent * ordered.len()).div_ceil(100);
    ordered[rank.max(1) - 1]
}

impl Measurement {
    pub fn from_samples(name: impl Into<String>, samples: Vec<u64>) -> Self {
        Measurement {
            name: name.into(),
            median: percentile(&samples, 50),
            p95: percentile(&samples, 95),
            samples,
            ..Measurement::default()
        }
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.thread_count = threads as u32;
        self
    }
}

fn time(operation: &mut impl FnMut()) -> u64 {
    let begin = Instant::now();
    operation();
    begin.elapsed().as_nanos() as u64
}

/// Times `operation` after `warmup` untimed runs.
pub fn run(name: impl Into<String>, options: &Options, mut operation: impl FnMut()) -> Measurement {
    for _ in 0..options.warmup {
        operation();
    }
    let samples = (0..options.iterations).map(|_| time(&mut operation)).collect();
    Measurement::from_samples(name, samples)
}

/// Like [`run`], failing if `validate` rejects the output after any run (validation is untimed).
pub fn run_checked(
    name: impl Into<String>,
    options: &Options,
    mut operation: impl FnMut(),
    mut validate: impl FnMut() -> bool,
) -> Outcome<Measurement> {
    let name = name.into();
    let check = |ok: bool| {
        if ok {
            Ok(())
        } else {
            Err(Failure::Failed(format!(
                "benchmark correctness check failed: {name}"
            )))
        }
    };
    for _ in 0..options.warmup {
        operation();
        check(validate())?;
    }
    let mut samples = Vec::with_capacity(options.iterations);
    for _ in 0..options.iterations {
        samples.push(time(&mut operation));
        check(validate())?;
    }
    Ok(Measurement::from_samples(name, samples))
}

/// Collects samples that `operation` measures itself (device timestamps); `None` if any is missing.
pub fn run_timestamps(
    name: impl Into<String>,
    options: &Options,
    mut operation: impl FnMut() -> Option<u64>,
) -> Option<Measurement> {
    for _ in 0..options.warmup {
        operation()?;
    }
    let samples = (0..options.iterations)
        .map(|_| operation())
        .collect::<Option<Vec<_>>>()?;
    Some(Measurement::from_samples(name, samples))
}

/// [`run`] plus the bytes moved per run and the median throughput.
pub fn run_transfer(
    name: impl Into<String>,
    bytes: u64,
    options: &Options,
    operation: impl FnMut(),
) -> Measurement {
    let mut result = run(name, options, operation);
    result.bytes_transferred = bytes;
    result.throughput_bytes_per_second = if result.median == 0 {
        0
    } else {
        (u128::from(bytes) * 1_000_000_000 / u128::from(result.median)) as u64
    };
    result
}

/// Which clock or API boundary a result measures.
pub fn timing_domain(name: &str) -> &'static str {
    if name.starts_with("profile_kora_") {
        "runtime_dispatch_phase"
    } else if name.ends_with("_gpu_execution") {
        "gpu_execution_timestamp"
    } else if name.starts_with("cpu_") || name.starts_with("staging_ring_reserve_") {
        "host_cpu_execution"
    } else if name.starts_with("gpu_") || name.starts_with("raw_vulkan_") || name.starts_with("cluster_") {
        "host_submit_completion"
    } else if name.starts_with("transfer_") || name.starts_with("kora_transfer_copy_") {
        "host_transfer_completion"
    } else if name.starts_with("command_pool_") {
        "host_api_only"
    } else {
        "host_operation"
    }
}

/// Which implementation a GPU result exercises.
pub fn gpu_path(name: &str) -> Option<&'static str> {
    if name.starts_with("gpu_") {
        Some("kora_runtime")
    } else if name.starts_with("kora_transfer_copy_") || name.starts_with("transfer_") {
        Some("kora_transfer_engine")
    } else if name.starts_with("raw_vulkan_") || name.starts_with("command_pool_") {
        Some("raw_vulkan")
    } else {
        None
    }
}

/// Facts about the GPU the measurements ran on.
#[derive(Debug, Clone, Default)]
pub struct GpuDevice {
    pub index: u32,
    pub name: String,
    pub api_version: String,
    pub driver_name: String,
    pub device_kind: String,
    pub compute_queue_family: u32,
    pub kora_timestamp_sizes: Vec<u32>,
    pub raw_timestamp_sizes: Vec<u32>,
}

/// Build facts captured by `build.rs`.
pub struct Environment {
    pub os: &'static str,
    pub architecture: &'static str,
    pub compiler: String,
    pub cpu_model: String,
    pub build_type: &'static str,
    pub build_flags: String,
    pub source_revision: String,
    pub logical_cpu_count: usize,
}

fn number(value: impl ToString) -> Value {
    Value::Number(value.to_string())
}

fn string(value: impl Into<String>) -> Value {
    Value::String(value.into())
}

const COMPUTE_WORKLOADS: [&str; 5] = ["saxpy_", "reduction_", "tiled_scan_", "tiled_matmul_", "stencil_"];

fn compute_timestamp_workloads(results: &[Measurement], prefix: &str) -> Vec<Value> {
    results
        .iter()
        .filter(|r| {
            r.name.ends_with("_gpu_execution")
                && COMPUTE_WORKLOADS
                    .iter()
                    .any(|w| r.name.starts_with(&format!("{prefix}{w}")))
        })
        .map(|r| string(r.name.clone()))
        .collect()
}

/// Serializes the schema-version-4 report.
pub fn render(
    options: &Options,
    environment: &Environment,
    gpu: Option<&GpuDevice>,
    results: &[Measurement],
) -> String {
    let mut members = vec![
        ("schema_version", number(4)),
        ("implementation", string("rust")),
        ("clock", string("steady_clock")),
        ("iterations", number(options.iterations)),
        ("warmup", number(options.warmup)),
        ("diagnostic_profile", Value::Bool(options.profile_phases)),
        (
            "environment",
            json_object(vec![
                ("os", string(environment.os)),
                ("architecture", string(environment.architecture)),
                ("compiler", string(environment.compiler.clone())),
                ("cpu_model", string(environment.cpu_model.clone())),
                ("build_type", string(environment.build_type)),
                ("build_flags", string(environment.build_flags.clone())),
                ("source_revision", string(environment.source_revision.clone())),
                ("logical_cpu_count", number(environment.logical_cpu_count)),
            ]),
        ),
    ];
    if let Some(gpu) = gpu {
        let sizes = |sizes: &[u32]| Value::Array(sizes.iter().map(|s| number(*s)).collect());
        let kora_compute = compute_timestamp_workloads(results, "gpu_");
        let raw_compute = compute_timestamp_workloads(results, "raw_vulkan_");
        members.push((
            "gpu_measurements",
            json_object(vec![
                ("backend", string("vulkan")),
                ("device_index", number(gpu.index)),
                ("device_name", string(gpu.name.clone())),
                ("api_version", string(gpu.api_version.clone())),
                ("driver_name", string(gpu.driver_name.clone())),
                ("device_kind", string(gpu.device_kind.clone())),
                ("compute_queue_family", number(gpu.compute_queue_family)),
                ("same_physical_device_for_kora_and_raw_vulkan", Value::Bool(true)),
                (
                    "kora_gpu_timestamps_available",
                    Value::Bool(!gpu.kora_timestamp_sizes.is_empty()),
                ),
                (
                    "raw_gpu_timestamps_available",
                    Value::Bool(!gpu.raw_timestamp_sizes.is_empty()),
                ),
                ("kora_gpu_timestamp_sizes", sizes(&gpu.kora_timestamp_sizes)),
                ("raw_gpu_timestamp_sizes", sizes(&gpu.raw_timestamp_sizes)),
                (
                    "kora_gpu_compute_timestamps_available",
                    Value::Bool(!kora_compute.is_empty()),
                ),
                ("kora_gpu_compute_timestamp_workloads", Value::Array(kora_compute)),
                (
                    "raw_gpu_compute_timestamps_available",
                    Value::Bool(!raw_compute.is_empty()),
                ),
                ("raw_gpu_compute_timestamp_workloads", Value::Array(raw_compute)),
                (
                    "paths",
                    Value::Array(vec![
                        string("kora_runtime"),
                        string("kora_transfer_engine"),
                        string("raw_vulkan"),
                    ]),
                ),
            ]),
        ));
    }
    let results = results
        .iter()
        .map(|r| {
            let mut fields = vec![
                ("name", string(r.name.clone())),
                ("unit", string("ns")),
                ("median", number(r.median)),
                ("p95", number(r.p95)),
                ("timing_domain", string(timing_domain(&r.name))),
            ];
            if let (Some(gpu), Some(path)) = (gpu, gpu_path(&r.name)) {
                fields.push(("path", string(path)));
                fields.push(("device_index", number(gpu.index)));
            }
            if r.bytes_transferred != 0 {
                fields.push(("bytes_transferred", number(r.bytes_transferred)));
                fields.push((
                    "throughput_bytes_per_second",
                    number(r.throughput_bytes_per_second),
                ));
            }
            if r.thread_count != 0 {
                fields.push(("thread_count", number(r.thread_count)));
            }
            fields.push((
                "samples",
                Value::Array(r.samples.iter().map(|s| number(*s)).collect()),
            ));
            json_object(fields)
        })
        .collect();
    members.push(("results", Value::Array(results)));
    let mut out = String::new();
    json::write_value(&mut out, &json_object(members));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_use_nearest_rank() {
        assert_eq!(percentile(&[5], 50), 5);
        assert_eq!(percentile(&[4, 1, 3, 2], 50), 2);
        assert_eq!(percentile(&[4, 1, 3, 2], 95), 4);
        let values: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile(&values, 50), 50);
        assert_eq!(percentile(&values, 95), 95);
    }

    #[test]
    fn classifies_timing_domains_and_paths() {
        assert_eq!(
            timing_domain("gpu_vector_add_64_gpu_execution"),
            "gpu_execution_timestamp"
        );
        assert_eq!(timing_domain("raw_vulkan_copy_4096"), "host_submit_completion");
        assert_eq!(
            timing_domain("transfer_direct_upload_4096"),
            "host_transfer_completion"
        );
        assert_eq!(timing_domain("cpu_saxpy_4096"), "host_cpu_execution");
        assert_eq!(timing_domain("allocator_churn_4096"), "host_operation");
        assert_eq!(gpu_path("command_pool_reuse_steady_state"), Some("raw_vulkan"));
        assert_eq!(gpu_path("scheduler_cost"), None);
    }
}
