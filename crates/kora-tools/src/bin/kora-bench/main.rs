//! kora-bench: CPU, KORA Runtime, raw Vulkan, transfer, and Cluster microbenchmarks with a
//! schema-versioned JSON report on stdout.
//!
//! ```text
//! kora-bench [--iterations N] [--warmup N] [--smoke] [--gpu] [--profile-phases] [--cluster]
//! ```
//!
//! Exit codes: 0 success, 1 a workload failed, 2 invalid arguments, 3 the GPU path is unavailable.

mod cluster;
mod gpu;
mod host;
mod raw;
mod report;

use std::process::ExitCode;

use report::{Environment, Failure, GpuDevice, Options};

const USAGE: &str =
    "Usage: kora-bench [--iterations N] [--warmup N] [--smoke] [--gpu] [--profile-phases] [--cluster]
Runs CPU microbenchmarks; --gpu adds Vulkan workloads, --profile-phases adds
telemetry-instrumented dispatch phase diagnostics, and --cluster adds kora::Cluster
dispatches run locally and on a loopback kora-worker on the same device.";

fn parse(args: impl IntoIterator<Item = String>) -> Option<Options> {
    let mut options = Options {
        iterations: 100,
        warmup: 10,
        gpu: false,
        profile_phases: false,
        cluster: false,
    };
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => {
                options.iterations = 3;
                options.warmup = 1;
            }
            "--gpu" => options.gpu = true,
            "--cluster" => {
                options.gpu = true;
                options.cluster = true;
            }
            "--profile-phases" => {
                options.gpu = true;
                options.profile_phases = true;
            }
            "--iterations" | "--warmup" => {
                let value: usize = args
                    .next()?
                    .parse()
                    .ok()
                    .filter(|v| (1..=1_000_000).contains(v))?;
                if arg == "--iterations" {
                    options.iterations = value;
                } else {
                    options.warmup = value;
                }
            }
            _ => return None,
        }
    }
    Some(options)
}

fn cpu_model() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            let model = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !model.is_empty() {
                return model;
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if let Some((key, value)) = line.split_once(':')
                    && matches!(key.trim(), "model name" | "Hardware" | "Processor")
                    && !value.trim().is_empty()
                {
                    return value.trim().to_string();
                }
            }
        }
    }
    #[cfg(windows)]
    {
        if let Ok(model) = std::env::var("PROCESSOR_IDENTIFIER") {
            return model;
        }
    }
    "unknown".into()
}

fn environment() -> Environment {
    Environment {
        os: match std::env::consts::OS {
            "windows" => "windows",
            "macos" => "macos",
            "linux" => "linux",
            _ => "unknown",
        },
        architecture: match std::env::consts::ARCH {
            "aarch64" => "arm64",
            "x86_64" => "x86_64",
            "x86" => "x86",
            _ => "unknown",
        },
        compiler: env!("KORA_RUSTC_VERSION").into(),
        cpu_model: cpu_model(),
        build_type: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        build_flags: env!("KORA_BUILD_FLAGS").into(),
        source_revision: env!("KORA_SOURCE_REVISION").into(),
        logical_cpu_count: std::thread::available_parallelism().map_or(1, usize::from),
    }
}

fn exit(failure: Failure) -> ExitCode {
    match failure {
        Failure::Unavailable(message) => {
            eprintln!("{message}");
            ExitCode::from(3)
        }
        Failure::Failed(message) => {
            eprintln!("benchmark workload failed: {message}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let Some(options) = parse(args) else {
        eprintln!("invalid kora-bench arguments; use --help for usage");
        return ExitCode::from(2);
    };
    let mut results = Vec::new();
    if let Err(failure) = host::add_host_benchmarks(&mut results, &options) {
        return exit(failure);
    }
    let mut gpu = None;
    if options.gpu {
        let mut device = GpuDevice::default();
        let outcome =
            gpu::add_gpu_benchmarks(&mut results, &options, &mut device).and_then(|(runtime, selected)| {
                raw::add_raw_benchmarks(&mut results, &options, &selected, &mut device)?;
                drop(runtime);
                if options.cluster {
                    cluster::add_cluster_benchmarks(&mut results, &options)?;
                }
                Ok(())
            });
        if let Err(failure) = outcome {
            return exit(failure);
        }
        gpu = Some(device);
    }
    println!(
        "{}",
        report::render(&options, &environment(), gpu.as_ref(), &results)
    );
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn parses_flags_and_rejects_invalid_counts() {
        let smoke = parse(args(&["--smoke", "--cluster"])).unwrap();
        assert_eq!(
            (smoke.iterations, smoke.warmup, smoke.gpu, smoke.cluster),
            (3, 1, true, true)
        );
        let counted = parse(args(&["--iterations", "7", "--warmup", "2"])).unwrap();
        assert_eq!((counted.iterations, counted.warmup), (7, 2));
        assert!(parse(args(&["--iterations", "0"])).is_none());
        assert!(parse(args(&["--iterations"])).is_none());
        assert!(parse(args(&["--bogus"])).is_none());
    }
}
