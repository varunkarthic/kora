//! `kora::Cluster` SAXPY (y += 2x) run locally and on an in-process loopback worker that uses the
//! same device. The remote rows measure the worker protocol's cost (request, execution on the
//! worker's own Runtime, and the output returned to the host), not a second machine's speed.

use std::time::{Duration, Instant};

use kora::kernel::spirv::words_from_bytes;
use kora::log::Level;
use kora::{Cluster, ClusterOptions, OffloadPolicy, RuntimeOptions, WorkerOptions, WorkerServer};

use crate::gpu::{COMPUTE_SIZES, SAXPY, SaxpyParams, saxpy_metadata};
use crate::report::{Failure, Measurement, Options, Outcome, failed, run_checked, unavailable};

/// Default runtime options (including startup calibration, as in the C++ benchmark) without
/// log output.
fn quiet() -> RuntimeOptions {
    RuntimeOptions {
        log_level: Some(Level::Off),
        ..RuntimeOptions::default()
    }
}

pub fn add_cluster_benchmarks(results: &mut Vec<Measurement>, options: &Options) -> Outcome {
    let worker = WorkerServer::start(WorkerOptions {
        listen: "127.0.0.1:0".into(),
        runtime: quiet(),
        ..WorkerOptions::default()
    })
    .map_err(unavailable("cluster benchmarks unavailable"))?;
    let words = words_from_bytes(SAXPY).map_err(failed("cluster kernel"))?;
    for (policy, prefix) in [
        (OffloadPolicy::LocalOnly, "cluster_local"),
        (OffloadPolicy::PreferRemote, "cluster_remote_loopback"),
    ] {
        let cluster = Cluster::new(ClusterOptions {
            local: quiet(),
            workers: vec![format!("127.0.0.1:{}", worker.port())],
            policy,
            ..ClusterOptions::default()
        })
        .map_err(unavailable("cluster benchmarks unavailable"))?;
        if policy == OffloadPolicy::PreferRemote {
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline && !cluster.workers()[0].connected {
                std::thread::sleep(Duration::from_millis(5));
            }
            if !cluster.workers()[0].connected {
                return Err(Failure::Unavailable(
                    "cluster benchmarks unavailable: loopback worker did not connect".into(),
                ));
            }
        }
        let kernel = cluster
            .load_kernel(&words, "main", saxpy_metadata())
            .map_err(failed("cluster kernel load"))?;
        for count in COMPUTE_SIZES {
            let n = count as usize;
            let x = cluster
                .create_buffer::<f32>(u64::from(count))
                .map_err(failed("cluster buffer"))?;
            let y = cluster
                .create_buffer::<f32>(u64::from(count))
                .map_err(failed("cluster buffer"))?;
            x.write(&vec![1.0; n]).map_err(failed("cluster buffer"))?;
            y.write(&vec![0.0; n]).map_err(failed("cluster buffer"))?;
            let expected = std::cell::Cell::new(0.0_f32);
            let mut error = None;
            let measured = run_checked(
                format!("{prefix}_saxpy_{count}_dispatch_wait"),
                options,
                || {
                    let status = cluster
                        .dispatch(&kernel)
                        .global(u64::from(count))
                        .read(&x)
                        .read_write(&y)
                        .push(&SaxpyParams { scale: 2.0, count })
                        .submit()
                        .and_then(|event| event.wait());
                    match status {
                        Ok(()) => expected.set(expected.get() + 2.0),
                        Err(e) => {
                            error.get_or_insert(e);
                        }
                    }
                },
                || {
                    y.read()
                        .is_ok_and(|v| v.iter().all(|value| *value == expected.get()))
                },
            );
            if let Some(error) = error {
                return Err(Failure::Failed(format!("cluster dispatch failed: {error}")));
            }
            results.push(measured?);
        }
        let stats = cluster.stats();
        if policy == OffloadPolicy::PreferRemote && (stats.local_dispatches != 0 || stats.failovers != 0) {
            return Err(Failure::Failed(
                "loopback cluster benchmark did not run every dispatch remotely".into(),
            ));
        }
    }
    Ok(())
}
