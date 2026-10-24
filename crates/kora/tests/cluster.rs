//! Cluster end to end: loopback workers, failover under random disconnects, hot swap, split
//! dispatches across devices, the status stream, and local device-loss recovery. Every test runs
//! under the validation layer when it is installed and fails on any validation error.

mod common;

use std::io::{BufRead, BufReader};
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use kora::kernel::spirv::words_from_bytes;
use kora::{
    Buffer, Cluster, ClusterKernel, ClusterOptions, DevicePolicy, ErrorCode, Event, KernelAccess,
    KernelBindingKind, KernelBindingMetadata, KernelMetadata, OffloadPolicy, PushConstantFieldMetadata,
    RuntimeOptions, WorkerOptions, WorkerServer,
};

const SAXPY: &[u8] = include_bytes!("fixtures/saxpy.spv");
const SPLIT_SAXPY: &[u8] = include_bytes!("fixtures/split_saxpy.spv");
const COUNT: u32 = 4099;

fn runtime_options(name: &str) -> RuntimeOptions {
    RuntimeOptions {
        enable_validation: Some(true),
        cache_dir: Some(
            std::env::temp_dir()
                .join(format!("kora-cluster-test-{}-{name}", std::process::id()))
                .to_string_lossy()
                .into_owned(),
        ),
        enable_startup_calibration: Some(false),
        ..RuntimeOptions::default()
    }
}

fn binding(binding: u32, name: &str, access: KernelAccess) -> KernelBindingMetadata {
    KernelBindingMetadata {
        set: 0,
        binding,
        name: name.into(),
        kind: KernelBindingKind::StorageBuffer,
        access,
    }
}

fn field(name: &str, type_name: &str, offset: u32) -> PushConstantFieldMetadata {
    PushConstantFieldMetadata {
        name: name.into(),
        type_name: type_name.into(),
        offset,
        size: 4,
    }
}

fn saxpy_metadata() -> KernelMetadata {
    KernelMetadata {
        kernel_id: "integration.cluster.saxpy".into(),
        local_size: [64, 1, 1],
        bindings: vec![
            binding(0, "input_x", KernelAccess::Read),
            binding(1, "input_y", KernelAccess::ReadWrite),
        ],
        push_constant_size: 8,
        push_constants: vec![field("scale", "f32", 0), field("count", "u32", 4)],
        ..KernelMetadata::default()
    }
}

fn split_saxpy_metadata() -> KernelMetadata {
    let mut metadata = saxpy_metadata();
    metadata.kernel_id = "integration.cluster.split_saxpy".into();
    metadata.push_constant_size = 12;
    metadata.push_constants.push(field("base", "u32", 8));
    metadata
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SaxpyParams {
    scale: f32,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SplitParams {
    scale: f32,
    count: u32,
    base: u32,
}

/// Runs `test` with validation-error capture (which also serializes the tests of this file).
fn with_validation(test: impl FnOnce()) {
    let (errors, _guard) = common::ValidationErrors::capture();
    test();
    errors.assert_none();
}

/// Starts a loopback worker, retrying briefly while a previous instance releases the port.
/// `None` when this machine has no usable Vulkan device.
fn start_worker(port: u16, token: &str) -> Option<WorkerServer> {
    let options = WorkerOptions {
        listen: format!("127.0.0.1:{port}"),
        token: token.into(),
        runtime: runtime_options("worker"),
        io_timeout: Duration::from_secs(2),
        ..WorkerOptions::default()
    };
    for _ in 0..50 {
        match WorkerServer::start(options.clone()) {
            Ok(server) => return Some(server),
            Err(error)
                if matches!(
                    error.code(),
                    ErrorCode::NoVulkanImplementation | ErrorCode::NoEligibleDevice
                ) =>
            {
                eprintln!("skipping: {error}");
                return None;
            }
            Err(_) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    None
}

fn fast_options(port: u16, policy: OffloadPolicy) -> ClusterOptions {
    ClusterOptions {
        local: runtime_options("cluster"),
        workers: vec![format!("127.0.0.1:{port}")],
        policy,
        connect_timeout: Duration::from_millis(200),
        reconnect_interval: Duration::from_millis(20),
        request_timeout: Duration::from_secs(3),
        heartbeat_interval: Duration::from_millis(100),
        ..ClusterOptions::default()
    }
}

fn split_options() -> ClusterOptions {
    ClusterOptions {
        local: RuntimeOptions {
            device_policy: Some(DevicePolicy::AllEligible),
            ..runtime_options("split")
        },
        policy: OffloadPolicy::LocalOnly,
        split_min_elements: 1024,
        split_overhead: Duration::ZERO,
        device_rescan_interval: Duration::ZERO,
        ..ClusterOptions::default()
    }
}

fn wait_until(limit: Duration, mut predicate: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if predicate() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    predicate()
}

fn any_connected(cluster: &Cluster) -> bool {
    cluster.workers().iter().any(|w| w.connected)
}

struct Fixture {
    cluster: Cluster,
    kernel: ClusterKernel,
    x: Buffer<f32>,
    y: Buffer<f32>,
}

fn fixture(options: ClusterOptions) -> Fixture {
    let cluster = Cluster::new(options).expect("cluster creation");
    let kernel = cluster
        .load_kernel(&words_from_bytes(SAXPY).unwrap(), "main", saxpy_metadata())
        .unwrap();
    let x = cluster.create_buffer::<f32>(u64::from(COUNT)).unwrap();
    let y = cluster.create_buffer::<f32>(u64::from(COUNT)).unwrap();
    x.write(&vec![1.0; COUNT as usize]).unwrap();
    y.write(&vec![0.0; COUNT as usize]).unwrap();
    Fixture {
        cluster,
        kernel,
        x,
        y,
    }
}

/// Submits `jobs` chained `y += x` dispatches; each depends on the previous one through `y`.
fn run_chain(f: &Fixture, jobs: usize, mut between: impl FnMut(usize)) {
    let mut events = Vec::with_capacity(jobs);
    for job in 0..jobs {
        let event = f
            .cluster
            .dispatch(&f.kernel)
            .global(u64::from(COUNT))
            .read(&f.x)
            .read_write(&f.y)
            .push(&SaxpyParams {
                scale: 1.0,
                count: COUNT,
            })
            .submit()
            .unwrap();
        events.push(event);
        between(job);
    }
    for event in events {
        event.wait().unwrap();
    }
}

fn expect_value(y: &Buffer<f32>, expected: f32) {
    let values = y.read().unwrap();
    for (i, value) in values.iter().enumerate() {
        assert_eq!(*value, expected, "index {i}");
    }
}

#[test]
fn offloads_to_a_connected_worker_with_identical_results() {
    with_validation(|| {
        let Some(worker) = start_worker(0, "") else { return };
        let f = fixture(fast_options(worker.port(), OffloadPolicy::PreferRemote));
        assert!(wait_until(Duration::from_secs(10), || any_connected(&f.cluster)));
        run_chain(&f, 20, |_| {});
        expect_value(&f.y, 20.0);
        let stats = f.cluster.stats();
        assert_eq!(stats.remote_dispatches, 20);
        assert_eq!(stats.failovers, 0);
        assert_eq!(worker.stats().dispatches, 20);
        assert_eq!(worker.stats().kernels_loaded, 1);
        // x never changes, so after the first dispatch only y travels; y's result is cached too.
        assert!(f.cluster.workers()[0].cached_bytes >= 2 * u64::from(COUNT) * 4);
    });
}

#[test]
fn picks_up_a_worker_that_starts_after_the_cluster() {
    with_validation(|| {
        let Some(probe) = start_worker(0, "") else { return };
        let port = probe.port();
        drop(probe);
        let f = fixture(fast_options(port, OffloadPolicy::PreferRemote));
        run_chain(&f, 5, |_| {}); // No worker yet: local.
        assert_eq!(f.cluster.stats().remote_dispatches, 0);
        let _worker = start_worker(port, "").expect("worker restarts on its port");
        assert!(wait_until(Duration::from_secs(10), || any_connected(&f.cluster)));
        run_chain(&f, 5, |_| {});
        expect_value(&f.y, 10.0);
        assert!(f.cluster.stats().remote_dispatches > 0);
    });
}

#[test]
fn survives_worker_stop_restart_and_detach_mid_stream() {
    with_validation(|| {
        let Some(worker) = start_worker(0, "") else { return };
        let port = worker.port();
        let mut worker = Some(worker);
        let f = fixture(fast_options(port, OffloadPolicy::PreferRemote));
        assert!(wait_until(Duration::from_secs(10), || any_connected(&f.cluster)));
        run_chain(&f, 60, |job| {
            if job == 20 {
                worker = None; // Unplug while work is queued.
            }
            if job == 40 {
                worker = start_worker(port, ""); // Plug back in.
                assert!(worker.is_some());
                assert!(wait_until(Duration::from_secs(10), || any_connected(&f.cluster)));
            }
        });
        expect_value(&f.y, 60.0);
        let status = f.cluster.workers();
        assert_eq!(status.len(), 1);
        assert!(status[0].connects >= 2);
        assert!(status[0].disconnects >= 1);
        let stats = f.cluster.stats();
        assert_eq!(stats.local_dispatches + stats.remote_dispatches, 60);

        f.cluster.detach(&format!("127.0.0.1:{port}")).unwrap();
        assert_eq!(
            f.cluster.detach(&format!("127.0.0.1:{port}")).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        run_chain(&f, 5, |_| {});
        expect_value(&f.y, 65.0);
        assert!(f.cluster.workers().is_empty());
    });
}

#[test]
fn random_disconnects_never_corrupt_or_lose_work() {
    with_validation(|| {
        let Some(worker) = start_worker(0, "") else { return };
        let port = worker.port();
        let f = fixture(fast_options(port, OffloadPolicy::PreferRemote));
        let done = Arc::new(AtomicBool::new(false));
        let chaos = {
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                // Deterministic pseudo-random delays of 5-60 ms.
                let mut state = 1234_u32;
                let mut worker = Some(worker);
                while !done.load(Ordering::Acquire) {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    std::thread::sleep(Duration::from_millis(5 + u64::from(state >> 16) % 56));
                    worker = if worker.is_some() {
                        None
                    } else {
                        start_worker(port, "")
                    };
                }
            })
        };
        // Keep submitting paced batches until the worker has been unplugged and replugged several
        // times, with at least 300 dependent dispatches overall.
        let mut jobs = 0;
        let deadline = Instant::now() + Duration::from_secs(60);
        while (jobs < 300 || f.cluster.workers()[0].connects < 3) && Instant::now() < deadline {
            run_chain(&f, 25, |job| {
                if job % 5 == 4 {
                    std::thread::sleep(Duration::from_millis(10));
                }
            });
            jobs += 25;
        }
        done.store(true, Ordering::Release);
        chaos.join().unwrap();
        expect_value(&f.y, jobs as f32);
        let stats = f.cluster.stats();
        assert_eq!(stats.local_dispatches + stats.remote_dispatches, jobs as u64);
        assert!(stats.remote_dispatches > 0);
        assert!(f.cluster.workers()[0].connects >= 3);
    });
}

#[test]
fn rejects_workers_with_the_wrong_token_and_runs_locally() {
    with_validation(|| {
        let Some(worker) = start_worker(0, "correct horse battery staple") else {
            return;
        };
        let mut options = fast_options(worker.port(), OffloadPolicy::PreferRemote);
        options.token = "wrong token".into();
        let f = fixture(options);
        run_chain(&f, 5, |_| {});
        expect_value(&f.y, 5.0);
        assert_eq!(f.cluster.stats().remote_dispatches, 0);
        assert!(wait_until(Duration::from_secs(10), || worker
            .stats()
            .rejected_connections
            > 0));
        assert!(!any_connected(&f.cluster));
        assert!(wait_until(Duration::from_secs(10), || {
            f.cluster
                .events()
                .iter()
                .any(|e| e.message.contains("authentication failed"))
        }));
    });
}

#[test]
fn accepts_workers_with_the_right_token() {
    with_validation(|| {
        let Some(worker) = start_worker(0, "s3cret") else {
            return;
        };
        let mut options = fast_options(worker.port(), OffloadPolicy::PreferRemote);
        options.token = "s3cret".into();
        let f = fixture(options);
        assert!(wait_until(Duration::from_secs(10), || any_connected(&f.cluster)));
        run_chain(&f, 3, |_| {});
        expect_value(&f.y, 3.0);
        assert_eq!(f.cluster.stats().remote_dispatches, 3);
        assert_eq!(worker.stats().rejected_connections, 0);
    });
}

#[test]
fn released_buffers_are_dropped_from_the_worker_cache() {
    with_validation(|| {
        let Some(worker) = start_worker(0, "") else { return };
        let f = fixture(fast_options(worker.port(), OffloadPolicy::PreferRemote));
        assert!(wait_until(Duration::from_secs(10), || any_connected(&f.cluster)));
        let extra = f.cluster.create_buffer::<f32>(u64::from(COUNT)).unwrap();
        extra.write(&vec![0.0; COUNT as usize]).unwrap();
        f.cluster
            .dispatch(&f.kernel)
            .global(u64::from(COUNT))
            .read(&f.x)
            .read_write(&extra)
            .push(&SaxpyParams {
                scale: 3.0,
                count: COUNT,
            })
            .submit()
            .unwrap()
            .wait()
            .unwrap();
        expect_value(&extra, 3.0);
        let with_extra = worker.stats().cached_bytes;
        assert_eq!(with_extra, 2 * u64::from(COUNT) * 4);
        drop(extra);
        assert!(
            wait_until(Duration::from_secs(10), || worker.stats().cached_bytes
                < with_extra),
            "worker still caches the released buffer"
        );
    });
}

#[test]
fn routes_across_every_local_device() {
    with_validation(|| {
        let cluster = match Cluster::new(split_options()) {
            Ok(cluster) => cluster,
            Err(error) => return eprintln!("skipping: {error}"),
        };
        let f = Fixture {
            kernel: cluster
                .load_kernel(&words_from_bytes(SAXPY).unwrap(), "main", saxpy_metadata())
                .unwrap(),
            x: cluster.create_buffer(u64::from(COUNT)).unwrap(),
            y: cluster.create_buffer(u64::from(COUNT)).unwrap(),
            cluster,
        };
        f.x.write(&vec![1.0; COUNT as usize]).unwrap();
        f.y.write(&vec![0.0; COUNT as usize]).unwrap();
        let devices = f.cluster.local_devices().len();
        let jobs = devices * 4;
        run_chain(&f, jobs, |_| {});
        expect_value(&f.y, jobs as f32);
        let status = f.cluster.local_devices();
        assert_eq!(status.iter().map(|d| d.dispatches).sum::<u64>(), jobs as u64);
        assert_eq!(
            status.iter().filter(|d| d.dispatches > 0).count(),
            devices,
            "Auto measures every local device before choosing"
        );
    });
}

fn split_dispatch(
    cluster: &Cluster,
    kernel: &ClusterKernel,
    x: &Buffer<f32>,
    y: &Buffer<f32>,
    scale: f32,
) -> Event {
    let count = x.len() as u32;
    cluster
        .dispatch(kernel)
        .global(u64::from(count))
        .read(x)
        .read_write(y)
        .push(&SplitParams {
            scale,
            count,
            base: 0,
        })
        .split(4)
        .submit()
        .unwrap()
}

#[test]
fn split_dispatch_matches_cpu_reference_across_devices() {
    with_validation(|| {
        let cluster = match Cluster::new(split_options()) {
            Ok(cluster) => cluster,
            Err(error) => return eprintln!("skipping: {error}"),
        };
        if cluster.local_devices().len() < 2 {
            return eprintln!("skipping: needs at least two local devices");
        }
        let kernel = cluster
            .load_kernel(
                &words_from_bytes(SPLIT_SAXPY).unwrap(),
                "main",
                split_saxpy_metadata(),
            )
            .unwrap();
        let count = 200_003_usize; // Not a multiple of the 64-wide workgroup.
        let x_values: Vec<f32> = (0..count).map(|i| (i % 97) as f32).collect();
        let mut y_values: Vec<f32> = (0..count).map(|i| (i % 13) as f32).collect();
        let x = cluster.create_buffer::<f32>(count as u64).unwrap();
        let y = cluster.create_buffer::<f32>(count as u64).unwrap();
        x.write(&x_values).unwrap();
        y.write(&y_values).unwrap();
        for _ in 0..6 {
            split_dispatch(&cluster, &kernel, &x, &y, 2.0).wait().unwrap();
            for (y, x) in y_values.iter_mut().zip(&x_values) {
                *y += 2.0 * x;
            }
        }
        assert_eq!(y.read().unwrap(), y_values);
        let stats = cluster.stats();
        assert!(stats.split_dispatches > 0);
        assert!(stats.slices >= 2 * stats.split_dispatches);
    });
}

#[test]
fn split_survives_worker_chaos() {
    with_validation(|| {
        let Some(worker) = start_worker(0, "") else { return };
        let port = worker.port();
        let mut options = fast_options(port, OffloadPolicy::PreferRemote);
        options.local.device_policy = Some(DevicePolicy::AllEligible);
        options.split_min_elements = 1024;
        options.split_overhead = Duration::ZERO;
        options.device_rescan_interval = Duration::ZERO;
        let cluster = Cluster::new(options).unwrap();
        let kernel = cluster
            .load_kernel(
                &words_from_bytes(SPLIT_SAXPY).unwrap(),
                "main",
                split_saxpy_metadata(),
            )
            .unwrap();
        let count = 65_537_usize;
        let x = cluster.create_buffer::<f32>(count as u64).unwrap();
        let y = cluster.create_buffer::<f32>(count as u64).unwrap();
        x.write(&vec![1.0; count]).unwrap();
        y.write(&vec![0.0; count]).unwrap();
        let done = Arc::new(AtomicBool::new(false));
        let chaos = {
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                let mut state = 99_u32;
                let mut worker = Some(worker);
                while !done.load(Ordering::Acquire) {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    std::thread::sleep(Duration::from_millis(5 + u64::from(state >> 16) % 36));
                    worker = if worker.is_some() {
                        None
                    } else {
                        start_worker(port, "")
                    };
                }
            })
        };
        let rounds = 120;
        let mut events = Vec::new();
        for round in 0..rounds {
            events.push(split_dispatch(&cluster, &kernel, &x, &y, 1.0));
            if round % 4 == 3 {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        for event in events {
            event.wait().unwrap();
        }
        done.store(true, Ordering::Release);
        chaos.join().unwrap();
        expect_value(&y, rounds as f32);
        assert!(cluster.stats().split_dispatches > 0);
    });
}

/// Reads one newline-terminated status snapshot and parses it.
fn read_snapshot(reader: &mut BufReader<TcpStream>) -> kora::json::Value {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    kora::json::parse(line.as_bytes()).unwrap_or_else(|e| panic!("{e}: {line}"))
}

fn events_text(snapshot: &kora::json::Value) -> String {
    snapshot
        .find("events")
        .and_then(kora::json::Value::as_array)
        .unwrap_or_default()
        .iter()
        .filter_map(|e| e.find("message").and_then(kora::json::Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn status_stream_reports_devices_workers_memory_and_events() {
    with_validation(|| {
        let Some(worker) = start_worker(0, "") else { return };
        let mut options = fast_options(worker.port(), OffloadPolicy::PreferRemote);
        options.status_listen = "127.0.0.1:0".into();
        options.status_interval = Duration::from_millis(20);
        let f = fixture(options);
        assert_ne!(f.cluster.status_port(), 0);
        assert!(wait_until(Duration::from_secs(10), || any_connected(&f.cluster)));
        run_chain(&f, 1, |_| {});

        let stream = TcpStream::connect(("127.0.0.1", f.cluster.status_port())).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut reader = BufReader::new(stream);
        let snapshot = read_snapshot(&mut reader);
        assert_eq!(
            snapshot.find("version").unwrap(),
            &kora::json::Value::Number("1".into())
        );
        let devices = snapshot
            .find("devices")
            .and_then(kora::json::Value::as_array)
            .unwrap();
        assert!(!devices.is_empty());
        assert!(devices[0].find("allocated_bytes").is_some());
        let workers = snapshot
            .find("workers")
            .and_then(kora::json::Value::as_array)
            .unwrap();
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].find("connected"), Some(&kora::json::Value::Bool(true)));
        assert!(
            snapshot
                .find("stats")
                .and_then(|s| s.find("remote_dispatches"))
                .is_some()
        );
        assert!(f.cluster.workers()[0].cached_bytes >= 2 * u64::from(COUNT) * 4);
        assert!(events_text(&snapshot).contains("connected"));

        drop(worker); // Unplug: the stream must report the disconnect.
        assert!(wait_until(Duration::from_secs(10), || !any_connected(&f.cluster)));
        assert!(wait_until(Duration::from_secs(10), || {
            events_text(&read_snapshot(&mut reader)).contains("disconnected")
        }));
    });
}

#[test]
fn status_listen_accepts_only_loopback() {
    let options = ClusterOptions {
        status_listen: "0.0.0.0:0".into(),
        ..ClusterOptions::default()
    };
    assert_eq!(
        Cluster::new(options).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
    let options = ClusterOptions {
        workers: vec!["no-port".into()],
        ..ClusterOptions::default()
    };
    assert_eq!(
        Cluster::new(options).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
}

#[test]
fn split_requires_a_base_field() {
    with_validation(|| {
        let cluster = match Cluster::new(split_options()) {
            Ok(cluster) => cluster,
            Err(error) => return eprintln!("skipping: {error}"),
        };
        let kernel = cluster
            .load_kernel(&words_from_bytes(SAXPY).unwrap(), "main", saxpy_metadata())
            .unwrap();
        let x = cluster.create_buffer::<f32>(u64::from(COUNT)).unwrap();
        let y = cluster.create_buffer::<f32>(u64::from(COUNT)).unwrap();
        let error = cluster
            .dispatch(&kernel)
            .global(u64::from(COUNT))
            .read(&x)
            .read_write(&y)
            .push(&SaxpyParams {
                scale: 1.0,
                count: COUNT,
            })
            .split(4)
            .submit()
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidArgument);
        let error = cluster.dispatch(&kernel).read(&x).submit().unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidKernelBinding);
    });
}

#[test]
fn recovers_from_local_device_loss() {
    with_validation(|| {
        if let Err(error) = kora::Runtime::new(&runtime_options("probe")) {
            return eprintln!("skipping: {error}");
        }
        let f = fixture(split_options());
        run_chain(&f, 2, |_| {});
        let devices: Vec<_> = f.cluster.local_devices().into_iter().map(|d| d.id).collect();
        for device in &devices {
            f.cluster.runtime().simulate_device_loss(device).unwrap();
        }
        run_chain(&f, 3, |_| {});
        expect_value(&f.y, 5.0);
        assert!(f.cluster.stats().local_recoveries >= 1);
        assert!(
            f.cluster
                .events()
                .iter()
                .any(|e| e.message.starts_with("local device lost"))
        );
    });
}
