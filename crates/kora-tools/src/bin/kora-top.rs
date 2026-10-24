//! kora-top: live terminal view of a `kora::Cluster`.
//!
//! Attaches to the status stream an application opens with `ClusterOptions::status_listen` (or
//! `KORA_STATUS_LISTEN`), or runs a self-contained demo with `--demo`.

use std::collections::{BTreeMap, VecDeque};
use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::net::TcpStream;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kora::json::Value;
use kora::kernel::spirv::words_from_bytes;
use kora::log::Level;
use kora::{
    Buffer, Cluster, ClusterOptions, ErrorCode, KernelAccess, KernelBindingKind, KernelBindingMetadata,
    KernelMetadata, OffloadPolicy, PushConstantFieldMetadata, RuntimeOptions, WorkerOptions, WorkerServer,
};

const DEFAULT_ENDPOINT: &str = "127.0.0.1:7979";
const DEMO_SAXPY: &[u8] = include_bytes!("../../../kora/tests/fixtures/split_saxpy.spv");

const USAGE: &str = "usage: kora-top [host:port] [--demo] [--frames N] [--no-color]

Shows a running kora::Cluster: local devices and remote workers, where work is going
now, device memory, and recent connect, disconnect, failover, and routing events.
The application opts in with ClusterOptions::status_listen or
KORA_STATUS_LISTEN=127.0.0.1:7979 (the default kora-top endpoint).

  --demo       run a built-in Cluster with two loopback workers that are
               unplugged and replugged while verified work runs
  --frames N   draw N snapshots and exit (no screen control)
  --no-color   plain text (also when NO_COLOR is set)";

struct Args {
    endpoint: String,
    demo: bool,
    /// 0 = run until interrupted.
    frames: usize,
    color: bool,
}

// --- JSON access ---------------------------------------------------------------------------------

fn number(value: &Value, key: &str) -> u64 {
    match value.find(key) {
        Some(Value::Number(text)) => text.parse().unwrap_or(0),
        _ => 0,
    }
}

fn text(value: &Value, key: &str) -> String {
    value
        .find(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn flag(value: &Value, key: &str) -> bool {
    matches!(value.find(key), Some(Value::Bool(true)))
}

fn items<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value.find(key).and_then(Value::as_array).unwrap_or_default()
}

// --- Formatting ----------------------------------------------------------------------------------

fn bytes_text(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value:.0} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i != 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn fit(value: &str, width: usize) -> String {
    let count = value.chars().count();
    if count > width {
        let mut cut: String = value.chars().take(width.saturating_sub(1)).collect();
        cut.push('~');
        return cut;
    }
    format!("{value}{}", " ".repeat(width - count))
}

fn terminal_width() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|c| c.parse().ok())
        .filter(|w| *w > 0)
        .unwrap_or(100)
}

#[derive(Clone, Copy)]
struct Style {
    on: bool,
}

impl Style {
    fn paint(self, code: &str, value: &str) -> String {
        if self.on {
            format!("\x1b[{code}m{value}\x1b[0m")
        } else {
            value.to_string()
        }
    }

    fn bar(self, fraction: f64, width: usize, color: &str) -> String {
        let filled = (fraction.clamp(0.0, 1.0) * width as f64 + 0.5) as usize;
        let full = "\u{2588}".repeat(filled);
        let empty = "\u{2591}".repeat(width - filled);
        if self.on {
            self.paint(color, &full) + &self.paint("2", &empty)
        } else {
            full + &empty
        }
    }
}

// --- Rendering -----------------------------------------------------------------------------------

/// Remembers recent work counts per target so the view can show where work is going right now.
#[derive(Default)]
struct Activity {
    history: VecDeque<(u64, BTreeMap<String, u64>)>,
}

impl Activity {
    /// Each target's share of the work completed during roughly the last second.
    fn update(&mut self, now_ms: u64, totals: BTreeMap<String, u64>) -> BTreeMap<String, f64> {
        self.history.push_back((now_ms, totals));
        while self.history.len() > 2 && now_ms - self.history[1].0 >= 1_000 {
            self.history.pop_front();
        }
        let before = &self.history[0].1;
        let totals = &self.history[self.history.len() - 1].1;
        let mut share: BTreeMap<String, f64> = totals
            .iter()
            .map(|(key, total)| {
                let previous = before.get(key).copied().unwrap_or(0);
                let delta = if *total >= previous {
                    total - previous
                } else {
                    *total
                };
                (key.clone(), delta as f64)
            })
            .collect();
        let sum: f64 = share.values().sum();
        for value in share.values_mut() {
            *value = if sum > 0.0 { *value / sum } else { 0.0 };
        }
        share
    }
}

fn render(snapshot: &Value, activity: &mut Activity, endpoint: &str, style: Style, footer: &str) -> String {
    let width = terminal_width().max(72);
    let stats = snapshot.find("stats").cloned().unwrap_or(Value::Null);
    let mut totals = BTreeMap::new();
    for device in items(snapshot, "devices") {
        totals.insert(
            format!("local:{}", number(device, "index")),
            number(device, "dispatches") + number(device, "slices"),
        );
    }
    for worker in items(snapshot, "workers") {
        totals.insert(
            format!("worker:{}", text(worker, "endpoint")),
            number(worker, "dispatches"),
        );
    }
    let share = activity.update(number(snapshot, "uptime_ms"), totals);

    let mut out = String::new();
    let mut line = |value: &str| {
        out.push_str(value);
        out.push_str(if style.on { "\x1b[K\n" } else { "\n" });
    };
    line(&format!(
        "{}  {endpoint}  up {:.1} s  policy {}",
        style.paint("1", "kora-top"),
        number(snapshot, "uptime_ms") as f64 / 1000.0,
        text(snapshot, "policy")
    ));
    line(&format!(
        "dispatches: local {}, remote {}, split {} ({} slices)   failovers {}   device rescans {}   recoveries {}",
        grouped(number(&stats, "local_dispatches")),
        grouped(number(&stats, "remote_dispatches")),
        grouped(number(&stats, "split_dispatches")),
        grouped(number(&stats, "slices")),
        grouped(number(&stats, "failovers")),
        grouped(number(&stats, "device_rescans")),
        grouped(number(&stats, "local_recoveries"))
    ));
    line("");

    let name_width = width.saturating_sub(68).clamp(24, 44);
    let bar_width = 16;
    line(&style.paint(
        "1;36",
        &format!(
            "{}{}{}{}MEMORY",
            fit("  TARGET", 12 + name_width),
            fit("WORK NOW", bar_width + 7),
            fit("TOTAL", 10),
            fit("RUN", 5)
        ),
    ));
    for device in items(snapshot, "devices") {
        let key = format!("local:{}", number(device, "index"));
        let fraction = share.get(&key).copied().unwrap_or(0.0);
        let allocated = number(device, "allocated_bytes");
        let budget = number(device, "budget_bytes");
        let memory = if budget > 0 {
            format!(
                "{} {} of {}",
                style.bar(allocated as f64 / budget as f64, 8, "35"),
                bytes_text(allocated),
                bytes_text(budget)
            )
        } else {
            bytes_text(allocated)
        };
        line(&format!(
            "{}{}{} {} {:4.0}%  {}{}{memory}",
            style.paint("32", "* "),
            fit(&format!("local {}", number(device, "index")), 10),
            fit(&text(device, "name"), name_width - 1),
            style.bar(fraction, bar_width, "32"),
            fraction * 100.0,
            fit(
                &grouped(number(device, "dispatches") + number(device, "slices")),
                10
            ),
            fit(&number(device, "inflight").to_string(), 5)
        ));
    }
    for worker in items(snapshot, "workers") {
        let key = format!("worker:{}", text(worker, "endpoint"));
        let connected = flag(worker, "connected");
        let fraction = share.get(&key).copied().unwrap_or(0.0);
        let device = if connected {
            text(worker, "device")
        } else {
            "unplugged".into()
        };
        let status = if connected {
            format!("cached {}", bytes_text(number(worker, "cached_bytes")))
        } else {
            format!(
                "{} connects, {} failovers",
                number(worker, "connects"),
                number(worker, "failovers")
            )
        };
        line(&format!(
            "{}{}{} {} {:4.0}%  {}{}{status}",
            if connected {
                style.paint("33", "* ")
            } else {
                style.paint("2", "o ")
            },
            fit("worker", 10),
            fit(&format!("{}  {device}", text(worker, "endpoint")), name_width - 1),
            style.bar(fraction, bar_width, "33"),
            fraction * 100.0,
            fit(&grouped(number(worker, "dispatches")), 10),
            fit(&number(worker, "inflight").to_string(), 5)
        ));
    }
    if items(snapshot, "workers").is_empty() {
        line(&style.paint("2", "  no remote workers configured"));
    }
    line("");
    line(&style.paint("1;36", "  EVENTS"));
    let events = items(snapshot, "events");
    for event in &events[events.len().saturating_sub(10)..] {
        let mut message = text(event, "message");
        let repeats = number(event, "repeats");
        if repeats > 1 {
            message += &format!("  (x{})", grouped(repeats));
        }
        line(&format!(
            "{}{}",
            style.paint("2", &format!("{:8.1}s  ", number(event, "at_ms") as f64 / 1000.0)),
            fit(&message, width - 12)
        ));
    }
    if events.is_empty() {
        line(&style.paint("2", "  none yet"));
    }
    if !footer.is_empty() {
        line("");
        line(footer);
    }
    out
}

/// Reads newline-delimited snapshots from the status endpoint and redraws, reconnecting as needed.
fn attach(args: &Args, footer: &dyn Fn() -> String, keep_running: &dyn Fn() -> bool) -> ExitCode {
    let stdout_terminal = std::io::stdout().is_terminal();
    let terminal = stdout_terminal && args.frames == 0;
    let style = Style {
        on: args.color && stdout_terminal,
    };
    let Ok(address) = args
        .endpoint
        .parse::<std::net::SocketAddr>()
        .or_else(|_| std::net::ToSocketAddrs::to_socket_addrs(&args.endpoint).map(|mut a| a.next().unwrap()))
    else {
        eprintln!("kora-top: invalid endpoint '{}'", args.endpoint);
        return ExitCode::from(2);
    };
    let mut activity = Activity::default();
    let mut frames = 0;
    let mut waiting_shown = false;
    let mut stdout = std::io::stdout();
    while keep_running() {
        let Ok(stream) = TcpStream::connect_timeout(&address, Duration::from_millis(500)) else {
            if !waiting_shown {
                let clear = if terminal { "\x1b[H\x1b[2J" } else { "" };
                let endpoint = &args.endpoint;
                let _ = writeln!(
                    stdout,
                    "{clear}kora-top: waiting for a KORA application at {endpoint}\n  start it with KORA_STATUS_LISTEN={endpoint}, or run kora-top --demo"
                );
                let _ = stdout.flush();
                waiting_shown = true;
            }
            std::thread::sleep(Duration::from_millis(500));
            continue;
        };
        waiting_shown = false;
        let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
        let mut reader = BufReader::new(stream);
        let mut pending = Vec::new();
        while keep_running() {
            match reader.read_until(b'\n', &mut pending) {
                Ok(0) => break,
                Ok(_) if pending.ends_with(b"\n") => {}
                Ok(_) => continue,
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    if pending.len() > 1 << 20 {
                        break; // Not a status stream.
                    }
                    continue;
                }
                Err(_) => break,
            }
            let parsed = kora::json::parse(&pending[..pending.len() - 1]);
            pending.clear();
            let Ok(snapshot) = parsed else { continue };
            let frame = render(&snapshot, &mut activity, &args.endpoint, style, &footer());
            let _ = write!(
                stdout,
                "{}{frame}{}",
                if terminal { "\x1b[H" } else { "" },
                if terminal { "\x1b[J" } else { "\n" }
            );
            let _ = stdout.flush();
            frames += 1;
            if args.frames > 0 && frames >= args.frames {
                return ExitCode::SUCCESS;
            }
        }
    }
    ExitCode::SUCCESS
}

// --- Demo ----------------------------------------------------------------------------------------

fn quiet() -> RuntimeOptions {
    RuntimeOptions {
        log_level: Some(Level::Off),
        ..RuntimeOptions::default()
    }
}

/// Starts a loopback worker, retrying briefly while a previous instance releases the port.
fn start_worker(port: u16) -> kora::Result<WorkerServer> {
    let options = WorkerOptions {
        listen: format!("127.0.0.1:{port}"),
        runtime: quiet(),
        ..WorkerOptions::default()
    };
    let mut attempt = 0;
    loop {
        match WorkerServer::start(options.clone()) {
            Err(error)
                if attempt < 99
                    && !matches!(
                        error.code(),
                        ErrorCode::NoVulkanImplementation | ErrorCode::NoEligibleDevice
                    ) =>
            {
                attempt += 1;
                std::thread::sleep(Duration::from_millis(20));
            }
            result => return result,
        }
    }
}

/// 77 tells test harnesses the demo was skipped because this machine has no usable Vulkan device.
fn exit_code_for(error: &kora::Error) -> ExitCode {
    if matches!(
        error.code(),
        ErrorCode::NoVulkanImplementation | ErrorCode::NoEligibleDevice
    ) {
        ExitCode::from(77)
    } else {
        ExitCode::FAILURE
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DemoParams {
    scale: f32,
    count: u32,
    base: u32,
}

fn demo_metadata() -> KernelMetadata {
    let field = |name: &str, type_name: &str, offset| PushConstantFieldMetadata {
        name: name.into(),
        type_name: type_name.into(),
        offset,
        size: 4,
    };
    let binding = |binding, name: &str, access| KernelBindingMetadata {
        set: 0,
        binding,
        name: name.into(),
        kind: KernelBindingKind::StorageBuffer,
        access,
    };
    KernelMetadata {
        kernel_id: "demo.saxpy".into(),
        local_size: [64, 1, 1],
        push_constant_size: 12,
        push_constants: vec![
            field("scale", "f32", 0),
            field("count", "u32", 4),
            field("base", "u32", 8),
        ],
        bindings: vec![
            binding(0, "input_x", KernelAccess::Read),
            binding(1, "input_y", KernelAccess::ReadWrite),
        ],
        ..KernelMetadata::default()
    }
}

#[derive(Default)]
struct Tally {
    verified: AtomicU64,
    mismatches: AtomicU64,
}

/// Runs paired small and split large SAXPY dispatches, checking every eighth round exactly.
fn workload(cluster: &Cluster, kernel: &kora::ClusterKernel, stop: &AtomicBool, tally: &Tally) {
    const SMALL: u32 = 4_096;
    const LARGE: u32 = 1 << 20;
    let make = |count: u32, value: f32| -> Option<Buffer<f32>> {
        let buffer = cluster.create_buffer::<f32>(u64::from(count)).ok()?;
        buffer.write(&vec![value; count as usize]).ok()?;
        Some(buffer)
    };
    let (Some(x_small), Some(y_small), Some(x_large), Some(y_large)) = (
        make(SMALL, 1.0),
        make(SMALL, 0.0),
        make(LARGE, 1.0),
        make(LARGE, 0.0),
    ) else {
        tally.mismatches.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let mut expected = 0.0_f32;
    let check = |buffer: &Buffer<f32>, expected: f32| {
        let exact = buffer.read().is_ok_and(|v| v.iter().all(|x| *x == expected));
        let counter = if exact { &tally.verified } else { &tally.mismatches };
        counter.fetch_add(1, Ordering::Relaxed);
    };
    let mut round = 0_u64;
    while !stop.load(Ordering::Acquire) {
        let first = cluster
            .dispatch(kernel)
            .global(u64::from(SMALL))
            .read(&x_small)
            .read_write(&y_small)
            .push(&DemoParams {
                scale: 2.0,
                count: SMALL,
                base: 0,
            })
            .submit();
        let second = cluster
            .dispatch(kernel)
            .global(u64::from(LARGE))
            .read(&x_large)
            .read_write(&y_large)
            .push(&DemoParams {
                scale: 2.0,
                count: LARGE,
                base: 0,
            })
            .split(4)
            .submit();
        let ok = matches!((&first, &second), (Ok(a), Ok(b)) if a.wait().is_ok() && b.wait().is_ok());
        if !ok {
            tally.mismatches.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        expected += 2.0;
        if round.is_multiple_of(8) {
            check(&y_small, expected);
            check(&y_large, expected);
        }
        round += 1;
        std::thread::sleep(Duration::from_millis(4));
    }
}

/// Two loopback workers stand in for attached devices; a control thread unplugs and replugs them
/// while verified work runs.
fn run_demo(mut args: Args) -> ExitCode {
    let mut first = Vec::new();
    for _ in 0..2 {
        match start_worker(0) {
            Ok(worker) => first.push(worker),
            Err(error) => {
                eprintln!("kora-top --demo: could not start a loopback worker: {error}");
                return exit_code_for(&error);
            }
        }
    }
    let ports = [first[0].port(), first[1].port()];
    let workers: Arc<Mutex<[Option<WorkerServer>; 2]>> =
        Arc::new(Mutex::new([first.drain(..1).next(), None]));
    drop(first); // Worker B starts unplugged.

    let options = ClusterOptions {
        local: quiet(),
        workers: ports.iter().map(|p| format!("127.0.0.1:{p}")).collect(),
        policy: OffloadPolicy::PreferRemote,
        status_listen: "127.0.0.1:0".into(),
        connect_timeout: Duration::from_millis(200),
        reconnect_interval: Duration::from_millis(200),
        heartbeat_interval: Duration::from_millis(250),
        ..ClusterOptions::default()
    };
    let cluster = match Cluster::new(options) {
        Ok(cluster) => cluster,
        Err(error) => {
            eprintln!("kora-top --demo: {error}");
            return exit_code_for(&error);
        }
    };
    let kernel =
        match words_from_bytes(DEMO_SAXPY).and_then(|w| cluster.load_kernel(&w, "main", demo_metadata())) {
            Ok(kernel) => kernel,
            Err(error) => {
                eprintln!("kora-top --demo: {error}");
                return ExitCode::FAILURE;
            }
        };

    let stop = AtomicBool::new(false);
    let tally = Tally::default();
    let next_change = Mutex::new(String::new());
    let start = Instant::now();
    args.endpoint = format!("127.0.0.1:{}", cluster.status_port());
    let style = Style {
        on: args.color && std::io::stdout().is_terminal(),
    };
    let status = std::thread::scope(|scope| {
        scope.spawn(|| workload(&cluster, &kernel, &stop, &tally));
        scope.spawn(|| {
            // Repeating 16-second script: plug B in, unplug A, plug A back, unplug B.
            const STEPS: [(u64, usize, bool); 4] =
                [(3, 1, true), (7, 0, false), (11, 0, true), (15, 1, false)];
            for cycle in 0.. {
                for (at_s, worker, plug) in STEPS {
                    let when = start + Duration::from_secs(16 * cycle + at_s);
                    *next_change.lock().unwrap() = format!(
                        "{} worker {} (127.0.0.1:{})",
                        if plug { "plug in" } else { "unplug" },
                        if worker == 0 { "A" } else { "B" },
                        ports[worker]
                    );
                    while !stop.load(Ordering::Acquire) && Instant::now() < when {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    let replacement = if plug {
                        start_worker(ports[worker]).ok()
                    } else {
                        None
                    };
                    workers.lock().unwrap()[worker] = replacement;
                }
            }
        });
        let footer = || {
            let elapsed = start.elapsed().as_secs() % 16;
            let next = [3, 7, 11, 15]
                .into_iter()
                .find(|at| elapsed < *at)
                .map_or(19 - elapsed, |at| at - elapsed);
            let bad = tally.mismatches.load(Ordering::Relaxed);
            format!(
                "{}  results verified {}, {}   next: {} in {next} s   (Ctrl-C to quit)",
                style.paint("1", "demo"),
                grouped(tally.verified.load(Ordering::Relaxed)),
                if bad == 0 {
                    style.paint("32", "0 mismatches")
                } else {
                    style.paint("31", &format!("{} mismatches", grouped(bad)))
                },
                next_change.lock().unwrap()
            )
        };
        let status = attach(&args, &footer, &|| true);
        stop.store(true, Ordering::Release);
        status
    });
    let bad = tally.mismatches.load(Ordering::Relaxed);
    if bad != 0 {
        eprintln!("kora-top --demo: {bad} results did not match");
        return ExitCode::FAILURE;
    }
    status
}

fn main() -> ExitCode {
    let mut args = Args {
        endpoint: DEFAULT_ENDPOINT.into(),
        demo: false,
        frames: 0,
        color: std::env::var_os("NO_COLOR").is_none(),
    };
    let mut argv = std::env::args().skip(1);
    while let Some(argument) = argv.next() {
        match argument.as_str() {
            "--demo" => args.demo = true,
            "--frames" => match argv.next().and_then(|n| n.parse::<usize>().ok()) {
                Some(frames) => args.frames = frames.max(1),
                None => {
                    println!("{USAGE}");
                    return ExitCode::from(2);
                }
            },
            "--no-color" => args.color = false,
            "--help" | "-h" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other if !other.starts_with('-') => args.endpoint = other.into(),
            _ => {
                println!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    if args.demo {
        run_demo(args)
    } else {
        attach(&args, &String::new, &|| true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_sizes_counts_and_columns() {
        assert_eq!(bytes_text(512), "512 B");
        assert_eq!(bytes_text(1536), "1.5 KiB");
        assert_eq!(bytes_text(3 << 30), "3.0 GiB");
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(1234567), "1,234,567");
        assert_eq!(grouped(100), "100");
        assert_eq!(fit("abc", 5), "abc  ");
        assert_eq!(fit("abcdef", 4), "abc~");
    }

    #[test]
    fn activity_reports_recent_shares() {
        let mut activity = Activity::default();
        let totals = |a, b| BTreeMap::from([("a".to_string(), a), ("b".to_string(), b)]);
        activity.update(0, totals(0, 0));
        let share = activity.update(500, totals(30, 10));
        assert!((share["a"] - 0.75).abs() < 1e-9);
        assert!((share["b"] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn renders_a_snapshot_without_color() {
        let snapshot = kora::json::parse(
            br#"{"version":1,"uptime_ms":1500,"policy":"auto","devices":[{"index":0,"name":"GPU","dispatches":4,"slices":0,"inflight":1,"allocated_bytes":2048,"budget_bytes":0}],"workers":[{"endpoint":"127.0.0.1:1","connected":false,"name":"","device":"","inflight":0,"dispatches":0,"failovers":2,"connects":1,"disconnects":1,"cached_bytes":0}],"stats":{"local_dispatches":4,"remote_dispatches":0,"failovers":2,"local_recoveries":0,"split_dispatches":0,"slices":0,"device_rescans":0},"events":[{"at_ms":10,"message":"worker 127.0.0.1:1 disconnected","repeats":3}]}"#,
        )
        .unwrap();
        let frame = render(
            &snapshot,
            &mut Activity::default(),
            "x:1",
            Style { on: false },
            "",
        );
        assert!(frame.contains("up 1.5 s  policy auto"));
        assert!(frame.contains("local 0"));
        assert!(frame.contains("2.0 KiB"));
        assert!(frame.contains("unplugged"));
        assert!(frame.contains("1 connects, 2 failovers"));
        assert!(frame.contains("disconnected  (x3)"));
        assert!(!frame.contains('\x1b'));
    }
}
