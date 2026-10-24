//! kora-worker: runs KORA dispatches for a remote `kora::Cluster`.
//!
//! ```text
//! kora-worker [--listen HOST:PORT] [--device INDEX] [--cache-mb N] [--name NAME]
//! ```
//!
//! Set `KORA_WORKER_TOKEN` to accept authenticated non-loopback clients. Without it, only clients
//! on this machine can connect. Traffic is not encrypted: use a trusted link or an SSH/VPN tunnel.
//! Stop it with Ctrl-C; connected clients re-run outstanding work locally.

use std::process::ExitCode;

use kora::{DevicePolicy, WorkerOptions, WorkerServer};

const USAGE: &str = "usage: kora-worker [--listen HOST:PORT] [--device INDEX] [--cache-mb N] [--name NAME]";

fn parse(args: impl IntoIterator<Item = String>) -> Result<WorkerOptions, String> {
    let mut options = WorkerOptions::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--listen" => options.listen = value()?,
            "--device" => {
                options.runtime.device_policy = Some(DevicePolicy::ByIndex);
                options.runtime.device_index = Some(value()?.parse().map_err(|_| "--device needs an index")?);
            }
            "--cache-mb" => {
                let megabytes: u64 = value()?.parse().map_err(|_| "--cache-mb needs a number")?;
                options.max_cache_bytes = megabytes.saturating_mul(1024 * 1024);
            }
            "--name" => options.name = value()?,
            _ => return Err(format!("unknown argument '{arg}'")),
        }
    }
    Ok(options)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let options = match parse(args) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("kora-worker: {message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let server = match WorkerServer::start(options) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("kora-worker: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("kora-worker listening on port {}", server.port());
    loop {
        std::thread::park();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn parses_every_option() {
        let options = parse(args(&[
            "--listen",
            "0.0.0.0:9000",
            "--device",
            "2",
            "--cache-mb",
            "64",
            "--name",
            "box",
        ]))
        .unwrap();
        assert_eq!(options.listen, "0.0.0.0:9000");
        assert_eq!(options.runtime.device_policy, Some(DevicePolicy::ByIndex));
        assert_eq!(options.runtime.device_index, Some(2));
        assert_eq!(options.max_cache_bytes, 64 << 20);
        assert_eq!(options.name, "box");
    }

    #[test]
    fn rejects_unknown_and_incomplete_arguments() {
        assert!(parse(args(&["--bogus"])).is_err());
        assert!(parse(args(&["--listen"])).is_err());
        assert!(parse(args(&["--device", "x"])).is_err());
    }
}
