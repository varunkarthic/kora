//! Records build facts that kora-bench reports alongside its measurements.

use std::process::Command;

fn output(command: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(command).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = output(&rustc, &["--version"]).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=KORA_RUSTC_VERSION={version}");
    let profile = std::env::var("PROFILE").unwrap_or_default();
    let opt = std::env::var("OPT_LEVEL").unwrap_or_default();
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .replace('\x1f', " ");
    println!("cargo:rustc-env=KORA_BUILD_FLAGS=profile={profile} opt-level={opt} {flags}");
    let revision = output("git", &["rev-parse", "HEAD"])
        .filter(|r| r.len() == 40 && r.bytes().all(|b| b.is_ascii_hexdigit()))
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=KORA_SOURCE_REVISION={revision}");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
}
