//! Derives the runtime version from `VERSION` (MAJOR.MINOR) and `KORA_VERSION_PATCH`, matching the
//! release pipeline contract: the highest `vMAJOR.MINOR.*` tag supplies the next patch number.

use std::{env, fs, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let version_file = manifest.join("../../VERSION");
    println!("cargo:rerun-if-changed={}", version_file.display());
    println!("cargo:rerun-if-env-changed=KORA_VERSION_PATCH");

    let base = fs::read_to_string(&version_file).expect("VERSION file is readable");
    let base = base.trim();
    let (major, minor) = base.split_once('.').expect("VERSION must contain MAJOR.MINOR");
    let major: u32 = major.parse().expect("VERSION major is numeric");
    let minor: u32 = minor.parse().expect("VERSION minor is numeric");
    let patch: u32 = match env::var("KORA_VERSION_PATCH") {
        Ok(value) if !value.is_empty() => value.parse().expect("KORA_VERSION_PATCH is numeric"),
        _ => 0,
    };
    println!("cargo:rustc-env=KORA_VERSION_MAJOR={major}");
    println!("cargo:rustc-env=KORA_VERSION_MINOR={minor}");
    println!("cargo:rustc-env=KORA_VERSION_PATCH={patch}");
}
