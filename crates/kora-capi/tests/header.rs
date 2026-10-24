//! `include/kora.h` matches the cbindgen output for the current sources. Run with
//! `KORA_BLESS=1` to regenerate it.

use std::path::Path;

#[test]
fn committed_header_is_current() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")).expect("cbindgen.toml");
    let mut generated = Vec::new();
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("header generation")
        .write(&mut generated);
    let path = crate_dir.join("include/kora.h");
    if std::env::var_os("KORA_BLESS").is_some() {
        std::fs::write(&path, &generated).expect("write header");
        return;
    }
    let committed = std::fs::read(&path).unwrap_or_default();
    // Windows checkouts may convert the header to CRLF line endings.
    let lf = |bytes: &[u8]| bytes.iter().copied().filter(|&b| b != b'\r').collect::<Vec<u8>>();
    assert!(
        lf(&committed) == lf(&generated),
        "include/kora.h is stale; run `KORA_BLESS=1 cargo test -p kora-capi --test header`"
    );
}
