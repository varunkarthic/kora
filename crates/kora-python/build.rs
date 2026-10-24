//! Lets plain `cargo build`/`cargo test` link the extension module (maturin does this itself).

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
