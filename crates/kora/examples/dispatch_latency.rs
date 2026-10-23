//! Measures round-trip latency of a small vector-add dispatch (upload, dispatch, readback).
//!
//! ```sh
//! cargo run --release --example dispatch_latency -- [elements] [iterations]
//! ```

use std::time::Instant;

use kora::kernel::spirv::words_from_bytes;
use kora::{
    KernelAccess, KernelBindingKind, KernelBindingMetadata, KernelMetadata, PushConstantFieldMetadata,
    Runtime, RuntimeOptions,
};

const VECTOR_ADD: &[u8] = include_bytes!("../tests/fixtures/vector_add.spv");

fn main() -> kora::Result<()> {
    let mut args = std::env::args().skip(1);
    let n: u32 = args.next().and_then(|a| a.parse().ok()).unwrap_or(1024);
    let iterations: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(2000);
    let runtime = Runtime::new(&RuntimeOptions {
        enable_startup_calibration: Some(false),
        ..RuntimeOptions::default()
    })?;
    let binding = |binding, name: &str, access| KernelBindingMetadata {
        set: 0,
        binding,
        name: name.into(),
        kind: KernelBindingKind::StorageBuffer,
        access,
    };
    let metadata = KernelMetadata {
        kernel_id: "example.vector_add".into(),
        local_size: [64, 1, 1],
        bindings: vec![
            binding(0, "input_a", KernelAccess::Read),
            binding(1, "input_b", KernelAccess::Read),
            binding(2, "output_data", KernelAccess::Write),
        ],
        push_constant_size: 4,
        push_constants: vec![PushConstantFieldMetadata {
            name: "count".into(),
            type_name: "u32".into(),
            offset: 0,
            size: 4,
        }],
        ..KernelMetadata::default()
    };
    let kernel = runtime.load_kernel(&words_from_bytes(VECTOR_ADD)?, "main", metadata)?;
    let a = runtime.create_buffer::<u32>(u64::from(n))?;
    let b = runtime.create_buffer::<u32>(u64::from(n))?;
    let out = runtime.create_buffer::<u32>(u64::from(n))?;
    b.write(&vec![1; n as usize])?;
    let mut samples = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        a.write(&vec![iteration as u32; n as usize])?;
        let started = Instant::now();
        runtime
            .dispatch(&kernel)
            .global(u64::from(n))
            .read(&a)
            .read(&b)
            .write(&out)
            .push(&n)
            .submit()?
            .wait()?;
        samples.push(started.elapsed().as_nanos() as u64);
    }
    assert!(out.read()?.iter().all(|&v| v == iterations as u32));
    samples.sort_unstable();
    let pick = |q: f64| samples[((samples.len() - 1) as f64 * q) as usize] as f64 / 1000.0;
    println!(
        "device: {}\nelements: {n}, iterations: {iterations}\nlatency us: p50 {:.1}  p90 {:.1}  p99 {:.1}",
        runtime.device_info(&runtime.devices()[0])?.name,
        pick(0.5),
        pick(0.9),
        pick(0.99)
    );
    Ok(())
}
