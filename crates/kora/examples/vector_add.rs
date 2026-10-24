//! Adds two vectors on the first KORA device.
//!
//! ```text
//! cargo run --release -p kora --example vector_add
//! ```

use kora::kernel::spirv::words_from_bytes;
use kora::{Runtime, RuntimeOptions, parse_kernel_metadata};

const SPIRV: &[u8] = include_bytes!("../../../examples/kernels/vector_add.spv");
const MANIFEST: &str = include_str!("../../../examples/kernels/vector_add.json");

fn main() -> kora::Result<()> {
    let runtime = Runtime::new(&RuntimeOptions::default())?;
    let device = &runtime.devices()[0];
    println!(
        "KORA {} on {}",
        kora::version::version_string(),
        runtime.device_info(device)?.name
    );

    let kernel = runtime.load_kernel(
        &words_from_bytes(SPIRV)?,
        "main",
        parse_kernel_metadata(MANIFEST)?,
    )?;
    let count: u32 = 1 << 20;
    let a = runtime.create_buffer::<u32>(u64::from(count))?;
    let b = runtime.create_buffer::<u32>(u64::from(count))?;
    let sums = runtime.create_buffer::<u32>(u64::from(count))?;
    a.write(&(0..count).collect::<Vec<_>>())?;
    b.write(&vec![10; count as usize])?;

    runtime
        .dispatch(&kernel)
        .global(u64::from(count))
        .read(&a)
        .read(&b)
        .write(&sums)
        .push(&count)
        .submit()?
        .wait()?;

    let values = sums.read()?;
    let mismatches = values
        .iter()
        .enumerate()
        .filter(|(i, v)| **v != *i as u32 + 10)
        .count();
    println!("vector add of {count} elements: {mismatches} mismatches");
    Ok(())
}
