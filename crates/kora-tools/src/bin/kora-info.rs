//! kora-info: reports the Vulkan loader, instance, and every physical device's capabilities.
//!
//! ```text
//! kora-info [--json]
//! ```
//!
//! `--json` prints one object (schema version 2). kora-info does not create a Runtime, so it never
//! touches the pipeline cache.

use std::process::ExitCode;

use kora::device::{DeviceInfo, MemoryTopology};
use kora::json::{self, Value};
use kora::vulkan::{InstanceOptions, Loader, PhysicalDeviceInfo, device_info};
use kora_tools::json_object;

fn version_string(version: u32) -> String {
    kora::device::ApiVersion::from_packed(version).to_string()
}

fn topology_name(topology: MemoryTopology) -> &'static str {
    match topology {
        MemoryTopology::Discrete => "discrete",
        MemoryTopology::Uma => "uma",
        MemoryTopology::DiscreteWithBar => "discrete_with_bar",
        MemoryTopology::Unknown => "unknown",
    }
}

fn is_moltenvk(raw: &PhysicalDeviceInfo) -> bool {
    raw.driver_name.contains("MoltenVK")
}

fn device_json(index: usize, raw: &PhysicalDeviceInfo, device: &DeviceInfo) -> Value {
    let f = &device.features;
    let s = &device.subgroup;
    let n = |value: u64| Value::Number(value.to_string());
    let b = Value::Bool;
    let queue_families = device
        .queue_families
        .iter()
        .enumerate()
        .map(|(i, q)| {
            json_object(vec![
                ("index", n(i as u64)),
                ("count", n(u64::from(q.queue_count))),
                ("timestamp_valid_bits", n(u64::from(q.timestamp_valid_bits))),
                ("graphics", b(q.graphics)),
                ("compute", b(q.compute)),
                ("transfer", b(q.transfer)),
                ("sparse_binding", b(q.sparse_binding)),
                ("dedicated_compute", b(q.dedicated_compute)),
                ("dedicated_transfer", b(q.dedicated_transfer)),
            ])
        })
        .collect();
    let heaps = device
        .memory_heaps
        .iter()
        .enumerate()
        .map(|(i, heap)| {
            let mut members = vec![
                ("index", n(i as u64)),
                ("size_bytes", n(heap.size_bytes)),
                ("device_local", b(heap.device_local)),
                ("multi_instance", b(heap.multi_instance)),
            ];
            if let Some(budget) = &raw.memory_budget {
                members.push(("budget_bytes", n(budget.heap_budget[i])));
                members.push(("usage_bytes", n(budget.heap_usage[i])));
            }
            json_object(members)
        })
        .collect();
    let types = device
        .memory_types
        .iter()
        .enumerate()
        .map(|(i, t)| {
            json_object(vec![
                ("index", n(i as u64)),
                ("heap_index", n(u64::from(t.heap_index))),
                ("device_local", b(t.device_local)),
                ("host_visible", b(t.host_visible)),
                ("host_coherent", b(t.host_coherent)),
                ("host_cached", b(t.host_cached)),
                ("lazily_allocated", b(t.lazily_allocated)),
                ("protected", b(t.protected_memory)),
            ])
        })
        .collect();
    json_object(vec![
        ("index", n(index as u64)),
        ("name", Value::String(device.name.clone())),
        ("api_version", Value::String(device.api_version.to_string())),
        ("driver_name", Value::String(raw.driver_name.clone())),
        ("portability_subset", b(f.portability_subset)),
        ("moltenvk", b(is_moltenvk(raw))),
        (
            "memory_topology",
            Value::String(topology_name(device.memory_topology).into()),
        ),
        (
            "memory_topology_needs_benchmark",
            b(device.memory_topology_needs_benchmark),
        ),
        (
            "features",
            json_object(vec![
                ("timeline_semaphore", b(f.timeline_semaphore)),
                ("synchronization2", b(f.synchronization2)),
                ("buffer_device_address", b(f.buffer_device_address)),
                ("descriptor_indexing", b(f.descriptor_indexing)),
                ("runtime_descriptor_array", b(f.runtime_descriptor_array)),
            ]),
        ),
        (
            "subgroup",
            json_object(vec![
                ("supported", b(s.supported)),
                ("size", n(u64::from(s.size))),
                ("basic", b(s.basic)),
                ("vote", b(s.vote)),
                ("arithmetic", b(s.arithmetic)),
                ("ballot", b(s.ballot)),
                ("shuffle", b(s.shuffle)),
                ("shuffle_relative", b(s.shuffle_relative)),
                ("clustered", b(s.clustered)),
                ("quad", b(s.quad)),
            ]),
        ),
        ("queue_families", Value::Array(queue_families)),
        (
            "memory",
            json_object(vec![
                ("budget_available", b(raw.memory_budget.is_some())),
                ("heaps", Value::Array(heaps)),
                ("types", Value::Array(types)),
            ]),
        ),
    ])
}

fn print_human_device(index: usize, raw: &PhysicalDeviceInfo, device: &DeviceInfo) {
    let f = &device.features;
    let s = &device.subgroup;
    let yes = |value: bool| if value { "yes" } else { "no" };
    println!("\nDevice [{index}]: {}", device.name);
    println!("  Vulkan API: {}", device.api_version);
    println!(
        "  Driver: {}",
        if raw.driver_name.is_empty() {
            "unreported"
        } else {
            &raw.driver_name
        }
    );
    println!(
        "  Portability subset: {} (MoltenVK: {})",
        yes(f.portability_subset),
        yes(is_moltenvk(raw))
    );
    println!(
        "  Memory topology: {} (benchmark needed: {})",
        topology_name(device.memory_topology),
        yes(device.memory_topology_needs_benchmark)
    );
    println!(
        "  Features: timeline={} sync2={} bda={} descriptor_indexing={} runtime_descriptor_array={}",
        f.timeline_semaphore,
        f.synchronization2,
        f.buffer_device_address,
        f.descriptor_indexing,
        f.runtime_descriptor_array
    );
    println!(
        "  Subgroup: supported={} size={} basic={} arithmetic={}",
        s.supported, s.size, s.basic, s.arithmetic
    );
    println!("  Queue families: {}", device.queue_families.len());
    for (i, q) in device.queue_families.iter().enumerate() {
        println!(
            "    [{i}] count={} graphics={} compute={} transfer={} dedicated_compute={} dedicated_transfer={}",
            q.queue_count, q.graphics, q.compute, q.transfer, q.dedicated_compute, q.dedicated_transfer
        );
    }
    println!(
        "  Memory heaps: {} (budget extension: {})",
        device.memory_heaps.len(),
        yes(raw.memory_budget.is_some())
    );
    for (i, heap) in device.memory_heaps.iter().enumerate() {
        let budget = raw.memory_budget.as_ref().map_or_else(String::new, |b| {
            format!(" budget={} usage={}", b.heap_budget[i], b.heap_usage[i])
        });
        println!(
            "    [{i}] size={} device_local={}{budget}",
            heap.size_bytes, heap.device_local
        );
    }
    println!("  Memory types: {}", device.memory_types.len());
    for (i, t) in device.memory_types.iter().enumerate() {
        println!(
            "    [{i}] heap={} device_local={} host_visible={} coherent={} cached={}",
            t.heap_index, t.device_local, t.host_visible, t.host_coherent, t.host_cached
        );
    }
}

fn fail(operation: &str, error: &kora::Error) -> ExitCode {
    eprintln!("kora-info: {operation}: {}", error.describe());
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.len() == 1 && args[0] == "--json";
    if !args.is_empty() && !json {
        eprintln!("Usage: kora-info [--json]");
        return ExitCode::FAILURE;
    }
    let loader = match Loader::load() {
        Ok(loader) => loader,
        Err(error) => return fail("unable to load Vulkan", &error),
    };
    let instance = match loader.create_instance(&InstanceOptions::default()) {
        Ok(instance) => instance,
        Err(error) => return fail("unable to create Vulkan instance", &error),
    };
    let devices = match instance.physical_devices() {
        Ok(devices) => devices,
        Err(error) => return fail("unable to enumerate physical devices", &error),
    };
    if json {
        let report = json_object(vec![
            ("schema_version", Value::Number("2".into())),
            ("kora_version", Value::String(kora::version::version_string())),
            (
                "loader_api_version",
                Value::String(version_string(loader.supported_api_version())),
            ),
            (
                "instance_api_version",
                Value::String(version_string(instance.api_version())),
            ),
            ("pipeline_cache_state", Value::String("not_queried".into())),
            (
                "devices",
                Value::Array(
                    devices
                        .iter()
                        .enumerate()
                        .map(|(i, raw)| device_json(i, raw, &device_info(raw)))
                        .collect(),
                ),
            ),
        ]);
        let mut out = String::new();
        json::write_value(&mut out, &report);
        println!("{out}");
        return ExitCode::SUCCESS;
    }
    println!("KORA version: {}", kora::version::version_string());
    println!(
        "Vulkan loader API version: {}",
        version_string(loader.supported_api_version())
    );
    println!(
        "Vulkan instance API version: {}",
        version_string(instance.api_version())
    );
    println!("Physical devices: {}", devices.len());
    println!("Pipeline cache state: not queried (kora-info does not initialize a Runtime)");
    for (index, raw) in devices.iter().enumerate() {
        print_human_device(index, raw, &device_info(raw));
    }
    ExitCode::SUCCESS
}
