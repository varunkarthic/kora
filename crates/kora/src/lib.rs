//! # KORA — Kernel Offload Runtime Architecture
//!
//! KORA is an explicit, vendor-neutral runtime for CPU and Vulkan compute workloads. It provides
//! device-bound kernels, logical buffers with host shadows and per-device replicas, transfers,
//! completion events, task graphs, and diagnostics, while keeping Vulkan capabilities and their
//! limits visible to the caller.

pub mod version;
