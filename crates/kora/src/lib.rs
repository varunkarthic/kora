//! # KORA — Kernel Offload Runtime Architecture
//!
//! KORA is an explicit, vendor-neutral runtime for CPU and Vulkan compute workloads. It provides
//! device-bound kernels, logical buffers with host shadows and per-device replicas, transfers,
//! completion events, task graphs, and diagnostics, while keeping Vulkan capabilities and their
//! limits visible to the caller.
//!
//! Expected failures are values: fallible operations return [`Result<T>`] carrying a structured
//! [`Error`] with a stable [`ErrorCode`].

pub mod checked;
pub mod error;
pub mod json;
pub mod log;
pub mod metadata;
pub mod options;
pub mod verification;
pub mod version;

pub use error::{Error, ErrorCode, Result, Status};
pub use metadata::{
    KernelAccess, KernelBindingKind, KernelBindingMetadata, KernelMetadata, PushConstantFieldMetadata,
    SpecializationConstantMetadata, SpecializationValue, parse_kernel_metadata, validate_kernel_metadata,
};
pub use options::{DevicePolicy, ResolvedOptions, RuntimeOptions, ShutdownMode, TelemetryLevel};
pub use verification::{VerificationDataType, VerificationPolicy};
