//! # KORA — Kernel Offload Runtime Architecture
//!
//! KORA is an explicit, vendor-neutral runtime for CPU and Vulkan compute workloads. It provides
//! device-bound kernels, logical buffers with host shadows and per-device replicas, transfers,
//! completion events, task graphs, and diagnostics, while keeping Vulkan capabilities and their
//! limits visible to the caller.
//!
//! Expected failures are values: fallible operations return [`Result<T>`] carrying a structured
//! [`Error`] with a stable [`ErrorCode`].

pub mod aliasing;
pub mod buffer;
pub mod checked;
pub mod cpu_executor;
pub mod device;
pub mod error;
pub mod event;
pub mod execution_plan;
pub mod fsutil;
pub mod ids;
pub mod json;
pub mod kernel;
pub mod log;
pub mod memory;
pub mod metadata;
pub mod options;
pub mod partition;
pub mod performance_db;
pub mod profiling;
pub mod scheduler;
pub mod task_graph;
pub mod transfer_model;
pub mod verification;
pub mod version;
pub mod vulkan;

pub use buffer::{Buffer, BufferHandle, BufferInfo, BufferManager, BufferUsage};
pub use bytemuck;
pub use cpu_executor::{CpuExecutor, CpuExecutorOptions, CpuTaskPriority};
pub use device::{CapabilityTier, DeviceCandidate, DeviceInfo, DeviceKind, MemoryTopology, select_devices};
pub use error::{Error, ErrorCode, Result, Status};
pub use event::{Event, EventPromise};
pub use execution_plan::ExecutionPlan;
pub use ids::{BufferId, DeviceId};
pub use memory::{DirtyRange, MemoryStats};
pub use metadata::{
    KernelAccess, KernelBindingKind, KernelBindingMetadata, KernelMetadata, PushConstantFieldMetadata,
    SpecializationConstantMetadata, SpecializationValue, parse_kernel_metadata, validate_kernel_metadata,
};
pub use options::{DevicePolicy, ResolvedOptions, RuntimeOptions, ShutdownMode, TelemetryLevel};
pub use task_graph::{
    AccessMode, BufferAccess, TaskGraph, TaskGraphPlan, TaskId, TaskMetadata, TaskNode, TaskType,
};
pub use verification::{VerificationDataType, VerificationPolicy};
