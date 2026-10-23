//! Kernel packages: SPIR-V validation and reflection, compute pipelines, descriptor management,
//! and workgroup-size autotuning.

mod autotuner;
mod pipeline;
pub mod spirv;

pub use autotuner::{
    AutotuneOptions, Trial, WorkgroupAutotuner, WorkgroupCandidate, generate_workgroup_candidates,
    workgroup_performance_key,
};
pub use pipeline::{
    ComputePipeline, DescriptorSetLease, PipelineCacheIdentity, PipelineKey, PipelineManager,
    PipelineManagerOptions, PipelineManagerStats, PipelineRef,
};
pub use spirv::{
    DeviceCapabilities, ReflectedBinding, Reflection, validate_and_reflect, validate_module_structure,
};
