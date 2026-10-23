//! Device-bound compute kernels.

use std::sync::Arc;

use super::context::DeviceContext;
use crate::error::{Error, ErrorCode, Result};
use crate::ids::DeviceId;
use crate::kernel::{
    AutotuneOptions, Reflection, WorkgroupAutotuner, generate_workgroup_candidates, validate_and_reflect,
    validate_module_structure, workgroup_performance_key,
};
use crate::metadata::{KernelMetadata, validate_kernel_metadata};

const MAXIMUM_MODULE_WORDS: usize = 4 * 1024 * 1024;

pub(crate) struct KernelState {
    pub(crate) context: Arc<DeviceContext>,
    pub(crate) entry_point: String,
    pub(crate) metadata: KernelMetadata,
    pub(crate) words: Arc<[u32]>,
    pub(crate) reflection: Reflection,
    pub(crate) autotuner: Option<Arc<WorkgroupAutotuner>>,
    pub(crate) autotune_keys: Vec<String>,
}

/// A validated SPIR-V compute kernel bound to one device of a [`Runtime`](super::Runtime).
/// Cheap to clone.
#[derive(Clone)]
pub struct Kernel {
    pub(crate) state: Arc<KernelState>,
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("kernel_id", &self.state.metadata.kernel_id)
            .field("entry_point", &self.state.entry_point)
            .field("device", &self.state.context.id)
            .finish()
    }
}

impl Kernel {
    /// Kernel metadata.
    pub fn metadata(&self) -> &KernelMetadata {
        &self.state.metadata
    }

    /// Entry point name.
    pub fn entry_point(&self) -> &str {
        &self.state.entry_point
    }

    /// Device the kernel is bound to.
    pub fn device(&self) -> &DeviceId {
        &self.state.context.id
    }

    /// Reflected compute interface.
    pub fn reflection(&self) -> &Reflection {
        &self.state.reflection
    }

    /// Workgroup size the next dispatch would use (the autotuned size once tuning finished).
    pub fn local_size(&self) -> [u32; 3] {
        self.state
            .autotuner
            .as_ref()
            .map_or(self.state.reflection.local_size, |t| t.selected_local_size())
    }

    /// `true` when the kernel declares `LocalSizeId` and is being autotuned.
    pub fn is_autotuned(&self) -> bool {
        self.state.autotuner.is_some()
    }
}

/// Validates, reflects, and pre-compiles a kernel for `context`.
pub(crate) fn load(
    context: &Arc<DeviceContext>,
    words: &[u32],
    entry_point: &str,
    metadata: KernelMetadata,
) -> Result<Kernel> {
    context.check_accepting()?;
    validate_kernel_metadata(&metadata)?;
    if words.is_empty() || words.len() > MAXIMUM_MODULE_WORDS {
        return Err(Error::new(
            ErrorCode::InvalidKernel,
            "SPIR-V module is empty or exceeds 16 MiB",
        ));
    }
    validate_module_structure(words)?;
    let reflection = validate_and_reflect(words, &metadata, &context.capabilities, entry_point)?;
    let words: Arc<[u32]> = words.into();

    let mut autotuner = None;
    let mut autotune_keys = Vec::new();
    let ids = reflection.local_size_spec_ids;
    // Only X is tuned, so it must be specialized independently of Y and Z.
    if reflection.local_size_id_based && ids[0] != ids[1] && ids[0] != ids[2] {
        let candidates = generate_workgroup_candidates(
            reflection.local_size,
            true,
            &context.info.limits,
            &context.info.subgroup,
            8,
        )?;
        if candidates.len() > 1 {
            let base = candidates
                .iter()
                .position(|c| c.local_size == reflection.local_size)
                .unwrap_or(0);
            let tuner = WorkgroupAutotuner::new(candidates, base, AutotuneOptions::default());
            let device_key = if context.id.uuid.is_empty() {
                format!("{}/{}", context.info.name, context.id.index)
            } else {
                context.id.uuid.to_string()
            };
            let module_hash = words.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &w| {
                (h ^ u64::from(w)).wrapping_mul(0x0100_0000_01b3)
            });
            let tuning_id = format!("{}/{module_hash}", metadata.kernel_id);
            for (index, candidate) in tuner.candidates().iter().enumerate() {
                let key = workgroup_performance_key(
                    &device_key,
                    context.info.driver_version,
                    &tuning_id,
                    entry_point,
                    candidate,
                );
                if let Ok(saved) = context.performance.query(&key) {
                    tuner.restore(index, &saved)?;
                }
                autotune_keys.push(key);
            }
            autotuner = Some(Arc::new(tuner));
        }
    }
    // Compile now so driver rejections surface at load time and the first dispatch only looks up
    // the cached pipeline.
    context
        .pipelines
        .get_or_create(&words, &metadata, &[], entry_point)?;
    Ok(Kernel {
        state: Arc::new(KernelState {
            context: Arc::clone(context),
            entry_point: entry_point.to_owned(),
            metadata,
            words,
            reflection,
            autotuner,
            autotune_keys,
        }),
    })
}
