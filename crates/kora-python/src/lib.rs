//! `kora._core`: the PyO3 extension behind the `kora` Python package.
//!
//! Names follow the 0.x pybind11 module; behavior follows the Rust runtime (see the CHANGELOG for
//! the differences). Blocking calls release the GIL.

use std::ffi::c_int;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use kora::buffer::{MappedRead, MappedWrite};
use pyo3::buffer::PyBuffer;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

create_exception!(
    _core,
    KoraError,
    PyException,
    "A KORA operation failed; `code` holds the ErrorCode."
);

macro_rules! mirror_enum {
    ($(#[$meta:meta])* $name:ident : $rust:ty { $($variant:ident = $value:literal => $target:expr),* $(,)? }) => {
        $(#[$meta])*
        #[pyclass(module = "kora._core", eq, eq_int, hash, frozen, from_py_object, rename_all = "SCREAMING_SNAKE_CASE")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $($variant = $value),*
        }

        impl From<$name> for $rust {
            fn from(value: $name) -> $rust {
                match value {
                    $($name::$variant => $target),*
                }
            }
        }

        impl From<$rust> for $name {
            #[allow(unreachable_patterns)]
            fn from(value: $rust) -> $name {
                match value {
                    $(v if v == $target => $name::$variant,)*
                    _ => unreachable!("every variant is mirrored"),
                }
            }
        }
    };
}

/// Stable error codes (`OK` is never raised; it exists for comparisons).
#[pyclass(
    module = "kora._core",
    eq,
    eq_int,
    hash,
    frozen,
    from_py_object,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    Ok = 0,
    NoVulkanImplementation = 1,
    NoEligibleDevice = 2,
    NoComputeQueue = 3,
    UnsupportedFeature = 4,
    InvalidKernel = 5,
    InvalidKernelBinding = 6,
    InvalidDispatch = 7,
    AllocationFailed = 8,
    OutOfMemoryBudget = 9,
    TransferFailed = 10,
    Timeout = 11,
    DeviceLost = 12,
    PipelineCreationFailed = 13,
    SynchronizationError = 14,
    ValidationError = 15,
    PythonBindingError = 16,
    InternalInvariantViolation = 17,
    InvalidArgument = 18,
    Overflow = 19,
    GraphCycle = 20,
    Cancelled = 21,
    StaleHandle = 22,
    DataLoss = 23,
    IoError = 24,
}

const ERROR_CODES: [ErrorCode; 25] = [
    ErrorCode::Ok,
    ErrorCode::NoVulkanImplementation,
    ErrorCode::NoEligibleDevice,
    ErrorCode::NoComputeQueue,
    ErrorCode::UnsupportedFeature,
    ErrorCode::InvalidKernel,
    ErrorCode::InvalidKernelBinding,
    ErrorCode::InvalidDispatch,
    ErrorCode::AllocationFailed,
    ErrorCode::OutOfMemoryBudget,
    ErrorCode::TransferFailed,
    ErrorCode::Timeout,
    ErrorCode::DeviceLost,
    ErrorCode::PipelineCreationFailed,
    ErrorCode::SynchronizationError,
    ErrorCode::ValidationError,
    ErrorCode::PythonBindingError,
    ErrorCode::InternalInvariantViolation,
    ErrorCode::InvalidArgument,
    ErrorCode::Overflow,
    ErrorCode::GraphCycle,
    ErrorCode::Cancelled,
    ErrorCode::StaleHandle,
    ErrorCode::DataLoss,
    ErrorCode::IoError,
];

fn error_code(code: kora::ErrorCode) -> ErrorCode {
    ERROR_CODES
        .into_iter()
        .find(|c| *c as i32 == code as i32)
        .unwrap_or(ErrorCode::InternalInvariantViolation)
}

/// Converts a runtime error into a `KoraError` carrying every structured field.
fn to_py(error: kora::Error) -> PyErr {
    Python::attach(|py| {
        let exception = KoraError::new_err(error.describe());
        let value = exception.value(py);
        // Attribute assignment on a fresh exception instance cannot meaningfully fail.
        let _ = value.setattr("code", error_code(error.code()));
        let _ = value.setattr("code_name", error.code().name());
        let _ = value.setattr("message", error.message());
        let _ = value.setattr("kernel_id", error.kernel_id());
        let _ = value.setattr("device", error.device());
        let _ = value.setattr("recovery_action", error.recovery_action());
        let _ = value.setattr("vk_result", error.vk_result());
        let _ = value.setattr("task_id", error.task_id());
        exception
    })
}

trait OrRaise<T> {
    fn or_raise(self) -> PyResult<T>;
}

impl<T> OrRaise<T> for kora::Result<T> {
    fn or_raise(self) -> PyResult<T> {
        self.map_err(to_py)
    }
}

fn invalid(message: &str) -> PyErr {
    to_py(kora::Error::new(kora::ErrorCode::InvalidArgument, message))
}

mirror_enum!(
    /// How a runtime picks its devices.
    DevicePolicy: kora::DevicePolicy {
        Auto = 0 => kora::DevicePolicy::Auto,
        PreferDiscrete = 1 => kora::DevicePolicy::PreferDiscrete,
        PreferIntegrated = 2 => kora::DevicePolicy::PreferIntegrated,
        ByIndex = 3 => kora::DevicePolicy::ByIndex,
        ByUuid = 4 => kora::DevicePolicy::ByUuid,
        AllEligible = 5 => kora::DevicePolicy::AllEligible,
    }
);

mirror_enum!(
    /// Intended use of a buffer.
    BufferUsage: kora::BufferUsage {
        Generic = 0 => kora::BufferUsage::Generic,
        Storage = 1 => kora::BufferUsage::Storage,
        Uniform = 2 => kora::BufferUsage::Uniform,
        TransferSource = 3 => kora::BufferUsage::TransferSource,
        TransferDestination = 4 => kora::BufferUsage::TransferDestination,
    }
);

mirror_enum!(
    /// Descriptor kind of a kernel binding.
    KernelBindingKind: kora::KernelBindingKind {
        StorageBuffer = 0 => kora::KernelBindingKind::StorageBuffer,
        UniformBuffer = 1 => kora::KernelBindingKind::UniformBuffer,
        StorageImage = 2 => kora::KernelBindingKind::StorageImage,
        SampledImage = 3 => kora::KernelBindingKind::SampledImage,
    }
);

mirror_enum!(
    /// How a kernel accesses a binding.
    KernelAccess: kora::KernelAccess {
        Read = 0 => kora::KernelAccess::Read,
        Write = 1 => kora::KernelAccess::Write,
        ReadWrite = 2 => kora::KernelAccess::ReadWrite,
    }
);

mirror_enum!(
    /// Kind of task-graph node.
    TaskType: kora::TaskType {
        Upload = 0 => kora::TaskType::Upload,
        Download = 1 => kora::TaskType::Download,
        Copy = 2 => kora::TaskType::Copy,
        Fill = 3 => kora::TaskType::Fill,
        Dispatch = 4 => kora::TaskType::Dispatch,
        CpuTask = 5 => kora::TaskType::CpuTask,
        HostCallback = 6 => kora::TaskType::HostCallback,
    }
);

mirror_enum!(
    /// Buffer access declared by a task.
    AccessMode: kora::AccessMode {
        Read = 0 => kora::AccessMode::Read,
        Write = 1 => kora::AccessMode::Write,
        ReadWrite = 2 => kora::AccessMode::ReadWrite,
    }
);

/// Access requested by `Buffer.map`.
#[pyclass(
    module = "kora._core",
    eq,
    eq_int,
    hash,
    frozen,
    from_py_object,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MapAccess {
    Read = 0,
    Write = 1,
    ReadWrite = 2,
}

/// A device selected by a runtime.
#[pyclass(module = "kora._core", get_all, set_all, eq, from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceId {
    pub index: u32,
    pub uuid: String,
}

#[pymethods]
impl DeviceId {
    #[new]
    #[pyo3(signature = (index = 0, uuid = String::new()))]
    fn new(index: u32, uuid: String) -> Self {
        Self { index, uuid }
    }
    fn __repr__(&self) -> String {
        format!("DeviceId(index={}, uuid={:?})", self.index, self.uuid)
    }
}

impl DeviceId {
    fn rust(&self) -> kora::DeviceId {
        kora::DeviceId::new(self.index, self.uuid.as_str())
    }
}

/// Identity of a logical buffer.
#[pyclass(module = "kora._core", get_all, set_all, eq, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BufferId {
    pub slot: u32,
    pub generation: u64,
}

#[pymethods]
impl BufferId {
    #[new]
    #[pyo3(signature = (slot = 0, generation = 0))]
    fn new(slot: u32, generation: u64) -> Self {
        Self { slot, generation }
    }
}

/// A task's declared access to one buffer.
#[pyclass(module = "kora._core", get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct BufferAccess {
    pub buffer: BufferId,
    pub mode: AccessMode,
}

#[pymethods]
impl BufferAccess {
    #[new]
    #[pyo3(signature = (buffer = BufferId::default(), mode = AccessMode::Read))]
    fn new(buffer: BufferId, mode: AccessMode) -> Self {
        Self { buffer, mode }
    }
}

/// Scheduling metadata of a task-graph node.
#[pyclass(module = "kora._core", get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct TaskMetadata {
    #[pyo3(name = "type")]
    pub task_type: TaskType,
    pub kernel_id: String,
    pub cpu_function_id: String,
    pub accesses: Vec<BufferAccess>,
    pub problem_dimensions: [u64; 3],
    pub priority: i32,
    pub deadline_hint_ns: Option<u64>,
    pub allowed_devices: Vec<String>,
    pub preferred_device: Option<String>,
    pub cpu_fallback_available: bool,
    pub estimated_bytes_read: u64,
    pub estimated_bytes_written: u64,
    pub estimated_work_units: u64,
}

#[pymethods]
impl TaskMetadata {
    #[new]
    fn new() -> Self {
        Self::from(&kora::TaskMetadata::default())
    }
}

impl From<&kora::TaskMetadata> for TaskMetadata {
    fn from(m: &kora::TaskMetadata) -> Self {
        Self {
            task_type: m.task_type.into(),
            kernel_id: m.kernel_id.clone(),
            cpu_function_id: m.cpu_function_id.clone(),
            accesses: m
                .accesses
                .iter()
                .map(|a| BufferAccess {
                    buffer: BufferId {
                        slot: a.buffer.slot,
                        generation: a.buffer.generation,
                    },
                    mode: a.mode.into(),
                })
                .collect(),
            problem_dimensions: m.problem_dimensions,
            priority: m.priority,
            deadline_hint_ns: m.deadline_hint_ns,
            allowed_devices: m.allowed_devices.clone(),
            preferred_device: m.preferred_device.clone(),
            cpu_fallback_available: m.cpu_fallback_available,
            estimated_bytes_read: m.estimated_bytes_read,
            estimated_bytes_written: m.estimated_bytes_written,
            estimated_work_units: m.estimated_work_units,
        }
    }
}

impl From<&TaskMetadata> for kora::TaskMetadata {
    fn from(m: &TaskMetadata) -> Self {
        Self {
            task_type: m.task_type.into(),
            kernel_id: m.kernel_id.clone(),
            cpu_function_id: m.cpu_function_id.clone(),
            accesses: m
                .accesses
                .iter()
                .map(|a| kora::BufferAccess {
                    buffer: kora::BufferId::new(a.buffer.slot, a.buffer.generation),
                    mode: a.mode.into(),
                })
                .collect(),
            problem_dimensions: m.problem_dimensions,
            priority: m.priority,
            deadline_hint_ns: m.deadline_hint_ns,
            allowed_devices: m.allowed_devices.clone(),
            preferred_device: m.preferred_device.clone(),
            cpu_fallback_available: m.cpu_fallback_available,
            estimated_bytes_read: m.estimated_bytes_read,
            estimated_bytes_written: m.estimated_bytes_written,
            estimated_work_units: m.estimated_work_units,
        }
    }
}

/// A finalized task.
#[pyclass(module = "kora._core", get_all, frozen, from_py_object)]
#[derive(Debug, Clone)]
pub struct TaskNode {
    pub id: u64,
    pub metadata: TaskMetadata,
    pub dependencies: Vec<u64>,
}

impl From<&kora::TaskNode> for TaskNode {
    fn from(node: &kora::TaskNode) -> Self {
        Self {
            id: node.id,
            metadata: TaskMetadata::from(&node.metadata),
            dependencies: node.dependencies.clone(),
        }
    }
}

/// Result of `TaskGraph.finalize`.
#[pyclass(module = "kora._core", get_all, frozen)]
#[derive(Debug, Clone)]
pub struct TaskGraphPlan {
    pub tasks: Vec<TaskNode>,
    pub topological_order: Vec<u64>,
}

/// Dependency analysis over declared buffer accesses and explicit edges.
#[pyclass(module = "kora._core")]
#[derive(Debug, Default)]
pub struct TaskGraph(kora::TaskGraph);

#[pymethods]
impl TaskGraph {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Adds a task and returns its id (ids start at 1).
    fn add_task(&mut self, metadata: TaskMetadata) -> PyResult<u64> {
        self.0.add_task(kora::TaskMetadata::from(&metadata)).or_raise()
    }

    /// Orders `dependent` after `prerequisite`.
    fn add_dependency(&mut self, prerequisite: u64, dependent: u64) -> PyResult<()> {
        self.0.add_dependency(prerequisite, dependent).or_raise()
    }

    /// Validates the graph and returns its tasks in a topological order.
    fn finalize(&self) -> PyResult<TaskGraphPlan> {
        let plan = self.0.finalize().or_raise()?;
        Ok(TaskGraphPlan {
            tasks: plan.tasks.iter().map(TaskNode::from).collect(),
            topological_order: plan.topological_order,
        })
    }
}

/// A reusable, allocation-free replay of a finalized graph.
#[pyclass(module = "kora._core", frozen)]
#[derive(Debug)]
pub struct ExecutionPlan(kora::ExecutionPlan);

#[pymethods]
impl ExecutionPlan {
    #[staticmethod]
    fn create(graph: &TaskGraph) -> PyResult<Self> {
        kora::ExecutionPlan::new(&graph.0).map(Self).or_raise()
    }

    /// Calls `executor(task)` in order; it must return `None` or `True`. `False`, another value,
    /// or an exception stops replay with `ErrorCode.PYTHON_BINDING_ERROR`.
    fn replay(&self, py: Python<'_>, executor: Bound<'_, PyAny>) -> PyResult<()> {
        self.0
            .replay(|node| {
                let failure = |message: String| {
                    kora::Error::new(kora::ErrorCode::PythonBindingError, message).with_task(node.id)
                };
                let task = Bound::new(py, TaskNode::from(node)).map_err(|e| failure(e.to_string()))?;
                match executor.call1((task,)) {
                    Ok(result) if result.is_none() => Ok(()),
                    Ok(result) => match result.extract::<bool>() {
                        Ok(true) => Ok(()),
                        Ok(false) => Err(failure("Python TaskGraph executor returned False".into())),
                        Err(_) => Err(failure(
                            "Python TaskGraph executor must return None or True".into(),
                        )),
                    },
                    Err(error) => Err(failure(error.to_string())),
                }
            })
            .or_raise()
    }
}

/// Runtime configuration; `None` fields defer to `KORA_*` environment variables and defaults.
#[pyclass(module = "kora._core", get_all, set_all, from_py_object)]
#[derive(Debug, Clone, Default)]
pub struct RuntimeOptions {
    pub device_policy: Option<DevicePolicy>,
    pub device_index: Option<u32>,
    pub device_uuid: Option<String>,
    pub cache_dir: Option<String>,
    pub enable_validation: Option<bool>,
    pub enable_startup_calibration: Option<bool>,
    pub disable_autotune: Option<bool>,
    pub force_legacy_sync: Option<bool>,
}

#[pymethods]
impl RuntimeOptions {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

impl From<&RuntimeOptions> for kora::RuntimeOptions {
    fn from(o: &RuntimeOptions) -> Self {
        kora::RuntimeOptions {
            device_policy: o.device_policy.map(Into::into),
            device_index: o.device_index,
            device_uuid: o.device_uuid.clone(),
            cache_dir: o.cache_dir.clone(),
            enable_validation: o.enable_validation,
            enable_startup_calibration: o.enable_startup_calibration,
            disable_autotune: o.disable_autotune,
            force_legacy_sync: o.force_legacy_sync,
            ..kora::RuntimeOptions::default()
        }
    }
}

/// One kernel binding.
#[pyclass(module = "kora._core", get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct KernelBindingMetadata {
    pub set: u32,
    pub binding: u32,
    pub name: String,
    pub kind: KernelBindingKind,
    pub access: KernelAccess,
}

#[pymethods]
impl KernelBindingMetadata {
    #[new]
    #[pyo3(signature = (set = 0, binding = 0, name = String::new(), kind = KernelBindingKind::StorageBuffer, access = KernelAccess::Read))]
    fn new(set: u32, binding: u32, name: String, kind: KernelBindingKind, access: KernelAccess) -> Self {
        Self {
            set,
            binding,
            name,
            kind,
            access,
        }
    }
}

/// One named push-constant field.
#[pyclass(module = "kora._core", get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct PushConstantFieldMetadata {
    pub name: String,
    #[pyo3(name = "type")]
    pub type_name: String,
    pub offset: u32,
    pub size: u32,
}

#[pymethods]
impl PushConstantFieldMetadata {
    #[new]
    #[pyo3(signature = (name = String::new(), r#type = String::new(), offset = 0, size = 0))]
    fn new(name: String, r#type: String, offset: u32, size: u32) -> Self {
        Self {
            name,
            type_name: r#type,
            offset,
            size,
        }
    }
}

/// One specialization constant; `default_value` is a bool, int, or float.
#[pyclass(module = "kora._core", from_py_object)]
#[derive(Debug, Clone)]
pub struct SpecializationConstantMetadata {
    #[pyo3(get, set)]
    pub id: u32,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set, name = "type")]
    pub type_name: String,
    value: kora::SpecializationValue,
}

#[pymethods]
impl SpecializationConstantMetadata {
    #[new]
    #[pyo3(signature = (id = 0, name = String::new(), r#type = String::new(), default_value = None))]
    fn new(id: u32, name: String, r#type: String, default_value: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut constant = Self {
            id,
            name,
            type_name: r#type,
            value: kora::SpecializationValue::U64(0),
        };
        if let Some(value) = default_value {
            constant.set_default_value(value)?;
        }
        Ok(constant)
    }

    #[getter]
    fn default_value<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self.value {
            kora::SpecializationValue::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
            kora::SpecializationValue::I64(i) => i.into_pyobject(py)?.into_any(),
            kora::SpecializationValue::U64(u) => u.into_pyobject(py)?.into_any(),
            kora::SpecializationValue::F64(f) => f.into_pyobject(py)?.into_any(),
        })
    }

    #[setter]
    fn set_default_value(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.value = if let Ok(b) = value.cast::<pyo3::types::PyBool>() {
            kora::SpecializationValue::Bool(b.is_true())
        } else if let Ok(u) = value.extract::<u64>() {
            kora::SpecializationValue::U64(u)
        } else if let Ok(i) = value.extract::<i64>() {
            kora::SpecializationValue::I64(i)
        } else if let Ok(f) = value.extract::<f64>() {
            kora::SpecializationValue::F64(f)
        } else {
            return Err(invalid("default_value must be a bool, int, or float"));
        };
        Ok(())
    }
}

/// Kernel manifest: bindings, push constants, specialization constants, and workgroup size.
#[pyclass(module = "kora._core", get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct KernelMetadata {
    pub schema_version: u32,
    pub kernel_id: String,
    pub bindings: Vec<KernelBindingMetadata>,
    pub push_constant_size: u32,
    pub push_constants: Vec<PushConstantFieldMetadata>,
    pub specialization_constants: Vec<SpecializationConstantMetadata>,
    pub local_size: (u32, u32, u32),
    pub cpu_fallback_key: String,
    pub semantics: Vec<String>,
}

#[pymethods]
impl KernelMetadata {
    #[new]
    fn new() -> Self {
        let defaults = kora::KernelMetadata::default();
        Self {
            schema_version: defaults.schema_version,
            kernel_id: String::new(),
            bindings: Vec::new(),
            push_constant_size: 0,
            push_constants: Vec::new(),
            specialization_constants: Vec::new(),
            local_size: (
                defaults.local_size[0],
                defaults.local_size[1],
                defaults.local_size[2],
            ),
            cpu_fallback_key: String::new(),
            semantics: Vec::new(),
        }
    }

    /// Parses the documented JSON manifest format.
    #[staticmethod]
    fn from_json(document: &str) -> PyResult<Self> {
        kora::parse_kernel_metadata(document)
            .map(|m| Self::from(&m))
            .or_raise()
    }

    /// Serializes to the documented JSON manifest format.
    fn to_json(&self) -> String {
        kora::KernelMetadata::from(self).to_json()
    }
}

impl From<&kora::KernelMetadata> for KernelMetadata {
    fn from(m: &kora::KernelMetadata) -> Self {
        Self {
            schema_version: m.schema_version,
            kernel_id: m.kernel_id.clone(),
            bindings: m
                .bindings
                .iter()
                .map(|b| KernelBindingMetadata {
                    set: b.set,
                    binding: b.binding,
                    name: b.name.clone(),
                    kind: b.kind.into(),
                    access: b.access.into(),
                })
                .collect(),
            push_constant_size: m.push_constant_size,
            push_constants: m
                .push_constants
                .iter()
                .map(|f| PushConstantFieldMetadata {
                    name: f.name.clone(),
                    type_name: f.type_name.clone(),
                    offset: f.offset,
                    size: f.size,
                })
                .collect(),
            specialization_constants: m
                .specialization_constants
                .iter()
                .map(|s| SpecializationConstantMetadata {
                    id: s.id,
                    name: s.name.clone(),
                    type_name: s.type_name.clone(),
                    value: s.default_value,
                })
                .collect(),
            local_size: (m.local_size[0], m.local_size[1], m.local_size[2]),
            cpu_fallback_key: m.cpu_fallback_key.clone(),
            semantics: m.semantics.clone(),
        }
    }
}

impl From<&KernelMetadata> for kora::KernelMetadata {
    fn from(m: &KernelMetadata) -> Self {
        Self {
            schema_version: m.schema_version,
            kernel_id: m.kernel_id.clone(),
            bindings: m
                .bindings
                .iter()
                .map(|b| kora::KernelBindingMetadata {
                    set: b.set,
                    binding: b.binding,
                    name: b.name.clone(),
                    kind: b.kind.into(),
                    access: b.access.into(),
                })
                .collect(),
            push_constant_size: m.push_constant_size,
            push_constants: m
                .push_constants
                .iter()
                .map(|f| kora::PushConstantFieldMetadata {
                    name: f.name.clone(),
                    type_name: f.type_name.clone(),
                    offset: f.offset,
                    size: f.size,
                })
                .collect(),
            specialization_constants: m
                .specialization_constants
                .iter()
                .map(|s| kora::SpecializationConstantMetadata {
                    id: s.id,
                    name: s.name.clone(),
                    type_name: s.type_name.clone(),
                    default_value: s.value,
                })
                .collect(),
            local_size: [m.local_size.0, m.local_size.1, m.local_size.2],
            cpu_fallback_key: m.cpu_fallback_key.clone(),
            semantics: m.semantics.clone(),
        }
    }
}

/// Copies the bytes of any C-contiguous buffer-protocol object (bytes, bytearray, memoryview,
/// array.array, NumPy arrays of any dtype).
fn contiguous_bytes(data: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let py = data.py();
    let view = py
        .import("builtins")?
        .getattr("memoryview")?
        .call1((data,))?
        .call_method1("cast", ("B",))
        .map_err(|_| invalid("expected a C-contiguous object that supports the buffer protocol"))?;
    let buffer = PyBuffer::<u8>::get(&view)?;
    buffer.to_vec(py)
}

/// The native runtime.
#[pyclass(module = "kora._core", frozen)]
#[derive(Debug)]
pub struct Runtime(Arc<kora::Runtime>);

/// A logical buffer: a host copy plus per-device replicas.
#[pyclass(module = "kora._core", frozen)]
#[derive(Debug)]
pub struct Buffer {
    handle: kora::BufferHandle,
    owner: Arc<kora::Runtime>,
}

/// Held only for its `Drop`, which ends the mapping.
#[allow(dead_code)]
enum Mapping {
    Read(MappedRead<'static>),
    Write(MappedWrite<'static>),
}

/// A live mapping of a buffer's host copy, exported through the buffer protocol (wrap it in
/// `memoryview` or `numpy.frombuffer`). It keeps the buffer alive and blocks other access from
/// this thread until it is garbage collected or `release()` is called.
#[pyclass(module = "kora._core", unsendable)]
pub struct MappedView {
    // Declared before `_handle`: the mapping borrows the buffer and must be dropped first.
    mapping: Option<Mapping>,
    data: *mut u8,
    len: usize,
    readonly: bool,
    exports: usize,
    _handle: Box<kora::BufferHandle>,
}

impl std::fmt::Debug for MappedView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedView")
            .field("len", &self.len)
            .field("readonly", &self.readonly)
            .finish()
    }
}

#[pymethods]
impl MappedView {
    fn __len__(&self) -> usize {
        if self.mapping.is_some() { self.len } else { 0 }
    }

    /// Ends the mapping early; fails while a memoryview of it is still alive.
    fn release(&mut self) -> PyResult<()> {
        if self.exports != 0 {
            return Err(invalid("release the memoryviews of this mapping first"));
        }
        self.mapping = None;
        Ok(())
    }

    /// # Safety
    /// Called by the interpreter with a valid `Py_buffer` to fill.
    unsafe fn __getbuffer__(
        mut slf: PyRefMut<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if slf.mapping.is_none() {
            return Err(invalid("the mapping was released"));
        }
        if slf.readonly && (flags & ffi::PyBUF_WRITABLE) != 0 {
            return Err(pyo3::exceptions::PyBufferError::new_err("mapping is read-only"));
        }
        let (data, len, readonly) = (slf.data, slf.len, slf.readonly);
        // SAFETY: `view` is valid per the protocol; the mapping keeps `data[..len]` alive and
        // writable (unless read-only) for as long as this object, which the view references,
        // lives.
        let status = unsafe {
            ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                data.cast(),
                len as ffi::Py_ssize_t,
                c_int::from(readonly),
                flags,
            )
        };
        if status != 0 {
            return Err(PyErr::fetch(slf.py()));
        }
        slf.exports += 1;
        Ok(())
    }

    /// # Safety
    /// Called by the interpreter for a view filled by `__getbuffer__`.
    unsafe fn __releasebuffer__(&mut self, _view: *mut ffi::Py_buffer) {
        self.exports = self.exports.saturating_sub(1);
    }
}

#[pymethods]
impl Buffer {
    fn __len__(&self) -> usize {
        self.handle.size_bytes() as usize
    }

    #[getter]
    fn id(&self) -> BufferId {
        let id = self.handle.id();
        BufferId {
            slot: id.slot,
            generation: id.generation,
        }
    }

    /// Copies the host copy to `bytes`.
    fn read<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let handle = self.handle.clone();
        let bytes = py.detach(move || handle.read_bytes()).or_raise()?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Replaces the whole host copy from any C-contiguous buffer-protocol object of exactly
    /// `len(buffer)` bytes (bytes, bytearray, memoryview, NumPy arrays).
    fn write(&self, py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = contiguous_bytes(&data)?;
        let handle = self.handle.clone();
        py.detach(move || handle.write_bytes(&bytes)).or_raise()
    }

    /// Maps the host copy (see `MappedView`).
    #[pyo3(signature = (access = MapAccess::ReadWrite))]
    fn map(&self, access: MapAccess) -> PyResult<MappedView> {
        let handle = Box::new(self.handle.clone());
        // The boxed handle never moves and is stored in the view after the mapping, so the
        // mapping is dropped first and its borrow never dangles.
        let address: *const kora::BufferHandle = &raw const *handle;
        // SAFETY: as above; `address` points into the box, which outlives the mapping.
        let extended: &'static kora::BufferHandle = unsafe { &*address };
        let (mapping, data, len, readonly) = match access {
            MapAccess::Read => {
                let mapped = extended.map_read().or_raise()?;
                let (data, len) = (mapped.as_ptr().cast_mut(), mapped.len());
                (Mapping::Read(mapped), data, len, true)
            }
            MapAccess::Write | MapAccess::ReadWrite => {
                let mut mapped = extended.map_write().or_raise()?;
                let (data, len) = (mapped.as_mut_ptr(), mapped.len());
                (Mapping::Write(mapped), data, len, false)
            }
        };
        Ok(MappedView {
            mapping: Some(mapping),
            data,
            len,
            readonly,
            exports: 0,
            _handle: handle,
        })
    }

    /// Copies the host copy into a new one-dimensional NumPy array of `dtype`.
    #[pyo3(signature = (dtype = None))]
    fn to_numpy<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let numpy = py.import("numpy")?;
        let dtype = match dtype {
            Some(dtype) => dtype,
            None => numpy.getattr("uint8")?,
        };
        let bytes = self.read(py)?;
        numpy
            .call_method1("frombuffer", (bytes, dtype))?
            .call_method0("copy")
    }

    fn __repr__(&self) -> String {
        format!("Buffer(size_bytes={})", self.handle.size_bytes())
    }
}

/// A validated kernel bound to one device.
#[pyclass(module = "kora._core", frozen)]
#[derive(Debug)]
pub struct Kernel {
    kernel: kora::Kernel,
    owner: Arc<kora::Runtime>,
}

/// Completion of submitted work. `await event` works in asyncio code.
#[pyclass(module = "kora._core", frozen)]
#[derive(Debug)]
pub struct Event(kora::Event);

#[pymethods]
impl Event {
    /// Blocks until completion (releasing the GIL) and raises `KoraError` on failure.
    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        let event = self.0.clone();
        py.detach(move || event.wait()).or_raise()
    }

    /// Like `wait`, raising `KoraError` with `ErrorCode.TIMEOUT` after `timeout_seconds`.
    fn wait_for(&self, py: Python<'_>, timeout_seconds: f64) -> PyResult<()> {
        if !(timeout_seconds >= 0.0 && timeout_seconds.is_finite()) {
            return Err(invalid("event wait timeout must be a non-negative number"));
        }
        let event = self.0.clone();
        py.detach(move || event.wait_timeout(Duration::from_secs_f64(timeout_seconds)))
            .or_raise()
    }

    /// Lets coroutines `await event` without blocking the event loop.
    fn __await__<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py().import("kora._aio")?.call_method1("await_event", (slf,))
    }

    /// `True` once the work has completed (successfully or not).
    #[getter]
    fn done(&self) -> bool {
        self.0.is_done()
    }

    /// Calls `loop.call_soon_threadsafe(resolver, future, error)` on completion (`error` is
    /// `None` on success). Used by `Event.__await__`.
    fn _register_awaitable(
        &self,
        event_loop: Py<PyAny>,
        future: Py<PyAny>,
        resolver: Py<PyAny>,
    ) -> PyResult<()> {
        self.0
            .on_complete(move |status| {
                Python::attach(|py| {
                    let error = match status {
                        Ok(()) => py.None(),
                        Err(error) => to_py(error).into_value(py).into_any(),
                    };
                    if let Err(failure) =
                        event_loop.call_method1(py, "call_soon_threadsafe", (resolver, future, error))
                    {
                        // The loop may already be closed; the awaiting coroutine is gone.
                        failure.write_unraisable(py, None);
                    }
                });
            })
            .or_raise()
    }
}

/// Describes one dispatch; methods return the builder for chaining.
#[pyclass(module = "kora._core")]
#[derive(Debug)]
pub struct DispatchBuilder {
    builder: Option<kora::DispatchBuilder>,
    owner: Arc<kora::Runtime>,
}

impl DispatchBuilder {
    fn update(
        mut slf: PyRefMut<'_, Self>,
        change: impl FnOnce(kora::DispatchBuilder) -> kora::DispatchBuilder,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let builder = slf
            .builder
            .take()
            .ok_or_else(|| invalid("dispatch was already submitted"))?;
        slf.builder = Some(change(builder));
        Ok(slf)
    }

    fn bind<'a>(
        slf: PyRefMut<'a, Self>,
        buffer: &Buffer,
        access: KernelAccess,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if !Arc::ptr_eq(&slf.owner, &buffer.owner) {
            return Err(invalid("buffer belongs to a different Runtime"));
        }
        let handle = buffer.handle.clone();
        Self::update(slf, move |b| match access {
            KernelAccess::Read => b.read(&handle),
            KernelAccess::Write => b.write(&handle),
            KernelAccess::ReadWrite => b.read_write(&handle),
        })
    }
}

#[pymethods]
impl DispatchBuilder {
    /// Global size in invocations.
    #[pyo3(signature = (x, y = 1, z = 1))]
    fn global_size(slf: PyRefMut<'_, Self>, x: u64, y: u64, z: u64) -> PyResult<PyRefMut<'_, Self>> {
        Self::update(slf, move |b| b.global_3d(x, y, z))
    }

    /// Alias of `global_size`.
    #[pyo3(name = "global", signature = (x, y = 1, z = 1))]
    fn global_alias(slf: PyRefMut<'_, Self>, x: u64, y: u64, z: u64) -> PyResult<PyRefMut<'_, Self>> {
        Self::update(slf, move |b| b.global_3d(x, y, z))
    }

    fn read<'a>(slf: PyRefMut<'a, Self>, buffer: PyRef<'a, Buffer>) -> PyResult<PyRefMut<'a, Self>> {
        Self::bind(slf, &buffer, KernelAccess::Read)
    }

    fn write<'a>(slf: PyRefMut<'a, Self>, buffer: PyRef<'a, Buffer>) -> PyResult<PyRefMut<'a, Self>> {
        Self::bind(slf, &buffer, KernelAccess::Write)
    }

    fn read_write<'a>(slf: PyRefMut<'a, Self>, buffer: PyRef<'a, Buffer>) -> PyResult<PyRefMut<'a, Self>> {
        Self::bind(slf, &buffer, KernelAccess::ReadWrite)
    }

    /// Push-constant bytes from any C-contiguous buffer-protocol object.
    fn push<'a>(slf: PyRefMut<'a, Self>, data: Bound<'a, PyAny>) -> PyResult<PyRefMut<'a, Self>> {
        let bytes = contiguous_bytes(&data)?;
        Self::update(slf, move |b| b.push_bytes(&bytes))
    }

    /// Keeps written outputs on the device until `Runtime.readback`.
    fn defer_readback(slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        Self::update(slf, kora::DispatchBuilder::defer_readback)
    }

    /// Submits the dispatch (once) and returns its `Event`.
    fn submit(&mut self, py: Python<'_>) -> PyResult<Event> {
        let builder = self
            .builder
            .take()
            .ok_or_else(|| invalid("dispatch was already submitted"))?;
        py.detach(move || builder.submit()).map(Event).or_raise()
    }
}

#[pymethods]
impl Runtime {
    /// Creates the runtime (releasing the GIL during initialization).
    #[staticmethod]
    #[pyo3(signature = (options = None))]
    fn create(py: Python<'_>, options: Option<RuntimeOptions>) -> PyResult<Self> {
        let options = kora::RuntimeOptions::from(&options.unwrap_or_default());
        py.detach(move || kora::Runtime::new(&options))
            .map(|r| Self(Arc::new(r)))
            .or_raise()
    }

    /// Initialized devices, in selection order.
    #[getter]
    fn devices(&self) -> Vec<DeviceId> {
        self.0
            .devices()
            .into_iter()
            .map(|d| DeviceId {
                index: d.index,
                uuid: d.uuid.to_string(),
            })
            .collect()
    }

    /// Creates a zero-filled buffer.
    #[pyo3(signature = (size_bytes, usage = BufferUsage::Storage))]
    fn create_buffer(&self, py: Python<'_>, size_bytes: u64, usage: BufferUsage) -> PyResult<Buffer> {
        let runtime = Arc::clone(&self.0);
        let handle = py
            .detach(move || runtime.create_buffer_bytes(size_bytes, usage.into()))
            .or_raise()?;
        Ok(Buffer {
            handle,
            owner: Arc::clone(&self.0),
        })
    }

    /// Creates a buffer holding a copy of `array` (any C-contiguous buffer-protocol object, such
    /// as a NumPy array).
    #[pyo3(signature = (array, usage = BufferUsage::Storage))]
    fn create_buffer_from(
        &self,
        py: Python<'_>,
        array: Bound<'_, PyAny>,
        usage: BufferUsage,
    ) -> PyResult<Buffer> {
        let bytes = contiguous_bytes(&array)?;
        let buffer = self.create_buffer(py, bytes.len() as u64, usage)?;
        let handle = buffer.handle.clone();
        py.detach(move || handle.write_bytes(&bytes)).or_raise()?;
        Ok(buffer)
    }

    /// Loads a kernel from a `.spv` path or SPIR-V bytes, for `device` or the first device.
    #[pyo3(signature = (spirv, entry_point, metadata, device = None))]
    fn load_kernel(
        &self,
        py: Python<'_>,
        spirv: Bound<'_, PyAny>,
        entry_point: String,
        metadata: KernelMetadata,
        device: Option<DeviceId>,
    ) -> PyResult<Kernel> {
        let words = if let Ok(path) = spirv.extract::<PathBuf>() {
            let bytes = std::fs::read(&path).map_err(|e| {
                to_py(kora::Error::new(
                    kora::ErrorCode::InvalidKernel,
                    format!("cannot read SPIR-V module {}: {e}", path.display()),
                ))
            })?;
            kora::kernel::spirv::words_from_bytes(&bytes).or_raise()?
        } else {
            kora::kernel::spirv::words_from_bytes(&contiguous_bytes(&spirv)?).or_raise()?
        };
        let metadata = kora::KernelMetadata::from(&metadata);
        let runtime = Arc::clone(&self.0);
        let kernel = py
            .detach(move || match device {
                Some(device) => runtime.load_kernel_on(&device.rust(), &words, &entry_point, metadata),
                None => runtime.load_kernel(&words, &entry_point, metadata),
            })
            .or_raise()?;
        Ok(Kernel {
            kernel,
            owner: Arc::clone(&self.0),
        })
    }

    /// Starts describing a dispatch of `kernel`.
    fn dispatch(&self, kernel: &Kernel) -> PyResult<DispatchBuilder> {
        if !Arc::ptr_eq(&self.0, &kernel.owner) {
            return Err(invalid("kernel belongs to a different Runtime"));
        }
        Ok(DispatchBuilder {
            builder: Some(self.0.dispatch(&kernel.kernel)),
            owner: Arc::clone(&self.0),
        })
    }

    /// Makes device-resident results visible in the buffer's host copy (from `device` only when
    /// given).
    #[pyo3(signature = (buffer, device = None))]
    fn readback(&self, py: Python<'_>, buffer: &Buffer, device: Option<DeviceId>) -> PyResult<Event> {
        if !Arc::ptr_eq(&self.0, &buffer.owner) {
            return Err(invalid("buffer belongs to a different Runtime"));
        }
        let runtime = Arc::clone(&self.0);
        let handle = buffer.handle.clone();
        py.detach(move || match device {
            Some(device) => runtime.readback_from(&handle, &device.rust()),
            None => runtime.readback(&handle),
        })
        .map(Event)
        .or_raise()
    }

    fn __repr__(&self) -> String {
        format!("Runtime(devices={})", self.0.devices().len())
    }
}

/// The linked KORA version.
#[pyfunction]
fn version() -> String {
    kora::version::version_string()
}

/// The KORA metadata/report schema version.
#[pyfunction]
fn schema_version() -> u32 {
    kora::version::SCHEMA_VERSION
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("KoraError", py.get_type::<KoraError>())?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(schema_version, module)?)?;
    module.add_class::<ErrorCode>()?;
    module.add_class::<DevicePolicy>()?;
    module.add_class::<BufferUsage>()?;
    module.add_class::<MapAccess>()?;
    module.add_class::<KernelBindingKind>()?;
    module.add_class::<KernelAccess>()?;
    module.add_class::<TaskType>()?;
    module.add_class::<AccessMode>()?;
    module.add_class::<DeviceId>()?;
    module.add_class::<BufferId>()?;
    module.add_class::<BufferAccess>()?;
    module.add_class::<TaskMetadata>()?;
    module.add_class::<TaskNode>()?;
    module.add_class::<TaskGraphPlan>()?;
    module.add_class::<TaskGraph>()?;
    module.add_class::<ExecutionPlan>()?;
    module.add_class::<RuntimeOptions>()?;
    module.add_class::<KernelBindingMetadata>()?;
    module.add_class::<PushConstantFieldMetadata>()?;
    module.add_class::<SpecializationConstantMetadata>()?;
    module.add_class::<KernelMetadata>()?;
    module.add_class::<Runtime>()?;
    module.add_class::<Buffer>()?;
    module.add_class::<MappedView>()?;
    module.add_class::<Kernel>()?;
    module.add_class::<Event>()?;
    module.add_class::<DispatchBuilder>()?;
    Ok(())
}
