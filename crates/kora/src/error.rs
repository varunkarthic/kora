//! Structured error model.
//!
//! Expected failures are values: every fallible KORA operation returns [`Result<T>`]. An [`Error`]
//! carries a stable [`ErrorCode`], a human-readable message, and optional structured context (Vulkan
//! result, device identity, task, kernel, attempted recovery, and a cause chain). The payload is boxed
//! so `Result<T>` stays small on hot paths.

use std::fmt;
use std::sync::Arc;

/// Stable error categories. Discriminants match the C ABI and the historical C++ numbering
/// (`0` is reserved for success).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
#[non_exhaustive]
pub enum ErrorCode {
    /// No Vulkan loader or implementation could be loaded.
    NoVulkanImplementation = 1,
    /// No physical device satisfied the selection policy and requirements.
    NoEligibleDevice,
    /// The selected device exposes no compute-capable queue family.
    NoComputeQueue,
    /// A required feature, extension, or capability is not available.
    UnsupportedFeature,
    /// A kernel module or its metadata is malformed or inconsistent.
    InvalidKernel,
    /// Dispatch bindings do not match the kernel interface.
    InvalidKernelBinding,
    /// Dispatch dimensions or parameters are outside device limits.
    InvalidDispatch,
    /// A host or device allocation failed.
    AllocationFailed,
    /// The request does not fit the configured memory budget.
    OutOfMemoryBudget,
    /// A host/device transfer failed.
    TransferFailed,
    /// A wait expired before the operation completed (the operation itself continues).
    Timeout,
    /// The logical device was lost.
    DeviceLost,
    /// Compute pipeline creation failed.
    PipelineCreationFailed,
    /// A synchronization primitive failed or was misused.
    SynchronizationError,
    /// Validation detected an error.
    ValidationError,
    /// A Python binding boundary failed.
    PythonBindingError,
    /// An internal invariant was violated (a KORA bug).
    InternalInvariantViolation,
    /// Bad caller input (out-of-range, mismatched sizes, ...).
    InvalidArgument,
    /// Checked arithmetic overflowed.
    Overflow,
    /// A task graph contains a cycle.
    GraphCycle,
    /// Work was cancelled by shutdown or by the caller.
    Cancelled,
    /// A buffer handle refers to a released or reused slot.
    StaleHandle,
    /// Authoritative data existed only on a lost device.
    DataLoss,
    /// A persistent cache or file operation failed.
    IoError,
}

impl ErrorCode {
    /// Every code, in discriminant order.
    pub const ALL: [ErrorCode; 24] = [
        Self::NoVulkanImplementation,
        Self::NoEligibleDevice,
        Self::NoComputeQueue,
        Self::UnsupportedFeature,
        Self::InvalidKernel,
        Self::InvalidKernelBinding,
        Self::InvalidDispatch,
        Self::AllocationFailed,
        Self::OutOfMemoryBudget,
        Self::TransferFailed,
        Self::Timeout,
        Self::DeviceLost,
        Self::PipelineCreationFailed,
        Self::SynchronizationError,
        Self::ValidationError,
        Self::PythonBindingError,
        Self::InternalInvariantViolation,
        Self::InvalidArgument,
        Self::Overflow,
        Self::GraphCycle,
        Self::Cancelled,
        Self::StaleHandle,
        Self::DataLoss,
        Self::IoError,
    ];

    /// Stable name such as `"DeviceLost"`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::NoVulkanImplementation => "NoVulkanImplementation",
            Self::NoEligibleDevice => "NoEligibleDevice",
            Self::NoComputeQueue => "NoComputeQueue",
            Self::UnsupportedFeature => "UnsupportedFeature",
            Self::InvalidKernel => "InvalidKernel",
            Self::InvalidKernelBinding => "InvalidKernelBinding",
            Self::InvalidDispatch => "InvalidDispatch",
            Self::AllocationFailed => "AllocationFailed",
            Self::OutOfMemoryBudget => "OutOfMemoryBudget",
            Self::TransferFailed => "TransferFailed",
            Self::Timeout => "Timeout",
            Self::DeviceLost => "DeviceLost",
            Self::PipelineCreationFailed => "PipelineCreationFailed",
            Self::SynchronizationError => "SynchronizationError",
            Self::ValidationError => "ValidationError",
            Self::PythonBindingError => "PythonBindingError",
            Self::InternalInvariantViolation => "InternalInvariantViolation",
            Self::InvalidArgument => "InvalidArgument",
            Self::Overflow => "Overflow",
            Self::GraphCycle => "GraphCycle",
            Self::Cancelled => "Cancelled",
            Self::StaleHandle => "StaleHandle",
            Self::DataLoss => "DataLoss",
            Self::IoError => "IoError",
        }
    }

    /// Looks a code up by its stable discriminant.
    pub fn from_i32(value: i32) -> Option<Self> {
        Self::ALL.into_iter().find(|code| *code as i32 == value)
    }

    /// Looks a code up by its stable name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|code| code.name() == name)
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ErrorInner {
    code: ErrorCode,
    message: String,
    vk_result: i32,
    device: String,
    task_id: u64,
    kernel_id: String,
    recovery_action: String,
    cause: Option<Arc<Error>>,
}

/// A structured KORA error. Cloning is cheap enough for diagnostics; the cause chain is shared.
#[derive(Clone, PartialEq)]
pub struct Error(Box<ErrorInner>);

impl Error {
    /// Creates an error with a code and message and no further context.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self(Box::new(ErrorInner {
            code,
            message: message.into(),
            vk_result: 0,
            device: String::new(),
            task_id: 0,
            kernel_id: String::new(),
            recovery_action: String::new(),
            cause: None,
        }))
    }

    /// Stable error category.
    pub fn code(&self) -> ErrorCode {
        self.0.code
    }
    /// Human-readable message (without context fields).
    pub fn message(&self) -> &str {
        &self.0.message
    }
    /// Raw `VkResult`, or `0` (`VK_SUCCESS`) when not relevant.
    pub fn vk_result(&self) -> i32 {
        self.0.vk_result
    }
    /// Device identity (name or UUID), or empty.
    pub fn device(&self) -> &str {
        &self.0.device
    }
    /// Task identifier, or `0` for none.
    pub fn task_id(&self) -> u64 {
        self.0.task_id
    }
    /// Kernel identifier, or empty.
    pub fn kernel_id(&self) -> &str {
        &self.0.kernel_id
    }
    /// Recovery action that was attempted, or empty.
    pub fn recovery_action(&self) -> &str {
        &self.0.recovery_action
    }
    /// The underlying error, if this error wraps another.
    pub fn cause(&self) -> Option<&Error> {
        self.0.cause.as_deref()
    }

    /// Attaches a raw `VkResult`.
    #[must_use]
    pub fn with_vk(mut self, result: i32) -> Self {
        self.0.vk_result = result;
        self
    }
    /// Attaches a device identity.
    #[must_use]
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.0.device = device.into();
        self
    }
    /// Attaches a task identifier.
    #[must_use]
    pub fn with_task(mut self, task: u64) -> Self {
        self.0.task_id = task;
        self
    }
    /// Attaches a kernel identifier.
    #[must_use]
    pub fn with_kernel(mut self, kernel: impl Into<String>) -> Self {
        self.0.kernel_id = kernel.into();
        self
    }
    /// Records the attempted recovery action.
    #[must_use]
    pub fn with_recovery(mut self, action: impl Into<String>) -> Self {
        self.0.recovery_action = action.into();
        self
    }
    /// Chains an underlying cause.
    #[must_use]
    pub fn with_cause(mut self, cause: Error) -> Self {
        self.0.cause = Some(Arc::new(cause));
        self
    }
    /// Replaces the message, keeping all context.
    #[must_use]
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.0.message = message.into();
        self
    }

    /// Single-line description: `"Code: message [vk=..] [device=..] ... <- cause"`.
    pub fn describe(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let e = &self.0;
        write!(f, "{}: {}", e.code, e.message)?;
        if e.vk_result != 0 {
            write!(f, " [vk={}]", e.vk_result)?;
        }
        if !e.device.is_empty() {
            write!(f, " [device={}]", e.device)?;
        }
        if e.task_id != 0 {
            write!(f, " [task={}]", e.task_id)?;
        }
        if !e.kernel_id.is_empty() {
            write!(f, " [kernel={}]", e.kernel_id)?;
        }
        if !e.recovery_action.is_empty() {
            write!(f, " [recovery={}]", e.recovery_action)?;
        }
        if let Some(cause) = &e.cause {
            write!(f, " <- {cause}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Error({self})")
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0
            .cause
            .as_deref()
            .map(|cause| cause as &(dyn std::error::Error + 'static))
    }
}

/// Value-or-[`Error`]. Expected failures never panic.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// `Result<()>`.
pub type Status = Result<()>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_has_a_unique_stable_name_and_discriminant() {
        let mut names = std::collections::HashSet::new();
        for (index, code) in ErrorCode::ALL.into_iter().enumerate() {
            assert_eq!(code as i32, index as i32 + 1);
            assert!(names.insert(code.name()));
            assert_eq!(ErrorCode::from_name(code.name()), Some(code));
            assert_eq!(ErrorCode::from_i32(code as i32), Some(code));
        }
        assert_eq!(ErrorCode::from_i32(0), None);
        assert_eq!(ErrorCode::from_i32(25), None);
        assert_eq!(ErrorCode::from_name("Unknown"), None);
    }

    fn fail<T>(code: ErrorCode, message: &str) -> Result<T> {
        Err(Error::new(code, message))
    }

    #[test]
    fn map_and_then_propagate_codes() {
        let b: Result<i32> = fail(ErrorCode::Timeout, "slow");
        assert_eq!(b.unwrap_err().code(), ErrorCode::Timeout);
        let r: Result<i32> = Ok::<i32, Error>(2).map(|x| x * 3);
        assert_eq!(r.unwrap(), 6);
        let e: Result<i32> = fail::<i32>(ErrorCode::Cancelled, "x").map(|x| x + 1);
        assert_eq!(e.unwrap_err().code(), ErrorCode::Cancelled);
        let c: Result<i32> = Ok::<i32, Error>(4).and_then(|_| fail(ErrorCode::Overflow, "o"));
        assert_eq!(c.unwrap_err().code(), ErrorCode::Overflow);
    }

    #[test]
    fn display_matches_the_historical_single_line_format() {
        let failed = Error::new(ErrorCode::DeviceLost, "gone");
        assert_eq!(failed.to_string(), "DeviceLost: gone");
    }

    #[test]
    fn cause_chain_describe_includes_every_context_field() {
        let inner = Error::new(ErrorCode::AllocationFailed, "vkAllocateMemory").with_vk(-2);
        let outer = Error::new(ErrorCode::OutOfMemoryBudget, "no space")
            .with_device("dev0")
            .with_task(7)
            .with_kernel("k")
            .with_recovery("evict")
            .with_cause(inner.clone());
        let d = outer.describe();
        assert!(d.contains("OutOfMemoryBudget: no space"));
        assert!(d.contains("[device=dev0]"));
        assert!(d.contains("[task=7]"));
        assert!(d.contains("[kernel=k]"));
        assert!(d.contains("[recovery=evict]"));
        assert!(d.contains("<- AllocationFailed: vkAllocateMemory [vk=-2]"));
        assert_eq!(outer.cause(), Some(&inner));
        let source = std::error::Error::source(&outer).expect("cause is exposed as source");
        assert_eq!(source.to_string(), inner.to_string());
    }

    fn half(x: i32) -> Result<i32> {
        if x % 2 != 0 {
            return fail(ErrorCode::InvalidArgument, "odd");
        }
        Ok(x / 2)
    }

    fn quarter(x: i32) -> Result<i32> {
        let h = half(x)?;
        half(h)
    }

    #[test]
    fn question_mark_propagates() {
        assert_eq!(quarter(8).unwrap(), 2);
        assert_eq!(quarter(6).unwrap_err().code(), ErrorCode::InvalidArgument);
    }

    #[test]
    fn result_is_pointer_sized_for_small_values() {
        assert_eq!(std::mem::size_of::<Error>(), std::mem::size_of::<usize>());
        assert!(std::mem::size_of::<Result<u64>>() <= 2 * std::mem::size_of::<usize>());
    }

    #[test]
    fn with_message_keeps_context() {
        let e = Error::new(ErrorCode::IoError, "a")
            .with_device("d")
            .with_message("b");
        assert_eq!(e.message(), "b");
        assert_eq!(e.device(), "d");
    }
}
