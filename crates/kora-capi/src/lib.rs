//! Stable C ABI for KORA: `libkora` plus the cbindgen-generated `include/kora.h`.
//!
//! Every function catches panics and reports failures as a [`KoraStatus`]; the calling thread's
//! last error message is available from [`kora_last_error_message`]. Handles are opaque boxes the
//! caller releases with the matching `*_destroy` function, which accepts `NULL`.
//!
//! The ABI version ([`KORA_ABI_VERSION`]) changes only when an existing declaration changes
//! incompatibly; new functions and new trailing `KoraRuntimeOptions` fields keep it.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;
use std::time::Duration;

use kora_rt::{
    BufferHandle, BufferUsage, DevicePolicy, DispatchBuilder, Error, ErrorCode, Event, Kernel, Runtime,
    RuntimeOptions, parse_kernel_metadata,
};

/// Version of this C ABI.
pub const KORA_ABI_VERSION: u32 = 1;

/// Result of a fallible call: `KORA_STATUS_OK`, or the stable KORA error code.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KoraStatus {
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

impl From<ErrorCode> for KoraStatus {
    fn from(code: ErrorCode) -> Self {
        match code {
            ErrorCode::NoVulkanImplementation => Self::NoVulkanImplementation,
            ErrorCode::NoEligibleDevice => Self::NoEligibleDevice,
            ErrorCode::NoComputeQueue => Self::NoComputeQueue,
            ErrorCode::UnsupportedFeature => Self::UnsupportedFeature,
            ErrorCode::InvalidKernel => Self::InvalidKernel,
            ErrorCode::InvalidKernelBinding => Self::InvalidKernelBinding,
            ErrorCode::InvalidDispatch => Self::InvalidDispatch,
            ErrorCode::AllocationFailed => Self::AllocationFailed,
            ErrorCode::OutOfMemoryBudget => Self::OutOfMemoryBudget,
            ErrorCode::TransferFailed => Self::TransferFailed,
            ErrorCode::Timeout => Self::Timeout,
            ErrorCode::DeviceLost => Self::DeviceLost,
            ErrorCode::PipelineCreationFailed => Self::PipelineCreationFailed,
            ErrorCode::SynchronizationError => Self::SynchronizationError,
            ErrorCode::ValidationError => Self::ValidationError,
            ErrorCode::PythonBindingError => Self::PythonBindingError,
            ErrorCode::InvalidArgument => Self::InvalidArgument,
            ErrorCode::Overflow => Self::Overflow,
            ErrorCode::GraphCycle => Self::GraphCycle,
            ErrorCode::Cancelled => Self::Cancelled,
            ErrorCode::StaleHandle => Self::StaleHandle,
            ErrorCode::DataLoss => Self::DataLoss,
            ErrorCode::IoError => Self::IoError,
            _ => Self::InternalInvariantViolation,
        }
    }
}

/// Leave a `KoraRuntimeOptions` setting to the environment (`KORA_*`) and built-in defaults.
pub const KORA_DEFAULT: i32 = -1;

/// `device_policy` values.
pub const KORA_DEVICE_POLICY_AUTO: i32 = 0;
pub const KORA_DEVICE_POLICY_PREFER_DISCRETE: i32 = 1;
pub const KORA_DEVICE_POLICY_PREFER_INTEGRATED: i32 = 2;
pub const KORA_DEVICE_POLICY_BY_INDEX: i32 = 3;
pub const KORA_DEVICE_POLICY_BY_UUID: i32 = 4;
pub const KORA_DEVICE_POLICY_ALL_ELIGIBLE: i32 = 5;

/// Binding access passed to `kora_dispatch_bind`.
pub const KORA_ACCESS_READ: u32 = 0;
pub const KORA_ACCESS_WRITE: u32 = 1;
pub const KORA_ACCESS_READ_WRITE: u32 = 2;

/// Runtime configuration. Initialize with `kora_runtime_options_default`, then change fields.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KoraRuntimeOptions {
    /// `sizeof(KoraRuntimeOptions)` as compiled by the caller; later ABI revisions only append
    /// fields.
    pub struct_size: u32,
    /// One of `KORA_DEVICE_POLICY_*`, or `KORA_DEFAULT`.
    pub device_policy: i32,
    /// Device enumeration index for `KORA_DEVICE_POLICY_BY_INDEX`, or `KORA_DEFAULT`.
    pub device_index: i32,
    /// Lower-case hex device UUID for `KORA_DEVICE_POLICY_BY_UUID`, or `NULL`.
    pub device_uuid: *const c_char,
    /// 0 or 1 to disable or enable Vulkan validation, or `KORA_DEFAULT`.
    pub enable_validation: i32,
    /// 0 or 1 to skip or run startup calibration, or `KORA_DEFAULT`.
    pub enable_startup_calibration: i32,
    /// Directory for the pipeline cache, calibration, and performance database, or `NULL`.
    pub cache_dir: *const c_char,
}

/// A runtime: the Vulkan instance, its selected devices, and their queues and memory.
pub struct KoraRuntime(Runtime);
/// A logical buffer (host copy plus per-device replicas). Independent of its runtime's lifetime.
pub struct KoraBuffer(BufferHandle);
/// A validated kernel bound to one device.
pub struct KoraKernel(Kernel);
/// A dispatch being described; consumed by `kora_dispatch_submit`.
pub struct KoraDispatch(Option<DispatchBuilder>);
/// Completion of submitted work.
pub struct KoraEvent(Event);

impl std::fmt::Debug for KoraRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::fmt::Debug for KoraBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::fmt::Debug for KoraKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::fmt::Debug for KoraDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("KoraDispatch")
    }
}
impl std::fmt::Debug for KoraEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(message: &str) {
    let message = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|last| *last.borrow_mut() = message);
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorCode::InvalidArgument, message)
}

/// Runs `body`, converting errors and panics into a status and the thread's last error.
fn guard(body: impl FnOnce() -> kora_rt::Status) -> KoraStatus {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => KoraStatus::Ok,
        Ok(Err(error)) => {
            set_last_error(&error.describe());
            error.code().into()
        }
        Err(_) => {
            set_last_error("KORA panicked inside a C API call");
            KoraStatus::InternalInvariantViolation
        }
    }
}

/// Stores `value` as a new handle in `*out`.
///
/// # Safety
/// `out` is null (rejected) or valid for a pointer write.
unsafe fn emit<T>(out: *mut *mut T, value: T) -> kora_rt::Status {
    if out.is_null() {
        return Err(invalid("output pointer is NULL"));
    }
    // SAFETY: checked non-null; the caller guarantees it is writable.
    unsafe { *out = Box::into_raw(Box::new(value)) };
    Ok(())
}

/// # Safety
/// `handle` is null or a live handle of type `T` created by this library.
unsafe fn borrow<'a, T>(handle: *const T, what: &str) -> kora_rt::Result<&'a T> {
    // SAFETY: forwarded from the caller.
    unsafe { handle.as_ref() }.ok_or_else(|| invalid(&format!("{what} handle is NULL")))
}

/// # Safety
/// `handle` is null or a live handle of type `T` created by this library.
unsafe fn borrow_mut<'a, T>(handle: *mut T, what: &str) -> kora_rt::Result<&'a mut T> {
    // SAFETY: forwarded from the caller.
    unsafe { handle.as_mut() }.ok_or_else(|| invalid(&format!("{what} handle is NULL")))
}

/// # Safety
/// `text` is null or a NUL-terminated string.
unsafe fn string(text: *const c_char, what: &str) -> kora_rt::Result<Option<String>> {
    if text.is_null() {
        return Ok(None);
    }
    // SAFETY: forwarded from the caller.
    let text = unsafe { CStr::from_ptr(text) };
    text.to_str()
        .map(|s| Some(s.to_string()))
        .map_err(|_| invalid(&format!("{what} is not UTF-8")))
}

/// # Safety
/// `handle` is null or a live, owned handle; it must not be used afterwards.
unsafe fn destroy<T>(handle: *mut T) {
    if !handle.is_null() {
        // Dropping can run KORA teardown; a panic must not cross the C boundary.
        let _ = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: the caller transfers ownership of a pointer from Box::into_raw.
            drop(unsafe { Box::from_raw(handle) });
        }));
    }
}

/// The ABI version this library implements (`KORA_ABI_VERSION`).
#[unsafe(no_mangle)]
pub extern "C" fn kora_abi_version() -> u32 {
    KORA_ABI_VERSION
}

/// The library version, e.g. "1.0.3". The string is static.
#[unsafe(no_mangle)]
pub extern "C" fn kora_version_string() -> *const c_char {
    static VERSION: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(kora_rt::version::version_string()).unwrap_or_default())
        .as_ptr()
}

/// The stable name of a status, e.g. "DeviceLost". The string is static.
#[unsafe(no_mangle)]
pub extern "C" fn kora_status_name(status: KoraStatus) -> *const c_char {
    let name: &'static CStr = match ErrorCode::from_i32(status as i32) {
        None => c"Ok",
        Some(code) => match code {
            ErrorCode::NoVulkanImplementation => c"NoVulkanImplementation",
            ErrorCode::NoEligibleDevice => c"NoEligibleDevice",
            ErrorCode::NoComputeQueue => c"NoComputeQueue",
            ErrorCode::UnsupportedFeature => c"UnsupportedFeature",
            ErrorCode::InvalidKernel => c"InvalidKernel",
            ErrorCode::InvalidKernelBinding => c"InvalidKernelBinding",
            ErrorCode::InvalidDispatch => c"InvalidDispatch",
            ErrorCode::AllocationFailed => c"AllocationFailed",
            ErrorCode::OutOfMemoryBudget => c"OutOfMemoryBudget",
            ErrorCode::TransferFailed => c"TransferFailed",
            ErrorCode::Timeout => c"Timeout",
            ErrorCode::DeviceLost => c"DeviceLost",
            ErrorCode::PipelineCreationFailed => c"PipelineCreationFailed",
            ErrorCode::SynchronizationError => c"SynchronizationError",
            ErrorCode::ValidationError => c"ValidationError",
            ErrorCode::PythonBindingError => c"PythonBindingError",
            ErrorCode::InternalInvariantViolation => c"InternalInvariantViolation",
            ErrorCode::InvalidArgument => c"InvalidArgument",
            ErrorCode::Overflow => c"Overflow",
            ErrorCode::GraphCycle => c"GraphCycle",
            ErrorCode::Cancelled => c"Cancelled",
            ErrorCode::StaleHandle => c"StaleHandle",
            ErrorCode::DataLoss => c"DataLoss",
            ErrorCode::IoError => c"IoError",
            _ => c"Unknown",
        },
    };
    name.as_ptr()
}

/// The calling thread's most recent error description (empty if none). Valid until the next
/// failing call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn kora_last_error_message() -> *const c_char {
    LAST_ERROR.with(|last| last.borrow().as_ptr())
}

/// Fills `options` with defaults: every setting `KORA_DEFAULT`, strings `NULL`.
///
/// # Safety
/// `options` is null (ignored) or valid for writing one `KoraRuntimeOptions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_runtime_options_default(options: *mut KoraRuntimeOptions) {
    if options.is_null() {
        return;
    }
    // SAFETY: checked non-null; the caller guarantees it is writable.
    unsafe {
        options.write(KoraRuntimeOptions {
            struct_size: size_of::<KoraRuntimeOptions>() as u32,
            device_policy: KORA_DEFAULT,
            device_index: KORA_DEFAULT,
            device_uuid: ptr::null(),
            enable_validation: KORA_DEFAULT,
            enable_startup_calibration: KORA_DEFAULT,
            cache_dir: ptr::null(),
        });
    }
}

fn tristate(value: i32, what: &str) -> kora_rt::Result<Option<bool>> {
    match value {
        KORA_DEFAULT => Ok(None),
        0 => Ok(Some(false)),
        1 => Ok(Some(true)),
        _ => Err(invalid(&format!("{what} must be 0, 1, or KORA_DEFAULT"))),
    }
}

/// # Safety
/// `options` is null or points to a `KoraRuntimeOptions` of at least `struct_size` bytes whose
/// strings are null or NUL-terminated.
unsafe fn runtime_options(options: *const KoraRuntimeOptions) -> kora_rt::Result<RuntimeOptions> {
    // SAFETY: forwarded from the caller.
    let Some(options) = (unsafe { options.as_ref() }) else {
        return Ok(RuntimeOptions::default());
    };
    if (options.struct_size as usize) < size_of::<KoraRuntimeOptions>() {
        return Err(invalid(
            "KoraRuntimeOptions.struct_size is smaller than this ABI's struct",
        ));
    }
    let device_policy = match options.device_policy {
        KORA_DEFAULT => None,
        KORA_DEVICE_POLICY_AUTO => Some(DevicePolicy::Auto),
        KORA_DEVICE_POLICY_PREFER_DISCRETE => Some(DevicePolicy::PreferDiscrete),
        KORA_DEVICE_POLICY_PREFER_INTEGRATED => Some(DevicePolicy::PreferIntegrated),
        KORA_DEVICE_POLICY_BY_INDEX => Some(DevicePolicy::ByIndex),
        KORA_DEVICE_POLICY_BY_UUID => Some(DevicePolicy::ByUuid),
        KORA_DEVICE_POLICY_ALL_ELIGIBLE => Some(DevicePolicy::AllEligible),
        _ => return Err(invalid("unknown device_policy")),
    };
    let device_index = match options.device_index {
        KORA_DEFAULT => None,
        index if index >= 0 => Some(index as u32),
        _ => return Err(invalid("device_index must be non-negative or KORA_DEFAULT")),
    };
    Ok(RuntimeOptions {
        device_policy,
        device_index,
        // SAFETY: forwarded from the caller.
        device_uuid: unsafe { string(options.device_uuid, "device_uuid") }?,
        enable_validation: tristate(options.enable_validation, "enable_validation")?,
        enable_startup_calibration: tristate(
            options.enable_startup_calibration,
            "enable_startup_calibration",
        )?,
        // SAFETY: forwarded from the caller.
        cache_dir: unsafe { string(options.cache_dir, "cache_dir") }?,
        ..RuntimeOptions::default()
    })
}

/// Creates a runtime. `options` may be `NULL` for defaults.
///
/// # Safety
/// `options` is null or a valid `KoraRuntimeOptions`; `out` is valid for a pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_runtime_create(
    options: *const KoraRuntimeOptions,
    out: *mut *mut KoraRuntime,
) -> KoraStatus {
    guard(|| {
        // SAFETY: forwarded from the caller.
        let options = unsafe { runtime_options(options) }?;
        let runtime = Runtime::new(&options)?;
        // SAFETY: forwarded from the caller.
        unsafe { emit(out, KoraRuntime(runtime)) }
    })
}

/// Drains (or cancels, per `KORA_SHUTDOWN_MODE`) queued work and releases the runtime. Buffers,
/// kernels, and events created from it stay valid to destroy.
///
/// # Safety
/// `runtime` is null or a live runtime handle, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_runtime_destroy(runtime: *mut KoraRuntime) {
    // SAFETY: forwarded from the caller.
    unsafe { destroy(runtime) }
}

/// Number of initialized devices (0 for a null handle).
///
/// # Safety
/// `runtime` is null or a live runtime handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_runtime_device_count(runtime: *const KoraRuntime) -> u32 {
    // SAFETY: forwarded from the caller.
    unsafe { runtime.as_ref() }.map_or(0, |r| r.0.devices().len() as u32)
}

/// Copies the NUL-terminated name of device `index` into `buffer` (truncated to `capacity`) and
/// stores the full length plus one in `*required` when it is not `NULL`.
///
/// # Safety
/// `runtime` is a live runtime handle; `buffer` is null or valid for `capacity` bytes;
/// `required` is null or valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_runtime_device_name(
    runtime: *const KoraRuntime,
    index: u32,
    buffer: *mut c_char,
    capacity: usize,
    required: *mut usize,
) -> KoraStatus {
    guard(|| {
        // SAFETY: forwarded from the caller.
        let runtime = &unsafe { borrow(runtime, "runtime") }?.0;
        let device = runtime
            .devices()
            .into_iter()
            .nth(index as usize)
            .ok_or_else(|| invalid("device index is out of range"))?;
        let name = runtime.device_info(&device)?.name.clone();
        let bytes = name.as_bytes();
        if !required.is_null() {
            // SAFETY: checked non-null; the caller guarantees it is writable.
            unsafe { *required = bytes.len() + 1 };
        }
        if !buffer.is_null() && capacity > 0 {
            let count = bytes.len().min(capacity - 1);
            // SAFETY: the caller guarantees `capacity` writable bytes at `buffer`.
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast(), count);
                *buffer.add(count) = 0;
            }
        }
        Ok(())
    })
}

/// Creates a zero-initialized storage buffer of `size_bytes` bytes.
///
/// # Safety
/// `runtime` is a live runtime handle; `out` is valid for a pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_buffer_create(
    runtime: *const KoraRuntime,
    size_bytes: u64,
    out: *mut *mut KoraBuffer,
) -> KoraStatus {
    guard(|| {
        // SAFETY: forwarded from the caller.
        let runtime = &unsafe { borrow(runtime, "runtime") }?.0;
        let buffer = runtime.create_buffer_bytes(size_bytes, BufferUsage::Storage)?;
        // SAFETY: forwarded from the caller.
        unsafe { emit(out, KoraBuffer(buffer)) }
    })
}

/// Releases the caller's reference; the buffer lives on while submitted work uses it.
///
/// # Safety
/// `buffer` is null or a live buffer handle, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_buffer_destroy(buffer: *mut KoraBuffer) {
    // SAFETY: forwarded from the caller.
    unsafe { destroy(buffer) }
}

/// Size in bytes (0 for a null handle).
///
/// # Safety
/// `buffer` is null or a live buffer handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_buffer_size(buffer: *const KoraBuffer) -> u64 {
    // SAFETY: forwarded from the caller.
    unsafe { buffer.as_ref() }.map_or(0, |b| b.0.size_bytes())
}

/// Replaces the whole host copy with `size` bytes from `data` (`size` must equal the buffer size).
///
/// # Safety
/// `buffer` is a live buffer handle; `data` is valid for `size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_buffer_write(
    buffer: *const KoraBuffer,
    data: *const u8,
    size: u64,
) -> KoraStatus {
    guard(|| {
        // SAFETY: forwarded from the caller.
        let buffer = &unsafe { borrow(buffer, "buffer") }?.0;
        if data.is_null() && size != 0 {
            return Err(invalid("data is NULL"));
        }
        let len = usize::try_from(size).map_err(|_| Error::new(ErrorCode::Overflow, "size"))?;
        let bytes = if len == 0 {
            &[][..]
        } else {
            // SAFETY: the caller guarantees `size` readable bytes at `data`.
            unsafe { std::slice::from_raw_parts(data, len) }
        };
        buffer.write_bytes(bytes)
    })
}

/// Copies the host copy into `data` (`size` must equal the buffer size). Call `kora_readback`
/// first for results of a dispatch submitted with `kora_dispatch_defer_readback`.
///
/// # Safety
/// `buffer` is a live buffer handle; `data` is valid for writing `size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_buffer_read(buffer: *const KoraBuffer, data: *mut u8, size: u64) -> KoraStatus {
    guard(|| {
        // SAFETY: forwarded from the caller.
        let buffer = &unsafe { borrow(buffer, "buffer") }?.0;
        if data.is_null() && size != 0 {
            return Err(invalid("data is NULL"));
        }
        let len = usize::try_from(size).map_err(|_| Error::new(ErrorCode::Overflow, "size"))?;
        if len == 0 {
            return buffer.read_into(&mut []);
        }
        // SAFETY: the caller guarantees `size` writable bytes at `data`.
        buffer.read_into(unsafe { std::slice::from_raw_parts_mut(data, len) })
    })
}

/// Validates and compiles a kernel for the runtime's first device. `metadata_json` uses the KORA
/// kernel metadata format (`kernel-format` in the documentation).
///
/// # Safety
/// `runtime` is a live runtime handle; `words` is valid for `word_count` words; the strings are
/// NUL-terminated; `out` is valid for a pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_kernel_create(
    runtime: *const KoraRuntime,
    words: *const u32,
    word_count: usize,
    entry_point: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut KoraKernel,
) -> KoraStatus {
    guard(|| {
        // SAFETY: forwarded from the caller.
        let runtime = &unsafe { borrow(runtime, "runtime") }?.0;
        if words.is_null() || word_count == 0 {
            return Err(Error::new(ErrorCode::InvalidKernel, "SPIR-V module is empty"));
        }
        // SAFETY: the caller guarantees `word_count` readable words at `words`.
        let words = unsafe { std::slice::from_raw_parts(words, word_count) };
        // SAFETY: forwarded from the caller.
        let entry =
            unsafe { string(entry_point, "entry_point") }?.ok_or_else(|| invalid("entry_point is NULL"))?;
        // SAFETY: forwarded from the caller.
        let metadata = unsafe { string(metadata_json, "metadata_json") }?
            .ok_or_else(|| invalid("metadata_json is NULL"))?;
        let kernel = runtime.load_kernel(words, &entry, parse_kernel_metadata(&metadata)?)?;
        // SAFETY: forwarded from the caller.
        unsafe { emit(out, KoraKernel(kernel)) }
    })
}

/// Like `kora_kernel_create`, reading the SPIR-V module from a `.spv` file (at most 16 MiB).
///
/// # Safety
/// As `kora_kernel_create`; `path` is NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_kernel_create_from_file(
    runtime: *const KoraRuntime,
    path: *const c_char,
    entry_point: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut KoraKernel,
) -> KoraStatus {
    guard(|| {
        // SAFETY: forwarded from the caller.
        let runtime = &unsafe { borrow(runtime, "runtime") }?.0;
        // SAFETY: forwarded from the caller.
        let path = unsafe { string(path, "path") }?.ok_or_else(|| invalid("path is NULL"))?;
        // SAFETY: forwarded from the caller.
        let entry =
            unsafe { string(entry_point, "entry_point") }?.ok_or_else(|| invalid("entry_point is NULL"))?;
        // SAFETY: forwarded from the caller.
        let metadata = unsafe { string(metadata_json, "metadata_json") }?
            .ok_or_else(|| invalid("metadata_json is NULL"))?;
        let kernel = runtime.load_kernel_file(Path::new(&path), &entry, parse_kernel_metadata(&metadata)?)?;
        // SAFETY: forwarded from the caller.
        unsafe { emit(out, KoraKernel(kernel)) }
    })
}

/// Releases a kernel.
///
/// # Safety
/// `kernel` is null or a live kernel handle, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_kernel_destroy(kernel: *mut KoraKernel) {
    // SAFETY: forwarded from the caller.
    unsafe { destroy(kernel) }
}

/// Starts describing a dispatch of `kernel` (global size 1×1×1, no bindings).
///
/// # Safety
/// `runtime` and `kernel` are live handles (the kernel loaded by this runtime); `out` is valid
/// for a pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_dispatch_create(
    runtime: *const KoraRuntime,
    kernel: *const KoraKernel,
    out: *mut *mut KoraDispatch,
) -> KoraStatus {
    guard(|| {
        // SAFETY: forwarded from the caller.
        let runtime = &unsafe { borrow(runtime, "runtime") }?.0;
        // SAFETY: forwarded from the caller.
        let kernel = &unsafe { borrow(kernel, "kernel") }?.0;
        if !runtime.owns(kernel) {
            return Err(invalid("kernel was loaded by a different runtime"));
        }
        // SAFETY: forwarded from the caller.
        unsafe { emit(out, KoraDispatch(Some(runtime.dispatch(kernel)))) }
    })
}

/// Applies `change` to a dispatch that has not been submitted.
///
/// # Safety
/// `dispatch` is null or a live dispatch handle.
unsafe fn update(
    dispatch: *mut KoraDispatch,
    change: impl FnOnce(DispatchBuilder) -> kora_rt::Result<DispatchBuilder>,
) -> KoraStatus {
    guard(|| {
        // SAFETY: forwarded from the caller.
        let slot = &mut unsafe { borrow_mut(dispatch, "dispatch") }?.0;
        let builder = slot
            .take()
            .ok_or_else(|| invalid("dispatch was already submitted"))?;
        *slot = Some(change(builder)?);
        Ok(())
    })
}

/// Sets the global size in invocations (each dimension at least 1).
///
/// # Safety
/// `dispatch` is a live dispatch handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_dispatch_set_global(
    dispatch: *mut KoraDispatch,
    x: u64,
    y: u64,
    z: u64,
) -> KoraStatus {
    // SAFETY: forwarded from the caller.
    unsafe { update(dispatch, |b| Ok(b.global_3d(x, y, z))) }
}

/// Binds the next buffer in binding order with one of `KORA_ACCESS_*`.
///
/// # Safety
/// `dispatch` and `buffer` are live handles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_dispatch_bind(
    dispatch: *mut KoraDispatch,
    buffer: *const KoraBuffer,
    access: u32,
) -> KoraStatus {
    // SAFETY: forwarded from the caller.
    let buffer = match unsafe { buffer.as_ref() } {
        Some(buffer) => buffer.0.clone(),
        None => return guard(|| Err(invalid("buffer handle is NULL"))),
    };
    // SAFETY: forwarded from the caller.
    unsafe {
        update(dispatch, |b| match access {
            KORA_ACCESS_READ => Ok(b.read(&buffer)),
            KORA_ACCESS_WRITE => Ok(b.write(&buffer)),
            KORA_ACCESS_READ_WRITE => Ok(b.read_write(&buffer)),
            _ => Err(invalid("access must be one of KORA_ACCESS_*")),
        })
    }
}

/// Sets the push-constant bytes (copied).
///
/// # Safety
/// `dispatch` is a live dispatch handle; `data` is valid for `size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_dispatch_push(
    dispatch: *mut KoraDispatch,
    data: *const u8,
    size: usize,
) -> KoraStatus {
    if data.is_null() && size != 0 {
        return guard(|| Err(invalid("data is NULL")));
    }
    let bytes = if size == 0 {
        Vec::new()
    } else {
        // SAFETY: the caller guarantees `size` readable bytes at `data`.
        unsafe { std::slice::from_raw_parts(data, size) }.to_vec()
    };
    // SAFETY: forwarded from the caller.
    unsafe { update(dispatch, |b| Ok(b.push_bytes(&bytes))) }
}

/// Keeps written outputs on the device until `kora_readback`.
///
/// # Safety
/// `dispatch` is a live dispatch handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_dispatch_defer_readback(dispatch: *mut KoraDispatch) -> KoraStatus {
    // SAFETY: forwarded from the caller.
    unsafe { update(dispatch, |b| Ok(b.defer_readback())) }
}

/// Submits the dispatch. The dispatch handle is spent either way; destroy it afterwards. On
/// success `*event` completes after the kernel (and, by default, output readback) finishes.
///
/// # Safety
/// `dispatch` is a live dispatch handle; `event` is valid for a pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_dispatch_submit(
    dispatch: *mut KoraDispatch,
    event: *mut *mut KoraEvent,
) -> KoraStatus {
    guard(|| {
        if event.is_null() {
            return Err(invalid("output pointer is NULL"));
        }
        // SAFETY: forwarded from the caller.
        let builder = unsafe { borrow_mut(dispatch, "dispatch") }?
            .0
            .take()
            .ok_or_else(|| invalid("dispatch was already submitted"))?;
        let submitted = builder.submit()?;
        // SAFETY: forwarded from the caller.
        unsafe { emit(event, KoraEvent(submitted)) }
    })
}

/// Releases a dispatch handle (submitted or not).
///
/// # Safety
/// `dispatch` is null or a live dispatch handle, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_dispatch_destroy(dispatch: *mut KoraDispatch) {
    // SAFETY: forwarded from the caller.
    unsafe { destroy(dispatch) }
}

/// Makes device-resident results of `buffer` visible in its host copy.
///
/// # Safety
/// `runtime` and `buffer` are live handles; `event` is valid for a pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_readback(
    runtime: *const KoraRuntime,
    buffer: *const KoraBuffer,
    event: *mut *mut KoraEvent,
) -> KoraStatus {
    guard(|| {
        // SAFETY: forwarded from the caller.
        let runtime = &unsafe { borrow(runtime, "runtime") }?.0;
        // SAFETY: forwarded from the caller.
        let buffer = &unsafe { borrow(buffer, "buffer") }?.0;
        let completion = runtime.readback(buffer)?;
        // SAFETY: forwarded from the caller.
        unsafe { emit(event, KoraEvent(completion)) }
    })
}

/// Blocks until the event completes and returns its status.
///
/// # Safety
/// `event` is a live event handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_event_wait(event: *const KoraEvent) -> KoraStatus {
    guard(|| {
        // SAFETY: forwarded from the caller.
        let event = unsafe { borrow(event, "event") }?;
        event.0.wait()
    })
}

/// Waits up to `timeout_ns`; returns `KORA_STATUS_TIMEOUT` if the event is still pending.
///
/// # Safety
/// `event` is a live event handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_event_wait_timeout(event: *const KoraEvent, timeout_ns: u64) -> KoraStatus {
    guard(|| {
        // SAFETY: forwarded from the caller.
        let event = unsafe { borrow(event, "event") }?;
        event.0.wait_timeout(Duration::from_nanos(timeout_ns))
    })
}

/// 1 when the event has completed, 0 when pending or null.
///
/// # Safety
/// `event` is null or a live event handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_event_is_done(event: *const KoraEvent) -> i32 {
    // SAFETY: forwarded from the caller.
    i32::from(unsafe { event.as_ref() }.is_some_and(|e| e.0.is_done()))
}

/// Releases the caller's reference; the operation itself continues.
///
/// # Safety
/// `event` is null or a live event handle, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kora_event_destroy(event: *mut KoraEvent) {
    // SAFETY: forwarded from the caller.
    unsafe { destroy(event) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_mirror_every_error_code() {
        for code in ErrorCode::ALL {
            let status = KoraStatus::from(code);
            assert_eq!(status as i32, code as i32, "{code}");
            // SAFETY: kora_status_name returns a static NUL-terminated string.
            let name = unsafe { CStr::from_ptr(kora_status_name(status)) };
            assert_eq!(name.to_str().unwrap(), code.name());
        }
        // SAFETY: as above.
        assert_eq!(unsafe { CStr::from_ptr(kora_status_name(KoraStatus::Ok)) }, c"Ok");
    }

    #[test]
    fn null_handles_fail_cleanly_with_a_message() {
        let mut runtime = ptr::null_mut();
        // SAFETY: null handles and a valid output pointer are part of the contract.
        unsafe {
            assert_eq!(
                kora_buffer_create(ptr::null(), 4, &mut ptr::null_mut()),
                KoraStatus::InvalidArgument
            );
            let message = CStr::from_ptr(kora_last_error_message()).to_str().unwrap();
            assert!(message.contains("runtime handle is NULL"), "{message}");
            assert_eq!(
                kora_runtime_create(ptr::null(), ptr::null_mut()),
                KoraStatus::InvalidArgument
            );
            assert_eq!(kora_event_wait(ptr::null()), KoraStatus::InvalidArgument);
            assert_eq!(kora_event_is_done(ptr::null()), 0);
            assert_eq!(kora_runtime_device_count(ptr::null()), 0);
            kora_runtime_destroy(runtime);
            runtime = ptr::null_mut();
            kora_buffer_destroy(ptr::null_mut());
        }
        assert!(runtime.is_null());
    }

    #[test]
    fn options_are_validated() {
        let mut options = std::mem::MaybeUninit::<KoraRuntimeOptions>::uninit();
        // SAFETY: the pointer is valid for one write.
        unsafe { kora_runtime_options_default(options.as_mut_ptr()) };
        // SAFETY: just initialized.
        let mut options = unsafe { options.assume_init() };
        // SAFETY: valid options.
        assert!(unsafe { runtime_options(&options) }.is_ok());
        options.enable_validation = 7;
        // SAFETY: valid options.
        assert!(unsafe { runtime_options(&options) }.is_err());
        options.enable_validation = KORA_DEFAULT;
        options.struct_size = 4;
        // SAFETY: valid options.
        assert!(unsafe { runtime_options(&options) }.is_err());
        options.struct_size = size_of::<KoraRuntimeOptions>() as u32;
        options.device_policy = 99;
        // SAFETY: valid options.
        assert!(unsafe { runtime_options(&options) }.is_err());
    }
}
