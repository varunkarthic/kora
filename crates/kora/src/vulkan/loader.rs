//! Runtime Vulkan loader discovery and instance lifetime.

use std::ffi::{CStr, CString, c_char, c_void};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use ash::vk;

use super::physical::{PhysicalDeviceInfo, query_physical_device};
use super::vk_error;
use crate::error::{Error, ErrorCode, Result};
use crate::log::{self, Category, Level};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";
/// Highest API version KORA negotiates.
pub(crate) const MAX_API_VERSION: u32 = vk::API_VERSION_1_3;

fn library_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if cfg!(windows) {
        candidates.push("vulkan-1.dll".into());
    } else if cfg!(target_os = "macos") {
        candidates.push("libvulkan.1.dylib".into());
        if let Some(sdk) = std::env::var_os("VULKAN_SDK").filter(|s| !s.is_empty()) {
            let sdk = PathBuf::from(sdk);
            candidates.push(sdk.join("lib/libvulkan.1.dylib"));
            candidates.push(sdk.join("macOS/lib/libvulkan.1.dylib"));
        }
        candidates.push("/opt/homebrew/lib/libvulkan.1.dylib".into());
        candidates.push("/usr/local/lib/libvulkan.1.dylib".into());
    } else {
        candidates.push("libvulkan.so.1".into());
        candidates.push("libvulkan.so".into());
    }
    candidates
}

/// A dynamically loaded Vulkan loader.
#[derive(Clone)]
pub struct Loader {
    entry: Arc<ash::Entry>,
    supported_api_version: u32,
}

impl fmt::Debug for Loader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Loader(api={})",
            super::version_string(self.supported_api_version)
        )
    }
}

/// Instance creation options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceOptions {
    /// Requested API version (clamped to the loader and to Vulkan 1.3).
    pub requested_api_version: u32,
    /// Application name reported to drivers.
    pub application_name: String,
    /// Application version reported to drivers.
    pub application_version: u32,
    /// Enable `VK_LAYER_KHRONOS_validation` when available; messages go to the KORA log.
    pub enable_validation: bool,
}

impl Default for InstanceOptions {
    fn default() -> Self {
        Self {
            requested_api_version: MAX_API_VERSION,
            application_name: "KORA".into(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            enable_validation: false,
        }
    }
}

impl Loader {
    /// Loads the system Vulkan loader (standard name first, then SDK/package locations on macOS).
    pub fn load() -> Result<Loader> {
        let mut last_error = String::new();
        for candidate in library_candidates() {
            // SAFETY: loading the Vulkan loader runs its library initializers. Only the platform's
            // standard loader names and documented SDK/package locations are tried, and the
            // resulting entry is kept alive (Arc) for as long as any instance derived from it.
            match unsafe { ash::Entry::load_from(&candidate) } {
                Ok(entry) => {
                    // SAFETY: `entry` holds a valid vkGetInstanceProcAddr; this global query has no
                    // preconditions.
                    let version = unsafe { entry.try_enumerate_instance_version() }
                        .map_err(|r| {
                            vk_error(
                                ErrorCode::NoVulkanImplementation,
                                "Failed to query Vulkan loader version",
                                r,
                            )
                        })?
                        .unwrap_or(vk::API_VERSION_1_0);
                    return Ok(Loader {
                        entry: Arc::new(entry),
                        supported_api_version: version,
                    });
                }
                Err(error) => last_error = format!("{}: {error}", candidate.display()),
            }
        }
        Err(Error::new(
            ErrorCode::NoVulkanImplementation,
            format!("Unable to load Vulkan loader ({last_error})"),
        ))
    }

    /// Highest API version the loader reports (Vulkan 1.0 when it predates the query).
    pub fn supported_api_version(&self) -> u32 {
        self.supported_api_version
    }

    /// Creates an instance, negotiating version, portability enumeration, and optional validation.
    /// Validation is a debugging aid: when the layer is advertised but fails to load, the instance
    /// is created without it and a warning is logged.
    pub fn create_instance(&self, options: &InstanceOptions) -> Result<Instance> {
        match self.create_instance_once(options) {
            Err(error)
                if options.enable_validation
                    && error.vk_result() == vk::Result::ERROR_LAYER_NOT_PRESENT.as_raw() =>
            {
                crate::kora_log!(
                    Device,
                    Warn,
                    "Vulkan validation layer failed to load; continuing without validation"
                );
                self.create_instance_once(&InstanceOptions {
                    enable_validation: false,
                    ..options.clone()
                })
            }
            result => result,
        }
    }

    fn create_instance_once(&self, options: &InstanceOptions) -> Result<Instance> {
        let entry = &self.entry;
        let requested = if options.requested_api_version == 0 {
            vk::API_VERSION_1_0
        } else {
            options.requested_api_version
        };
        let api_version = requested.min(self.supported_api_version).min(MAX_API_VERSION);
        let enumerate_error = |what: &'static str| {
            move |r| {
                vk_error(
                    ErrorCode::NoVulkanImplementation,
                    format!("Failed to enumerate {what}"),
                    r,
                )
            }
        };
        // SAFETY: global loader queries with no preconditions.
        let extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(enumerate_error("instance extensions"))?;
        // SAFETY: as above.
        let layers = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(enumerate_error("instance layers"))?;
        let has_extension = |name: &CStr| extensions.iter().any(|e| e.extension_name_as_c_str() == Ok(name));

        let mut enabled_extensions: Vec<*const c_char> = Vec::new();
        let mut flags = vk::InstanceCreateFlags::empty();
        if has_extension(ash::khr::portability_enumeration::NAME) {
            enabled_extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        let properties2_extension = api_version == vk::API_VERSION_1_0
            && has_extension(ash::khr::get_physical_device_properties2::NAME);
        if properties2_extension {
            enabled_extensions.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }
        let validation_available = layers
            .iter()
            .any(|l| l.layer_name_as_c_str() == Ok(VALIDATION_LAYER));
        let enable_validation = options.enable_validation && validation_available;
        if options.enable_validation && !validation_available {
            crate::kora_log!(
                Device,
                Warn,
                "Vulkan validation requested but VK_LAYER_KHRONOS_validation is unavailable"
            );
        }
        let enabled_layers: Vec<*const c_char> = if enable_validation {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };
        let enable_debug_utils = enable_validation && has_extension(ash::ext::debug_utils::NAME);
        if enable_debug_utils {
            enabled_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let application_name =
            CString::new(options.application_name.replace('\0', "")).unwrap_or_else(|_| c"KORA".into());
        let application = vk::ApplicationInfo::default()
            .application_name(&application_name)
            .application_version(options.application_version)
            .engine_name(c"KORA")
            .engine_version(vk::make_api_version(
                0,
                crate::version::MAJOR,
                crate::version::MINOR,
                crate::version::PATCH,
            ))
            .api_version(api_version);
        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_message));
        let mut create_info = vk::InstanceCreateInfo::default()
            .flags(flags)
            .application_info(&application)
            .enabled_layer_names(&enabled_layers)
            .enabled_extension_names(&enabled_extensions);
        if enable_debug_utils {
            create_info = create_info.push_next(&mut debug_info);
        }
        // SAFETY: every pointer in `create_info` (names, application info, debug chain) refers to
        // locals that outlive this call; the extensions and layers were checked as available.
        let raw = unsafe { entry.create_instance(&create_info, None) }.map_err(|r| {
            vk_error(
                ErrorCode::NoVulkanImplementation,
                "Failed to create Vulkan instance",
                r,
            )
        })?;
        let mut inner = InstanceInner {
            entry: Arc::clone(entry),
            raw,
            debug: None,
            api_version,
            validation_enabled: enable_validation,
            properties2_extension,
        };
        // SAFETY: `inner.raw` is the live instance created above.
        let devices =
            unsafe { inner.raw.enumerate_physical_devices() }.map_err(enumerate_error("physical devices"))?;
        if devices.is_empty() {
            return Err(Error::new(
                ErrorCode::NoVulkanImplementation,
                "Vulkan loader found no physical devices",
            ));
        }
        if enable_debug_utils {
            let debug_utils = ash::ext::debug_utils::Instance::new(entry, &inner.raw);
            let standalone = vk::DebugUtilsMessengerCreateInfoEXT {
                p_next: std::ptr::null(),
                ..debug_info
            };
            // SAFETY: the instance enabled VK_EXT_debug_utils; the callback is a 'static function.
            let messenger =
                unsafe { debug_utils.create_debug_utils_messenger(&standalone, None) }.map_err(|r| {
                    vk_error(
                        ErrorCode::NoVulkanImplementation,
                        "Failed to create Vulkan debug messenger",
                        r,
                    )
                })?;
            inner.debug = Some((debug_utils, messenger));
        }
        Ok(Instance {
            inner: Arc::new(inner),
        })
    }
}

/// Routes validation messages to the KORA log (category `Device`). Only validation and
/// performance findings keep error severity; loader/general diagnostics are capped at warnings.
unsafe extern "system" fn debug_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        Level::Error
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        Level::Warn
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        Level::Info
    } else {
        Level::Debug
    };
    let validation = types.intersects(
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
    );
    let level = if validation { level } else { level.min(Level::Warn) };
    // SAFETY: Vulkan passes a valid callback-data pointer (or null) whose message, when non-null, is
    // a NUL-terminated string valid for the duration of this callback.
    let message = unsafe {
        data.as_ref()
            .filter(|d| !d.p_message.is_null())
            .map(|d| CStr::from_ptr(d.p_message).to_string_lossy().into_owned())
    };
    if log::enabled(Category::Device, level) {
        log::write(
            level,
            Category::Device,
            message.as_deref().unwrap_or("Vulkan validation message"),
        );
    }
    vk::FALSE
}

pub(crate) struct InstanceInner {
    pub(crate) entry: Arc<ash::Entry>,
    pub(crate) raw: ash::Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub(crate) api_version: u32,
    validation_enabled: bool,
    pub(crate) properties2_extension: bool,
}

impl Drop for InstanceInner {
    fn drop(&mut self) {
        // SAFETY: every Device holds an Arc to this instance, so no child object outlives it; the
        // messenger was created from this instance and is destroyed first.
        unsafe {
            if let Some((debug_utils, messenger)) = self.debug.take() {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.raw.destroy_instance(None);
        }
    }
}

/// A Vulkan instance. Cloning shares it; it is destroyed after the last clone and every device
/// created from it are dropped.
#[derive(Clone)]
pub struct Instance {
    pub(crate) inner: Arc<InstanceInner>,
}

impl fmt::Debug for Instance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Instance")
            .field("api_version", &super::version_string(self.inner.api_version))
            .field("validation", &self.inner.validation_enabled)
            .finish()
    }
}

impl Instance {
    /// Raw handle.
    pub fn handle(&self) -> vk::Instance {
        self.inner.raw.handle()
    }

    /// Negotiated API version.
    pub fn api_version(&self) -> u32 {
        self.inner.api_version
    }

    /// Whether the validation layer is active.
    pub fn validation_enabled(&self) -> bool {
        self.inner.validation_enabled
    }

    /// The ash instance (for interop).
    pub fn raw(&self) -> &ash::Instance {
        &self.inner.raw
    }

    /// Enumerates and queries every physical device.
    pub fn physical_devices(&self) -> Result<Vec<PhysicalDeviceInfo>> {
        // SAFETY: the instance is alive for the duration of the borrow.
        let handles = unsafe { self.inner.raw.enumerate_physical_devices() }.map_err(|r| {
            vk_error(
                ErrorCode::NoVulkanImplementation,
                "Failed to enumerate physical devices",
                r,
            )
        })?;
        handles
            .into_iter()
            .map(|handle| query_physical_device(&self.inner, handle))
            .collect()
    }
}
