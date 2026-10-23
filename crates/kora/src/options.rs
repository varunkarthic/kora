//! Runtime options.
//!
//! Every [`RuntimeOptions`] field is optional. Unset fields fall back to `KORA_*` environment
//! variables and then to built-in defaults; API values always win over the environment. Bad
//! environment values add a warning to [`ResolvedOptions::warnings`] and never fail.

use crate::log::Level;

/// How the runtime chooses physical devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DevicePolicy {
    /// Score by capability tier, local memory, and compute queues.
    #[default]
    Auto,
    /// Prefer discrete GPUs, then fall back to [`DevicePolicy::Auto`] ordering.
    PreferDiscrete,
    /// Prefer integrated GPUs, then fall back to [`DevicePolicy::Auto`] ordering.
    PreferIntegrated,
    /// Exactly the device at [`RuntimeOptions::device_index`].
    ByIndex,
    /// Exactly the device with [`RuntimeOptions::device_uuid`].
    ByUuid,
    /// Every eligible device; failing devices are quarantined instead of failing creation.
    AllEligible,
}

/// Telemetry collection level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum TelemetryLevel {
    /// No GPU timestamps.
    Off,
    /// Sampled GPU timestamps (1 in 16 dispatches).
    #[default]
    Basic,
    /// Timestamps and host phase timings on every dispatch.
    Profile,
    /// Profile plus a bounded trace history for Chrome trace export.
    Trace,
}

/// What happens to queued work at shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ShutdownMode {
    /// Drain queued work.
    #[default]
    Graceful,
    /// Cancel queued, not-yet-submitted work with [`ErrorCode::Cancelled`](crate::ErrorCode::Cancelled).
    CancelPending,
}

/// Caller-supplied runtime options; `None` means "use the environment, then the default".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeOptions {
    /// Enable the Khronos validation layer (`KORA_VALIDATION`).
    pub enable_validation: Option<bool>,
    /// Extra diagnostics; defaults on in debug builds (`KORA_DEBUG_SAFETY`).
    pub debug_safety: Option<bool>,
    /// Device selection policy (`KORA_DEVICE`).
    pub device_policy: Option<DevicePolicy>,
    /// Index for [`DevicePolicy::ByIndex`].
    pub device_index: Option<u32>,
    /// 32 hex characters for [`DevicePolicy::ByUuid`].
    pub device_uuid: Option<String>,
    /// Fraction of each heap KORA may use, in `(0, 1]` (`KORA_MEMORY_BUDGET_FRACTION`, default 0.80).
    pub memory_budget_fraction: Option<f64>,
    /// Telemetry level (`KORA_PROFILE`).
    pub telemetry: Option<TelemetryLevel>,
    /// Log threshold (`KORA_LOG_LEVEL`).
    pub log_level: Option<Level>,
    /// Persistent cache directory (`KORA_CACHE_DIR`).
    pub cache_dir: Option<String>,
    /// Disable exploration and other nondeterministic choices (`KORA_DETERMINISTIC`).
    pub deterministic: Option<bool>,
    /// Shutdown behavior (`KORA_SHUTDOWN_MODE`).
    pub shutdown_mode: Option<ShutdownMode>,
    /// Disable workgroup autotuning (`KORA_DISABLE_AUTOTUNE`).
    pub disable_autotune: Option<bool>,
    /// Use one queue per device (`KORA_DISABLE_MULTI_QUEUE`).
    pub disable_multi_queue: Option<bool>,
    /// Disable the CPU executor (`KORA_DISABLE_CPU_BACKEND`).
    pub disable_cpu_backend: Option<bool>,
    /// Test hook: skip timeline semaphores and synchronization2 (`KORA_FORCE_LEGACY_SYNC`).
    pub force_legacy_sync: Option<bool>,
    /// Shorthand for [`TelemetryLevel::Trace`] (`KORA_TRACE`).
    pub trace: Option<bool>,
    /// Run bounded startup measurements for selected devices (`KORA_STARTUP_CALIBRATION`).
    pub enable_startup_calibration: Option<bool>,
    /// Per-device calibration upper bound in milliseconds (`KORA_CALIBRATION_BUDGET_MS`, default 250).
    pub calibration_budget_ms: Option<u32>,
}

/// Fully concrete options after merging API values, the environment, and defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedOptions {
    /// See [`RuntimeOptions::enable_validation`].
    pub enable_validation: bool,
    /// See [`RuntimeOptions::debug_safety`].
    pub debug_safety: bool,
    /// See [`RuntimeOptions::device_policy`].
    pub device_policy: DevicePolicy,
    /// See [`RuntimeOptions::device_index`].
    pub device_index: u32,
    /// See [`RuntimeOptions::device_uuid`] (lower-case hex, or empty).
    pub device_uuid: String,
    /// See [`RuntimeOptions::memory_budget_fraction`].
    pub memory_budget_fraction: f64,
    /// See [`RuntimeOptions::telemetry`].
    pub telemetry: TelemetryLevel,
    /// See [`RuntimeOptions::log_level`].
    pub log_level: Level,
    /// See [`RuntimeOptions::cache_dir`].
    pub cache_dir: String,
    /// See [`RuntimeOptions::deterministic`].
    pub deterministic: bool,
    /// See [`RuntimeOptions::shutdown_mode`].
    pub shutdown_mode: ShutdownMode,
    /// See [`RuntimeOptions::disable_autotune`].
    pub disable_autotune: bool,
    /// See [`RuntimeOptions::disable_multi_queue`].
    pub disable_multi_queue: bool,
    /// See [`RuntimeOptions::disable_cpu_backend`].
    pub disable_cpu_backend: bool,
    /// See [`RuntimeOptions::force_legacy_sync`].
    pub force_legacy_sync: bool,
    /// See [`RuntimeOptions::enable_startup_calibration`].
    pub enable_startup_calibration: bool,
    /// See [`RuntimeOptions::calibration_budget_ms`].
    pub calibration_budget_ms: u32,
    /// Problems found while parsing the environment.
    pub warnings: Vec<String>,
}

impl Default for ResolvedOptions {
    fn default() -> Self {
        Self {
            enable_validation: false,
            debug_safety: cfg!(debug_assertions),
            device_policy: DevicePolicy::Auto,
            device_index: 0,
            device_uuid: String::new(),
            memory_budget_fraction: DEFAULT_BUDGET_FRACTION,
            telemetry: TelemetryLevel::Basic,
            log_level: Level::Warn,
            cache_dir: String::new(),
            deterministic: false,
            shutdown_mode: ShutdownMode::Graceful,
            disable_autotune: false,
            disable_multi_queue: false,
            disable_cpu_backend: false,
            force_legacy_sync: false,
            enable_startup_calibration: true,
            calibration_budget_ms: DEFAULT_CALIBRATION_BUDGET_MS,
            warnings: Vec::new(),
        }
    }
}

const DEFAULT_BUDGET_FRACTION: f64 = 0.80;
const DEFAULT_CALIBRATION_BUDGET_MS: u32 = 250;

/// Reads the process environment.
pub fn system_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" | "" => Some(false),
        _ => None,
    }
}

fn is_uuid_hex(value: &str) -> bool {
    value.len() == 32 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn default_cache_dir(env: &dyn Fn(&str) -> Option<String>) -> String {
    if cfg!(windows) {
        env("LOCALAPPDATA")
            .map(|v| format!("{v}\\kora"))
            .unwrap_or_else(|| ".kora-cache".into())
    } else if cfg!(target_os = "macos") {
        env("HOME")
            .map(|h| format!("{h}/Library/Caches/kora"))
            .unwrap_or_else(|| ".kora-cache".into())
    } else if let Some(xdg) = env("XDG_CACHE_HOME").filter(|x| !x.is_empty()) {
        format!("{xdg}/kora")
    } else {
        env("HOME")
            .map(|h| format!("{h}/.cache/kora"))
            .unwrap_or_else(|| ".kora-cache".into())
    }
}

impl RuntimeOptions {
    /// Merges these options over the process environment and defaults.
    pub fn resolve(&self) -> ResolvedOptions {
        resolve(self, &system_env)
    }
}

/// Merges `options` over `env` over defaults. `env` is injectable for tests.
pub fn resolve(options: &RuntimeOptions, env: &dyn Fn(&str) -> Option<String>) -> ResolvedOptions {
    let mut r = ResolvedOptions::default();
    let mut warnings = Vec::new();
    let mut env_bool = |name: &str, api: Option<bool>, default: bool| -> bool {
        if let Some(value) = api {
            return value;
        }
        if let Some(value) = env(name) {
            match parse_bool(&value) {
                Some(parsed) => return parsed,
                None => warnings.push(format!("{name}: invalid boolean '{value}', ignored")),
            }
        }
        default
    };

    r.debug_safety = env_bool("KORA_DEBUG_SAFETY", options.debug_safety, cfg!(debug_assertions));
    r.enable_validation = env_bool("KORA_VALIDATION", options.enable_validation, false);
    r.disable_autotune = env_bool("KORA_DISABLE_AUTOTUNE", options.disable_autotune, false);
    r.disable_multi_queue = env_bool("KORA_DISABLE_MULTI_QUEUE", options.disable_multi_queue, false);
    r.disable_cpu_backend = env_bool("KORA_DISABLE_CPU_BACKEND", options.disable_cpu_backend, false);
    r.deterministic = env_bool("KORA_DETERMINISTIC", options.deterministic, false);
    r.force_legacy_sync = env_bool("KORA_FORCE_LEGACY_SYNC", options.force_legacy_sync, false);
    r.enable_startup_calibration = env_bool(
        "KORA_STARTUP_CALIBRATION",
        options.enable_startup_calibration,
        true,
    );
    let trace = env_bool("KORA_TRACE", options.trace, false);

    r.calibration_budget_ms = options
        .calibration_budget_ms
        .unwrap_or(DEFAULT_CALIBRATION_BUDGET_MS);
    if options.calibration_budget_ms.is_none()
        && let Some(value) = env("KORA_CALIBRATION_BUDGET_MS")
    {
        match value.parse::<u32>() {
            Ok(parsed) if (1..=60_000).contains(&parsed) => r.calibration_budget_ms = parsed,
            _ => warnings.push("KORA_CALIBRATION_BUDGET_MS: expected 1..60000, using 250".into()),
        }
    }

    if let Some(mode) = options.shutdown_mode {
        r.shutdown_mode = mode;
    } else if let Some(value) = env("KORA_SHUTDOWN_MODE") {
        match value.to_ascii_lowercase().as_str() {
            "graceful" => r.shutdown_mode = ShutdownMode::Graceful,
            "cancel_pending" | "cancel-pending" => r.shutdown_mode = ShutdownMode::CancelPending,
            _ => warnings.push(format!(
                "KORA_SHUTDOWN_MODE: expected graceful or cancel_pending, got '{value}', ignored"
            )),
        }
    }

    // Device policy: API > KORA_DEVICE ("auto|discrete|integrated|all|<index>|<uuid>").
    if let Some(policy) = options.device_policy {
        r.device_policy = policy;
        r.device_index = options.device_index.unwrap_or(0);
        r.device_uuid = options.device_uuid.clone().unwrap_or_default();
    } else if let Some(value) = env("KORA_DEVICE") {
        let lower = value.to_ascii_lowercase();
        match lower.as_str() {
            "auto" | "" => r.device_policy = DevicePolicy::Auto,
            "discrete" => r.device_policy = DevicePolicy::PreferDiscrete,
            "integrated" => r.device_policy = DevicePolicy::PreferIntegrated,
            "all" => r.device_policy = DevicePolicy::AllEligible,
            s if is_uuid_hex(s) => {
                r.device_policy = DevicePolicy::ByUuid;
                r.device_uuid = lower.clone();
            }
            s if s.len() < 9 && s.bytes().all(|b| b.is_ascii_digit()) => {
                r.device_policy = DevicePolicy::ByIndex;
                r.device_index = s.parse().unwrap_or(0);
            }
            _ => warnings.push(format!("KORA_DEVICE: unrecognized value '{value}', using auto")),
        }
    }

    if let Some(fraction) = options.memory_budget_fraction {
        r.memory_budget_fraction = fraction;
    } else if let Some(value) = env("KORA_MEMORY_BUDGET_FRACTION") {
        match value.parse::<f64>() {
            Ok(parsed) if parsed > 0.0 && parsed <= 1.0 => r.memory_budget_fraction = parsed,
            _ => warnings.push(format!(
                "KORA_MEMORY_BUDGET_FRACTION: expected (0,1], got '{value}', ignored"
            )),
        }
    }
    if !(r.memory_budget_fraction > 0.0 && r.memory_budget_fraction <= 1.0) {
        warnings.push("memory_budget_fraction out of range, reset to 0.80".into());
        r.memory_budget_fraction = DEFAULT_BUDGET_FRACTION;
    }

    if let Some(level) = options.telemetry {
        r.telemetry = level;
    } else if let Some(value) = env("KORA_PROFILE") {
        match value.to_ascii_lowercase().as_str() {
            "off" => r.telemetry = TelemetryLevel::Off,
            "basic" => r.telemetry = TelemetryLevel::Basic,
            "profile" | "1" => r.telemetry = TelemetryLevel::Profile,
            "trace" => r.telemetry = TelemetryLevel::Trace,
            _ => warnings.push(format!("KORA_PROFILE: unrecognized value '{value}', ignored")),
        }
    }
    if trace {
        r.telemetry = TelemetryLevel::Trace;
    }

    if let Some(level) = options.log_level {
        r.log_level = level;
    } else if let Some(value) = env("KORA_LOG_LEVEL") {
        match Level::parse(&value) {
            Some(level) => r.log_level = level,
            None => warnings.push(format!("KORA_LOG_LEVEL: unrecognized value '{value}', ignored")),
        }
    }

    r.cache_dir = match &options.cache_dir {
        Some(dir) => dir.clone(),
        None => env("KORA_CACHE_DIR").unwrap_or_else(|| default_cache_dir(env)),
    };
    r.warnings = warnings;
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn fake(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn defaults() {
        let r = resolve(&RuntimeOptions::default(), &fake(&[("HOME", "/h")]));
        assert!(!r.enable_validation);
        assert_eq!(r.device_policy, DevicePolicy::Auto);
        assert_eq!(r.memory_budget_fraction, 0.80);
        assert_eq!(r.telemetry, TelemetryLevel::Basic);
        assert!(r.warnings.is_empty());
        assert!(!r.cache_dir.is_empty());
        assert!(r.enable_startup_calibration);
        assert_eq!(r.calibration_budget_ms, 250);
    }

    #[test]
    fn cache_dir_has_a_relative_fallback_without_home() {
        let r = resolve(&RuntimeOptions::default(), &fake(&[]));
        assert!(!r.cache_dir.is_empty());
    }

    #[test]
    fn debug_safety_defaults_to_build_mode_and_can_be_overridden() {
        let defaults = resolve(&RuntimeOptions::default(), &fake(&[]));
        assert_eq!(defaults.debug_safety, cfg!(debug_assertions));
        assert!(
            resolve(
                &RuntimeOptions::default(),
                &fake(&[("KORA_DEBUG_SAFETY", "true")])
            )
            .debug_safety
        );
        assert!(
            !resolve(
                &RuntimeOptions::default(),
                &fake(&[("KORA_DEBUG_SAFETY", "false")])
            )
            .debug_safety
        );
        let explicit = RuntimeOptions {
            debug_safety: Some(false),
            ..Default::default()
        };
        assert!(!resolve(&explicit, &fake(&[("KORA_DEBUG_SAFETY", "true")])).debug_safety);
    }

    #[test]
    fn startup_calibration_can_be_disabled_and_budget_configured() {
        let env = fake(&[
            ("KORA_STARTUP_CALIBRATION", "false"),
            ("KORA_CALIBRATION_BUDGET_MS", "500"),
        ]);
        let r = resolve(&RuntimeOptions::default(), &env);
        assert!(!r.enable_startup_calibration);
        assert_eq!(r.calibration_budget_ms, 500);
        let explicit = RuntimeOptions {
            enable_startup_calibration: Some(true),
            calibration_budget_ms: Some(75),
            ..Default::default()
        };
        let env = fake(&[
            ("KORA_STARTUP_CALIBRATION", "false"),
            ("KORA_CALIBRATION_BUDGET_MS", "900"),
        ]);
        let r = resolve(&explicit, &env);
        assert!(r.enable_startup_calibration);
        assert_eq!(r.calibration_budget_ms, 75);
    }

    #[test]
    fn calibration_budget_rejects_zero_and_huge_values() {
        for bad in ["0", "60001", "-1", "abc", "1.5"] {
            let r = resolve(
                &RuntimeOptions::default(),
                &fake(&[("KORA_CALIBRATION_BUDGET_MS", bad)]),
            );
            assert_eq!(r.calibration_budget_ms, 250, "{bad}");
            assert_eq!(r.warnings.len(), 1, "{bad}");
        }
    }

    #[test]
    fn environment_applies() {
        let env = fake(&[
            ("KORA_VALIDATION", "1"),
            ("KORA_DEVICE", "2"),
            ("KORA_LOG_LEVEL", "DEBUG"),
            ("KORA_MEMORY_BUDGET_FRACTION", "0.5"),
            ("KORA_PROFILE", "trace"),
            ("KORA_CACHE_DIR", "/c"),
            ("KORA_DISABLE_AUTOTUNE", "true"),
        ]);
        let r = resolve(&RuntimeOptions::default(), &env);
        assert!(r.enable_validation);
        assert_eq!(r.device_policy, DevicePolicy::ByIndex);
        assert_eq!(r.device_index, 2);
        assert_eq!(r.log_level, Level::Debug);
        assert_eq!(r.memory_budget_fraction, 0.5);
        assert_eq!(r.telemetry, TelemetryLevel::Trace);
        assert_eq!(r.cache_dir, "/c");
        assert!(r.disable_autotune);
    }

    #[test]
    fn api_beats_environment() {
        let o = RuntimeOptions {
            enable_validation: Some(false),
            memory_budget_fraction: Some(0.9),
            device_policy: Some(DevicePolicy::PreferIntegrated),
            ..Default::default()
        };
        let env = fake(&[
            ("KORA_VALIDATION", "1"),
            ("KORA_MEMORY_BUDGET_FRACTION", "0.3"),
            ("KORA_DEVICE", "discrete"),
        ]);
        let r = resolve(&o, &env);
        assert!(!r.enable_validation);
        assert_eq!(r.memory_budget_fraction, 0.9);
        assert_eq!(r.device_policy, DevicePolicy::PreferIntegrated);
    }

    #[test]
    fn bad_environment_warns_and_never_fails() {
        let env = fake(&[
            ("KORA_VALIDATION", "maybe"),
            ("KORA_MEMORY_BUDGET_FRACTION", "7"),
            ("KORA_DEVICE", "???"),
            ("KORA_LOG_LEVEL", "loud"),
            ("KORA_PROFILE", "x"),
        ]);
        let r = resolve(&RuntimeOptions::default(), &env);
        assert_eq!(r.warnings.len(), 5);
        assert!(!r.enable_validation);
        assert_eq!(r.memory_budget_fraction, 0.80);
    }

    #[test]
    fn out_of_range_api_fraction_is_reset() {
        for bad in [2.0, 0.0, -0.5, f64::NAN] {
            let o = RuntimeOptions {
                memory_budget_fraction: Some(bad),
                ..Default::default()
            };
            let r = resolve(&o, &fake(&[]));
            assert_eq!(r.memory_budget_fraction, 0.80);
            assert_eq!(r.warnings.len(), 1);
        }
    }

    #[test]
    fn uuid_device_is_normalized_to_lower_case() {
        let r = resolve(
            &RuntimeOptions::default(),
            &fake(&[("KORA_DEVICE", "00112233445566778899AABBCCDDEEFF")]),
        );
        assert_eq!(r.device_policy, DevicePolicy::ByUuid);
        assert_eq!(r.device_uuid, "00112233445566778899aabbccddeeff");
    }

    #[test]
    fn device_keywords() {
        for (value, policy) in [
            ("auto", DevicePolicy::Auto),
            ("", DevicePolicy::Auto),
            ("Discrete", DevicePolicy::PreferDiscrete),
            ("integrated", DevicePolicy::PreferIntegrated),
            ("ALL", DevicePolicy::AllEligible),
        ] {
            let r = resolve(&RuntimeOptions::default(), &fake(&[("KORA_DEVICE", value)]));
            assert_eq!(r.device_policy, policy, "{value}");
            assert!(r.warnings.is_empty());
        }
        let too_long = resolve(&RuntimeOptions::default(), &fake(&[("KORA_DEVICE", "123456789")]));
        assert_eq!(too_long.device_policy, DevicePolicy::Auto);
        assert_eq!(too_long.warnings.len(), 1);
    }

    #[test]
    fn operational_environment_knobs() {
        let env = fake(&[
            ("KORA_DETERMINISTIC", "true"),
            ("KORA_DISABLE_MULTI_QUEUE", "1"),
            ("KORA_DISABLE_CPU_BACKEND", "yes"),
            ("KORA_FORCE_LEGACY_SYNC", "on"),
            ("KORA_SHUTDOWN_MODE", "cancel_pending"),
            ("KORA_TRACE", "true"),
        ]);
        let r = resolve(&RuntimeOptions::default(), &env);
        assert!(r.deterministic);
        assert!(r.disable_multi_queue);
        assert!(r.disable_cpu_backend);
        assert!(r.force_legacy_sync);
        assert_eq!(r.shutdown_mode, ShutdownMode::CancelPending);
        assert_eq!(r.telemetry, TelemetryLevel::Trace);
    }

    #[test]
    fn explicit_operational_knobs_beat_environment() {
        let o = RuntimeOptions {
            deterministic: Some(false),
            force_legacy_sync: Some(false),
            shutdown_mode: Some(ShutdownMode::Graceful),
            ..Default::default()
        };
        let env = fake(&[
            ("KORA_DETERMINISTIC", "true"),
            ("KORA_FORCE_LEGACY_SYNC", "true"),
            ("KORA_SHUTDOWN_MODE", "cancel_pending"),
        ]);
        let r = resolve(&o, &env);
        assert!(!r.deterministic);
        assert!(!r.force_legacy_sync);
        assert_eq!(r.shutdown_mode, ShutdownMode::Graceful);
    }

    #[test]
    fn bad_shutdown_mode_warns() {
        let r = resolve(
            &RuntimeOptions::default(),
            &fake(&[("KORA_SHUTDOWN_MODE", "immediate")]),
        );
        assert_eq!(r.shutdown_mode, ShutdownMode::Graceful);
        assert_eq!(r.warnings.len(), 1);
        assert!(r.warnings[0].contains("KORA_SHUTDOWN_MODE"));
    }

    #[test]
    fn profile_aliases() {
        for (value, level) in [
            ("off", TelemetryLevel::Off),
            ("basic", TelemetryLevel::Basic),
            ("1", TelemetryLevel::Profile),
            ("PROFILE", TelemetryLevel::Profile),
        ] {
            let r = resolve(&RuntimeOptions::default(), &fake(&[("KORA_PROFILE", value)]));
            assert_eq!(r.telemetry, level, "{value}");
        }
    }
}
