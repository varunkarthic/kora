//! Structured logging with categories and levels.
//!
//! Thresholds are per category and lock-free to query, so a disabled [`kora_log!`](crate::kora_log)
//! statement costs one relaxed atomic load and never evaluates its format arguments. Messages go to
//! a replaceable sink (default: `"[CAT] message"` on stderr); sink calls are serialized.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

/// Severity threshold. `Off` disables a category entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Level {
    /// Very detailed tracing.
    Trace = 0,
    /// Debugging detail.
    Debug,
    /// Informational events.
    Info,
    /// Recoverable problems (default threshold).
    Warn,
    /// Failures.
    Error,
    /// Disabled.
    Off,
}

impl Level {
    const ALL: [Level; 6] = [
        Level::Trace,
        Level::Debug,
        Level::Info,
        Level::Warn,
        Level::Error,
        Level::Off,
    ];

    /// Upper-case name such as `"WARN"`.
    pub const fn name(self) -> &'static str {
        match self {
            Level::Trace => "TRACE",
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warn => "WARN",
            Level::Error => "ERROR",
            Level::Off => "OFF",
        }
    }

    /// Parses `trace|debug|info|warn|error|off`, case-insensitively.
    pub fn parse(text: &str) -> Option<Level> {
        Self::ALL
            .into_iter()
            .find(|level| level.name().eq_ignore_ascii_case(text))
    }

    fn from_u8(value: u8) -> Level {
        Self::ALL.get(usize::from(value)).copied().unwrap_or(Level::Off)
    }
}

/// Subsystem that emitted a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Category {
    /// Runtime lifecycle.
    Runtime = 0,
    /// Device discovery and initialization.
    Device,
    /// Memory management.
    Memory,
    /// Scheduling and placement.
    Scheduler,
    /// Queue submission and completion.
    Queue,
    /// Synchronization.
    Sync,
    /// Kernels and pipelines.
    Kernel,
    /// Profiling and telemetry.
    Profiling,
    /// Device-loss recovery.
    Recovery,
    /// Python bindings.
    Python,
}

const CATEGORY_COUNT: usize = 10;

impl Category {
    /// Every category, in discriminant order.
    pub const ALL: [Category; CATEGORY_COUNT] = [
        Category::Runtime,
        Category::Device,
        Category::Memory,
        Category::Scheduler,
        Category::Queue,
        Category::Sync,
        Category::Kernel,
        Category::Profiling,
        Category::Recovery,
        Category::Python,
    ];

    /// Short tag such as `"MEM"`.
    pub const fn name(self) -> &'static str {
        match self {
            Category::Runtime => "RUNTIME",
            Category::Device => "DEVICE",
            Category::Memory => "MEM",
            Category::Scheduler => "SCHED",
            Category::Queue => "QUEUE",
            Category::Sync => "SYNC",
            Category::Kernel => "KERNEL",
            Category::Profiling => "PROF",
            Category::Recovery => "RECOVERY",
            Category::Python => "PYTHON",
        }
    }
}

/// Destination for enabled messages.
pub type Sink = Box<dyn Fn(Level, Category, &str) + Send + Sync>;

fn levels() -> &'static [AtomicU8; CATEGORY_COUNT] {
    static LEVELS: OnceLock<[AtomicU8; CATEGORY_COUNT]> = OnceLock::new();
    LEVELS.get_or_init(|| {
        let initial = std::env::var("KORA_LOG_LEVEL")
            .ok()
            .and_then(|value| Level::parse(&value))
            .unwrap_or(Level::Warn);
        std::array::from_fn(|_| AtomicU8::new(initial as u8))
    })
}

fn sink() -> &'static Mutex<Option<Sink>> {
    static SINK: Mutex<Option<Sink>> = Mutex::new(None);
    &SINK
}

/// Applies `KORA_LOG_LEVEL` to every category. Returns `false` when it is absent or invalid.
pub fn configure_from_environment() -> bool {
    configure_from(std::env::var("KORA_LOG_LEVEL").ok().as_deref())
}

fn configure_from(value: Option<&str>) -> bool {
    match value.and_then(Level::parse) {
        Some(level) => {
            set_level(level);
            true
        }
        None => false,
    }
}

/// Replaces the sink. `None` restores the default stderr sink.
pub fn set_sink(new_sink: Option<Sink>) {
    *sink().lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = new_sink;
}

/// Sets the threshold of every category.
pub fn set_level(level: Level) {
    for slot in levels() {
        slot.store(level as u8, Ordering::Relaxed);
    }
}

/// Sets one category's threshold.
pub fn set_category_level(category: Category, level: Level) {
    levels()[category as usize].store(level as u8, Ordering::Relaxed);
}

/// Current threshold of a category.
pub fn level(category: Category) -> Level {
    Level::from_u8(levels()[category as usize].load(Ordering::Relaxed))
}

/// `true` when a message at `level` in `category` would be written.
#[inline]
pub fn enabled(category: Category, level_: Level) -> bool {
    let threshold = level(category);
    level_ != Level::Off && threshold != Level::Off && level_ >= threshold
}

/// Writes a message unconditionally (callers normally check [`enabled`] first).
pub fn write(level_: Level, category: Category, message: &str) {
    let guard = sink().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    match guard.as_ref() {
        Some(sink) => sink(level_, category, message),
        None => eprintln!("[{}] {message}", category.name()),
    }
}

/// Logs a formatted message when the category/level is enabled; arguments are not evaluated
/// otherwise. `kora_log!(Memory, Warn, "budget {} exceeded", n)`.
#[macro_export]
macro_rules! kora_log {
    ($category:ident, $level:ident, $($arg:tt)+) => {
        if $crate::log::enabled($crate::log::Category::$category, $crate::log::Level::$level) {
            $crate::log::write(
                $crate::log::Level::$level,
                $crate::log::Category::$category,
                &::std::format!($($arg)+),
            );
        }
    };
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Arc;

    /// Serializes tests that mutate the process-wide logger.
    pub(crate) static LOGGER_LOCK: Mutex<()> = Mutex::new(());

    struct Capture {
        lines: Arc<Mutex<Vec<String>>>,
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    impl Capture {
        fn new() -> Self {
            let guard = LOGGER_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            set_level(Level::Info);
            let lines = Arc::new(Mutex::new(Vec::new()));
            let sink_lines = Arc::clone(&lines);
            set_sink(Some(Box::new(move |_, category, message| {
                sink_lines
                    .lock()
                    .unwrap()
                    .push(format!("{}:{message}", category.name()));
            })));
            Self { lines, _guard: guard }
        }
        fn lines(&self) -> Vec<String> {
            self.lines.lock().unwrap().clone()
        }
    }

    impl Drop for Capture {
        fn drop(&mut self) {
            set_sink(None);
            set_level(Level::Warn);
        }
    }

    #[test]
    fn environment_value_configures_thresholds() {
        let _guard = LOGGER_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        assert!(configure_from(Some("DeBuG")));
        assert_eq!(level(Category::Runtime), Level::Debug);
        assert!(!configure_from(Some("")));
        assert!(!configure_from(None));
        set_level(Level::Warn);
    }

    #[test]
    fn level_filtering() {
        let cap = Capture::new();
        crate::kora_log!(Runtime, Debug, "hidden");
        crate::kora_log!(Runtime, Info, "shown {}", 1);
        assert_eq!(cap.lines(), vec!["RUNTIME:shown 1".to_string()]);
    }

    #[test]
    fn disabled_statements_do_not_evaluate_arguments() {
        let cap = Capture::new();
        let mut evaluated = 0;
        let mut f = || {
            evaluated += 1;
            1
        };
        crate::kora_log!(Memory, Trace, "x {}", f());
        crate::kora_log!(Memory, Error, "x {}", f());
        assert_eq!(evaluated, 1);
        assert_eq!(cap.lines().len(), 1);
    }

    #[test]
    fn category_level_overrides_global() {
        let cap = Capture::new();
        set_category_level(Category::Sync, Level::Trace);
        crate::kora_log!(Sync, Trace, "t");
        crate::kora_log!(Queue, Trace, "q");
        assert_eq!(cap.lines(), vec!["SYNC:t".to_string()]);
    }

    #[test]
    fn off_disables_every_severity() {
        let cap = Capture::new();
        set_level(Level::Off);
        crate::kora_log!(Runtime, Error, "hidden");
        crate::kora_log!(Runtime, Off, "also hidden");
        assert!(cap.lines().is_empty());
    }

    #[test]
    fn parse_level_is_case_insensitive() {
        assert_eq!(Level::parse("WaRn"), Some(Level::Warn));
        assert_eq!(Level::parse("nope"), None);
        assert_eq!(Level::from_u8(200), Level::Off);
    }

    #[test]
    fn concurrent_writes_are_serialized() {
        let cap = Capture::new();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..200 {
                        crate::kora_log!(Runtime, Info, "m");
                    }
                });
            }
        });
        assert_eq!(cap.lines().len(), 1600);
    }

    #[test]
    fn category_names_are_stable() {
        let names: Vec<_> = Category::ALL.iter().map(|c| c.name()).collect();
        assert_eq!(
            names,
            [
                "RUNTIME", "DEVICE", "MEM", "SCHED", "QUEUE", "SYNC", "KERNEL", "PROF", "RECOVERY", "PYTHON"
            ]
        );
    }
}
