# Logging API

Header: `include/kora/log.hpp`

KORA logging has severity levels `Trace`, `Debug`, `Info`, `Warn`, `Error`, and `Off`, plus categories
for runtime, device, memory, scheduler, queue, synchronization, kernel, profiling, recovery, and
Python activity.

## Levels and categories

A statement is emitted when its level is at least the category threshold. `Trace` is the most
verbose threshold and `Error` is the least verbose emitting threshold. `Off` disables the category;
an `Off` statement is never emitted.

| Category | Intended events |
| --- | --- |
| `Runtime` | Runtime creation, shutdown, and global policy |
| `Device` | Adapter discovery, selection, and capability decisions |
| `Memory` | Allocation, residency, mapping, and budget decisions |
| `Scheduler` | Graph scheduling and dependency decisions |
| `Queue` | Queue selection, submission, and completion |
| `Sync` | Fences, semaphores, barriers, and host synchronization |
| `Kernel` | SPIR-V validation, pipeline creation, and dispatch |
| `Profiling` | Timing and telemetry collection |
| `Recovery` | Recoverable failures and device-loss handling |
| `Python` | Binding conversion and Python-facing lifecycle events |

## Function reference

`set_level(level)` writes the threshold to every category. It is useful for a process-wide reset and
overwrites earlier category-specific choices.

`set_category_level(category, level)` changes one category. Invalid enum values are ignored.

`level(category)` reads a category threshold with relaxed atomic ordering because thresholds affect
diagnostics, not correctness. It returns `Off` for an invalid category.

`enabled(category, level)` is the cheap check used by the macro. It performs no allocation or
formatting.

`set_sink(sink)` installs a process-wide callback. Passing `nullptr` restores the default stderr
sink. Sink callbacks are serialized; they should complete quickly and must not recursively call the
logger.

`KORA_LOG(Category, Level, format, args...)` checks the threshold before formatting, so arguments to
a disabled statement are not evaluated. Toolchains with a conforming C++20 `<format>` use
`std::format`. Other supported toolchains use KORA's zero-dependency fallback, which recognizes
sequential `{}` replacements and escaped braces `{{` and `}}`; it intentionally does not implement
format specifiers such as `{:08x}`.

`to_string(level)` and `to_string(category)` return stable uppercase diagnostic names. Unknown enum
values return `"?"`.

`parse_level(text, output)` accepts `trace`, `debug`, `info`, `warn`, `error`, or `off` without case
sensitivity. It returns `false` and leaves callers responsible for fallback behavior when parsing
fails.

`configure_from_environment()` re-reads `KORA_LOG_LEVEL`, applies a valid value to every category,
and returns `true`. If the variable is absent or invalid it returns `false` without changing current
thresholds.

## Environment initialization

On the first logging operation, every category threshold is initialized from `KORA_LOG_LEVEL`. An
unset or invalid value uses `Warn`. Environment variables should be set before creating threads or
using any logging function. Later environment changes do not alter initialized thresholds; use
`configure_from_environment`, `set_level`, or `set_category_level` instead.

`resolve(RuntimeOptions)` also parses `KORA_LOG_LEVEL` into `ResolvedOptions::log_level`. This keeps
configuration reporting deterministic. Runtime construction will apply the resolved value as that
layer is implemented; direct users of the logging API already receive the first-use initialization
described above.

## Example

```cpp
#include <kora/log.hpp>

kora::log::set_level(kora::log::Level::Info);
kora::log::set_category_level(kora::log::Category::Memory,
                              kora::log::Level::Debug);

KORA_LOG(Runtime, Info, "runtime {} started", 3);
KORA_LOG(Memory, Debug, "reserved {} bytes", 4096);
```

## Invariants

- Concurrent writes are serialized.
- Disabled log expressions do not evaluate arguments.
- Threshold reads and writes are thread-safe.
- Logging must not expose credentials, buffer contents, proprietary kernels, or private device data.
- Log filtering changes diagnostics only; it cannot change runtime correctness.
