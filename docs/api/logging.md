# Logging API

Header: `include/kora/log.hpp`

KORA logging has severity levels `Trace`, `Debug`, `Info`, `Warn`, `Error`, and `Off`, plus categories
for runtime, device, memory, scheduler, queue, synchronization, kernel, profiling, recovery, and
Python activity.

`set_level(level)` changes the default threshold for every category. `set_category_level(category,
level)` overrides one category. `level(category)` reads the active threshold, and `enabled` supports
cheap call-site filtering.

`set_sink(sink)` installs a process-wide callback. Passing `nullptr` restores the default stderr
sink. Sink callbacks are serialized; they should complete quickly and must not recursively call the
logger.

`KORA_LOG(Category, Level, format, args...)` checks the threshold before formatting, so arguments to
a disabled statement are not evaluated. Format strings follow C++20 `std::format` rules.

## Invariants

- Concurrent writes are serialized.
- Disabled log expressions do not evaluate arguments.
- Logging must not expose credentials, buffer contents, proprietary kernels, or private device data.
- Log filtering changes diagnostics only; it cannot change runtime correctness.
