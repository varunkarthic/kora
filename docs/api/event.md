# Event API

`kora::Event` is a copyable handle to shared asynchronous completion state. It is currently used by
the internal transfer engine and will be extended by the scheduler as dependency composition is
implemented.

```cpp
#include <chrono>
#include <kora/event.hpp>

kora::Event event = /* asynchronous operation */;
if (!event.valid()) {
    // No operation was associated with this handle.
}

auto status = event.wait_for(std::chrono::milliseconds{20});
if (!status && status.error().code == kora::ErrorCode::Timeout) {
    // The operation is still running; timeout does not cancel it.
}
status = event.wait();
if (!status) {
    // Inspect the structured error returned by the operation.
}
```

| Member | Behavior |
| --- | --- |
| `valid()` | True when the handle refers to a completion state. |
| `done()` | Non-blocking query; false for an empty or pending handle. |
| `wait()` | Blocks until completion and returns its success or failure status. |
| `wait_for(nanoseconds)` | Waits for the duration; returns `Timeout` without changing operation state. |
| `status()` | Returns `Timeout` while pending; otherwise returns the final status. |

Waiting on or querying the final status of an empty handle returns `InvalidArgument`. Event handles do
not extend Vulkan resource lifetimes: buffers, devices, queue contexts, and output spans referenced by
the operation must remain valid until completion. The duration overload accepts `std::chrono` values
implicitly convertible to nanoseconds.

This API does not yet provide continuations, dependency chaining, or `when_all`; those capabilities are
tracked separately from the basic completion contract.
