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
| `on_complete(callback)` | Registers one callback with a copy of the final status. Pending callbacks run once on completion; callbacks registered after completion run synchronously before return. Callback exceptions are contained by the event completion boundary. |
| `then(callback)` | Returns a new event completed with the callback's returned status after the source completes. The callback receives the source's success or failure and is responsible for propagating it when appropriate. |
| `Event::when_all(events)` | Returns an event that completes after all inputs; the first failure in input order becomes the aggregate status. Empty input succeeds immediately. |

Waiting on or querying the final status of an empty handle returns `InvalidArgument`. Event handles do
not extend Vulkan resource lifetimes: buffers, devices, queue contexts, and output spans referenced by
the operation must remain valid until completion. The duration overload accepts `std::chrono` values
implicitly convertible to nanoseconds.

Callbacks execute inline on the thread that completes the source event (or synchronously during
registration if it was already complete). Keep callbacks short and avoid blocking waits that depend on
the same completion processor. `then` is a status continuation, not a task scheduler: it does not
automatically submit work or establish GPU dependencies. The event shared state protects registration,
completion, and notification; callback invocation occurs outside its mutex.
