#pragma once

#include <functional>
#include <memory>

#include "kora/buffer.hpp"
#include "kora/runtime.hpp"

namespace kora::detail {

/// Runtime-internal access to BufferManager without widening its public API.
struct BufferManagerAccess {
    /// Calls `callback` after a buffer's last handle is released, for as long as `owner` is alive.
    /// The callback runs outside the registry lock and must not call back into the manager.
    static void add_release_listener(BufferManager& manager, std::weak_ptr<void> owner,
                                     std::function<void(BufferId)> callback);
    static BufferManager& buffers(Runtime& runtime) noexcept { return runtime.buffers_; }
};

}  // namespace kora::detail
