#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "kora/event.hpp"
#include "kora/runtime.hpp"

namespace kora::detail {

/// Runtime-internal access used to merge split dispatches.
struct RuntimeReplicaAccess {
    /// Starts downloading `destination.size()` bytes at `offset` from `device`'s copy of `buffer`
    /// into `destination`, without changing the host copy or residency bookkeeping. `destination`
    /// must stay valid until the returned Event completes. The caller guarantees the work that
    /// produced the bytes was submitted and no other work writes that range concurrently. Safe to
    /// call from a completion callback: it never waits.
    static Result<Event> read_replica(const Runtime& runtime, const BufferHandle& buffer,
                                      const DeviceId& device, std::uint64_t offset,
                                      std::span<std::byte> destination);
};

}  // namespace kora::detail
