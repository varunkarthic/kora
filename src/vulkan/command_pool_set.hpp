#pragma once

#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "vulkan/api.hpp"

namespace kora::vk {

/// Thread-owned command-buffer ring. Completion values are from one monotonic queue timeline.
class CommandPoolSet {
public:
    struct Options {
        uint32_t queue_family = 0;
        size_t ring_size = 8;
        bool debug_thread_checks = true;
    };

    CommandPoolSet(Api api, VkDevice device, const Options& options);
    ~CommandPoolSet();
    CommandPoolSet(const CommandPoolSet&) = delete;
    CommandPoolSet& operator=(const CommandPoolSet&) = delete;
    CommandPoolSet(CommandPoolSet&&) noexcept;
    CommandPoolSet& operator=(CommandPoolSet&&) noexcept;

    /// Acquire a reusable buffer. completed_value must be sampled from the owning queue timeline.
    Result<VkCommandBuffer> acquire(uint64_t completed_value);
    /// End recording before passing the buffer to queue submission.
    Status end_recording(VkCommandBuffer buffer);
    /// Discard a recorded command buffer when preparation or queue submission fails.
    Status discard(VkCommandBuffer buffer);
    /// Associate an ended buffer with its signal value after successful queue submission.
    Status submitted(VkCommandBuffer buffer, uint64_t signal_value);
    /// Number of ring slots, useful for capacity diagnostics and deterministic tests.
    size_t capacity() const noexcept;
    /// Number of command buffers allocated since construction (never changes during steady state).
    size_t allocation_count() const noexcept;
    /// Lowest nonzero completion value currently occupying the ring, or zero if none are pending.
    uint64_t oldest_pending_value() const noexcept;

private:
    struct Slot {
        VkCommandBuffer buffer = VK_NULL_HANDLE;
        uint64_t completion_value = 0;
        bool acquired = false;
        bool ended = false;
    };
    bool on_owner_thread() const noexcept;
    bool valid() const noexcept;
    Status check_thread() const;

    Api api_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    std::thread::id owner_;
    bool debug_thread_checks_ = true;
    size_t cursor_ = 0;
    uint64_t last_signal_value_ = 0;
    uint64_t completed_batch_value_ = 0;
    std::vector<Slot> slots_;
    bool valid_ = false;
};

}  // namespace kora::vk
