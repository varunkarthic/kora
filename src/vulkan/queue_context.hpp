#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "kora/options.hpp"
#include "vulkan/api.hpp"
#include "vulkan/device_fault_state.hpp"

namespace kora::vk {

/// Serializes submissions for one VkQueue and reports GPU completion on a separate thread.
class QueueContext {
public:
    struct CoalescingOptions {
        bool enabled = false;
        uint32_t maximum_batch_size = 8;
        uint64_t maximum_wait_ns = 100'000;
    };

    struct ResourceAccess {
        uint32_t slot = 0;
        uint64_t generation = 0;
        bool write = false;
    };

    struct Submission {
        std::vector<VkSemaphore> wait_semaphores;
        std::vector<VkPipelineStageFlags> wait_stages;
        std::vector<uint64_t> wait_values;
        std::vector<VkCommandBuffer> command_buffers;
        std::vector<VkSemaphore> signal_semaphores;
        std::vector<uint64_t> signal_values;
        VkFence fence = VK_NULL_HANDLE;
        bool coalescible = false;
        bool dependency_free = false;
        uint64_t coalescing_key = 0;
        uint8_t priority = 1;
        std::vector<ResourceAccess> resources;
        std::function<Status(uint64_t, std::vector<VkCommandBuffer>&)> record_commands;
        std::function<void(uint64_t, Status)> on_submitted;
        std::function<void(uint64_t, Status)> on_complete;
    };

    static Result<std::unique_ptr<QueueContext>> create(Api api, VkDevice device, VkQueue queue,
                                                        bool force_fence_fallback = false,
                                                        std::shared_ptr<DeviceFaultState> fault_state = {});
    static Result<std::unique_ptr<QueueContext>> create(Api api, VkDevice device, VkQueue queue,
                                                        bool force_fence_fallback,
                                                        CoalescingOptions coalescing,
                                                        std::shared_ptr<DeviceFaultState> fault_state = {});

    QueueContext() = default;
    ~QueueContext();
    QueueContext(const QueueContext&) = delete;
    QueueContext& operator=(const QueueContext&) = delete;
    QueueContext(QueueContext&&) = delete;
    QueueContext& operator=(QueueContext&&) = delete;

    /// Enqueue one owned submission record. The returned ID is stable for this context's lifetime.
    Result<uint64_t> submit(Submission submission);
    /// Samples or waits on this context's private monotonically increasing queue timeline.
    Result<uint64_t> completed_value() const;
    /// Borrowed timeline semaphore used to establish a dependency from another queue.
    VkSemaphore completion_semaphore() const noexcept { return use_timeline_ ? timeline_ : VK_NULL_HANDLE; }
    bool uses_synchronization2() const noexcept { return use_synchronization2_; }
    Status wait_for_value(uint64_t value, uint64_t timeout_ns = UINT64_MAX) const;
    /// Stop accepting work, then drain or cancel work that has not reached vkQueueSubmit.
    Status shutdown(ShutdownMode mode = ShutdownMode::Graceful);

private:
    struct QueuedSubmission {
        uint64_t id = 0;
        Submission submission;
    };
    struct Completion {
        uint64_t id = 0;
        uint64_t timeline_value = 0;
        Status status;
        std::function<void(uint64_t, Status)> callback;
        VkFence fence = VK_NULL_HANDLE;
        bool pooled_fence = false;
    };

    QueueContext(Api api, VkDevice device, VkQueue queue, VkSemaphore timeline, bool use_timeline,
                 bool use_synchronization2, std::vector<VkFence> fence_pool, CoalescingOptions coalescing,
                 std::shared_ptr<DeviceFaultState> fault_state);
    Status start();
    void submission_loop();
    void completion_loop();
    void enqueue_completion(Completion completion);
    static Status validate(const Submission& submission);
    Result<VkFence> acquire_fence();
    void release_fence(VkFence fence, bool reset) noexcept;

    Api api_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    bool use_timeline_ = true;
    bool use_synchronization2_ = false;
    CoalescingOptions coalescing_;
    std::vector<VkFence> fence_pool_;
    mutable std::mutex fence_mutex_;
    mutable std::condition_variable fence_changed_;
    std::deque<VkFence> available_fences_;
    bool fence_pool_failed_ = false;
    std::atomic<uint64_t> fallback_completed_value_{0};
    std::shared_ptr<DeviceFaultState> fault_state_;

    std::mutex queue_mutex_;
    std::condition_variable queue_changed_;
    std::deque<QueuedSubmission> ready_;
    uint64_t next_id_ = 1;
    bool accepting_ = true;
    bool stop_requested_ = false;
    ShutdownMode shutdown_mode_ = ShutdownMode::Graceful;
    std::thread submission_thread_;
    std::atomic_flag queue_submit_active_ = ATOMIC_FLAG_INIT;

    std::mutex completion_mutex_;
    std::condition_variable completion_changed_;
    std::deque<Completion> completions_;
    bool submission_finished_ = false;
    std::thread completion_thread_;

    std::mutex shutdown_mutex_;
    uint64_t next_timeline_value_ = 1;
};

}  // namespace kora::vk
