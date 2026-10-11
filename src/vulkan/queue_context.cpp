#include "vulkan/queue_context.hpp"

#include <cassert>
#include <limits>
#include <system_error>
#include <utility>

namespace kora::vk {
namespace {

thread_local const QueueContext* active_queue_context = nullptr;
constexpr std::uint64_t kCompletionPollIntervalNs = 250'000'000;
constexpr std::size_t kFencePoolSize = 8;

Status vk_error(const char* operation, VkResult result) {
    return Error(ErrorCode::TransferFailed, operation).with_vk(result);
}

}  // namespace

Result<std::unique_ptr<QueueContext>> QueueContext::create(Api api, VkDevice device, VkQueue queue,
                                                           bool force_fence_fallback) {
    if (device == VK_NULL_HANDLE || queue == VK_NULL_HANDLE || !api.queue_submit)
        return Error(ErrorCode::InvalidArgument,
                     "QueueContext requires a device, queue, and queue-submit dispatch");

    const bool timeline_available = api.create_semaphore && api.destroy_semaphore && api.wait_semaphores &&
                                    api.get_semaphore_counter_value;
    VkSemaphore timeline = VK_NULL_HANDLE;
    bool use_timeline = timeline_available && !force_fence_fallback;
    std::vector<VkFence> fences;
    if (use_timeline) {
        VkSemaphoreTypeCreateInfo type_info{};
        type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        create_info.pNext = &type_info;
        const VkResult result = api.create_semaphore(device, &create_info, nullptr, &timeline);
        if (result != VK_SUCCESS) {
            if (!api.create_fence || !api.destroy_fence || !api.wait_for_fences || !api.reset_fences)
                return Error(ErrorCode::UnsupportedFeature,
                             "Timeline semaphore creation failed and fence fallback is unavailable")
                    .with_vk(result);
            use_timeline = false;
        }
    }
    if (!use_timeline) {
        if (!api.create_fence || !api.destroy_fence || !api.wait_for_fences || !api.reset_fences)
            return Error(ErrorCode::UnsupportedFeature, "QueueContext requires timeline or fence dispatch");
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fences.reserve(kFencePoolSize);
        for (std::size_t index = 0; index < kFencePoolSize; ++index) {
            VkFence fence = VK_NULL_HANDLE;
            const VkResult result = api.create_fence(device, &fence_info, nullptr, &fence);
            if (result != VK_SUCCESS) {
                for (const auto allocated : fences) api.destroy_fence(device, allocated, nullptr);
                if (timeline != VK_NULL_HANDLE) api.destroy_semaphore(device, timeline, nullptr);
                return Error(ErrorCode::UnsupportedFeature, "Creating QueueContext fence pool failed")
                    .with_vk(result);
            }
            fences.push_back(fence);
        }
    }
    auto context = std::unique_ptr<QueueContext>(
        new QueueContext(std::move(api), device, queue, timeline, use_timeline, std::move(fences)));
    auto start_status = context->start();
    if (!start_status) return std::move(start_status).error();
    return context;
}

QueueContext::QueueContext(Api api, VkDevice device, VkQueue queue, VkSemaphore timeline, bool use_timeline,
                           std::vector<VkFence> fence_pool)
    : api_(std::move(api)),
      device_(device),
      queue_(queue),
      timeline_(timeline),
      use_timeline_(use_timeline),
      fence_pool_(std::move(fence_pool)),
      available_fences_(fence_pool_.begin(), fence_pool_.end()) {}

QueueContext::~QueueContext() {
    (void)shutdown(ShutdownMode::Graceful);
    if (timeline_ != VK_NULL_HANDLE) api_.destroy_semaphore(device_, timeline_, nullptr);
    for (const auto fence : fence_pool_) api_.destroy_fence(device_, fence, nullptr);
}

Status QueueContext::start() {
    try {
        completion_thread_ = std::thread([this] { completion_loop(); });
        submission_thread_ = std::thread([this] { submission_loop(); });
    } catch (const std::system_error&) {
        {
            std::lock_guard lock(completion_mutex_);
            submission_finished_ = true;
        }
        completion_changed_.notify_one();
        if (completion_thread_.joinable()) completion_thread_.join();
        return Error(ErrorCode::InternalInvariantViolation, "Failed to start QueueContext worker threads");
    }
    return ok_status();
}

Status QueueContext::validate(const Submission& submission) {
    const auto size_fits = [](size_t size) { return size <= std::numeric_limits<uint32_t>::max(); };
    if (!size_fits(submission.wait_semaphores.size()) || !size_fits(submission.wait_stages.size()) ||
        !size_fits(submission.wait_values.size()) || !size_fits(submission.command_buffers.size()) ||
        !size_fits(submission.signal_semaphores.size()) || !size_fits(submission.signal_values.size())) {
        return Error(ErrorCode::Overflow, "QueueContext submission exceeds Vulkan count limits");
    }
    if (submission.wait_stages.size() != submission.wait_semaphores.size()) {
        return Error(ErrorCode::InvalidArgument, "Every wait semaphore requires a destination stage mask");
    }
    if ((!submission.wait_values.empty() &&
         submission.wait_values.size() != submission.wait_semaphores.size()) ||
        (!submission.signal_values.empty() &&
         submission.signal_values.size() != submission.signal_semaphores.size())) {
        return Error(ErrorCode::InvalidArgument, "Timeline value arrays must match their semaphore arrays");
    }
    return ok_status();
}

Result<uint64_t> QueueContext::submit(Submission submission) {
    auto status = validate(submission);
    if (!status) return std::move(status).error();
    std::lock_guard lock(queue_mutex_);
    if (!accepting_) return Error(ErrorCode::Cancelled, "QueueContext is shutting down and rejects new work");
    if (next_id_ == std::numeric_limits<uint64_t>::max())
        return Error(ErrorCode::Overflow, "QueueContext submission ID space is exhausted");
    const uint64_t id = next_id_++;
    ready_.push_back(QueuedSubmission{id, std::move(submission)});
    queue_changed_.notify_one();
    return id;
}

Result<uint64_t> QueueContext::completed_value() const {
    if (!use_timeline_) return fallback_completed_value_.load(std::memory_order_acquire);
    uint64_t value = 0;
    const auto result = api_.get_semaphore_counter_value(device_, timeline_, &value);
    if (result != VK_SUCCESS)
        return std::move(vk_error("Reading queue completion timeline failed", result)).error();
    return value;
}

Status QueueContext::wait_for_value(uint64_t value, uint64_t timeout_ns) const {
    if (value == 0) return ok_status();
    if (!use_timeline_) {
        std::unique_lock lock(fence_mutex_);
        const auto completed = [this, value] {
            return fallback_completed_value_.load(std::memory_order_acquire) >= value;
        };
        if (timeout_ns == UINT64_MAX) {
            fence_changed_.wait(lock, completed);
            return ok_status();
        }
        if (!fence_changed_.wait_for(lock, std::chrono::nanoseconds(timeout_ns), completed))
            return Error(ErrorCode::Timeout, "Queue completion fence wait timed out");
        return ok_status();
    }
    VkSemaphoreWaitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &timeline_;
    wait_info.pValues = &value;
    const auto result = api_.wait_semaphores(device_, &wait_info, timeout_ns);
    if (result == VK_TIMEOUT) return Error(ErrorCode::Timeout, "Queue completion wait timed out");
    if (result != VK_SUCCESS) return vk_error("Waiting for queue completion timeline failed", result);
    return ok_status();
}

Result<VkFence> QueueContext::acquire_fence() {
    std::unique_lock lock(fence_mutex_);
    fence_changed_.wait(lock, [this] { return fence_pool_failed_ || !available_fences_.empty(); });
    if (fence_pool_failed_) return Error(ErrorCode::DeviceLost, "QueueContext fence pool is unavailable");
    const auto fence = available_fences_.front();
    available_fences_.pop_front();
    return fence;
}

void QueueContext::release_fence(VkFence fence, bool reset) noexcept {
    if (fence == VK_NULL_HANDLE) return;
    if (reset && api_.reset_fences(device_, 1, &fence) != VK_SUCCESS) {
        std::lock_guard lock(fence_mutex_);
        fence_pool_failed_ = true;
        fence_changed_.notify_all();
        return;
    }
    {
        std::lock_guard lock(fence_mutex_);
        available_fences_.push_back(fence);
    }
    fence_changed_.notify_all();
}

Status QueueContext::shutdown(ShutdownMode mode) {
    if (active_queue_context == this) {
        return Error(ErrorCode::InvalidArgument,
                     "QueueContext cannot be shut down from one of its callbacks");
    }
    std::lock_guard shutdown_lock(shutdown_mutex_);
    {
        std::lock_guard lock(queue_mutex_);
        accepting_ = false;
        stop_requested_ = true;
        if (mode == ShutdownMode::CancelPending) shutdown_mode_ = mode;
    }
    queue_changed_.notify_one();
    if (submission_thread_.joinable()) submission_thread_.join();
    completion_changed_.notify_one();
    if (completion_thread_.joinable()) completion_thread_.join();
    return ok_status();
}

void QueueContext::enqueue_completion(Completion completion) {
    {
        std::lock_guard lock(completion_mutex_);
        completions_.push_back(std::move(completion));
    }
    completion_changed_.notify_one();
}

void QueueContext::submission_loop() {
    active_queue_context = this;
    for (;;) {
        QueuedSubmission queued;
        bool cancel = false;
        {
            std::unique_lock lock(queue_mutex_);
            queue_changed_.wait(lock, [this] { return stop_requested_ || !ready_.empty(); });
            if (stop_requested_ && ready_.empty()) break;
            if (stop_requested_ && shutdown_mode_ == ShutdownMode::CancelPending) {
                queued = std::move(ready_.front());
                ready_.pop_front();
                cancel = true;
            } else {
                queued = std::move(ready_.front());
                ready_.pop_front();
            }
        }
        if (cancel) {
            enqueue_completion(Completion{
                queued.id, 0, Error(ErrorCode::Cancelled, "Submission cancelled before queue submit"),
                std::move(queued.submission.on_complete)});
            continue;
        }
        if (next_timeline_value_ == std::numeric_limits<uint64_t>::max()) {
            enqueue_completion(Completion{
                queued.id, 0, Error(ErrorCode::Overflow, "QueueContext timeline value space is exhausted"),
                std::move(queued.submission.on_complete)});
            continue;
        }

        const uint64_t timeline_value = next_timeline_value_++;
        auto& submission = queued.submission;
        if (submission.record_commands) {
            Status record_status;
            try {
                record_status = submission.record_commands(timeline_value, submission.command_buffers);
            } catch (...) {
                record_status = Error(ErrorCode::InternalInvariantViolation,
                                      "QueueContext command-recording callback threw an exception");
            }
            if (!record_status) {
                if (submission.on_submitted) {
                    try {
                        submission.on_submitted(timeline_value, record_status);
                    } catch (...) {
                    }
                }
                enqueue_completion(
                    Completion{queued.id, 0, std::move(record_status), std::move(submission.on_complete)});
                continue;
            }
        }
        if (submission.command_buffers.size() > std::numeric_limits<uint32_t>::max()) {
            const auto error =
                Error(ErrorCode::Overflow, "Recorded command buffer count exceeds Vulkan limit");
            if (submission.on_submitted) {
                try {
                    submission.on_submitted(timeline_value, error);
                } catch (...) {
                }
            }
            enqueue_completion(Completion{queued.id, 0, error, std::move(submission.on_complete)});
            continue;
        }
        std::vector<uint64_t> wait_values = submission.wait_values;
        if (wait_values.empty()) wait_values.resize(submission.wait_semaphores.size(), 0);
        std::vector<uint64_t> signal_values = submission.signal_values;
        if (signal_values.empty()) signal_values.resize(submission.signal_semaphores.size(), 0);
        if (!use_timeline_ &&
            (std::any_of(wait_values.begin(), wait_values.end(), [](auto value) { return value != 0; }) ||
             std::any_of(signal_values.begin(), signal_values.end(),
                         [](auto value) { return value != 0; }))) {
            const auto error = Error(ErrorCode::UnsupportedFeature,
                                     "Fence fallback accepts binary semaphore waits and signals only");
            enqueue_completion(Completion{queued.id, 0, error, std::move(submission.on_complete)});
            continue;
        }
        auto signal_semaphores = submission.signal_semaphores;
        VkTimelineSemaphoreSubmitInfo timeline_info{};
        if (use_timeline_) {
            signal_semaphores.push_back(timeline_);
            signal_values.push_back(timeline_value);
            timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_info.waitSemaphoreValueCount = static_cast<uint32_t>(wait_values.size());
            timeline_info.pWaitSemaphoreValues = wait_values.data();
            timeline_info.signalSemaphoreValueCount = static_cast<uint32_t>(signal_values.size());
            timeline_info.pSignalSemaphoreValues = signal_values.data();
        }

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pNext = use_timeline_ ? &timeline_info : nullptr;
        submit_info.waitSemaphoreCount = static_cast<uint32_t>(submission.wait_semaphores.size());
        submit_info.pWaitSemaphores = submission.wait_semaphores.data();
        submit_info.pWaitDstStageMask = submission.wait_stages.data();
        submit_info.commandBufferCount = static_cast<uint32_t>(submission.command_buffers.size());
        submit_info.pCommandBuffers = submission.command_buffers.data();
        submit_info.signalSemaphoreCount = static_cast<uint32_t>(signal_semaphores.size());
        submit_info.pSignalSemaphores = signal_semaphores.data();

        const bool was_active = queue_submit_active_.test_and_set(std::memory_order_acquire);
        assert(!was_active && "Concurrent host submission to one Vulkan queue");
        if (was_active) {
            enqueue_completion(Completion{queued.id, 0,
                                          Error(ErrorCode::InternalInvariantViolation,
                                                "Concurrent host submission to one Vulkan queue"),
                                          std::move(submission.on_complete)});
            continue;
        }
        VkFence completion_fence = VK_NULL_HANDLE;
        bool pooled_fence = false;
        if (!use_timeline_) {
            if (submission.fence != VK_NULL_HANDLE) {
                completion_fence = submission.fence;
            } else {
                auto fence_result = acquire_fence();
                if (!fence_result) {
                    enqueue_completion(Completion{queued.id, 0, std::move(fence_result).error(),
                                                  std::move(submission.on_complete)});
                    continue;
                }
                completion_fence = fence_result.value();
                pooled_fence = true;
            }
        }
        const VkResult result =
            api_.queue_submit(queue_, 1, &submit_info, use_timeline_ ? submission.fence : completion_fence);
        queue_submit_active_.clear(std::memory_order_release);
        const Status submit_status =
            result == VK_SUCCESS ? ok_status() : Status(vk_error("vkQueueSubmit failed", result));
        if (submission.on_submitted) {
            try {
                submission.on_submitted(timeline_value, submit_status);
            } catch (...) {
            }
        }
        if (result == VK_SUCCESS) {
            Completion completion{queued.id, timeline_value, ok_status(), std::move(submission.on_complete)};
            completion.fence = completion_fence;
            completion.pooled_fence = pooled_fence;
            enqueue_completion(std::move(completion));
        } else {
            if (pooled_fence) release_fence(completion_fence, false);
            enqueue_completion(Completion{queued.id, 0, vk_error("vkQueueSubmit failed", result),
                                          std::move(submission.on_complete)});
        }
    }
    {
        std::lock_guard lock(completion_mutex_);
        submission_finished_ = true;
    }
    completion_changed_.notify_one();
    active_queue_context = nullptr;
}

void QueueContext::completion_loop() {
    active_queue_context = this;
    for (;;) {
        Completion completion;
        {
            std::unique_lock lock(completion_mutex_);
            completion_changed_.wait(lock, [this] { return submission_finished_ || !completions_.empty(); });
            if (completions_.empty() && submission_finished_) {
                active_queue_context = nullptr;
                return;
            }
            completion = std::move(completions_.front());
            completions_.pop_front();
        }
        if (completion.timeline_value != 0) {
            if (use_timeline_) {
                for (;;) {
                    VkSemaphoreWaitInfo wait_info{};
                    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
                    wait_info.semaphoreCount = 1;
                    wait_info.pSemaphores = &timeline_;
                    wait_info.pValues = &completion.timeline_value;
                    const VkResult result =
                        api_.wait_semaphores(device_, &wait_info, kCompletionPollIntervalNs);
                    if (result == VK_SUCCESS) break;
                    if (result != VK_TIMEOUT) {
                        completion.status = vk_error("Waiting for queue completion failed", result);
                        break;
                    }

                    // A timeout is only a poll boundary: do not deliver completion or release
                    // referenced resources until the timeline confirms the submitted work finished.
                    uint64_t completed = 0;
                    const VkResult query_result =
                        api_.get_semaphore_counter_value(device_, timeline_, &completed);
                    if (query_result != VK_SUCCESS) {
                        completion.status =
                            vk_error("Polling queue completion timeline failed", query_result);
                        break;
                    }
                    if (completed >= completion.timeline_value) break;
                }
            } else {
                for (;;) {
                    const VkResult result = api_.wait_for_fences(device_, 1, &completion.fence, VK_TRUE,
                                                                 kCompletionPollIntervalNs);
                    if (result == VK_SUCCESS) {
                        fallback_completed_value_.store(completion.timeline_value, std::memory_order_release);
                        fence_changed_.notify_all();
                        if (completion.pooled_fence) release_fence(completion.fence, true);
                        break;
                    }
                    if (result != VK_TIMEOUT) {
                        completion.status = vk_error("Waiting for queue completion fence failed", result);
                        if (completion.pooled_fence) {
                            std::lock_guard lock(fence_mutex_);
                            fence_pool_failed_ = true;
                            fence_changed_.notify_all();
                        }
                        break;
                    }
                }
            }
        }
        if (completion.callback) {
            try {
                completion.callback(completion.id, std::move(completion.status));
            } catch (...) {
                // User callbacks are an exception boundary and must not terminate the completion worker.
            }
        }
    }
}

}  // namespace kora::vk
