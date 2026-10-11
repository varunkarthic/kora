#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "vulkan/queue_context.hpp"

namespace kora::vk {
namespace {

struct Mock {
    std::atomic<uint64_t> submit_calls{0};
    std::atomic<uint64_t> wait_calls{0};
    std::atomic<bool> timeout_next_wait{false};
    std::atomic<uint64_t> destroy_calls{0};
    std::atomic<uint64_t> completed_timeline{0};
    std::atomic<uint32_t> active_submit_calls{0};
    std::atomic<uint32_t> maximum_active{0};
    std::atomic<uint32_t> malformed_submit_infos{0};
    std::atomic<uint32_t> submitted_command_buffers{0};
    std::atomic<uint32_t> non_monotonic_timeline_values{0};
    std::atomic<uint32_t> non_null_fences{0};
    std::atomic<uint32_t> external_wait_calls{0};
    std::atomic<uint64_t> external_wait_value{0};
    std::atomic<uint32_t> created_fences{0};
    std::atomic<uint32_t> destroyed_fences{0};
    std::atomic<uint32_t> waited_fences{0};
    std::atomic<uint32_t> reset_fences{0};
    std::atomic<uint64_t> next_fence{100};
    std::mutex gate_mutex;
    std::condition_variable gate_changed;
    bool block_first_submit = false;
    bool fail_first_submit = false;
    bool first_submit_entered = false;
    bool release_first_submit = false;
};

VkResult VKAPI_PTR create_semaphore(void*, VkDevice, const VkSemaphoreCreateInfo* info,
                                    const VkAllocationCallbacks*, VkSemaphore* semaphore) noexcept {
    const auto* type = static_cast<const VkSemaphoreTypeCreateInfo*>(info->pNext);
    if (type == nullptr || type->semaphoreType != VK_SEMAPHORE_TYPE_TIMELINE)
        return VK_ERROR_INITIALIZATION_FAILED;
    *semaphore = reinterpret_cast<VkSemaphore>(uintptr_t{77});
    return VK_SUCCESS;
}
void VKAPI_PTR destroy_semaphore(void* context, VkDevice, VkSemaphore,
                                 const VkAllocationCallbacks*) noexcept {
    ++static_cast<Mock*>(context)->destroy_calls;
}
VkResult VKAPI_PTR wait_semaphores(void* context, VkDevice, const VkSemaphoreWaitInfo*, uint64_t) noexcept {
    auto& mock = *static_cast<Mock*>(context);
    ++mock.wait_calls;
    if (mock.timeout_next_wait.exchange(false)) return VK_TIMEOUT;
    return VK_SUCCESS;
}
VkResult VKAPI_PTR create_fence(void* context, VkDevice, const VkFenceCreateInfo*,
                                const VkAllocationCallbacks*, VkFence* fence) noexcept {
    auto& mock = *static_cast<Mock*>(context);
    ++mock.created_fences;
    *fence = reinterpret_cast<VkFence>(mock.next_fence.fetch_add(1));
    return VK_SUCCESS;
}
void VKAPI_PTR destroy_fence(void* context, VkDevice, VkFence, const VkAllocationCallbacks*) noexcept {
    ++static_cast<Mock*>(context)->destroyed_fences;
}
VkResult VKAPI_PTR wait_for_fences(void* context, VkDevice, uint32_t count, const VkFence* fences, VkBool32,
                                   uint64_t) noexcept {
    auto& mock = *static_cast<Mock*>(context);
    if (count != 1 || fences == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    ++mock.waited_fences;
    return VK_SUCCESS;
}
VkResult VKAPI_PTR reset_fences(void* context, VkDevice, uint32_t count, const VkFence* fences) noexcept {
    auto& mock = *static_cast<Mock*>(context);
    if (count != 1 || fences == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    ++mock.reset_fences;
    return VK_SUCCESS;
}
VkResult VKAPI_PTR get_semaphore_counter_value(void* context, VkDevice, VkSemaphore,
                                               uint64_t* value) noexcept {
    *value = static_cast<Mock*>(context)->completed_timeline.load();
    return VK_SUCCESS;
}
VkResult VKAPI_PTR queue_submit(void* context, VkQueue, uint32_t count, const VkSubmitInfo* submits,
                                VkFence fence) noexcept {
    auto& mock = *static_cast<Mock*>(context);
    if (submits != nullptr && submits[0].waitSemaphoreCount != 0) {
        if (submits[0].waitSemaphoreCount != 1 || submits[0].pWaitSemaphores == nullptr ||
            submits[0].pWaitSemaphores[0] != reinterpret_cast<VkSemaphore>(uintptr_t{88}) ||
            submits[0].pWaitDstStageMask == nullptr ||
            submits[0].pWaitDstStageMask[0] != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
            ++mock.malformed_submit_infos;
        } else {
            ++mock.external_wait_calls;
        }
    }
    if (fence != VK_NULL_HANDLE) ++mock.non_null_fences;
    if (count == 1 && submits != nullptr)
        mock.submitted_command_buffers.fetch_add(submits[0].commandBufferCount);
    const auto active = ++mock.active_submit_calls;
    auto maximum = mock.maximum_active.load();
    while (active > maximum && !mock.maximum_active.compare_exchange_weak(maximum, active)) {
    }
    if (fence != VK_NULL_HANDLE && submits != nullptr && submits[0].pNext == nullptr) {
        if (count != 1 || submits[0].signalSemaphoreCount != 0) ++mock.malformed_submit_infos;
    } else if (count != 1 || submits == nullptr || submits[0].signalSemaphoreCount != 1 ||
               submits[0].pSignalSemaphores == nullptr ||
               submits[0].pSignalSemaphores[0] != reinterpret_cast<VkSemaphore>(uintptr_t{77})) {
        ++mock.malformed_submit_infos;
    } else {
        const auto* timeline = static_cast<const VkTimelineSemaphoreSubmitInfo*>(submits[0].pNext);
        if (timeline == nullptr || timeline->signalSemaphoreValueCount != 1 ||
            timeline->pSignalSemaphoreValues == nullptr || timeline->pSignalSemaphoreValues[0] == 0) {
            ++mock.malformed_submit_infos;
        } else {
            if (submits[0].waitSemaphoreCount != 0) {
                if (timeline->waitSemaphoreValueCount != 1 || timeline->pWaitSemaphoreValues == nullptr) {
                    ++mock.malformed_submit_infos;
                } else {
                    mock.external_wait_value.store(timeline->pWaitSemaphoreValues[0]);
                }
            }
            const auto value = timeline->pSignalSemaphoreValues[0];
            if (value <= mock.completed_timeline.load()) ++mock.non_monotonic_timeline_values;
            mock.completed_timeline.store(value);
        }
    }
    ++mock.submit_calls;
    const bool fail = mock.fail_first_submit && mock.submit_calls.load() == 1;
    {
        std::unique_lock lock(mock.gate_mutex);
        if (mock.block_first_submit && mock.submit_calls.load() == 1) {
            mock.first_submit_entered = true;
            mock.gate_changed.notify_all();
            mock.gate_changed.wait(lock, [&] { return mock.release_first_submit; });
        }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(20));
    --mock.active_submit_calls;
    return fail ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}

Api api_for(Mock& mock) {
    Api api;
    api.create_semaphore = {&mock, create_semaphore};
    api.destroy_semaphore = {&mock, destroy_semaphore};
    api.wait_semaphores = {&mock, wait_semaphores};
    api.create_fence = {&mock, create_fence};
    api.destroy_fence = {&mock, destroy_fence};
    api.wait_for_fences = {&mock, wait_for_fences};
    api.reset_fences = {&mock, reset_fences};
    api.get_semaphore_counter_value = {&mock, get_semaphore_counter_value};
    api.queue_submit = {&mock, queue_submit};
    return api;
}

TEST(QueueContext, SerializesSixteenConcurrentProducersAndCompletesCallbacks) {
    Mock mock;
    auto result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                       reinterpret_cast<VkQueue>(uintptr_t{22}));
    ASSERT_TRUE(result) << result.error().describe();
    auto context = std::move(result).value();
    constexpr uint32_t kThreads = 16;
    constexpr uint32_t kPerThread = 128;
    std::atomic<uint64_t> callbacks{0};
    std::atomic<uint64_t> callback_errors{0};
    std::atomic<uint64_t> accepted{0};
    std::vector<std::thread> producers;
    producers.reserve(kThreads);
    for (uint32_t thread = 0; thread < kThreads; ++thread) {
        producers.emplace_back([&] {
            for (uint32_t i = 0; i < kPerThread; ++i) {
                QueueContext::Submission submission;
                submission.on_complete = [&](uint64_t, Status status) {
                    if (!status) ++callback_errors;
                    ++callbacks;
                };
                auto id = context->submit(std::move(submission));
                if (id) ++accepted;
            }
        });
    }
    for (auto& producer : producers) producer.join();
    ASSERT_TRUE(context->shutdown(ShutdownMode::Graceful));
    EXPECT_EQ(accepted.load(), kThreads * kPerThread);
    EXPECT_EQ(mock.submit_calls.load(), accepted.load());
    EXPECT_EQ(callbacks.load(), accepted.load());
    EXPECT_EQ(callback_errors.load(), 0u);
    EXPECT_EQ(mock.non_monotonic_timeline_values.load(), 0u);
    EXPECT_EQ(mock.non_null_fences.load(), 0u);
    EXPECT_EQ(mock.wait_calls.load(), accepted.load());
    EXPECT_EQ(mock.maximum_active.load(), 1u);
    EXPECT_EQ(mock.malformed_submit_infos.load(), 0u);
    EXPECT_EQ(context->submit({}).error().code, ErrorCode::Cancelled);
    context.reset();
    EXPECT_EQ(mock.destroy_calls.load(), 1u);
}

TEST(QueueContext, CancelPendingCancelsQueuedWorkButDrainsInflightCompletion) {
    Mock mock;
    mock.block_first_submit = true;
    auto result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                       reinterpret_cast<VkQueue>(uintptr_t{22}));
    ASSERT_TRUE(result) << result.error().describe();
    auto context = std::move(result).value();
    std::atomic<uint64_t> succeeded{0};
    std::atomic<uint64_t> cancelled{0};
    constexpr uint32_t kSubmissions = 32;
    for (uint32_t i = 0; i < kSubmissions; ++i) {
        QueueContext::Submission submission;
        submission.on_complete = [&](uint64_t, Status status) {
            if (status)
                ++succeeded;
            else if (status.error().code == ErrorCode::Cancelled)
                ++cancelled;
        };
        ASSERT_TRUE(context->submit(std::move(submission)));
    }
    {
        std::unique_lock lock(mock.gate_mutex);
        ASSERT_TRUE(mock.gate_changed.wait_for(lock, std::chrono::seconds(5),
                                               [&] { return mock.first_submit_entered; }));
    }
    std::thread shutdown([&] { EXPECT_TRUE(context->shutdown(ShutdownMode::CancelPending)); });
    {
        std::lock_guard lock(mock.gate_mutex);
        mock.release_first_submit = true;
    }
    mock.gate_changed.notify_all();
    shutdown.join();
    EXPECT_EQ(mock.submit_calls.load(), 1u);
    EXPECT_EQ(succeeded.load(), 1u);
    EXPECT_EQ(cancelled.load(), kSubmissions - 1);
    EXPECT_EQ(mock.wait_calls.load(), 1u);
}

TEST(QueueContext, RejectsMalformedSubmissionBeforeQueueing) {
    Mock mock;
    auto result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                       reinterpret_cast<VkQueue>(uintptr_t{22}));
    ASSERT_TRUE(result);
    auto context = std::move(result).value();
    QueueContext::Submission submission;
    submission.wait_semaphores.push_back(reinterpret_cast<VkSemaphore>(uintptr_t{33}));
    EXPECT_EQ(context->submit(std::move(submission)).error().code, ErrorCode::InvalidArgument);
    EXPECT_TRUE(context->shutdown());
    EXPECT_EQ(mock.submit_calls.load(), 0u);
}

TEST(QueueContext, ForwardsExternalTimelineWaitForCrossQueueDependency) {
    Mock mock;
    auto result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                       reinterpret_cast<VkQueue>(uintptr_t{22}));
    ASSERT_TRUE(result) << result.error().describe();
    auto context = std::move(result).value();
    QueueContext::Submission submission;
    submission.wait_semaphores.push_back(reinterpret_cast<VkSemaphore>(uintptr_t{88}));
    submission.wait_stages.push_back(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    submission.wait_values.push_back(42);
    ASSERT_TRUE(context->submit(std::move(submission)));
    ASSERT_TRUE(context->shutdown());
    EXPECT_EQ(mock.external_wait_calls.load(), 1U);
    EXPECT_EQ(mock.external_wait_value.load(), 42U);
    EXPECT_EQ(mock.malformed_submit_infos.load(), 0U);
}

TEST(QueueContext, FenceFallbackReusesBoundedPoolAndEmulatesCompletionValues) {
    Mock mock;
    auto created = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                        reinterpret_cast<VkQueue>(uintptr_t{22}), true);
    ASSERT_TRUE(created) << created.error().describe();
    auto context = std::move(created).value();
    constexpr uint64_t kSubmissionCount = 32;
    std::atomic<uint64_t> callbacks{0};
    uint64_t last_id = 0;
    for (uint64_t index = 0; index < kSubmissionCount; ++index) {
        QueueContext::Submission submission;
        submission.on_complete = [&](uint64_t, Status status) {
            EXPECT_TRUE(status);
            ++callbacks;
        };
        auto submitted = context->submit(std::move(submission));
        ASSERT_TRUE(submitted) << submitted.error().describe();
        last_id = submitted.value();
    }
    ASSERT_TRUE(context->wait_for_value(last_id, 5'000'000'000));
    auto completed = context->completed_value();
    ASSERT_TRUE(completed) << completed.error().describe();
    EXPECT_EQ(completed.value(), last_id);
    ASSERT_TRUE(context->shutdown(ShutdownMode::Graceful));
    EXPECT_EQ(callbacks.load(), kSubmissionCount);
    EXPECT_EQ(mock.submit_calls.load(), kSubmissionCount);
    EXPECT_EQ(mock.created_fences.load(), 8U);
    EXPECT_EQ(mock.waited_fences.load(), kSubmissionCount);
    EXPECT_EQ(mock.reset_fences.load(), kSubmissionCount);
    context.reset();
    EXPECT_EQ(mock.destroyed_fences.load(), 8U);
}

TEST(QueueContext, DeviceLostFailsQueuedSubmissionsAndPreventsFurtherQueueUse) {
    Mock mock;
    mock.fail_first_submit = true;
    mock.block_first_submit = true;
    auto fault_state = std::make_shared<DeviceFaultState>();
    auto result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                       reinterpret_cast<VkQueue>(uintptr_t{22}), false, fault_state);
    ASSERT_TRUE(result);
    auto context = std::move(result).value();
    auto sibling_result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                               reinterpret_cast<VkQueue>(uintptr_t{23}), false, fault_state);
    ASSERT_TRUE(sibling_result);
    auto sibling = std::move(sibling_result).value();
    std::atomic<uint64_t> callbacks{0};
    std::atomic<int32_t> first_vk_error{VK_SUCCESS};
    std::atomic<ErrorCode> first_error{ErrorCode::Ok};
    std::atomic<ErrorCode> second_error{ErrorCode::Ok};
    std::mutex callback_mutex;
    std::condition_variable callback_changed;
    QueueContext::Submission failed;
    failed.on_complete = [&](uint64_t, Status status) {
        if (!status) {
            first_vk_error = status.error().vk_result;
            first_error = status.error().code;
        }
        ++callbacks;
        callback_changed.notify_all();
    };
    QueueContext::Submission queued;
    queued.on_complete = [&](uint64_t, Status status) {
        if (!status) second_error = status.error().code;
        ++callbacks;
        callback_changed.notify_all();
    };
    ASSERT_TRUE(context->submit(std::move(failed)));
    {
        std::unique_lock lock(mock.gate_mutex);
        ASSERT_TRUE(mock.gate_changed.wait_for(lock, std::chrono::seconds(5),
                                               [&] { return mock.first_submit_entered; }));
    }
    ASSERT_TRUE(context->submit(std::move(queued)));
    {
        std::lock_guard lock(mock.gate_mutex);
        mock.release_first_submit = true;
    }
    mock.gate_changed.notify_all();
    {
        std::unique_lock lock(callback_mutex);
        ASSERT_TRUE(
            callback_changed.wait_for(lock, std::chrono::seconds(5), [&] { return callbacks.load() == 2; }));
    }
    ASSERT_TRUE(context->shutdown());
    EXPECT_EQ(callbacks.load(), 2u);
    EXPECT_EQ(first_vk_error.load(), VK_ERROR_DEVICE_LOST);
    EXPECT_EQ(first_error.load(), ErrorCode::DeviceLost);
    EXPECT_EQ(second_error.load(), ErrorCode::DeviceLost);
    EXPECT_FALSE(fault_state->healthy());
    EXPECT_EQ(fault_state->last_result(), VK_ERROR_DEVICE_LOST);
    EXPECT_EQ(sibling->submit({}).error().code, ErrorCode::DeviceLost);
    EXPECT_EQ(context->submit({}).error().code, ErrorCode::Cancelled);
    EXPECT_EQ(mock.submit_calls.load(), 1u);
    EXPECT_EQ(mock.wait_calls.load(), 0u);
    EXPECT_TRUE(sibling->shutdown());
}

TEST(QueueContext, ConcurrentShutdownCallersJoinWorkersSafely) {
    Mock mock;
    auto result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                       reinterpret_cast<VkQueue>(uintptr_t{22}));
    ASSERT_TRUE(result);
    auto context = std::move(result).value();
    std::atomic<uint64_t> callbacks{0};
    for (uint32_t i = 0; i < 8; ++i) {
        QueueContext::Submission submission;
        submission.on_complete = [&](uint64_t, Status status) {
            if (status) ++callbacks;
        };
        ASSERT_TRUE(context->submit(std::move(submission)));
    }
    std::atomic<uint32_t> shutdown_errors{0};
    std::thread first([&] {
        if (!context->shutdown(ShutdownMode::Graceful)) ++shutdown_errors;
    });
    std::thread second([&] {
        if (!context->shutdown(ShutdownMode::Graceful)) ++shutdown_errors;
    });
    first.join();
    second.join();
    EXPECT_EQ(shutdown_errors.load(), 0u);
    EXPECT_EQ(callbacks.load(), 8u);
}

TEST(QueueContext, RecordsOnSubmissionThreadAndReportsTimelineAcrossCallbacks) {
    Mock mock;
    auto result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                       reinterpret_cast<VkQueue>(uintptr_t{22}));
    ASSERT_TRUE(result);
    auto context = std::move(result).value();
    std::atomic<uint64_t> recorded_value{0};
    std::atomic<uint64_t> submitted_value{0};
    std::atomic<uint64_t> completed_id{0};
    std::atomic<bool> submit_succeeded{false};
    QueueContext::Submission submission;
    submission.record_commands = [&](uint64_t value, std::vector<VkCommandBuffer>& commands) {
        recorded_value.store(value);
        commands.push_back(reinterpret_cast<VkCommandBuffer>(uintptr_t{88}));
        return ok_status();
    };
    submission.on_submitted = [&](uint64_t value, Status status) {
        submitted_value.store(value);
        submit_succeeded.store(status.ok());
    };
    submission.on_complete = [&](uint64_t id, Status status) {
        completed_id.store(id);
        EXPECT_TRUE(status);
    };
    auto id = context->submit(std::move(submission));
    ASSERT_TRUE(id);
    ASSERT_TRUE(context->shutdown());
    EXPECT_EQ(recorded_value.load(), 1U);
    EXPECT_EQ(submitted_value.load(), 1U);
    EXPECT_TRUE(submit_succeeded.load());
    EXPECT_EQ(completed_id.load(), id.value());
    EXPECT_EQ(mock.submit_calls.load(), 1U);
}

TEST(QueueContext, RecordFailureSkipsQueueSubmitAndReportsSubmissionFailure) {
    Mock mock;
    auto result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                       reinterpret_cast<VkQueue>(uintptr_t{22}));
    ASSERT_TRUE(result);
    auto context = std::move(result).value();
    std::atomic<bool> submitted_failure{false};
    std::atomic<bool> completion_failure{false};
    QueueContext::Submission submission;
    submission.record_commands = [](uint64_t, std::vector<VkCommandBuffer>&) {
        return Status(Error(ErrorCode::TransferFailed, "mock recording failed"));
    };
    submission.on_submitted = [&](uint64_t, Status status) { submitted_failure.store(!status); };
    submission.on_complete = [&](uint64_t, Status status) { completion_failure.store(!status); };
    ASSERT_TRUE(context->submit(std::move(submission)));
    ASSERT_TRUE(context->shutdown());
    EXPECT_TRUE(submitted_failure.load());
    EXPECT_TRUE(completion_failure.load());
    EXPECT_EQ(mock.submit_calls.load(), 0U);
}

TEST(QueueContext, CoalescesCompatibleDispatchesIntoOneTimelineSubmission) {
    Mock mock;
    QueueContext::CoalescingOptions options;
    options.enabled = true;
    constexpr uint32_t kDispatches = 32;
    options.maximum_batch_size = 8;
    options.maximum_wait_ns = 5'000'000;
    auto result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                       reinterpret_cast<VkQueue>(uintptr_t{22}), false, options);
    ASSERT_TRUE(result);
    auto context = std::move(result).value();
    std::atomic<uint32_t> recorded{0};
    std::atomic<uint32_t> submitted{0};
    std::atomic<uint32_t> completed{0};
    std::atomic<uint32_t> callback_errors{0};
    std::mutex completed_ids_mutex;
    std::vector<uint64_t> completed_ids;
    std::vector<uint64_t> timeline_values(kDispatches, 0);
    for (uint32_t index = 0; index < kDispatches; ++index) {
        QueueContext::Submission submission;
        submission.coalescible = true;
        submission.dependency_free = true;
        submission.coalescing_key = 1234;
        submission.resources.push_back({index, 1, false});
        submission.record_commands = [&, index](uint64_t, std::vector<VkCommandBuffer>& commands) {
            ++recorded;
            commands.push_back(reinterpret_cast<VkCommandBuffer>(uintptr_t{200 + index}));
            return ok_status();
        };
        submission.on_submitted = [&, index](uint64_t value, Status status) {
            timeline_values[index] = value;
            if (!status) ++callback_errors;
            ++submitted;
        };
        submission.on_complete = [&](uint64_t id, Status status) {
            if (!status) ++callback_errors;
            {
                std::lock_guard lock(completed_ids_mutex);
                completed_ids.push_back(id);
            }
            ++completed;
        };
        ASSERT_TRUE(context->submit(std::move(submission)));
    }
    ASSERT_TRUE(context->shutdown());
    EXPECT_LT(mock.submit_calls.load(), kDispatches);
    EXPECT_EQ(mock.submitted_command_buffers.load(), kDispatches);
    EXPECT_EQ(recorded.load(), kDispatches);
    EXPECT_EQ(submitted.load(), kDispatches);
    EXPECT_EQ(completed.load(), kDispatches);
    std::sort(completed_ids.begin(), completed_ids.end());
    for (uint32_t index = 0; index < kDispatches; ++index)
        EXPECT_EQ(completed_ids[index], static_cast<uint64_t>(index + 1));
    EXPECT_EQ(callback_errors.load(), 0U);
    EXPECT_EQ(timeline_values[0], timeline_values[1]);
    EXPECT_EQ(timeline_values[1], timeline_values[2]);
    EXPECT_EQ(timeline_values[2], timeline_values[3]);
    EXPECT_LT(timeline_values[0], timeline_values.back());
}

TEST(QueueContext, DoesNotCoalesceHighPriorityOrConflictingDispatches) {
    Mock mock;
    QueueContext::CoalescingOptions options;
    options.enabled = true;
    options.maximum_wait_ns = 5'000'000;
    auto result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                       reinterpret_cast<VkQueue>(uintptr_t{22}), false, options);
    ASSERT_TRUE(result);
    auto context = std::move(result).value();
    for (uint32_t index = 0; index < 4; ++index) {
        QueueContext::Submission submission;
        submission.coalescible = true;
        submission.dependency_free = index < 2;
        submission.coalescing_key = 1234;
        submission.priority = index < 2 ? 2 : 1;
        submission.resources.push_back({index < 2 ? 1U : 9U, 1, index >= 2});
        ASSERT_TRUE(context->submit(std::move(submission)));
    }
    for (uint64_t value = 1; value <= 2; ++value) {
        QueueContext::Submission submission;
        submission.coalescible = true;
        submission.dependency_free = true;
        submission.coalescing_key = 5678;
        submission.resources.push_back({static_cast<uint32_t>(20 + value), 1, false});
        submission.wait_semaphores.push_back(reinterpret_cast<VkSemaphore>(uintptr_t{77}));
        submission.wait_stages.push_back(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        submission.wait_values.push_back(value);
        ASSERT_TRUE(context->submit(std::move(submission)));
    }
    ASSERT_TRUE(context->shutdown());
    EXPECT_EQ(mock.submit_calls.load(), 6U);
}

TEST(QueueContext, CompletionTimeoutPollsCounterBeforeDeliveringCallback) {
    Mock mock;
    mock.timeout_next_wait = true;
    auto result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                       reinterpret_cast<VkQueue>(uintptr_t{22}));
    ASSERT_TRUE(result);
    auto context = std::move(result).value();
    std::atomic<bool> callback_success{false};
    QueueContext::Submission submission;
    submission.on_complete = [&](uint64_t, Status status) { callback_success.store(status.ok()); };
    ASSERT_TRUE(context->submit(std::move(submission)));
    ASSERT_TRUE(context->shutdown());
    EXPECT_TRUE(callback_success.load());
    EXPECT_EQ(mock.wait_calls.load(), 1u);
}

}  // namespace
}  // namespace kora::vk
