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
    std::atomic<uint64_t> destroy_calls{0};
    std::atomic<uint32_t> active_submit_calls{0};
    std::atomic<uint32_t> maximum_active{0};
    std::atomic<uint32_t> malformed_submit_infos{0};
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
    ++static_cast<Mock*>(context)->wait_calls;
    return VK_SUCCESS;
}
VkResult VKAPI_PTR queue_submit(void* context, VkQueue, uint32_t count, const VkSubmitInfo* submits,
                                VkFence) noexcept {
    auto& mock = *static_cast<Mock*>(context);
    const auto active = ++mock.active_submit_calls;
    auto maximum = mock.maximum_active.load();
    while (active > maximum && !mock.maximum_active.compare_exchange_weak(maximum, active)) {
    }
    if (count != 1 || submits == nullptr || submits[0].signalSemaphoreCount != 1 ||
        submits[0].pSignalSemaphores == nullptr ||
        submits[0].pSignalSemaphores[0] != reinterpret_cast<VkSemaphore>(uintptr_t{77})) {
        ++mock.malformed_submit_infos;
    } else {
        const auto* timeline = static_cast<const VkTimelineSemaphoreSubmitInfo*>(submits[0].pNext);
        if (timeline == nullptr || timeline->signalSemaphoreValueCount != 1 ||
            timeline->pSignalSemaphoreValues == nullptr || timeline->pSignalSemaphoreValues[0] == 0) {
            ++mock.malformed_submit_infos;
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

TEST(QueueContext, ReportsQueueSubmitFailureWithoutBlockingLaterTimelineCompletions) {
    Mock mock;
    mock.fail_first_submit = true;
    auto result = QueueContext::create(api_for(mock), reinterpret_cast<VkDevice>(uintptr_t{11}),
                                       reinterpret_cast<VkQueue>(uintptr_t{22}));
    ASSERT_TRUE(result);
    auto context = std::move(result).value();
    std::atomic<uint64_t> callbacks{0};
    std::atomic<int32_t> first_vk_error{VK_SUCCESS};
    std::atomic<uint32_t> second_success{0};
    QueueContext::Submission failed;
    failed.on_complete = [&](uint64_t, Status status) {
        if (!status) first_vk_error = status.error().vk_result;
        ++callbacks;
    };
    QueueContext::Submission succeeded;
    succeeded.on_complete = [&](uint64_t, Status status) {
        if (status) ++second_success;
        ++callbacks;
    };
    ASSERT_TRUE(context->submit(std::move(failed)));
    ASSERT_TRUE(context->submit(std::move(succeeded)));
    ASSERT_TRUE(context->shutdown());
    EXPECT_EQ(callbacks.load(), 2u);
    EXPECT_EQ(first_vk_error.load(), VK_ERROR_DEVICE_LOST);
    EXPECT_EQ(second_success.load(), 1u);
    EXPECT_EQ(mock.wait_calls.load(), 1u);
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

}  // namespace
}  // namespace kora::vk
