#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "vulkan/api.hpp"
#include "vulkan/queue_context.hpp"

namespace kora::vk {
namespace {

TEST(QueueContext, CompletesConcurrentRealVulkanSubmissions) {
    auto loader = Loader::load();
    if (!loader) GTEST_SKIP() << loader.error().describe();
    auto instance = loader.value().create_instance();
    if (!instance) GTEST_SKIP() << instance.error().describe();
    auto physical_devices = instance.value().physical_devices();
    ASSERT_TRUE(physical_devices) << physical_devices.error().describe();
    if (physical_devices.value().empty()) GTEST_SKIP() << "No Vulkan physical device is available";
    auto device = instance.value().create_device(physical_devices.value().front());
    if (!device) GTEST_SKIP() << device.error().describe();
    if (!device.value().enabled_features().timeline_semaphore)
        GTEST_SKIP() << "Timeline semaphores are unavailable on this device";

    auto result =
        QueueContext::create(device.value().api(), device.value().handle(), device.value().compute_queue());
    if (!result) GTEST_SKIP() << result.error().describe();
    auto context = std::move(result).value();
    constexpr uint32_t kThreads = 16;
    constexpr uint32_t kSubmissionsPerThread = 64;
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> errors{0};
    std::vector<std::thread> producers;
    producers.reserve(kThreads);
    for (uint32_t thread = 0; thread < kThreads; ++thread) {
        producers.emplace_back([&] {
            for (uint32_t i = 0; i < kSubmissionsPerThread; ++i) {
                QueueContext::Submission submission;
                submission.on_complete = [&](uint64_t, Status status) {
                    if (!status) ++errors;
                    ++completed;
                };
                if (!context->submit(std::move(submission))) ++errors;
            }
        });
    }
    for (auto& producer : producers) producer.join();
    ASSERT_TRUE(context->shutdown(ShutdownMode::Graceful));
    EXPECT_EQ(completed.load(), kThreads * kSubmissionsPerThread);
    EXPECT_EQ(errors.load(), 0u);
}

}  // namespace
}  // namespace kora::vk
