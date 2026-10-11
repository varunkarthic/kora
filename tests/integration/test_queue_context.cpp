#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "vulkan/api.hpp"
#include "vulkan/queue_context.hpp"

namespace kora::vk {
namespace {

TEST(QueueContext, SelectsSynchronization2OnlyWhenEnabledOnLogicalDevice) {
    auto loader = Loader::load();
    if (!loader) GTEST_SKIP() << loader.error().describe();
    auto instance = loader.value().create_instance();
    if (!instance) GTEST_SKIP() << instance.error().describe();
    auto physical_devices = instance.value().physical_devices();
    ASSERT_TRUE(physical_devices) << physical_devices.error().describe();
    if (physical_devices.value().empty()) GTEST_SKIP() << "No Vulkan physical device is available";

    auto preferred_device = instance.value().create_device(physical_devices.value().front());
    if (!preferred_device) GTEST_SKIP() << preferred_device.error().describe();
    auto preferred_context =
        QueueContext::create(preferred_device.value().api(), preferred_device.value().handle(),
                             preferred_device.value().compute_queue());
    ASSERT_TRUE(preferred_context) << preferred_context.error().describe();
    EXPECT_EQ(preferred_context.value()->uses_synchronization2(),
              preferred_device.value().enabled_features().synchronization2);

    DeviceOptions legacy_options;
    legacy_options.enable_synchronization2 = false;
    auto legacy_device = instance.value().create_device(physical_devices.value().front(), legacy_options);
    ASSERT_TRUE(legacy_device) << legacy_device.error().describe();
    EXPECT_FALSE(legacy_device.value().enabled_features().synchronization2);
    auto legacy_context = QueueContext::create(legacy_device.value().api(), legacy_device.value().handle(),
                                               legacy_device.value().compute_queue());
    ASSERT_TRUE(legacy_context) << legacy_context.error().describe();
    EXPECT_FALSE(legacy_context.value()->uses_synchronization2());
}

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

TEST(QueueContext, RepeatedCreateSubmitShutdownDrainsCompletionBeforeDestruction) {
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

    constexpr std::uint32_t kIterations = 32;
    std::atomic<std::uint32_t> completed{0};
    for (std::uint32_t iteration = 0; iteration < kIterations; ++iteration) {
        auto created = QueueContext::create(device.value().api(), device.value().handle(),
                                            device.value().compute_queue());
        ASSERT_TRUE(created) << created.error().describe();
        auto context = std::move(created).value();
        auto lifetime_token = std::make_shared<std::uint32_t>(iteration);
        QueueContext::Submission submission;
        submission.on_complete = [&, lifetime_token](std::uint64_t, Status status) {
            EXPECT_TRUE(status);
            EXPECT_EQ(*lifetime_token, iteration);
            ++completed;
        };
        ASSERT_TRUE(context->submit(std::move(submission)));
        ASSERT_TRUE(context->shutdown(ShutdownMode::Graceful));
        EXPECT_EQ(completed.load(), iteration + 1);
    }
    EXPECT_EQ(completed.load(), kIterations);
}

TEST(QueueContext, WaitsForTimelineValueAndOrdersDependentSubmissions) {
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

    VkSemaphoreTypeCreateInfo type_info{};
    type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    create_info.pNext = &type_info;
    VkSemaphore dependency = VK_NULL_HANDLE;
    ASSERT_EQ(
        device.value().api().create_semaphore(device.value().handle(), &create_info, nullptr, &dependency),
        VK_SUCCESS);
    auto context_result =
        QueueContext::create(device.value().api(), device.value().handle(), device.value().compute_queue());
    ASSERT_TRUE(context_result) << context_result.error().describe();
    auto context = std::move(context_result).value();

    std::atomic<std::uint32_t> callbacks{0};
    QueueContext::Submission first;
    first.signal_semaphores.push_back(dependency);
    first.signal_values.push_back(1);
    first.on_complete = [&](std::uint64_t, Status status) {
        EXPECT_TRUE(status);
        ++callbacks;
    };
    auto first_id = context->submit(std::move(first));
    ASSERT_TRUE(first_id) << first_id.error().describe();

    QueueContext::Submission second;
    second.wait_semaphores.push_back(dependency);
    second.wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    second.wait_values.push_back(1);
    second.signal_semaphores.push_back(dependency);
    second.signal_values.push_back(2);
    second.on_complete = [&](std::uint64_t, Status status) {
        EXPECT_TRUE(status);
        ++callbacks;
    };
    auto second_id = context->submit(std::move(second));
    ASSERT_TRUE(second_id) << second_id.error().describe();

    ASSERT_TRUE(context->wait_for_value(second_id.value(), 5'000'000'000));
    auto completed = context->completed_value();
    ASSERT_TRUE(completed) << completed.error().describe();
    EXPECT_GE(completed.value(), second_id.value());
    ASSERT_TRUE(context->shutdown(ShutdownMode::Graceful));
    EXPECT_EQ(callbacks.load(), 2U);
    device.value().api().destroy_semaphore(device.value().handle(), dependency, nullptr);
}

}  // namespace
}  // namespace kora::vk
