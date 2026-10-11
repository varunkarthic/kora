#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "vulkan/api.hpp"
#include "vulkan/command_pool_set.hpp"

namespace kora::vk {
namespace {

TEST(CommandPoolSet, RepeatsTenThousandRecordSubmitAndReuseCycles) {
    auto loader = Loader::load();
    if (!loader) GTEST_SKIP() << loader.error().describe();
    auto instance = loader.value().create_instance();
    if (!instance) GTEST_SKIP() << instance.error().describe();
    auto physical_devices = instance.value().physical_devices();
    ASSERT_TRUE(physical_devices) << physical_devices.error().describe();
    if (physical_devices.value().empty()) GTEST_SKIP() << "No Vulkan physical device is available";
    auto device = instance.value().create_device(physical_devices.value().front());
    if (!device) GTEST_SKIP() << device.error().describe();

    CommandPoolSet::Options options;
    options.queue_family = device.value().compute_queue_family();
    options.ring_size = 16;
    CommandPoolSet pools(device.value().api(), device.value().handle(), options);
    ASSERT_EQ(pools.capacity(), options.ring_size);
    VkSemaphoreTypeCreateInfo type_info{};
    type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_info.pNext = &type_info;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    const auto& api = device.value().api();
    auto create_semaphore = api.create_semaphore;
    auto destroy_semaphore = api.destroy_semaphore;
    auto get_counter = api.get_semaphore_counter_value;
    auto wait_semaphores = api.wait_semaphores;
    std::vector<VkCommandBuffer> pending;
    pending.reserve(options.ring_size);
    if (!create_semaphore || !destroy_semaphore || !get_counter || !wait_semaphores || !api.queue_submit ||
        !device.value().enabled_features().timeline_semaphore ||
        !device.value().enabled_features().synchronization2) {
        GTEST_SKIP() << "Timeline semaphore and synchronization2 are unavailable";
    }
    ASSERT_EQ(create_semaphore(device.value().handle(), &semaphore_info, nullptr, &semaphore), VK_SUCCESS);

    for (uint64_t value = 1; value <= 10000; ++value) {
        uint64_t completed = 0;
        ASSERT_EQ(get_counter(device.value().handle(), semaphore, &completed), VK_SUCCESS);
        auto command_buffer = pools.acquire(completed);
        if (!command_buffer) {
            VkSemaphoreWaitInfo wait_info{};
            wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            wait_info.semaphoreCount = 1;
            wait_info.pSemaphores = &semaphore;
            const uint64_t target = value - options.ring_size;
            wait_info.pValues = &target;
            ASSERT_EQ(wait_semaphores(device.value().handle(), &wait_info, UINT64_MAX), VK_SUCCESS);
            ASSERT_EQ(get_counter(device.value().handle(), semaphore, &completed), VK_SUCCESS);
            command_buffer = pools.acquire(completed);
        }
        ASSERT_TRUE(command_buffer) << command_buffer.error().describe();
        ASSERT_TRUE(pools.end_recording(command_buffer.value()));
        pending.push_back(command_buffer.value());
        if (pending.size() == options.ring_size) {
            VkTimelineSemaphoreSubmitInfo timeline_info{};
            timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_info.signalSemaphoreValueCount = 1;
            timeline_info.pSignalSemaphoreValues = &value;
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.pNext = &timeline_info;
            submit_info.commandBufferCount = static_cast<uint32_t>(pending.size());
            submit_info.pCommandBuffers = pending.data();
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = &semaphore;
            ASSERT_EQ(api.queue_submit(device.value().compute_queue(), 1, &submit_info, VK_NULL_HANDLE),
                      VK_SUCCESS);
            const uint64_t first_value = value - pending.size() + 1;
            for (size_t i = 0; i < pending.size(); ++i) {
                ASSERT_TRUE(pools.submitted(pending[i], first_value + i));
            }
            pending.clear();
        }
    }
    if (!pending.empty()) {
        const uint64_t value = 10000;
        VkTimelineSemaphoreSubmitInfo timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &value;
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pNext = &timeline_info;
        submit_info.commandBufferCount = static_cast<uint32_t>(pending.size());
        submit_info.pCommandBuffers = pending.data();
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &semaphore;
        ASSERT_EQ(api.queue_submit(device.value().compute_queue(), 1, &submit_info, VK_NULL_HANDLE),
                  VK_SUCCESS);
        const uint64_t first_value = 10000 - pending.size() + 1;
        for (size_t i = 0; i < pending.size(); ++i) {
            ASSERT_TRUE(pools.submitted(pending[i], first_value + i));
        }
    }
    VkSemaphoreWaitInfo final_wait{};
    final_wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    final_wait.semaphoreCount = 1;
    final_wait.pSemaphores = &semaphore;
    const uint64_t final_value = 10000;
    final_wait.pValues = &final_value;
    ASSERT_EQ(wait_semaphores(device.value().handle(), &final_wait, UINT64_MAX), VK_SUCCESS);
    EXPECT_EQ(pools.allocation_count(), options.ring_size);
    destroy_semaphore(device.value().handle(), semaphore, nullptr);
}

}  // namespace
}  // namespace kora::vk
