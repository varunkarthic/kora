#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

#include "vulkan/api.hpp"

namespace {

struct FakeVulkan {
    int allocate_calls = 0;
    int submit_calls = 0;
    VkResult allocate_result = VK_SUCCESS;
    VkResult submit_result = VK_SUCCESS;
};

template <class Handle>
Handle fake_handle(uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<Handle>(value);
    else
        return static_cast<Handle>(value);
}

VkResult VKAPI_PTR allocate_memory(void* context, VkDevice, const VkMemoryAllocateInfo*,
                                   const VkAllocationCallbacks*, VkDeviceMemory* memory) noexcept {
    auto& fake = *static_cast<FakeVulkan*>(context);
    ++fake.allocate_calls;
    if (fake.allocate_result == VK_SUCCESS) *memory = fake_handle<VkDeviceMemory>(7);
    return fake.allocate_result;
}

VkResult VKAPI_PTR queue_submit(void* context, VkQueue, uint32_t, const VkSubmitInfo*, VkFence) noexcept {
    auto& fake = *static_cast<FakeVulkan*>(context);
    ++fake.submit_calls;
    return fake.submit_result;
}

kora::vk::Api fake_api(FakeVulkan& fake) {
    kora::vk::Api api;
    api.allocate_memory = {&fake, allocate_memory};
    api.queue_submit = {&fake, queue_submit};
    return api;
}

}  // namespace

TEST(VulkanFaultInjection, FailsNthAllocateMemoryCallAndThenForwards) {
    FakeVulkan fake;
    kora::vk::FaultInjector injector(fake_api(fake));
    ASSERT_TRUE(injector.fail_on_nth(kora::vk::Function::AllocateMemory, 2, VK_ERROR_OUT_OF_DEVICE_MEMORY));
    const auto api = injector.api();
    VkDeviceMemory memory = VK_NULL_HANDLE;

    EXPECT_EQ(api.allocate_memory(VK_NULL_HANDLE, nullptr, nullptr, &memory), VK_SUCCESS);
    EXPECT_EQ(memory, fake_handle<VkDeviceMemory>(7));
    EXPECT_EQ(api.allocate_memory(VK_NULL_HANDLE, nullptr, nullptr, &memory), VK_ERROR_OUT_OF_DEVICE_MEMORY);
    EXPECT_EQ(api.allocate_memory(VK_NULL_HANDLE, nullptr, nullptr, &memory), VK_SUCCESS);
    EXPECT_EQ(fake.allocate_calls, 2);
}

TEST(VulkanFaultInjection, FailsQueueSubmitAndPreservesForwardedResults) {
    FakeVulkan fake;
    fake.submit_result = VK_ERROR_DEVICE_LOST;
    kora::vk::FaultInjector injector(fake_api(fake));
    ASSERT_TRUE(injector.fail_on_nth(kora::vk::Function::QueueSubmit, 1, VK_ERROR_OUT_OF_HOST_MEMORY));
    const auto api = injector.api();

    EXPECT_EQ(api.queue_submit(VK_NULL_HANDLE, 0, nullptr, VK_NULL_HANDLE), VK_ERROR_OUT_OF_HOST_MEMORY);
    EXPECT_EQ(api.queue_submit(VK_NULL_HANDLE, 0, nullptr, VK_NULL_HANDLE), VK_ERROR_DEVICE_LOST);
    EXPECT_EQ(fake.submit_calls, 1);
}

TEST(VulkanFaultInjection, RejectsInvalidFailureConfiguration) {
    FakeVulkan fake;
    kora::vk::FaultInjector injector(fake_api(fake));

    EXPECT_FALSE(injector.fail_on_nth(kora::vk::Function::QueueSubmit, 0, VK_ERROR_DEVICE_LOST));
    EXPECT_FALSE(injector.fail_on_nth(kora::vk::Function::QueueSubmit, 1, VK_SUCCESS));
}

TEST(VulkanFaultInjection, ReturnedTableKeepsDecoratorStateAlive) {
    FakeVulkan fake;
    kora::vk::Api api;
    {
        kora::vk::FaultInjector injector(fake_api(fake));
        api = injector.api();
    }

    VkDeviceMemory memory = VK_NULL_HANDLE;
    EXPECT_EQ(api.allocate_memory(VK_NULL_HANDLE, nullptr, nullptr, &memory), VK_SUCCESS);
    EXPECT_EQ(fake.allocate_calls, 1);
}

TEST(VulkanLoader, LoadsDynamicallyOrReportsMissingImplementation) {
    auto loader = kora::vk::Loader::load();
    if (!loader) {
        EXPECT_EQ(loader.error().code, kora::ErrorCode::NoVulkanImplementation);
        return;
    }

    auto api = loader.value().device_api(VK_NULL_HANDLE, VK_NULL_HANDLE);
    ASSERT_FALSE(api);
    EXPECT_EQ(api.error().code, kora::ErrorCode::InvalidArgument);
}
