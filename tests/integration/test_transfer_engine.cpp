#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/device_caps.hpp"
#include "memory/memory_manager.hpp"
#include "memory/memory_pool.hpp"
#include "memory/transfer_engine.hpp"
#include "support/vulkan_test.hpp"
#include "vulkan/api.hpp"

namespace kora::detail {
namespace {

struct TestBuffer {
    vk::Api api;
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;

    ~TestBuffer() {
        if (buffer != VK_NULL_HANDLE) api.destroy_buffer(device, buffer, nullptr);
        if (memory != VK_NULL_HANDLE) api.free_memory(device, memory, nullptr);
    }
    TestBuffer(const TestBuffer&) = delete;
    TestBuffer& operator=(const TestBuffer&) = delete;
    TestBuffer(vk::Api api_value, VkDevice device_value, VkBuffer buffer_value, VkDeviceMemory memory_value,
               VkDeviceSize size_value)
        : api(std::move(api_value)),
          device(device_value),
          buffer(buffer_value),
          memory(memory_value),
          size(size_value) {}
};

Result<std::unique_ptr<TestBuffer>> make_buffer(const vk::Device& device,
                                                const vk::PhysicalDeviceInfo& physical,
                                                const DeviceInfo& info, VkDeviceSize size) {
    const auto& api = device.api();
    VkBufferCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_info.size = size;
    create_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    auto result = api.create_buffer(device.handle(), &create_info, nullptr, &buffer);
    if (result != VK_SUCCESS)
        return Error(ErrorCode::AllocationFailed, "transfer integration VkBuffer creation failed")
            .with_vk(result);
    VkMemoryRequirements requirements{};
    api.get_buffer_memory_requirements(device.handle(), buffer, &requirements);
    auto type = MemoryManager::select_memory_type(info.memory_types(), info.memory_type_preferences(),
                                                  MemoryPolicy::DevicePreferred, requirements.memoryTypeBits,
                                                  MemoryTypeRequirements{.device_local = true});
    if (!type) {
        api.destroy_buffer(device.handle(), buffer, nullptr);
        return std::move(type).error();
    }
    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = type.value();
    VkDeviceMemory memory = VK_NULL_HANDLE;
    result = api.allocate_memory(device.handle(), &allocation_info, nullptr, &memory);
    if (result != VK_SUCCESS) {
        api.destroy_buffer(device.handle(), buffer, nullptr);
        return Error(ErrorCode::AllocationFailed, "transfer integration VkDeviceMemory allocation failed")
            .with_vk(result);
    }
    result = api.bind_buffer_memory(device.handle(), buffer, memory, 0);
    if (result != VK_SUCCESS) {
        api.free_memory(device.handle(), memory, nullptr);
        api.destroy_buffer(device.handle(), buffer, nullptr);
        return Error(ErrorCode::AllocationFailed, "transfer integration VkBuffer binding failed")
            .with_vk(result);
    }
    static_cast<void>(physical);
    return std::make_unique<TestBuffer>(api, device.handle(), buffer, memory, size);
}

TEST(TransferEngine, RoundTripsOddOffsetsAndSizesThroughEveryLocalVulkanDevice) {
    auto loader = vk::Loader::load();
    if (!loader) KORA_REQUIRE_VULKAN(loader);
    auto instance = loader.value().create_instance();
    if (!instance && instance.error().code == ErrorCode::NoVulkanImplementation) KORA_REQUIRE_VULKAN(false);
    ASSERT_TRUE(instance) << instance.error().describe();
    auto physical_devices = instance.value().physical_devices();
    ASSERT_TRUE(physical_devices) << physical_devices.error().describe();
    if (physical_devices.value().empty()) GTEST_SKIP() << "No Vulkan physical devices are available";

    std::size_t tested_devices = 0;
    for (const auto& physical : physical_devices.value()) {
        auto device_result = instance.value().create_device(physical);
        if (!device_result) continue;
        auto device = std::move(device_result).value();
        const auto info = DeviceCaps::from_vulkan(physical).to_device_info();
        auto host_type = MemoryManager::select_memory_type(
            info.memory_types(), info.memory_type_preferences(), MemoryPolicy::HostVisible,
            std::numeric_limits<std::uint32_t>::max(), MemoryTypeRequirements{.host_visible = true});
        if (!host_type) continue;

        auto queue_result = vk::QueueContext::create(device.api(), device.handle(), device.transfer_queue());
        if (!queue_result) continue;
        auto queue = std::move(queue_result).value();
        const auto& selected_host_type = info.memory_types()[host_type.value()];
        MemoryPoolConfig pool_config;
        pool_config.memory_type_index = host_type.value();
        pool_config.host_visible = true;
        pool_config.host_coherent = selected_host_type.host_coherent;
        pool_config.heap_size = info.memory_heaps()[selected_host_type.heap_index].size_bytes;
        pool_config.budget = pool_config.heap_size;
        pool_config.block_size_hint = 8U * 1024U * 1024U;
        auto upload_pool_result = MemoryPool::create(device, pool_config);
        auto readback_pool_result = MemoryPool::create(device, pool_config);
        if (!upload_pool_result || !readback_pool_result) continue;

        TransferEngine::Options transfer_options;
        transfer_options.queue_family = device.transfer_queue_family();
        transfer_options.command_buffer_count = 8;
        transfer_options.staging_initial_capacity = 64U * 1024U * 1024U;
        transfer_options.staging_maximum_capacity = 128U * 1024U * 1024U;
        // The engine reserves readback slices until completion callbacks run; keep this
        // deliberately serial test under the default 128 MiB bounded-capacity contract.
        transfer_options.topology =
            MemoryTopology::Discrete;  // Exercise the staged transfer path on UMA too.
        transfer_options.non_coherent_atom_size = physical.properties.limits.nonCoherentAtomSize;
        auto engine_result = TransferEngine::create(device, *queue, *upload_pool_result.value(),
                                                    *readback_pool_result.value(), transfer_options);
        if (!engine_result) continue;
        auto engine = std::move(engine_result).value();

        constexpr VkDeviceSize kLargest = 64U * 1024U * 1024U;
        auto first_result = make_buffer(device, physical, info, kLargest + 32);
        auto second_result = make_buffer(device, physical, info, kLargest + 32);
        if (!first_result || !second_result) continue;
        auto first = std::move(first_result).value();
        auto second = std::move(second_result).value();
        TransferBuffer first_view{.buffer = first->buffer, .size = first->size};
        TransferBuffer second_view{.buffer = second->buffer, .size = second->size};

        const std::array<std::size_t, 10> sizes{1,
                                                2,
                                                3,
                                                5,
                                                17,
                                                4093,
                                                65539,
                                                1024U * 1024U + 3,
                                                8U * 1024U * 1024U + 1,
                                                static_cast<std::size_t>(kLargest)};
        std::mt19937 random(
            static_cast<std::uint32_t>(physical.properties.vendorID ^ physical.properties.deviceID));
        for (const auto size : sizes) {
            std::vector<std::byte> input(size);
            for (auto& byte : input) byte = static_cast<std::byte>(random() & 0xffU);
            auto uploaded = engine->upload(first_view, 3, input);
            ASSERT_TRUE(uploaded) << "upload size=" << size << ": " << uploaded.error().describe();
            ASSERT_TRUE(uploaded.value().wait()) << uploaded.value().status().error().describe();

            std::vector<std::byte> output(size);
            auto downloaded = engine->download(first_view, 3, output);
            ASSERT_TRUE(downloaded) << downloaded.error().describe();
            ASSERT_TRUE(downloaded.value().wait()) << downloaded.value().status().error().describe();
            EXPECT_EQ(output, input) << "size=" << size;

            auto copied = engine->copy(first_view, 3, second_view, 5, size);
            ASSERT_TRUE(copied) << copied.error().describe();
            ASSERT_TRUE(copied.value().wait()) << copied.value().status().error().describe();
            std::vector<std::byte> copy_output(size);
            auto copy_download = engine->download(second_view, 5, copy_output);
            ASSERT_TRUE(copy_download) << copy_download.error().describe();
            ASSERT_TRUE(copy_download.value().wait()) << copy_download.value().status().error().describe();
            EXPECT_EQ(copy_output, input) << "copied size=" << size;
        }

        auto filled = engine->fill(second_view, 0, 16, 0x12345678U);
        ASSERT_TRUE(filled) << filled.error().describe();
        ASSERT_TRUE(filled.value().wait()) << filled.value().status().error().describe();
        std::array<std::byte, 16> fill_output{};
        auto fill_download = engine->download(second_view, 0, fill_output);
        ASSERT_TRUE(fill_download) << fill_download.error().describe();
        ASSERT_TRUE(fill_download.value().wait()) << fill_download.value().status().error().describe();
        std::array<std::uint32_t, 4> expected_fill{0x12345678U, 0x12345678U, 0x12345678U, 0x12345678U};
        EXPECT_EQ(std::memcmp(fill_output.data(), expected_fill.data(), fill_output.size()), 0);
        ASSERT_TRUE(queue->shutdown());
        ++tested_devices;
    }
    EXPECT_GE(tested_devices, 1U);
}

}  // namespace
}  // namespace kora::detail
