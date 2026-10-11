#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "kora/runtime.hpp"
#include "support/vulkan_test.hpp"

TEST(Runtime, OwnsSelectedDeviceAndExposesMemoryStats) {
    auto runtime = kora::Runtime::create();
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    const auto devices = runtime.value().devices();
    ASSERT_FALSE(devices.empty());

    auto info = runtime.value().device_info(devices.front());
    ASSERT_TRUE(info) << info.error().describe();
    EXPECT_FALSE(info.value().name().empty());
    EXPECT_FALSE(info.value().memory_heaps().empty());
    if (info.value().api_version().major > 1 || info.value().api_version().minor >= 1) {
        EXPECT_EQ(devices.front().uuid.size(), 32U);
        kora::RuntimeOptions by_uuid;
        by_uuid.device_policy = kora::DevicePolicy::ByUuid;
        by_uuid.device_uuid = devices.front().uuid;
        auto uuid_runtime = kora::Runtime::create(by_uuid);
        ASSERT_TRUE(uuid_runtime) << uuid_runtime.error().describe();
        const auto uuid_devices = uuid_runtime.value().devices();
        ASSERT_EQ(uuid_devices.size(), 1U);
        EXPECT_EQ(uuid_devices.front().uuid, devices.front().uuid);
    }

    auto stats = runtime.value().memory_stats(devices.front());
    ASSERT_TRUE(stats) << stats.error().describe();
    EXPECT_GT(stats.value().pools, 0U);
    EXPECT_GT(stats.value().blocks, 0U);
    EXPECT_GE(stats.value().live_allocations, 2U);  // Upload and readback staging buffers.
    ASSERT_EQ(stats.value().heaps.size(), info.value().memory_heaps().size());
    for (std::size_t index = 0; index < stats.value().heaps.size(); ++index) {
        EXPECT_EQ(stats.value().heaps[index].heap_index, index);
        EXPECT_LE(stats.value().heaps[index].reserved_bytes, stats.value().heaps[index].budget_bytes);
        EXPECT_EQ(stats.value().heaps[index].available_bytes,
                  stats.value().heaps[index].budget_bytes - stats.value().heaps[index].reserved_bytes);
    }

    EXPECT_EQ(runtime.value().memory_stats(kora::DeviceId{UINT32_MAX, {}}).error().code,
              kora::ErrorCode::InvalidArgument);
}

TEST(Runtime, HonorsExplicitDeviceIndexSelection) {
    kora::RuntimeOptions options;
    options.device_policy = kora::DevicePolicy::ByIndex;
    options.device_index = 0;
    auto runtime = kora::Runtime::create(options);
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    const auto devices = runtime.value().devices();
    ASSERT_EQ(devices.size(), 1U);
    EXPECT_EQ(devices.front().index, 0U);
}

TEST(Runtime, DispatchesVectorAddOnEveryEnumeratedDevice) {
#ifndef KORA_VECTOR_ADD_FIXTURE_PATH
    GTEST_SKIP() << "Install glslangValidator to compile the vector-add fixture";
#else
    auto base_result = kora::Runtime::create();
    if (!base_result && (base_result.error().code == kora::ErrorCode::NoVulkanImplementation ||
                         base_result.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(base_result) << base_result.error().describe();
    auto base = std::move(base_result).value();
    const auto devices = base.devices();
    ASSERT_FALSE(devices.empty());
    base = kora::Runtime{};

    constexpr std::uint32_t count = 257;
    std::vector<std::uint32_t> a(count), b(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        a[index] = index * 3U;
        b[index] = index + 11U;
    }
    for (const auto& device : devices) {
        kora::RuntimeOptions options;
        options.device_policy = kora::DevicePolicy::ByIndex;
        options.device_index = device.index;
        options.cache_dir = (std::filesystem::temp_directory_path() / "kora-dispatch-test-cache").string();
        auto runtime = kora::Runtime::create(options);
        if (!runtime && runtime.error().code == kora::ErrorCode::UnsupportedFeature)
            GTEST_SKIP() << "device " << device.index
                         << " lacks runtime transfer support: " << runtime.error().describe();
        ASSERT_TRUE(runtime) << "device " << device.index << ": " << runtime.error().describe();
        kora::KernelMetadata metadata = [] {
            kora::KernelMetadata metadata;
            metadata.kernel_id = "integration.vector_add";
            metadata.local_size = {64, 1, 1};
            metadata.push_constant_size = sizeof(std::uint32_t);
            metadata.push_constants = {{"count", "u32", 0, sizeof(std::uint32_t)}};
            metadata.bindings = {
                {.set = 0,
                 .binding = 0,
                 .name = "input_a",
                 .kind = kora::KernelBindingKind::StorageBuffer,
                 .access = kora::KernelAccess::Read},
                {.set = 0,
                 .binding = 1,
                 .name = "input_b",
                 .kind = kora::KernelBindingKind::StorageBuffer,
                 .access = kora::KernelAccess::Read},
                {.set = 0,
                 .binding = 2,
                 .name = "output_data",
                 .kind = kora::KernelBindingKind::StorageBuffer,
                 .access = kora::KernelAccess::Write},
            };
            return metadata;
        }();
        auto kernel = runtime.value().load_kernel(KORA_VECTOR_ADD_FIXTURE_PATH, "main", std::move(metadata));
        ASSERT_TRUE(kernel) << kernel.error().describe();
        EXPECT_EQ(kernel.value().entry_point(), "main");
        EXPECT_EQ(kernel.value().metadata().kernel_id, "integration.vector_add");
        auto input_a = runtime.value().create_buffer<std::uint32_t>(count);
        auto input_b = runtime.value().create_buffer<std::uint32_t>(count);
        auto output = runtime.value().create_buffer<std::uint32_t>(count);
        ASSERT_TRUE(input_a) << input_a.error().describe();
        ASSERT_TRUE(input_b) << input_b.error().describe();
        ASSERT_TRUE(output) << output.error().describe();
        ASSERT_TRUE(input_a.value().write(a));
        ASSERT_TRUE(input_b.value().write(b));
        const auto byte_count = count;
        auto event = runtime.value()
                         .dispatch(kernel.value())
                         .global(count)
                         .read(input_a.value())
                         .read(input_b.value())
                         .write(output.value())
                         .push(byte_count)
                         .submit();
        ASSERT_TRUE(event) << event.error().describe();
        auto completed = event.value().wait();
        ASSERT_TRUE(completed) << completed.error().describe();
        auto values = output.value().read();
        ASSERT_TRUE(values) << values.error().describe();
        for (std::uint32_t index = 0; index < count; ++index)
            EXPECT_EQ(values.value()[index], a[index] + b[index])
                << "device " << device.index << ", index " << index;

        auto invalid_dimensions = runtime.value()
                                      .dispatch(kernel.value())
                                      .global(0)
                                      .read(input_a.value())
                                      .read(input_b.value())
                                      .write(output.value())
                                      .push(byte_count)
                                      .submit();
        ASSERT_FALSE(invalid_dimensions);
        EXPECT_EQ(invalid_dimensions.error().code, kora::ErrorCode::InvalidDispatch);
        auto invalid_access = runtime.value()
                                  .dispatch(kernel.value())
                                  .global(count)
                                  .write(input_a.value())
                                  .read(input_b.value())
                                  .write(output.value())
                                  .push(byte_count)
                                  .submit();
        ASSERT_FALSE(invalid_access);
        EXPECT_EQ(invalid_access.error().code, kora::ErrorCode::InvalidKernelBinding);
    }
#endif
}
