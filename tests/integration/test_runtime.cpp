#include <gtest/gtest.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/runtime_test_access.hpp"
#include "kora/runtime.hpp"
#include "kora/task_partition.hpp"
#include "support/vulkan_test.hpp"

namespace {

kora::KernelMetadata vector_add_metadata() {
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
}

kora::KernelMetadata partition_add_metadata() {
    auto metadata = vector_add_metadata();
    metadata.kernel_id = "integration.partition_add";
    metadata.push_constant_size = 2 * sizeof(std::uint32_t);
    metadata.push_constants = {{"base", "u32", 0, sizeof(std::uint32_t)},
                               {"count", "u32", sizeof(std::uint32_t), sizeof(std::uint32_t)}};
    return metadata;
}

}  // namespace

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
    EXPECT_EQ(runtime.value().recover().error().code, kora::ErrorCode::InvalidArgument);
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

TEST(Runtime, RecoversTransactionallyAndRetainsAuthoritativeBufferHandles) {
    auto created = kora::Runtime::create();
    if (!created && (created.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     created.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(created) << created.error().describe();
    auto runtime = std::move(created).value();
    const auto before_devices = runtime.devices();
    ASSERT_FALSE(before_devices.empty());
    auto buffer_result = runtime.create_buffer(4);
    ASSERT_TRUE(buffer_result) << buffer_result.error().describe();
    auto buffer = std::move(buffer_result).value();
    const std::array<std::byte, 4> original{std::byte{0x10}, std::byte{0x20}, std::byte{0x30},
                                            std::byte{0x40}};
    ASSERT_TRUE(buffer.write_bytes(original));
    ASSERT_TRUE(kora::detail::RuntimeTestAccess::mark_device_lost(runtime, before_devices.front()));
    auto report = runtime.device_fault_report(before_devices.front());
    ASSERT_TRUE(report) << report.error().describe();
    EXPECT_EQ(report.value().loss_result, VK_ERROR_DEVICE_LOST);

    ASSERT_TRUE(runtime.recover());
    EXPECT_EQ(runtime.devices(), before_devices);
    const auto restored = buffer.read_bytes();
    ASSERT_TRUE(restored) << restored.error().describe();
    EXPECT_EQ(restored.value(), std::vector<std::byte>(original.begin(), original.end()));

    auto subsequent_buffer = runtime.create_buffer(8);
    ASSERT_TRUE(subsequent_buffer) << subsequent_buffer.error().describe();
}

TEST(Runtime, StagesOutputsAcrossEverySelectedDevicePairThroughHostMemory) {
#ifndef KORA_VECTOR_ADD_FIXTURE_PATH
    GTEST_SKIP() << "vector-add SPIR-V fixture was not built";
#else
    kora::RuntimeOptions options;
    options.device_policy = kora::DevicePolicy::AllEligible;
    options.enable_startup_calibration = false;
    auto runtime = kora::Runtime::create(options);
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    const auto devices = runtime.value().devices();
    if (devices.size() < 2) GTEST_SKIP() << "cross-device staging requires two eligible Vulkan devices";

    constexpr std::uint32_t count = 257;
    auto input_a = runtime.value().create_buffer<std::uint32_t>(count);
    auto input_b = runtime.value().create_buffer<std::uint32_t>(count);
    auto intermediate = runtime.value().create_buffer<std::uint32_t>(count);
    auto output = runtime.value().create_buffer<std::uint32_t>(count);
    ASSERT_TRUE(input_a);
    ASSERT_TRUE(input_b);
    ASSERT_TRUE(intermediate);
    ASSERT_TRUE(output);
    std::vector<std::uint32_t> values_a(count);
    std::vector<std::uint32_t> values_b(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        values_a[index] = index * 3U;
        values_b[index] = index + 11U;
    }
    ASSERT_TRUE(input_a.value().write(values_a));
    ASSERT_TRUE(input_b.value().write(values_b));

    for (const auto& source : devices) {
        auto source_kernel =
            runtime.value().load_kernel(KORA_VECTOR_ADD_FIXTURE_PATH, "main", vector_add_metadata(), source);
        ASSERT_TRUE(source_kernel) << "source device " << source.index << ": "
                                   << source_kernel.error().describe();
        auto produced = runtime.value()
                            .dispatch(source_kernel.value())
                            .global(count)
                            .read(input_a.value())
                            .read(input_b.value())
                            .write(intermediate.value())
                            .push(count)
                            .defer_readback()
                            .submit();
        ASSERT_TRUE(produced) << produced.error().describe();
        ASSERT_TRUE(produced.value().wait()) << produced.value().status().error().describe();

        for (const auto& destination : devices) {
            if (destination == source) continue;
            auto staged = runtime.value().readback(intermediate.value().handle(), source);
            ASSERT_TRUE(staged) << staged.error().describe();
            ASSERT_TRUE(staged.value().wait()) << staged.value().status().error().describe();

            auto destination_kernel = runtime.value().load_kernel(KORA_VECTOR_ADD_FIXTURE_PATH, "main",
                                                                  vector_add_metadata(), destination);
            ASSERT_TRUE(destination_kernel) << "destination device " << destination.index << ": "
                                            << destination_kernel.error().describe();
            auto consumed = runtime.value()
                                .dispatch(destination_kernel.value())
                                .global(count)
                                .read(intermediate.value())
                                .read(intermediate.value())
                                .write(output.value())
                                .push(count)
                                .submit();
            ASSERT_TRUE(consumed) << consumed.error().describe();
            ASSERT_TRUE(consumed.value().wait()) << consumed.value().status().error().describe();
            auto actual = output.value().read();
            ASSERT_TRUE(actual) << actual.error().describe();
            for (std::uint32_t index = 0; index < count; ++index) {
                const auto source_value = values_a[index] + values_b[index];
                EXPECT_EQ(actual.value()[index], source_value * 2U)
                    << "source device " << source.index << ", destination device " << destination.index
                    << ", index " << index;
            }
        }
    }
    const auto transfers = runtime.value().transfer_stats();
    ASSERT_TRUE(transfers) << transfers.error().describe();
    EXPECT_GT(transfers.value().uploads, 0U);
    EXPECT_GE(transfers.value().downloads, devices.size() * devices.size());
#endif
}

TEST(Runtime, DynamicPartitionMatchesUnsplitResultAcrossTwoLocalDevices) {
#ifndef KORA_PARTITION_ADD_FIXTURE_PATH
    GTEST_SKIP() << "partition-add SPIR-V fixture was not built";
#else
    kora::RuntimeOptions options;
    options.device_policy = kora::DevicePolicy::AllEligible;
    options.enable_startup_calibration = false;
    auto runtime = kora::Runtime::create(options);
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    const auto devices = runtime.value().devices();
    if (devices.size() < 2) GTEST_SKIP() << "dynamic partition integration needs two eligible Vulkan devices";

    constexpr std::uint32_t count = 1024;
    std::vector<std::uint32_t> a(count), b(count), expected(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        a[i] = i * 5U + 7U;
        b[i] = i * 3U + 13U;
        expected[i] = a[i] + b[i];
    }
    auto input_a = runtime.value().create_buffer<std::uint32_t>(count);
    auto input_b = runtime.value().create_buffer<std::uint32_t>(count);
    ASSERT_TRUE(input_a);
    ASSERT_TRUE(input_b);
    ASSERT_TRUE(input_a.value().write(a));
    ASSERT_TRUE(input_b.value().write(b));

    kora::DynamicPartitioner partitioner(1.0);
    std::vector<kora::PartitionDevice> candidates;
    for (const auto& device : devices) {
        auto info = runtime.value().device_info(device);
        auto stats = runtime.value().memory_stats(device);
        ASSERT_TRUE(info);
        ASSERT_TRUE(stats);
        std::uint64_t memory = 0;
        for (const auto& heap : stats.value().heaps) memory = std::max(memory, heap.available_bytes);
        candidates.push_back({device, memory, 1.0});
        ASSERT_TRUE(partitioner.observe(device, 256, 100.0 + device.index));
    }
    kora::PartitionRequest request{.total_work_units = count,
                                   .bytes_per_work_unit = 12,
                                   .granularity = 64,
                                   .minimum_chunk_units = 64,
                                   .estimated_unsplit_duration_ns = 100000,
                                   .partition_overhead_ns = 0,
                                   .minimum_gain_fraction = 0};
    auto plan = partitioner.partition(request, candidates);
    ASSERT_TRUE(plan) << plan.error().describe();
    ASSERT_TRUE(plan.value().split);
    ASSERT_EQ(plan.value().slices.size(), candidates.size());

    std::vector<std::uint32_t> actual(count, 0);
    for (const auto& slice : plan.value().slices) {
        const auto local_count = static_cast<std::uint32_t>(slice.work_units);
        const auto base = static_cast<std::uint32_t>(slice.offset);
        auto output = runtime.value().create_buffer<std::uint32_t>(local_count);
        ASSERT_TRUE(output);
        auto kernel = runtime.value().load_kernel(KORA_PARTITION_ADD_FIXTURE_PATH, "main",
                                                  partition_add_metadata(), slice.device);
        ASSERT_TRUE(kernel) << kernel.error().describe();
        const std::array<std::uint32_t, 2> push{base, local_count};
        auto event = runtime.value()
                         .dispatch(kernel.value())
                         .global(local_count)
                         .read(input_a.value())
                         .read(input_b.value())
                         .write(output.value())
                         .push(push)
                         .submit();
        ASSERT_TRUE(event) << event.error().describe();
        ASSERT_TRUE(event.value().wait()) << event.value().status().error().describe();
        auto values = output.value().read();
        ASSERT_TRUE(values);
        std::copy(values.value().begin(), values.value().end(), actual.begin() + base);
    }
    EXPECT_EQ(actual, expected);
#endif
}

TEST(Runtime, CorruptCalibrationCacheNeverBlocksStartup) {
    const auto directory = std::filesystem::temp_directory_path() / "kora-calibration-runtime-test";
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    std::filesystem::create_directories(directory / "calibration", ignored);
    {
        std::ofstream corrupt(directory / "calibration" / "00112233445566778899aabbccddeeff.json");
        corrupt << "corrupt";
    }
    kora::RuntimeOptions options;
    options.cache_dir = directory.string();
    options.calibration_budget_ms = 50;
    auto runtime = kora::Runtime::create(options);
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        std::filesystem::remove_all(directory, ignored);
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    std::filesystem::remove_all(directory, ignored);
}

TEST(Runtime, CorruptPerformanceDatabaseNeverBlocksStartup) {
    const auto directory = std::filesystem::temp_directory_path() / "kora-performance-db-runtime-test";
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    std::filesystem::create_directories(directory, ignored);
    {
        std::ofstream corrupt(directory / "performance-db.json", std::ios::binary);
        corrupt << "{corrupt";
    }
    kora::RuntimeOptions options;
    options.cache_dir = directory.string();
    auto runtime = kora::Runtime::create(options);
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        std::filesystem::remove_all(directory, ignored);
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    std::filesystem::remove_all(directory, ignored);
}

TEST(Runtime, TunesLocalSizeIdDispatchesWithQueriedDeviceLimits) {
#ifndef KORA_AUTOTUNE_FIXTURE_PATH
    GTEST_SKIP() << "Install SPIRV-Tools to assemble the LocalSizeId fixture";
#else
    const auto directory = std::filesystem::temp_directory_path() / "kora-autotune-runtime-test";
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    kora::RuntimeOptions options;
    options.cache_dir = directory.string();
    options.enable_startup_calibration = false;
    auto runtime = kora::Runtime::create(options);
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        std::filesystem::remove_all(directory, ignored);
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();

    kora::KernelMetadata metadata;
    metadata.kernel_id = "integration.autotune_increment";
    metadata.local_size = {1, 1, 1};
    metadata.push_constant_size = sizeof(std::uint32_t);
    metadata.push_constants = {{"count", "u32", 0, sizeof(std::uint32_t)}};
    metadata.bindings = {{.set = 0,
                          .binding = 0,
                          .name = "data_buffer",
                          .kind = kora::KernelBindingKind::StorageBuffer,
                          .access = kora::KernelAccess::ReadWrite}};
    metadata.specialization_constants = {{7, "local_size_x", "u32", std::uint64_t{1}},
                                         {8, "local_size_y", "u32", std::uint64_t{1}},
                                         {9, "local_size_z", "u32", std::uint64_t{1}}};
    auto kernel = runtime.value().load_kernel(KORA_AUTOTUNE_FIXTURE_PATH, "main", std::move(metadata));
    if (!kernel && kernel.error().code == kora::ErrorCode::UnsupportedFeature) {
        std::filesystem::remove_all(directory, ignored);
        GTEST_SKIP() << "Selected Vulkan device does not validate LocalSizeId: " << kernel.error().describe();
    }
    ASSERT_TRUE(kernel) << kernel.error().describe();
    constexpr std::uint32_t count = 257;
    auto values = runtime.value().create_buffer<std::uint32_t>(count);
    ASSERT_TRUE(values) << values.error().describe();
    std::vector<std::uint32_t> initial(count, 0);
    ASSERT_TRUE(values.value().write(initial));
    for (std::uint32_t iteration = 0; iteration < 16; ++iteration) {
        auto event = runtime.value()
                         .dispatch(kernel.value())
                         .global(count)
                         .read_write(values.value())
                         .push(count)
                         .submit();
        ASSERT_TRUE(event) << event.error().describe();
        const auto completed = event.value().wait();
        ASSERT_TRUE(completed) << completed.error().describe();
        const auto profiling = runtime.value().profiling_stats(runtime.value().devices().front());
        if (!profiling && profiling.error().code == kora::ErrorCode::UnsupportedFeature) {
            std::filesystem::remove_all(directory, ignored);
            GTEST_SKIP() << "Selected Vulkan queue does not support timestamp profiling";
        }
        ASSERT_TRUE(profiling) << profiling.error().describe();
    }
    auto result = values.value().read();
    ASSERT_TRUE(result) << result.error().describe();
    for (const auto value : result.value()) EXPECT_EQ(value, 16U);
    std::filesystem::remove_all(directory, ignored);
#endif
}

TEST(Runtime, DispatchesVectorAddOnEveryDeviceWithTimelineAndFenceFallback) {
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
        for (const bool force_fallback : {false, true}) {
            kora::RuntimeOptions options;
            options.device_policy = kora::DevicePolicy::ByIndex;
            options.device_index = device.index;
            options.force_legacy_sync = force_fallback;
            options.cache_dir =
                (std::filesystem::temp_directory_path() / "kora-dispatch-test-cache").string();
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
            auto kernel =
                runtime.value().load_kernel(KORA_VECTOR_ADD_FIXTURE_PATH, "main", std::move(metadata));
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
            auto profiling = runtime.value().profiling_stats(device);
            if (profiling) {
                EXPECT_GT(profiling.value().samples_collected, 0U);
                EXPECT_GT(profiling.value().last_duration_nanoseconds, 0U);
            } else {
                EXPECT_EQ(profiling.error().code, kora::ErrorCode::UnsupportedFeature);
            }
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
    }
#endif
}

TEST(Runtime, ReusesDeviceResidentIntermediateAcrossDispatchChain) {
#ifndef KORA_VECTOR_ADD_FIXTURE_PATH
    GTEST_SKIP() << "Install glslangValidator to compile the vector-add fixture";
#else
    auto runtime = kora::Runtime::create();
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();

    constexpr std::uint32_t count = 257;
    std::vector<std::uint32_t> input(count), zero(count, 0);
    for (std::uint32_t index = 0; index < count; ++index) input[index] = index * 5U + 3U;
    kora::KernelMetadata metadata;
    metadata.kernel_id = "integration.vector_add.residency_chain";
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
    auto kernel = runtime.value().load_kernel(KORA_VECTOR_ADD_FIXTURE_PATH, "main", metadata);
    ASSERT_TRUE(kernel) << kernel.error().describe();
    auto input_buffer = runtime.value().create_buffer<std::uint32_t>(count);
    auto zero_buffer = runtime.value().create_buffer<std::uint32_t>(count);
    auto intermediate = runtime.value().create_buffer<std::uint32_t>(count);
    auto output = runtime.value().create_buffer<std::uint32_t>(count);
    auto host_updated_output = runtime.value().create_buffer<std::uint32_t>(count);
    ASSERT_TRUE(input_buffer);
    ASSERT_TRUE(zero_buffer);
    ASSERT_TRUE(intermediate);
    ASSERT_TRUE(output);
    ASSERT_TRUE(host_updated_output);
    ASSERT_TRUE(input_buffer.value().write(input));
    ASSERT_TRUE(zero_buffer.value().write(zero));

    const auto byte_count = count;
    auto first = runtime.value()
                     .dispatch(kernel.value())
                     .global(count)
                     .read(input_buffer.value())
                     .read(zero_buffer.value())
                     .write(intermediate.value())
                     .push(byte_count)
                     .defer_readback()
                     .submit();
    ASSERT_TRUE(first) << first.error().describe();
    ASSERT_TRUE(first.value().wait());
    auto before_second = runtime.value().transfer_stats();
    ASSERT_TRUE(before_second);
    EXPECT_EQ(before_second.value().uploads, 2U);
    EXPECT_EQ(before_second.value().downloads, 0U);

    auto second = runtime.value()
                      .dispatch(kernel.value())
                      .global(count)
                      .read(intermediate.value())
                      .read(zero_buffer.value())
                      .write(output.value())
                      .push(byte_count)
                      .defer_readback()
                      .submit();
    ASSERT_TRUE(second) << second.error().describe();
    ASSERT_TRUE(second.value().wait());
    auto before_readback = runtime.value().transfer_stats();
    ASSERT_TRUE(before_readback);
    EXPECT_EQ(before_readback.value().uploads, 2U);
    EXPECT_EQ(before_readback.value().downloads, 0U);

    auto readback = runtime.value().readback(output.value());
    ASSERT_TRUE(readback) << readback.error().describe();
    ASSERT_TRUE(readback.value().wait());
    auto after_readback = runtime.value().transfer_stats();
    ASSERT_TRUE(after_readback);
    EXPECT_EQ(after_readback.value().uploads, 2U);
    EXPECT_EQ(after_readback.value().downloads, 1U);
    const auto values = output.value().read();
    ASSERT_TRUE(values) << values.error().describe();
    EXPECT_EQ(values.value(), input);

    std::vector<std::uint32_t> host_override(count, 9U);
    ASSERT_TRUE(output.value().write(host_override));
    auto already_host_current = runtime.value().readback(output.value());
    ASSERT_TRUE(already_host_current);
    ASSERT_TRUE(already_host_current.value().wait());
    EXPECT_EQ(runtime.value().transfer_stats().value().downloads, 1U);
    EXPECT_EQ(output.value().read().value(), host_override);

    auto third = runtime.value()
                     .dispatch(kernel.value())
                     .global(count)
                     .read(output.value())
                     .read(zero_buffer.value())
                     .write(host_updated_output.value())
                     .push(byte_count)
                     .defer_readback()
                     .submit();
    ASSERT_TRUE(third) << third.error().describe();
    ASSERT_TRUE(third.value().wait());
    EXPECT_EQ(runtime.value().transfer_stats().value().uploads, 3U);
    auto final_readback = runtime.value().readback(host_updated_output.value());
    ASSERT_TRUE(final_readback);
    ASSERT_TRUE(final_readback.value().wait());
    EXPECT_EQ(host_updated_output.value().read().value(), host_override);
#endif
}

TEST(Runtime, RunsCpuFallbackBetweenGpuDispatches) {
#ifndef KORA_VECTOR_ADD_FIXTURE_PATH
    GTEST_SKIP() << "Install glslangValidator to compile the vector-add fixture";
#else
    auto runtime = kora::Runtime::create();
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();

    constexpr std::uint32_t count = 257;
    const std::vector<std::uint32_t> input(count, 2U), zeros(count, 0U);
    kora::KernelMetadata metadata;
    metadata.kernel_id = "integration.vector_add.mixed_cpu_gpu";
    metadata.local_size = {64, 1, 1};
    metadata.push_constant_size = sizeof(count);
    metadata.push_constants = {{"count", "u32", 0, sizeof(count)}};
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
    auto kernel = runtime.value().load_kernel(KORA_VECTOR_ADD_FIXTURE_PATH, "main", metadata);
    ASSERT_TRUE(kernel) << kernel.error().describe();
    auto input_buffer = runtime.value().create_buffer<std::uint32_t>(count);
    auto zero_buffer = runtime.value().create_buffer<std::uint32_t>(count);
    auto intermediate = runtime.value().create_buffer<std::uint32_t>(count);
    auto output = runtime.value().create_buffer<std::uint32_t>(count);
    auto final = runtime.value().create_buffer<std::uint32_t>(count);
    ASSERT_TRUE(input_buffer);
    ASSERT_TRUE(zero_buffer);
    ASSERT_TRUE(intermediate);
    ASSERT_TRUE(output);
    ASSERT_TRUE(final);
    ASSERT_TRUE(input_buffer.value().write(input));
    ASSERT_TRUE(zero_buffer.value().write(zeros));

    auto first = runtime.value()
                     .dispatch(kernel.value())
                     .global(count)
                     .read(input_buffer.value())
                     .read(zero_buffer.value())
                     .write(intermediate.value())
                     .push(count)
                     .defer_readback()
                     .submit();
    ASSERT_TRUE(first) << first.error().describe();
    ASSERT_TRUE(first.value().wait());

    ASSERT_TRUE(runtime.value().register_cpu_fallback(
        "integration.vector_add.bump", [](kora::CpuTaskContext& task) -> kora::Status {
            auto values = task.read_write(0);
            if (!values) return std::move(values).error();
            if (values.value().size() % sizeof(std::uint32_t) != 0)
                return kora::Error(kora::ErrorCode::InvalidArgument, "CPU vector has a partial element");
            auto* elements = reinterpret_cast<std::uint32_t*>(values.value().data());
            for (std::size_t index = 0; index < values.value().size() / sizeof(std::uint32_t); ++index)
                elements[index] += 5U;
            return {};
        }));
    const std::array buffers{intermediate.value().handle()};
    const std::array accesses{kora::KernelAccess::ReadWrite};
    auto cpu = runtime.value().run_cpu_fallback("integration.vector_add.bump", buffers, accesses);
    ASSERT_TRUE(cpu) << cpu.error().describe();
    ASSERT_TRUE(cpu.value().wait()) << cpu.value().status().error().describe();

    auto second = runtime.value()
                      .dispatch(kernel.value())
                      .global(count)
                      .read(intermediate.value())
                      .read(zero_buffer.value())
                      .write(output.value())
                      .push(count)
                      .defer_readback()
                      .submit();
    ASSERT_TRUE(second) << second.error().describe();
    ASSERT_TRUE(second.value().wait());
    auto third = runtime.value()
                     .dispatch(kernel.value())
                     .global(count)
                     .read(output.value())
                     .read(zero_buffer.value())
                     .write(final.value())
                     .push(count)
                     .defer_readback()
                     .submit();
    ASSERT_TRUE(third) << third.error().describe();
    ASSERT_TRUE(third.value().wait());
    auto readback = runtime.value().readback(final.value());
    ASSERT_TRUE(readback) << readback.error().describe();
    ASSERT_TRUE(readback.value().wait());
    std::vector<std::uint32_t> expected(count, 7U);
    EXPECT_EQ(final.value().read().value(), expected);

    auto stats = runtime.value().transfer_stats();
    ASSERT_TRUE(stats);
    EXPECT_EQ(stats.value().uploads, 3U);    // Two initial inputs plus CPU-updated intermediate.
    EXPECT_EQ(stats.value().downloads, 2U);  // CPU visibility plus explicit final readback.
#endif
}
