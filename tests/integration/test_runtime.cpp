#include <gtest/gtest.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/calibration.hpp"
#include "core/json.hpp"
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

TEST(Runtime, RapidCreateAndDestroyHonorsBothShutdownModes) {
    for (const auto mode : {kora::ShutdownMode::Graceful, kora::ShutdownMode::CancelPending}) {
        for (int iteration = 0; iteration < 4; ++iteration) {
            kora::RuntimeOptions options;
            options.shutdown_mode = mode;
            options.enable_startup_calibration = false;
            auto runtime = kora::Runtime::create(options);
            if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                             runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
                KORA_REQUIRE_VULKAN(false);
            }
            ASSERT_TRUE(runtime) << runtime.error().describe();
            ASSERT_FALSE(runtime.value().devices().empty());
        }
    }
}

TEST(Runtime, OwnsSelectedDeviceAndExposesMemoryStats) {
    auto runtime = kora::Runtime::create();
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    const auto devices = runtime.value().devices();
    ASSERT_FALSE(devices.empty());
    EXPECT_TRUE(runtime.value().quarantined_devices().empty());

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
    EXPECT_TRUE(runtime.value().quarantined_devices().empty());
}

TEST(Runtime, AllEligibleReportsQuarantinedDevicesSeparately) {
    kora::RuntimeOptions options;
    options.device_policy = kora::DevicePolicy::AllEligible;
    auto runtime = kora::Runtime::create(options);
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    EXPECT_FALSE(runtime.value().devices().empty());
    EXPECT_TRUE(runtime.value().quarantined_devices().empty());
}

TEST(Runtime, AdaptiveDispatchFallsBackToBoundDeviceWithoutCompleteCalibration) {
#ifndef KORA_VECTOR_ADD_FIXTURE_PATH
    GTEST_SKIP() << "Install glslangValidator to compile the vector-add fixture";
#else
    kora::RuntimeOptions options;
    options.enable_startup_calibration = false;
    options.telemetry = kora::TelemetryLevel::Trace;
    auto runtime = kora::Runtime::create(options);
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    const auto devices = runtime.value().devices();
    ASSERT_FALSE(devices.empty());
    const auto kernel =
        runtime.value().load_kernel(KORA_VECTOR_ADD_FIXTURE_PATH, "main", vector_add_metadata());
    ASSERT_TRUE(kernel) << kernel.error().describe();
    constexpr std::uint64_t count = 257;
    const kora::AdaptiveDispatchWorkload workload{
        .input_bytes = 2 * count * sizeof(std::uint32_t),
        .output_bytes = count * sizeof(std::uint32_t),
        .kernel_work = 3 * count * sizeof(std::uint32_t),
        .required_device_memory_bytes = 3 * count * sizeof(std::uint32_t),
        .kernel_kind = kora::AdaptiveKernelWorkKind::StreamingBytes,
    };
    auto adaptive = runtime.value().dispatch_adaptive(kernel.value(), workload);
    ASSERT_TRUE(adaptive) << adaptive.error().describe();
    ASSERT_TRUE(adaptive.value().selected_device());
    EXPECT_EQ(*adaptive.value().selected_device(), devices.front());

    auto input_a = runtime.value().create_buffer<std::uint32_t>(count);
    auto input_b = runtime.value().create_buffer<std::uint32_t>(count);
    auto output = runtime.value().create_buffer<std::uint32_t>(count);
    ASSERT_TRUE(input_a);
    ASSERT_TRUE(input_b);
    ASSERT_TRUE(output);
    std::vector<std::uint32_t> a(count, 2), b(count, 3);
    ASSERT_TRUE(input_a.value().write(a));
    ASSERT_TRUE(input_b.value().write(b));
    const auto push_count = static_cast<std::uint32_t>(count);
    auto event = adaptive.value()
                     .global(count)
                     .read(input_a.value())
                     .read(input_b.value())
                     .write(output.value())
                     .push(push_count)
                     .submit();
    ASSERT_TRUE(event) << event.error().describe();
    ASSERT_TRUE(event.value().wait());
    EXPECT_EQ(output.value().read().value(), std::vector<std::uint32_t>(count, 5));
    const auto trace = runtime.value().export_trace_json();
    ASSERT_TRUE(trace) << trace.error().describe();
    const auto parsed_trace = kora::detail::json::parse(trace.value());
    ASSERT_TRUE(parsed_trace) << parsed_trace.error().describe();
    const auto* trace_events_value = parsed_trace.value().find("traceEvents");
    ASSERT_NE(trace_events_value, nullptr);
    const auto* trace_events = std::get_if<kora::detail::json::Value::Array>(&trace_events_value->storage);
    ASSERT_NE(trace_events, nullptr);
    for (const auto& trace_event : *trace_events) {
        const auto* args = trace_event.find("args");
        ASSERT_NE(args, nullptr);
        EXPECT_EQ(args->find("scheduler_score"), nullptr);
    }
#endif
}

TEST(Runtime, AdaptiveDispatchPersistsCompletedExecutionObservations) {
#ifndef KORA_VECTOR_ADD_FIXTURE_PATH
    GTEST_SKIP() << "Install glslangValidator to compile the vector-add fixture";
#else
    const auto cache_directory =
        std::filesystem::temp_directory_path() /
        ("kora-adaptive-observations-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    kora::RuntimeOptions options;
    options.cache_dir = cache_directory.string();
    options.enable_startup_calibration = false;
    options.telemetry = kora::TelemetryLevel::Trace;
    auto runtime = kora::Runtime::create(options);
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    const auto device = runtime.value().devices().front();
    kora::detail::calibration::Record calibration;
    calibration.submit_overhead_ns = 100;
    calibration.tiny_dispatch_ns = 100;
    calibration.h2d_bytes_per_second = 100'000'000;
    calibration.d2h_bytes_per_second = 100'000'000;
    calibration.device_copy_bytes_per_second = 100'000'000;
    calibration.streaming_bytes_per_second = 100'000'000;
    calibration.alu_operations_per_second = 100'000'000;
    ASSERT_TRUE(kora::detail::RuntimeTestAccess::set_calibration_record(runtime.value(), device,
                                                                        std::move(calibration)));
    auto profiling = runtime.value().profiling_stats(device);
    if (!profiling && profiling.error().code == kora::ErrorCode::UnsupportedFeature)
        GTEST_SKIP() << "adaptive online observations require compute timestamp support";
    ASSERT_TRUE(profiling) << profiling.error().describe();
    const auto kernel =
        runtime.value().load_kernel(KORA_VECTOR_ADD_FIXTURE_PATH, "main", vector_add_metadata());
    ASSERT_TRUE(kernel) << kernel.error().describe();

    constexpr std::uint32_t count = 257;
    const kora::AdaptiveDispatchWorkload workload{
        .input_bytes = 2 * count * sizeof(std::uint32_t),
        .output_bytes = count * sizeof(std::uint32_t),
        .kernel_work = 3 * count * sizeof(std::uint32_t),
        .required_device_memory_bytes = 3 * count * sizeof(std::uint32_t),
        .kernel_kind = kora::AdaptiveKernelWorkKind::StreamingBytes,
    };
    const auto key = kora::detail::RuntimeTestAccess::performance_observation_key(
        runtime.value(), device, kernel.value(), workload,
        kora::detail::PerformanceTimingDomain::DeviceExecution);
    ASSERT_TRUE(key) << key.error().describe();
    auto input_a = runtime.value().create_buffer<std::uint32_t>(count);
    auto input_b = runtime.value().create_buffer<std::uint32_t>(count);
    auto output = runtime.value().create_buffer<std::uint32_t>(count);
    ASSERT_TRUE(input_a);
    ASSERT_TRUE(input_b);
    ASSERT_TRUE(output);
    const std::vector<std::uint32_t> values_a(count, 2);
    const std::vector<std::uint32_t> values_b(count, 3);
    ASSERT_TRUE(input_a.value().write(values_a));
    ASSERT_TRUE(input_b.value().write(values_b));
    const auto push_count = count;
    for (int sample = 0; sample < 8; ++sample) {
        auto builder = runtime.value().dispatch_adaptive(kernel.value(), workload);
        ASSERT_TRUE(builder) << builder.error().describe();
        auto event = builder.value()
                         .global(count)
                         .read(input_a.value())
                         .read(input_b.value())
                         .write(output.value())
                         .push(push_count)
                         .submit();
        ASSERT_TRUE(event) << event.error().describe();
        ASSERT_TRUE(event.value().wait());
    }
    auto observed = kora::detail::RuntimeTestAccess::performance_stats(runtime.value(), key.value());
    ASSERT_TRUE(observed) << observed.error().describe();
    EXPECT_GE(observed.value().count, 5U);
    kora::detail::PerformanceDatabase persisted;
    const auto load_now = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                         std::chrono::system_clock::now().time_since_epoch())
                                                         .count());
    const auto loaded = persisted.load(cache_directory / "performance-db.json", load_now);
    ASSERT_TRUE(loaded) << loaded.error().describe();
    const auto saved = persisted.query(key.value());
    ASSERT_TRUE(saved) << saved.error().describe();
    EXPECT_GE(saved.value().count, 5U);
    auto trace = runtime.value().export_trace_json();
    ASSERT_TRUE(trace) << trace.error().describe();
    auto parsed_trace = kora::detail::json::parse(trace.value());
    ASSERT_TRUE(parsed_trace) << parsed_trace.error().describe();
    const auto* trace_events_value = parsed_trace.value().find("traceEvents");
    ASSERT_NE(trace_events_value, nullptr);
    const auto* trace_events = std::get_if<kora::detail::json::Value::Array>(&trace_events_value->storage);
    ASSERT_NE(trace_events, nullptr);
    bool found_score = false;
    for (const auto& trace_event : *trace_events) {
        const auto* args = trace_event.find("args");
        if (!args) continue;
        const auto* score = args->find("scheduler_score");
        if (!score) continue;
        const auto* device_index = score->find("selected_device_index");
        const auto* predicted_finish = score->find("predicted_finish_ns");
        ASSERT_NE(device_index, nullptr);
        ASSERT_NE(predicted_finish, nullptr);
        const auto* device_number = std::get_if<kora::detail::json::Value::Number>(&device_index->storage);
        const auto* finish_number =
            std::get_if<kora::detail::json::Value::Number>(&predicted_finish->storage);
        ASSERT_NE(device_number, nullptr);
        ASSERT_NE(finish_number, nullptr);
        EXPECT_EQ(std::stoull(device_number->text), device.index);
        EXPECT_GT(std::stod(finish_number->text), 0.0);
        found_score = true;
    }
    EXPECT_TRUE(found_score);
    std::error_code ignored;
    std::filesystem::remove_all(cache_directory, ignored);
#endif
}

TEST(Runtime, AdaptiveDispatchReturnsToTheProducerDeviceForDeviceNewerInputs) {
#ifndef KORA_VECTOR_ADD_FIXTURE_PATH
    GTEST_SKIP() << "Install glslangValidator to compile the vector-add fixture";
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
    if (devices.size() < 2) GTEST_SKIP() << "Adaptive migration needs two initialized devices";

    for (std::size_t index = 0; index < devices.size(); ++index) {
        kora::detail::calibration::Record record;
        record.submit_overhead_ns = 100;
        record.tiny_dispatch_ns = 100;
        const auto rate = index == 0 ? 1'000'000ULL : 100'000'000ULL;
        record.h2d_bytes_per_second = rate;
        record.d2h_bytes_per_second = rate;
        record.device_copy_bytes_per_second = rate;
        record.streaming_bytes_per_second = rate;
        record.alu_operations_per_second = rate;
        ASSERT_TRUE(kora::detail::RuntimeTestAccess::set_calibration_record(runtime.value(), devices[index],
                                                                            std::move(record)));
    }
    const auto kernel = runtime.value().load_kernel(KORA_VECTOR_ADD_FIXTURE_PATH, "main",
                                                    vector_add_metadata(), devices.front());
    ASSERT_TRUE(kernel) << kernel.error().describe();

    constexpr std::uint64_t count = 257;
    auto input_a = runtime.value().create_buffer<std::uint32_t>(count);
    auto input_b = runtime.value().create_buffer<std::uint32_t>(count);
    auto intermediate = runtime.value().create_buffer<std::uint32_t>(count);
    auto output = runtime.value().create_buffer<std::uint32_t>(count);
    ASSERT_TRUE(input_a);
    ASSERT_TRUE(input_b);
    ASSERT_TRUE(intermediate);
    ASSERT_TRUE(output);
    std::vector<std::uint32_t> a(count), b(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        a[i] = i + 5;
        b[i] = i * 2;
    }
    ASSERT_TRUE(input_a.value().write(a));
    ASSERT_TRUE(input_b.value().write(b));
    const auto push_count = static_cast<std::uint32_t>(count);
    auto producer = runtime.value()
                        .dispatch(kernel.value())
                        .global(count)
                        .read(input_a.value())
                        .read(input_b.value())
                        .write(intermediate.value())
                        .push(push_count)
                        .defer_readback()
                        .submit();
    ASSERT_TRUE(producer) << producer.error().describe();
    ASSERT_TRUE(producer.value().wait());

    const kora::AdaptiveDispatchWorkload workload{
        .input_bytes = 2 * count * sizeof(std::uint32_t),
        .output_bytes = count * sizeof(std::uint32_t),
        .kernel_work = 3 * count * sizeof(std::uint32_t),
        .required_device_memory_bytes = 3 * count * sizeof(std::uint32_t),
        .kernel_kind = kora::AdaptiveKernelWorkKind::StreamingBytes,
    };
    auto adaptive = runtime.value().dispatch_adaptive(kernel.value(), workload);
    ASSERT_TRUE(adaptive) << adaptive.error().describe();
    ASSERT_TRUE(adaptive.value().selected_device());
    ASSERT_NE(*adaptive.value().selected_device(), devices.front());
    auto consumer = adaptive.value()
                        .global(count)
                        .read(intermediate.value())
                        .read(input_b.value())
                        .write(output.value())
                        .push(push_count)
                        .submit();
    ASSERT_TRUE(consumer) << consumer.error().describe();
    EXPECT_EQ(*adaptive.value().selected_device(), devices.front());
    ASSERT_TRUE(consumer.value().wait());
    const auto result = output.value().read();
    ASSERT_TRUE(result) << result.error().describe();
    for (std::uint32_t i = 0; i < count; ++i) EXPECT_EQ(result.value()[i], a[i] + 2 * b[i]);
#endif
}

TEST(Runtime, VerifiedDispatchPublishesMatchingResultsAndQuarantinesMismatches) {
#ifndef KORA_VECTOR_ADD_FIXTURE_PATH
    GTEST_SKIP() << "Install glslangValidator to compile the vector-add fixture";
#else
    kora::RuntimeOptions options;
    options.device_policy = kora::DevicePolicy::AllEligible;
    auto created = kora::Runtime::create(options);
    if (!created && (created.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     created.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(created) << created.error().describe();
    auto runtime = std::move(created).value();
    const auto devices = runtime.devices();
    if (devices.size() < 2) GTEST_SKIP() << "duplicate verification requires two initialized devices";

    constexpr std::uint32_t count = 32;
    auto input_a = runtime.create_buffer<std::uint32_t>(count);
    auto input_b = runtime.create_buffer<std::uint32_t>(count);
    auto output = runtime.create_buffer<std::uint32_t>(count);
    ASSERT_TRUE(input_a);
    ASSERT_TRUE(input_b);
    ASSERT_TRUE(output);
    std::vector<std::uint32_t> left(count);
    std::vector<std::uint32_t> right(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        left[index] = index * 2U;
        right[index] = index + 7U;
    }
    ASSERT_TRUE(input_a.value().write(left));
    ASSERT_TRUE(input_b.value().write(right));
    auto kernel =
        runtime.load_kernel(KORA_VECTOR_ADD_FIXTURE_PATH, "main", vector_add_metadata(), devices[0]);
    ASSERT_TRUE(kernel) << kernel.error().describe();

    kora::VerificationPolicy matching;
    matching.mode = kora::VerificationMode::Custom;
    matching.custom = [](std::span<const std::byte> first, std::span<const std::byte> second) {
        return std::equal(first.begin(), first.end(), second.begin(), second.end());
    };
    auto verified = runtime.dispatch(kernel.value())
                        .global(count)
                        .read(input_a.value())
                        .read(input_b.value())
                        .write(output.value())
                        .push(count)
                        .verify_with(matching)
                        .submit();
    ASSERT_TRUE(verified) << verified.error().describe();
    const auto success = verified.value().wait();
    ASSERT_TRUE(success) << success.error().describe();
    auto actual = output.value().read();
    ASSERT_TRUE(actual) << actual.error().describe();
    for (std::uint32_t index = 0; index < count; ++index)
        EXPECT_EQ(actual.value()[index], left[index] + right[index]);
    EXPECT_TRUE(runtime.quarantined_devices().empty());

    kora::VerificationPolicy injected_mismatch;
    injected_mismatch.mode = kora::VerificationMode::Custom;
    injected_mismatch.custom = [](std::span<const std::byte>, std::span<const std::byte>) { return false; };
    auto mismatched = runtime.dispatch(kernel.value())
                          .global(count)
                          .read(input_a.value())
                          .read(input_b.value())
                          .write(output.value())
                          .push(count)
                          .verify_with(injected_mismatch)
                          .submit();
    ASSERT_TRUE(mismatched) << mismatched.error().describe();
    const auto mismatch_status = mismatched.value().wait();
    ASSERT_FALSE(mismatch_status);
    EXPECT_EQ(mismatch_status.error().code, kora::ErrorCode::ValidationError);
    EXPECT_EQ(output.value().read().error().code, kora::ErrorCode::DataLoss);
    EXPECT_GE(runtime.quarantined_devices().size(), 2U);

    auto rejected = runtime.dispatch(kernel.value())
                        .global(count)
                        .read(input_a.value())
                        .read(input_b.value())
                        .write(output.value())
                        .push(count)
                        .submit();
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, kora::ErrorCode::ValidationError);
#endif
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

TEST(Runtime, CalibrationCanBeRequestedAndReplacesCorruptCache) {
    const auto directory = std::filesystem::temp_directory_path() / "kora-calibration-request-test";
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    kora::RuntimeOptions options;
    options.cache_dir = directory.string();
    options.enable_startup_calibration = false;
    options.calibration_budget_ms = 2000;
    auto runtime = kora::Runtime::create(options);
    if (!runtime && (runtime.error().code == kora::ErrorCode::NoVulkanImplementation ||
                     runtime.error().code == kora::ErrorCode::NoEligibleDevice)) {
        KORA_REQUIRE_VULKAN(false);
    }
    ASSERT_TRUE(runtime) << runtime.error().describe();
    const auto devices = runtime.value().devices();
    ASSERT_FALSE(devices.empty());
    if (devices.front().uuid.empty()) GTEST_SKIP() << "device does not expose a stable UUID";

    auto application_buffer_result = runtime.value().create_buffer<std::uint32_t>(4);
    ASSERT_TRUE(application_buffer_result);
    auto application_buffer = std::move(application_buffer_result).value();
    const std::array<std::uint32_t, 4> application_values{11, 22, 33, 44};
    ASSERT_TRUE(application_buffer.write(application_values));

    const auto path = directory / "calibration" / (devices.front().uuid + ".json");
    ASSERT_TRUE(runtime.value().calibrate());
    auto preserved_values = application_buffer.read();
    ASSERT_TRUE(preserved_values);
    EXPECT_EQ(preserved_values.value(),
              std::vector<std::uint32_t>(application_values.begin(), application_values.end()));
    ASSERT_TRUE(std::filesystem::exists(path));
    {
        std::ifstream calibration(path, std::ios::binary);
        ASSERT_TRUE(calibration);
        const std::string record((std::istreambuf_iterator<char>(calibration)),
                                 std::istreambuf_iterator<char>());
        const auto metric = [&](std::string_view name) {
            const auto key = "\"" + std::string(name) + "\":";
            const auto position = record.find(key);
            if (position == std::string::npos) return std::uint64_t{0};
            const auto start = position + key.size();
            try {
                return static_cast<std::uint64_t>(std::stoull(record.substr(start)));
            } catch (...) {
                return std::uint64_t{0};
            }
        };
        EXPECT_EQ(metric("schema"), kora::detail::calibration::kSchemaVersion);
        EXPECT_GT(metric("tiny_dispatch_ns"), 0U);
        EXPECT_GT(metric("streaming_bytes_per_second"), 0U);
        EXPECT_GT(metric("alu_operations_per_second"), 0U);
    }
    {
        std::ofstream corrupt(path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(corrupt);
        corrupt << "corrupt";
    }
    ASSERT_TRUE(runtime.value().calibrate());
    std::ifstream repaired(path, std::ios::binary);
    ASSERT_TRUE(repaired);
    const std::string contents((std::istreambuf_iterator<char>(repaired)), std::istreambuf_iterator<char>());
    EXPECT_NE(contents, "corrupt");
    EXPECT_NE(contents.find("device_uuid"), std::string::npos);
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
            options.telemetry = kora::TelemetryLevel::Trace;
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
                EXPECT_GT(profiling.value().last_profile_submission_id, 0U);
                EXPECT_GT(profiling.value().samples_collected, 0U);
                EXPECT_GT(profiling.value().last_duration_nanoseconds, 0U);
                EXPECT_GT(profiling.value().last_dispatch_validation_nanoseconds, 0U);
                EXPECT_GT(profiling.value().last_binding_preflight_nanoseconds, 0U);
                EXPECT_GT(profiling.value().last_pipeline_lookup_nanoseconds, 0U);
                EXPECT_GT(profiling.value().last_buffer_residency_nanoseconds, 0U);
                EXPECT_GT(profiling.value().last_submission_assembly_nanoseconds, 0U);
                EXPECT_GT(profiling.value().last_host_submission_nanoseconds, 0U);
                EXPECT_GT(profiling.value().last_queue_enqueue_nanoseconds, 0U);
                EXPECT_GT(profiling.value().last_command_buffer_acquire_nanoseconds, 0U);
                EXPECT_GT(profiling.value().last_descriptor_setup_nanoseconds, 0U);
                EXPECT_GT(profiling.value().last_command_recording_nanoseconds, 0U);
                EXPECT_GT(profiling.value().last_queue_completion_nanoseconds, 0U);
            } else {
                EXPECT_EQ(profiling.error().code, kora::ErrorCode::UnsupportedFeature);
            }
            auto trace = runtime.value().export_trace_json();
            ASSERT_TRUE(trace) << trace.error().describe();
            auto parsed_trace = kora::detail::json::parse(trace.value());
            ASSERT_TRUE(parsed_trace) << parsed_trace.error().describe();
            const auto* trace_events_value = parsed_trace.value().find("traceEvents");
            ASSERT_NE(trace_events_value, nullptr);
            const auto* trace_events =
                std::get_if<kora::detail::json::Value::Array>(&trace_events_value->storage);
            ASSERT_NE(trace_events, nullptr);
            if (profiling) ASSERT_FALSE(trace_events->empty());
            if (!trace_events->empty()) {
                EXPECT_NE(trace.value().find("\"ph\":\"X\""), std::string::npos);
                EXPECT_NE(trace.value().find("submission_id"), std::string::npos);
            }
            for (const auto& trace_event : *trace_events) {
                const auto* args = trace_event.find("args");
                ASSERT_NE(args, nullptr);
                EXPECT_EQ(args->find("scheduler_score"), nullptr);
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
