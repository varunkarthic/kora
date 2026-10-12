#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <spirv/unified1/spirv.hpp>
#include <string>
#include <vector>

#include "core/device_caps.hpp"
#include "kernel/pipeline_manager.hpp"
#include "support/vulkan_test.hpp"
#include "vulkan/api.hpp"

#ifdef KORA_SPIRV_FIXTURE_PATH
namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic<std::uint64_t> sequence{1};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("kora-pipeline-test-" + std::to_string(stamp) + "-" +
                 std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

std::vector<std::uint32_t> read_fixture() {
    std::ifstream input(KORA_SPIRV_FIXTURE_PATH, std::ios::binary);
    if (!input) return {};
    std::vector<char> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (bytes.empty() || bytes.size() % sizeof(std::uint32_t) != 0) return {};
    std::vector<std::uint32_t> words(bytes.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return words;
}

kora::KernelMetadata fixture_metadata(std::uint64_t scale = 2) {
    kora::KernelMetadata metadata;
    metadata.kernel_id = "pipeline-cache-integration";
    metadata.local_size = {8, 2, 1};
    metadata.bindings.push_back(
        {0, 0, "data_buffer", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::ReadWrite});
    metadata.bindings.push_back(
        {0, 1, "second_buffer", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::ReadWrite});
    metadata.push_constant_size = 4;
    metadata.push_constants.push_back({"count", "u32", 0, 4});
    metadata.specialization_constants.push_back({7, "SCALE", "u32", scale});
    return metadata;
}

kora::detail::kernel::PipelineCacheIdentity make_identity(const kora::vk::PhysicalDeviceInfo& physical) {
    kora::detail::kernel::PipelineCacheIdentity identity;
    identity.vendor_id = physical.properties.vendorID;
    identity.device_id = physical.properties.deviceID;
    identity.driver_version = physical.properties.driverVersion;
    std::copy(std::begin(physical.id_properties.deviceUUID), std::end(physical.id_properties.deviceUUID),
              identity.device_uuid.begin());
    std::copy(std::begin(physical.id_properties.driverUUID), std::end(physical.id_properties.driverUUID),
              identity.driver_uuid.begin());
    return identity;
}

std::vector<std::uint32_t> alternate_specialization_default(std::vector<std::uint32_t> words) {
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto instruction = words[cursor];
        const auto count = static_cast<std::size_t>(instruction >> 16U);
        const auto opcode = static_cast<spv::Op>(instruction & 0xffffU);
        if (count == 0 || count > words.size() - cursor) return {};
        if (opcode == spv::OpSpecConstant && count == 4) {
            words[cursor + 3] = 3;
            return words;
        }
        cursor += count;
    }
    return {};
}

}  // namespace

TEST(PipelineManager, CachesPipelinesRecyclesDescriptorsAndRecoversPersistentCache) {
    auto loader = kora::vk::Loader::load();
    if (!loader) KORA_REQUIRE_VULKAN(loader);
    auto instance = loader.value().create_instance();
    if (!instance && instance.error().code == kora::ErrorCode::NoVulkanImplementation)
        KORA_REQUIRE_VULKAN(false);
    ASSERT_TRUE(instance) << instance.error().describe();
    auto physical_devices = instance.value().physical_devices();
    ASSERT_TRUE(physical_devices) << physical_devices.error().describe();
    ASSERT_FALSE(physical_devices.value().empty());
    const auto& physical = physical_devices.value().front();
    auto device_result = instance.value().create_device(physical);
    if (!device_result && device_result.error().code == kora::ErrorCode::NoComputeQueue)
        KORA_REQUIRE_VULKAN(false);
    ASSERT_TRUE(device_result) << device_result.error().describe();
    auto device = std::move(device_result).value();
    auto device_info = kora::detail::DeviceCaps::from_vulkan(physical).to_device_info();
    const auto capabilities = kora::detail::spirv::capabilities_from(device_info);
    const auto identity = make_identity(physical);
    const auto module = read_fixture();
    ASSERT_FALSE(module.empty());

    TemporaryDirectory temporary;
    const auto cache_file = temporary.path() / "pipeline-cache.bin";
    kora::detail::kernel::PipelineManagerOptions options;
    options.max_cached_pipelines = 1;
    options.descriptor_sets_per_pool = 1;
    options.max_descriptor_pools_per_layout = 2;
    {
        auto manager_result = kora::detail::kernel::PipelineManager::create(
            device.api(), device.handle(), capabilities, identity, cache_file, options);
        ASSERT_TRUE(manager_result) << manager_result.error().describe();
        auto manager = std::move(manager_result).value();
        const auto metadata = fixture_metadata();
        auto first = manager->get_or_create(module, metadata);
        ASSERT_TRUE(first) << first.error().describe();
        ASSERT_NE(first.value()->handle, VK_NULL_HANDLE);

        auto fast_cached = manager->find_cached(module, metadata, metadata.specialization_constants);
        ASSERT_TRUE(fast_cached);
        EXPECT_EQ(first.value().get(), fast_cached->get());

        auto repeated = manager->get_or_create(module, metadata);
        ASSERT_TRUE(repeated) << repeated.error().describe();
        EXPECT_EQ(first.value().get(), repeated.value().get());
        EXPECT_EQ(manager->stats().pipeline_cache_hits, 2U);

        auto invalid_override = metadata.specialization_constants;
        invalid_override.front().type = "unsupported";
        EXPECT_FALSE(manager->find_cached(module, metadata, invalid_override));

        auto descriptor = manager->allocate_descriptor_set(first.value(), 0);
        ASSERT_TRUE(descriptor) << descriptor.error().describe();
        auto second_descriptor = manager->allocate_descriptor_set(first.value(), 0);
        ASSERT_TRUE(second_descriptor) << second_descriptor.error().describe();
        EXPECT_EQ(manager->stats().descriptor_pool_growths, 1U);
        ASSERT_TRUE(manager->free_descriptor_set(descriptor.value()));
        auto recycled = manager->allocate_descriptor_set(first.value(), 0);
        ASSERT_TRUE(recycled) << recycled.error().describe();
        EXPECT_EQ(recycled.value(), descriptor.value());
        ASSERT_TRUE(manager->free_descriptor_set(recycled.value()));
        ASSERT_TRUE(manager->free_descriptor_set(second_descriptor.value()));

        // Verify both reflected bindings share the cached layout across pipeline variants.
        auto two_binding_pipeline = manager->get_or_create(module, fixture_metadata());
        ASSERT_TRUE(two_binding_pipeline) << two_binding_pipeline.error().describe();
        auto descriptor_stats = manager->stats();
        auto first_set = manager->allocate_descriptor_set(two_binding_pipeline.value(), 0);
        ASSERT_TRUE(first_set) << first_set.error().describe();
        auto second_set = manager->allocate_descriptor_set(two_binding_pipeline.value(), 0);
        ASSERT_TRUE(second_set) << second_set.error().describe();
        const auto stats_after_two_sets = manager->stats();
        EXPECT_EQ(stats_after_two_sets.descriptor_pool_growths, 1U);
        EXPECT_EQ(stats_after_two_sets.descriptor_sets_allocated - descriptor_stats.descriptor_sets_allocated,
                  2U);
        ASSERT_TRUE(manager->free_descriptor_set(first_set.value()));
        ASSERT_TRUE(manager->free_descriptor_set(second_set.value()));

        const auto alternate = alternate_specialization_default(module);
        ASSERT_FALSE(alternate.empty());
        auto second = manager->get_or_create(alternate, fixture_metadata(3));
        ASSERT_TRUE(second) << second.error().describe();
        EXPECT_NE(first.value().get(), second.value().get());
        EXPECT_EQ(manager->stats().pipeline_evictions, 1U);
        ASSERT_TRUE(manager->save_pipeline_cache());
        EXPECT_TRUE(std::filesystem::exists(cache_file));
    }

    {
        auto manager_result = kora::detail::kernel::PipelineManager::create(
            device.api(), device.handle(), capabilities, identity, cache_file, options);
        ASSERT_TRUE(manager_result) << manager_result.error().describe();
        auto manager = std::move(manager_result).value();
        auto pipeline = manager->get_or_create(module, fixture_metadata());
        ASSERT_TRUE(pipeline) << pipeline.error().describe();
        EXPECT_TRUE(manager->stats().persisted_pipeline_cache_loaded);
    }

    {
        auto mismatched_identity = identity;
        mismatched_identity.device_uuid[0] ^= 0xffU;
        auto manager_result = kora::detail::kernel::PipelineManager::create(
            device.api(), device.handle(), capabilities, mismatched_identity, cache_file, options);
        ASSERT_TRUE(manager_result) << manager_result.error().describe();
        auto manager = std::move(manager_result).value();
        auto pipeline = manager->get_or_create(module, fixture_metadata());
        ASSERT_TRUE(pipeline) << pipeline.error().describe();
        EXPECT_FALSE(manager->stats().persisted_pipeline_cache_loaded);
    }

    {
        std::ofstream corrupt(cache_file, std::ios::binary | std::ios::trunc);
        corrupt << "not a KORA pipeline cache";
        ASSERT_TRUE(corrupt.good());
    }
    {
        auto manager_result = kora::detail::kernel::PipelineManager::create(
            device.api(), device.handle(), capabilities, identity, cache_file, options);
        ASSERT_TRUE(manager_result) << manager_result.error().describe();
        auto manager = std::move(manager_result).value();
        auto pipeline = manager->get_or_create(module, fixture_metadata());
        ASSERT_TRUE(pipeline) << pipeline.error().describe();
        EXPECT_FALSE(manager->stats().persisted_pipeline_cache_loaded);
    }
}
#else
TEST(PipelineManager, RequiresOptionalGlslFixtureCompiler) {
    GTEST_SKIP() << "Install glslangValidator to compile the pipeline test fixture";
}
#endif
