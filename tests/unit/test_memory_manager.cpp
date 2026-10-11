#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <vector>

#include "core/device_caps.hpp"
#include "memory/memory_manager.hpp"

namespace {

using kora::ErrorCode;
using kora::detail::BudgetLedger;
using kora::detail::DeviceCaps;
using kora::detail::HeapBudget;
using kora::detail::MemoryManager;
using kora::detail::MemoryPolicy;
using kora::detail::MemoryTypeRequirements;

kora::vk::PhysicalDeviceInfo make_device(const VkPhysicalDeviceMemoryProperties& properties) {
    kora::vk::PhysicalDeviceInfo raw;
    std::memcpy(raw.properties.deviceName, "Synthetic memory device", sizeof("Synthetic memory device"));
    raw.properties.apiVersion = VK_API_VERSION_1_3;
    raw.properties.limits.maxComputeWorkGroupCount[0] = 1;
    raw.properties.limits.maxComputeWorkGroupCount[1] = 1;
    raw.properties.limits.maxComputeWorkGroupCount[2] = 1;
    raw.properties.limits.maxComputeWorkGroupSize[0] = 1;
    raw.properties.limits.maxComputeWorkGroupSize[1] = 1;
    raw.properties.limits.maxComputeWorkGroupSize[2] = 1;
    raw.properties.limits.maxComputeWorkGroupInvocations = 1;
    raw.memory_properties = properties;
    return raw;
}

VkPhysicalDeviceMemoryProperties make_properties(
    std::vector<std::pair<VkMemoryPropertyFlags, std::uint32_t>> types, std::uint64_t heap_size = 1000) {
    VkPhysicalDeviceMemoryProperties properties{};
    properties.memoryHeapCount = 1;
    properties.memoryHeaps[0].size = heap_size;
    properties.memoryTypeCount = static_cast<std::uint32_t>(types.size());
    for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        properties.memoryTypes[index].propertyFlags = types[index].first;
        properties.memoryTypes[index].heapIndex = types[index].second;
    }
    return properties;
}

TEST(MemoryManager, AppliesPolicyOrderAndResourceCompatibilityMask) {
    const auto properties = make_properties({
        {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0},
        {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0},
        {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
         0},
    });
    const auto info = DeviceCaps::from_vulkan(make_device(properties)).to_device_info();

    auto device_choice = MemoryManager::select_memory_type(
        info.memory_types(), info.memory_type_preferences(), MemoryPolicy::DevicePreferred, 0b110);
    ASSERT_TRUE(device_choice) << device_choice.error().describe();
    EXPECT_EQ(device_choice.value(), 2U);

    auto host_choice = MemoryManager::select_memory_type(
        info.memory_types(), info.memory_type_preferences(), MemoryPolicy::HostVisible, 0b111,
        MemoryTypeRequirements{.host_visible = true, .host_coherent = true});
    ASSERT_TRUE(host_choice) << host_choice.error().describe();
    EXPECT_EQ(host_choice.value(), 1U);
}

TEST(MemoryManager, SyntheticUmaAndDiscreteTopologiesChooseDifferentAutomaticTypes) {
    const auto discrete_properties = make_properties({
        {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0},
        {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0},
    });
    const auto discrete = DeviceCaps::from_vulkan(make_device(discrete_properties)).to_device_info();
    const auto discrete_choice = MemoryManager::select_memory_type(
        discrete.memory_types(), discrete.memory_type_preferences(), MemoryPolicy::Auto, 0b11);
    ASSERT_TRUE(discrete_choice);
    EXPECT_EQ(discrete_choice.value(), 0U);
    EXPECT_EQ(discrete.memory_topology(), kora::MemoryTopology::Discrete);

    const auto uma_properties = make_properties({
        {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
         0},
        {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0},
    });
    const auto uma = DeviceCaps::from_vulkan(make_device(uma_properties)).to_device_info();
    const auto uma_choice = MemoryManager::select_memory_type(
        uma.memory_types(), uma.memory_type_preferences(), MemoryPolicy::Auto, 0b11);
    ASSERT_TRUE(uma_choice);
    EXPECT_EQ(uma_choice.value(), 0U);
    EXPECT_EQ(uma.memory_topology(), kora::MemoryTopology::Uma);
}

TEST(MemoryManager, RejectsEmptyMasksAndUnsatisfiedRequirements) {
    const auto properties = make_properties({{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0}});
    const auto info = DeviceCaps::from_vulkan(make_device(properties)).to_device_info();
    EXPECT_EQ(MemoryManager::select_memory_type(info.memory_types(), info.memory_type_preferences(),
                                                MemoryPolicy::Auto, 0)
                  .error()
                  .code,
              ErrorCode::InvalidArgument);
    EXPECT_EQ(MemoryManager::select_memory_type(info.memory_types(), info.memory_type_preferences(),
                                                MemoryPolicy::HostVisible, 1,
                                                MemoryTypeRequirements{.host_visible = true})
                  .error()
                  .code,
              ErrorCode::UnsupportedFeature);
}

TEST(BudgetLedger, EnforcesLimitAndRecoversAfterRelease) {
    BudgetLedger ledger({HeapBudget{100, 25, true}});
    ASSERT_TRUE(ledger.reserve(0, 80));
    EXPECT_EQ(ledger.reserve(0, 21).error().code, ErrorCode::OutOfMemoryBudget);
    auto snapshot = ledger.stats(0);
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(snapshot.value().reserved_bytes, 80U);
    EXPECT_EQ(snapshot.value().available_bytes, 20U);
    EXPECT_EQ(snapshot.value().externally_used_bytes, 25U);
    EXPECT_TRUE(snapshot.value().queried);

    ASSERT_TRUE(ledger.release(0, 30));
    ASSERT_TRUE(ledger.reserve(0, 20));
    EXPECT_EQ(ledger.stats(0).value().reserved_bytes, 70U);
}

TEST(BudgetLedger, CheckedArithmeticAndInvalidOperationsDoNotMutateState) {
    BudgetLedger ledger({HeapBudget{std::numeric_limits<std::uint64_t>::max(), 0, false}});
    ASSERT_TRUE(ledger.reserve(0, std::numeric_limits<std::uint64_t>::max() - 1));
    EXPECT_EQ(ledger.reserve(0, 2).error().code, ErrorCode::Overflow);
    EXPECT_EQ(ledger.release(0, std::numeric_limits<std::uint64_t>::max()).error().code,
              ErrorCode::InternalInvariantViolation);
    EXPECT_EQ(ledger.stats(0).value().reserved_bytes, std::numeric_limits<std::uint64_t>::max() - 1);
    EXPECT_EQ(ledger.reserve(1, 1).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(ledger.release(1, 1).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(ledger.stats(1).error().code, ErrorCode::InvalidArgument);
}

TEST(MemoryManager, AggregatesPoolTelemetryAndRejectsOverflow) {
    const std::vector<kora::detail::MemoryPoolStats> pools{
        {.reserved_bytes = 1024,
         .allocated_bytes = 256,
         .requested_bytes = 240,
         .free_bytes = 768,
         .internal_fragmentation_bytes = 16,
         .external_fragmentation_bytes = 128,
         .largest_free_block = 640,
         .high_water_mark = 512,
         .blocks = 1,
         .dedicated_allocations = 0,
         .live_allocations = 2,
         .device_allocation_calls = 1},
        {.reserved_bytes = 2048,
         .allocated_bytes = 512,
         .requested_bytes = 500,
         .free_bytes = 1536,
         .internal_fragmentation_bytes = 12,
         .external_fragmentation_bytes = 256,
         .largest_free_block = 1280,
         .high_water_mark = 1024,
         .blocks = 1,
         .dedicated_allocations = 1,
         .live_allocations = 3,
         .device_allocation_calls = 2},
    };
    const auto total = MemoryManager::aggregate_stats(pools);
    ASSERT_TRUE(total) << total.error().describe();
    EXPECT_EQ(total.value().reserved_bytes, 3072U);
    EXPECT_EQ(total.value().allocated_bytes, 768U);
    EXPECT_EQ(total.value().requested_bytes, 740U);
    EXPECT_EQ(total.value().free_bytes, 2304U);
    EXPECT_EQ(total.value().internal_fragmentation_bytes, 28U);
    EXPECT_EQ(total.value().external_fragmentation_bytes, 384U);
    EXPECT_EQ(total.value().largest_free_block, 1280U);
    EXPECT_EQ(total.value().high_water_mark, 1536U);
    EXPECT_EQ(total.value().blocks, 2U);
    EXPECT_EQ(total.value().dedicated_allocations, 1U);
    EXPECT_EQ(total.value().live_allocations, 5U);
    EXPECT_EQ(total.value().device_allocation_calls, 3U);

    const std::vector<kora::detail::MemoryPoolStats> overflowing{
        {.reserved_bytes = std::numeric_limits<std::uint64_t>::max()}, {.reserved_bytes = 1}};
    EXPECT_EQ(MemoryManager::aggregate_stats(overflowing).error().code, ErrorCode::Overflow);
}

TEST(MemoryManager, UsesBudgetExtensionOrConservativeHeapFraction) {
    auto raw = make_device(make_properties({{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0}}, 1000));
    const auto info = DeviceCaps::from_vulkan(raw).to_device_info();
    auto fallback = MemoryManager::create(raw, info, 0.5);
    ASSERT_TRUE(fallback) << fallback.error().describe();
    auto fallback_budget = fallback.value().budget(0);
    ASSERT_TRUE(fallback_budget);
    EXPECT_EQ(fallback_budget.value().limit_bytes, 500U);
    EXPECT_FALSE(fallback_budget.value().queried);

    VkExtensionProperties extension{};
    std::strncpy(extension.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
                 sizeof(extension.extensionName) - 1);
    raw.extensions.push_back(extension);
    raw.memory_budget_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    raw.memory_budget_properties.heapBudget[0] = 700;
    raw.memory_budget_properties.heapUsage[0] = 250;
    auto queried = MemoryManager::create(raw, info);
    ASSERT_TRUE(queried) << queried.error().describe();
    const auto queried_budget = queried.value().budget(0);
    ASSERT_TRUE(queried_budget);
    EXPECT_EQ(queried_budget.value().limit_bytes, 450U);
    EXPECT_EQ(queried_budget.value().externally_used_bytes, 250U);
    EXPECT_TRUE(queried_budget.value().queried);
    EXPECT_EQ(MemoryManager::create(raw, info, 1.1).error().code, ErrorCode::InvalidArgument);
}

}  // namespace
