#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

#include "kernel/spirv.hpp"

namespace kora::detail::spirv {
namespace {

KernelMetadata expected_metadata() {
    KernelMetadata metadata;
    metadata.kernel_id = "spirv-reflection-fixture";
    metadata.local_size = {8, 2, 1};
    metadata.bindings.push_back(
        {0, 0, "data_buffer", KernelBindingKind::StorageBuffer, KernelAccess::ReadWrite});
    metadata.bindings.push_back(
        {0, 1, "second_buffer", KernelBindingKind::StorageBuffer, KernelAccess::ReadWrite});
    metadata.push_constant_size = 4;
    metadata.push_constants.push_back({"count", "u32", 0, 4});
    metadata.specialization_constants.push_back({7, "SCALE", "u32", std::uint64_t{2}});
    return metadata;
}

std::vector<std::uint32_t> fixture_words() {
#ifdef KORA_SPIRV_FIXTURE_PATH
    std::ifstream input(KORA_SPIRV_FIXTURE_PATH, std::ios::binary);
    if (!input) return {};
    std::vector<char> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (bytes.size() % sizeof(std::uint32_t) != 0) return {};
    std::vector<std::uint32_t> words(bytes.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return words;
#else
    return {};
#endif
}

std::size_t instruction_end(const std::vector<std::uint32_t>& words, std::size_t cursor) {
    return cursor + (words[cursor] >> 16U);
}

TEST(SpirvReflection, RejectsTruncatedAndCorruptModulesWithoutTools) {
    KernelMetadata metadata;
    metadata.kernel_id = "invalid-module";
    const DeviceCapabilities capabilities{};
    EXPECT_FALSE(validate_and_reflect({}, metadata, capabilities));
    const std::vector<std::uint32_t> bad_magic{0, 0x00010000, 0, 1, 0};
    EXPECT_FALSE(validate_and_reflect(bad_magic, metadata, capabilities));
    const std::vector<std::uint32_t> truncated_instruction{0x07230203, 0x00010000, 0, 1, 0, 0x0002000b};
    EXPECT_FALSE(validate_and_reflect(truncated_instruction, metadata, capabilities));
}

#ifdef KORA_SPIRV_FIXTURE_PATH
TEST(SpirvReflection, ReflectsAndChecksComputeInterfaceAgainstMetadata) {
    const auto words = fixture_words();
    ASSERT_FALSE(words.empty());
    const DeviceCapabilities capabilities{};
    auto reflection = validate_and_reflect(words, expected_metadata(), capabilities);
    ASSERT_TRUE(reflection) << reflection.error().message;
    EXPECT_EQ(reflection.value().entry_point, "main");
    EXPECT_EQ(reflection.value().local_size, (std::array<std::uint32_t, 3>{8, 2, 1}));
    ASSERT_EQ(reflection.value().bindings.size(), 2U);
    EXPECT_EQ(reflection.value().bindings[0].set, 0U);
    EXPECT_EQ(reflection.value().bindings[0].binding, 0U);
    EXPECT_EQ(reflection.value().bindings[0].kind, KernelBindingKind::StorageBuffer);
    EXPECT_EQ(reflection.value().bindings[1].binding, 1U);
    EXPECT_EQ(reflection.value().bindings[1].kind, KernelBindingKind::StorageBuffer);
    EXPECT_EQ(reflection.value().push_constant_size, 4U);
    ASSERT_EQ(reflection.value().specialization_constants.size(), 1U);
    EXPECT_EQ(reflection.value().specialization_constants[0].id, 7U);
}

TEST(SpirvReflection, ReportsPreciseMetadataMismatch) {
    auto metadata = expected_metadata();
    metadata.local_size = {1, 1, 1};
    auto result = validate_and_reflect(fixture_words(), metadata, DeviceCapabilities{});
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().message.find("local_size"), std::string::npos);
}

TEST(SpirvReflection, RejectsCapabilitiesNotPresentInDeviceSnapshot) {
    auto words = fixture_words();
    ASSERT_FALSE(words.empty());
    std::size_t cursor = 5;
    while (cursor < words.size() && (words[cursor] & 0xffffU) == 17U) {
        cursor = instruction_end(words, cursor);
    }
    words.insert(words.begin() + static_cast<std::ptrdiff_t>(cursor), {0x00020011U, 10U});
    const auto result = validate_and_reflect(words, expected_metadata(), DeviceCapabilities{});
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().message.find("capability 10"), std::string::npos);
}

TEST(SpirvReflection, RejectsNonComputeEntryPoint) {
    auto words = fixture_words();
    ASSERT_FALSE(words.empty());
    for (std::size_t cursor = 5; cursor < words.size(); cursor = instruction_end(words, cursor)) {
        if ((words[cursor] & 0xffffU) == 15U) {  // OpEntryPoint
            words[cursor + 1] = 4U;              // Fragment execution model
            break;
        }
    }
    EXPECT_FALSE(validate_and_reflect(words, expected_metadata(), DeviceCapabilities{}));
}
#endif

}  // namespace
}  // namespace kora::detail::spirv
