#include <gtest/gtest.h>

#include "kora/kernel_metadata.hpp"

using namespace kora;

namespace {
constexpr auto valid_document = R"({
  "schema_version": 1,
  "kernel_id": "vector.add",
  "bindings": [{"set":0,"binding":0,"name":"a","kind":"storage_buffer","access":"read"}],
  "push_constant_size": 8,
  "push_constants": [{"name":"count","type":"u32","offset":0,"size":4},{"name":"stride","type":"u32","offset":4,"size":4}],
    "specialization_constants": [{"id":4,"name":"tile","type":"u64","default":18446744073709551615}],
  "local_size": [64,1,1],
  "cpu_fallback_key": "vector.add.cpu",
  "semantics": ["elementwise", "pure"]
})";
}

TEST(KernelMetadata, ParsesValidSchemaAndAllowsProgrammaticConstruction) {
    auto parsed = parse_kernel_metadata(valid_document);
    ASSERT_TRUE(parsed) << parsed.error().describe();
    EXPECT_EQ(parsed.value().kernel_id, "vector.add");
    EXPECT_EQ(parsed.value().bindings.size(), 1U);
    EXPECT_EQ(std::get<std::uint64_t>(parsed.value().specialization_constants[0].default_value), UINT64_MAX);
    KernelMetadata constructed;
    constructed.kernel_id = "manual";
    EXPECT_TRUE(validate_kernel_metadata(constructed));
}

TEST(KernelMetadata, RejectsUnknownFieldsAndInvalidMetadataPrecisely) {
    auto unknown = parse_kernel_metadata(
        R"({"schema_version":1,"kernel_id":"x","bindings":[],"push_constant_size":0,"push_constants":[],"specialization_constants":[],"local_size":[1,1,1],"surprise":true})");
    ASSERT_FALSE(unknown);
    EXPECT_EQ(unknown.error().code, ErrorCode::InvalidKernel);
    EXPECT_NE(unknown.error().message.find("$.surprise"), std::string::npos);
    auto duplicate = parse_kernel_metadata(
        R"({"schema_version":1,"kernel_id":"x","bindings":[{"set":0,"binding":0,"name":"a","kind":"storage_buffer","access":"read"},{"set":0,"binding":0,"name":"b","kind":"storage_buffer","access":"read"}],"push_constant_size":0,"push_constants":[],"specialization_constants":[],"local_size":[1,1,1]})");
    ASSERT_FALSE(duplicate);
    EXPECT_NE(duplicate.error().message.find("duplicate (set, binding)"), std::string::npos);
    auto mismatch = parse_kernel_metadata(
        R"({"schema_version":1,"kernel_id":"x","bindings":[],"push_constant_size":0,"push_constants":[],"specialization_constants":[{"id":0,"name":"flag","type":"bool","default":1}],"local_size":[1,1,1]})");
    ASSERT_FALSE(mismatch);
    EXPECT_NE(mismatch.error().message.find("does not match declared scalar type"), std::string::npos);
}

TEST(KernelMetadata, RejectsOverflowOverlapAndTruncatedDocuments) {
    KernelMetadata m;
    m.kernel_id = "range";
    m.push_constant_size = UINT32_MAX;
    EXPECT_NE(validate_kernel_metadata(m).error().message.find("portable metadata limit"), std::string::npos);
    m.push_constant_size = 8;
    m.push_constants = {{"x", "u32", 0, 4}, {"y", "u32", 2, 4}};
    EXPECT_NE(validate_kernel_metadata(m).error().message.find("overlap"), std::string::npos);
    auto truncated = parse_kernel_metadata("{\"schema_version\":1,\"kernel_id\":");
    ASSERT_FALSE(truncated);
    EXPECT_EQ(truncated.error().code, ErrorCode::InvalidKernel);
}

TEST(KernelMetadata, RejectsEveryTruncatedPrefixOfAValidManifest) {
    const std::string_view document{valid_document};
    for (std::size_t size = 0; size < document.size(); ++size) {
        auto parsed = parse_kernel_metadata(document.substr(0, size));
        EXPECT_FALSE(parsed) << "accepted truncated prefix of " << size << " bytes";
        if (!parsed) {
            EXPECT_EQ(parsed.error().code, ErrorCode::InvalidKernel);
        }
    }
}

TEST(KernelMetadata, ParsesFiniteFloatingSpecializationDefaults) {
    auto parsed = parse_kernel_metadata(
        R"({"schema_version":1,"kernel_id":"float","bindings":[],"push_constant_size":0,"push_constants":[],"specialization_constants":[{"id":0,"name":"scale","type":"f32","default":1.25e1}],"local_size":[1,1,1]})");
    ASSERT_TRUE(parsed) << parsed.error().describe();
    EXPECT_DOUBLE_EQ(std::get<double>(parsed.value().specialization_constants[0].default_value), 12.5);
}
