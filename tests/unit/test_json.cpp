#include <gtest/gtest.h>

#include "core/json.hpp"

using kora::ErrorCode;
using kora::detail::json::parse;

TEST(JsonParser, ParsesStrictValuesAndPreservesIntegerTokens) {
    auto parsed =
        parse(R"({"large":18446744073709551615,"unicode":"\ud83d\ude80","items":[true,null,-2.5e1]})");
    ASSERT_TRUE(parsed) << parsed.error().describe();
    const auto* large = parsed.value().find("large");
    ASSERT_NE(large, nullptr);
    const auto* number = std::get_if<kora::detail::json::Value::Number>(&large->storage);
    ASSERT_NE(number, nullptr);
    EXPECT_EQ(number->text, "18446744073709551615");
}

TEST(JsonParser, AcceptsFiniteDecimalAndExponentTokensOnly) {
    auto parsed = parse("[1.25,-2.5e1]");
    ASSERT_TRUE(parsed) << parsed.error().describe();
    const auto& values = std::get<kora::detail::json::Value::Array>(parsed.value().storage);
    EXPECT_EQ(std::get<kora::detail::json::Value::Number>(values[0].storage).text, "1.25");
    EXPECT_EQ(std::get<kora::detail::json::Value::Number>(values[1].storage).text, "-2.5e1");
    EXPECT_FALSE(parse("1e9999"));
    EXPECT_FALSE(parse("1e-9999"));
}

TEST(JsonParser, RejectsMalformedDuplicateAndTruncatedInput) {
    for (const auto input : {"{\"a\":1,\"a\":2}", "[1,", "01", "\"\\ud800\"", "true false"}) {
        auto parsed = parse(input);
        ASSERT_FALSE(parsed) << input;
        EXPECT_EQ(parsed.error().code, ErrorCode::InvalidKernel);
        EXPECT_NE(parsed.error().message.find("JSON byte"), std::string::npos);
    }
}

TEST(JsonParser, RejectsInvalidRawUtf8) {
    const std::string invalid_utf8{"\"\xc0\x80\"", 4};
    auto parsed = parse(invalid_utf8);
    ASSERT_FALSE(parsed);
    EXPECT_EQ(parsed.error().code, ErrorCode::InvalidKernel);
    EXPECT_NE(parsed.error().message.find("UTF-8"), std::string::npos);
}

TEST(JsonParser, EnforcesConfiguredLimits) {
    EXPECT_EQ(parse("[]", {.maximum_bytes = 1}).error().code, ErrorCode::InvalidKernel);
    EXPECT_EQ(parse("[[[]]]", {.maximum_depth = 1}).error().code, ErrorCode::InvalidKernel);
    EXPECT_EQ(parse("\"abcd\"", {.maximum_string_bytes = 3}).error().code, ErrorCode::InvalidKernel);
    EXPECT_EQ(parse("[1,2]", {.maximum_values = 2}).error().code, ErrorCode::InvalidKernel);
}
