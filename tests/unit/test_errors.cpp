#include <gtest/gtest.h>

#include "kora/errors.hpp"

using namespace kora;

TEST(Errors, ToStringCoversEveryCode) {
    for (int i = 0; i < static_cast<int>(ErrorCode::kCount_); ++i)
        EXPECT_STRNE(to_string(static_cast<ErrorCode>(i)), "Unknown") << i;
}

TEST(Errors, ResultValueAndError) {
    Result<int> a = 5;
    ASSERT_TRUE(a);
    EXPECT_EQ(a.value(), 5);
    Result<int> b = Error(ErrorCode::Timeout, "slow");
    ASSERT_FALSE(b);
    EXPECT_EQ(b.error().code, ErrorCode::Timeout);
    EXPECT_THROW((void)b.value_or_throw(), Exception);
}

TEST(Errors, MapAndThen) {
    auto r = Result<int>(2).map([](int x) { return x * 3; });
    ASSERT_TRUE(r);
    EXPECT_EQ(r.value(), 6);
    auto e = Result<int>(Error(ErrorCode::Cancelled, "x")).map([](int x) { return x; });
    EXPECT_EQ(e.error().code, ErrorCode::Cancelled);
    auto c = Result<int>(4).and_then([](int) -> Result<int> { return Error(ErrorCode::Overflow, "o"); });
    EXPECT_EQ(c.error().code, ErrorCode::Overflow);
}

TEST(Errors, StatusDefaultsOk) {
    Status s;
    EXPECT_TRUE(s.ok());
    Status f = Error(ErrorCode::DeviceLost, "gone");
    EXPECT_FALSE(f.ok());
    EXPECT_THROW(f.value_or_throw(), Exception);
}

TEST(Errors, CauseChainDescribe) {
    Error inner(ErrorCode::AllocationFailed, "vkAllocateMemory");
    inner.with_vk(-2);
    Error outer(ErrorCode::OutOfMemoryBudget, "no space");
    outer.with_device("dev0").with_task(7).with_kernel("k").with_recovery("evict").with_cause(inner);
    auto d = outer.describe();
    EXPECT_NE(d.find("OutOfMemoryBudget: no space"), std::string::npos);
    EXPECT_NE(d.find("[device=dev0]"), std::string::npos);
    EXPECT_NE(d.find("[task=7]"), std::string::npos);
    EXPECT_NE(d.find("[recovery=evict]"), std::string::npos);
    EXPECT_NE(d.find("<- AllocationFailed: vkAllocateMemory [vk=-2]"), std::string::npos);
}

namespace {
Result<int> half(int x) {
    if (x % 2) return Error(ErrorCode::InvalidArgument, "odd");
    return x / 2;
}
Result<int> quarter(int x) {
    KORA_TRY(h, half(x));
    KORA_TRY(q, half(h));
    return q;
}
}  // namespace

TEST(Errors, TryMacroPropagates) {
    EXPECT_EQ(quarter(8).value(), 2);
    EXPECT_EQ(quarter(6).error().code, ErrorCode::InvalidArgument);
}
