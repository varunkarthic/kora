#include <gtest/gtest.h>

#include <array>

#include "kora/runtime.hpp"

TEST(CpuFallback, ContextEnforcesDeclaredAccessAndExposesPushConstants) {
    kora::BufferManager manager;
    auto output = manager.create(8);
    ASSERT_TRUE(output);
    const std::array<std::byte, 2> push{std::byte{0x12}, std::byte{0x34}};
    auto runtime = kora::Runtime::create();
    if (!runtime && runtime.error().code == kora::ErrorCode::NoVulkanImplementation) GTEST_SKIP();
    ASSERT_TRUE(runtime) << runtime.error().describe();
    ASSERT_TRUE(runtime.value().register_cpu_fallback("unit.context", [](kora::CpuTaskContext& task) {
        EXPECT_EQ(task.buffer_count(), 1U);
        EXPECT_EQ(task.push_constants().size(), 2U);
        EXPECT_EQ(task.read(0).error().code, kora::ErrorCode::InvalidKernelBinding);
        auto writable = task.write(0);
        EXPECT_TRUE(writable);
        if (writable) writable.value()[0] = std::byte{0x5a};
        return kora::Status{};
    }));
    const std::array buffers{output.value()};
    const std::array accesses{kora::KernelAccess::Write};
    auto event = runtime.value().run_cpu_fallback("unit.context", buffers, accesses, push);
    ASSERT_TRUE(event) << event.error().describe();
    ASSERT_TRUE(event.value().wait());
    EXPECT_EQ(output.value().read_bytes().value()[0], std::byte{0x5a});
}

TEST(CpuFallback, RejectsUnknownKeyAndInvalidDeclarations) {
    auto runtime = kora::Runtime::create();
    if (!runtime && runtime.error().code == kora::ErrorCode::NoVulkanImplementation) GTEST_SKIP();
    ASSERT_TRUE(runtime) << runtime.error().describe();
    EXPECT_EQ(runtime.value()
                  .register_cpu_fallback("", [](kora::CpuTaskContext&) { return kora::Status{}; })
                  .error()
                  .code,
              kora::ErrorCode::InvalidArgument);
    auto missing = runtime.value().run_cpu_fallback("missing", {}, {});
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().code, kora::ErrorCode::InvalidKernel);
    auto mismatch = runtime.value().run_cpu_fallback("missing", {}, std::array{kora::KernelAccess::Read});
    ASSERT_FALSE(mismatch);
    EXPECT_EQ(mismatch.error().code, kora::ErrorCode::InvalidArgument);
}
