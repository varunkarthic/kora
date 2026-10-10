#include <gtest/gtest.h>

#include "support/vulkan_test.hpp"

TEST(VulkanTestSupport, MissingImplementationSkipsCleanly) {
    KORA_REQUIRE_VULKAN(false);
    FAIL() << "execution must stop after KORA_REQUIRE_VULKAN reports an unavailable implementation";
}
