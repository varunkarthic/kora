#pragma once

#include <gtest/gtest.h>

// Use this at the start of a test that requires a usable Vulkan compute implementation. The
// condition should come from the test fixture's normal discovery path so initialization failures
// are reported consistently instead of being converted into crashes or false failures.
#define KORA_REQUIRE_VULKAN(condition)                                              \
    do {                                                                            \
        if (!(condition)) {                                                         \
            GTEST_SKIP() << "No usable Vulkan compute implementation is available"; \
        }                                                                           \
    } while (false)
