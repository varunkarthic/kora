#include <gtest/gtest.h>

#include "kora/detail/assert.hpp"

TEST(KoraAssert, UsesAssertionInDebugAndDoesNotEvaluateInRelease) {
#if defined(NDEBUG)
    int evaluations = 0;
    KORA_ASSERT(++evaluations != 0);
    EXPECT_EQ(evaluations, 0);
#else
    EXPECT_DEATH(KORA_ASSERT(false && "KORA_ASSERT invariant sentinel"), "KORA_ASSERT invariant sentinel");
#endif
}
