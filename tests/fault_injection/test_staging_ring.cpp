#include <gtest/gtest.h>

#include "memory/staging_ring.hpp"

namespace kora::detail {
namespace {

TEST(StagingRingPolicy, UsesTopologyAndMeasuredBenefitWithoutVendorRules) {
    EXPECT_TRUE(StagingRing::should_use_staging(MemoryTopology::Discrete));
    EXPECT_FALSE(StagingRing::should_use_staging(MemoryTopology::Uma));
    EXPECT_TRUE(StagingRing::should_use_staging(MemoryTopology::Uma, true));
    EXPECT_FALSE(StagingRing::should_use_staging(MemoryTopology::DiscreteWithBar));
    EXPECT_TRUE(StagingRing::should_use_staging(MemoryTopology::DiscreteWithBar, true));
    EXPECT_FALSE(StagingRing::should_use_staging(MemoryTopology::Unknown));
    EXPECT_TRUE(StagingRing::should_use_staging(MemoryTopology::Unknown, true));
}

}  // namespace
}  // namespace kora::detail
