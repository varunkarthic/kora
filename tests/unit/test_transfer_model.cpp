#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "core/transfer_model.hpp"

namespace kora::detail {
namespace {

TEST(TransferSizeBucket, CoversLogarithmicBoundariesAndFullUint64Range) {
    EXPECT_EQ(transfer_size_bucket(0).index, 0);
    EXPECT_EQ(transfer_size_bucket(1).index, 0);
    EXPECT_EQ(transfer_size_bucket(1).lower_bytes, 0U);
    EXPECT_EQ(transfer_size_bucket(1).upper_bytes, 1U);
    EXPECT_EQ(transfer_size_bucket(2).index, 1);
    EXPECT_EQ(transfer_size_bucket(3).index, 1);
    EXPECT_EQ(transfer_size_bucket(4).index, 2);
    EXPECT_EQ(transfer_size_bucket(1024).index, 10);
    EXPECT_EQ(transfer_size_bucket(UINT64_MAX).index, 63);
    EXPECT_EQ(transfer_size_bucket(UINT64_MAX).upper_bytes, UINT64_MAX);
}

TEST(TransferModel, FitsLatencyAndBandwidthSeparatelyForEveryDirection) {
    TransferModelSet models;
    constexpr std::array paths{TransferPath::HostToDevice, TransferPath::DeviceToHost,
                               TransferPath::DeviceCopy, TransferPath::CrossDeviceViaHost,
                               TransferPath::UmaDirect};
    constexpr std::array<std::uint64_t, 3> sizes{4096, 65536, 1048576};
    for (const auto path : paths) {
        for (const auto bytes : sizes) {
            const auto duration = 500.0 + static_cast<double>(bytes) * 0.025;
            ASSERT_TRUE(models.observe(path, bytes, duration));
        }
        const auto model = models.model(path);
        ASSERT_TRUE(model);
        EXPECT_EQ(model.value().observations, 3U);
        EXPECT_NEAR(model.value().latency_ns, 500.0, 1e-7);
        EXPECT_NEAR(model.value().nanoseconds_per_byte, 0.025, 1e-10);
        double prior = 0;
        bool first = true;
        for (const auto bytes : {0ULL, 1ULL, 512ULL, 4096ULL, 16384ULL, 65536ULL, 1048576ULL, UINT64_MAX}) {
            const auto estimate = models.estimate_ns(path, bytes);
            ASSERT_TRUE(estimate);
            if (!first) EXPECT_GE(estimate.value(), prior);
            first = false;
            prior = estimate.value();
        }
    }
}

TEST(TransferModel, TracksPerPathBucketStatisticsAndEwma) {
    TransferModelSet::Options options;
    options.ewma_alpha = 0.5;
    TransferModelSet models(options);
    ASSERT_TRUE(models.observe(TransferPath::HostToDevice, 1024, 100));
    ASSERT_TRUE(models.observe(TransferPath::HostToDevice, 1500, 200));
    const auto bucket = transfer_size_bucket(1024).index;
    auto stats = models.bucket_stats(TransferPath::HostToDevice, bucket);
    ASSERT_TRUE(stats);
    EXPECT_EQ(stats.value().observations, 2U);
    EXPECT_DOUBLE_EQ(stats.value().ewma_bytes, 1262.0);
    EXPECT_DOUBLE_EQ(stats.value().ewma_duration_ns, 150.0);
    EXPECT_EQ(models.bucket_stats(TransferPath::DeviceToHost, bucket).value().observations, 0U);
    EXPECT_FALSE(models.estimate_ns(TransferPath::DeviceToHost, 1024));
}

TEST(TransferModel, AppliesConfigurableUmaPenaltyOnlyToDirectByteCost) {
    TransferModelSet::Options options;
    options.uma_direct_penalty = 2.0;
    TransferModelSet models(options);
    for (const auto path : {TransferPath::HostToDevice, TransferPath::UmaDirect}) {
        ASSERT_TRUE(models.observe(path, 4096, 1000));
        ASSERT_TRUE(models.observe(path, 65536, 2000));
    }
    const auto staged = models.estimate_ns(TransferPath::HostToDevice, 32768);
    const auto direct = models.estimate_ns(TransferPath::UmaDirect, 32768);
    ASSERT_TRUE(staged);
    ASSERT_TRUE(direct);
    EXPECT_GT(direct.value(), staged.value());
    EXPECT_DOUBLE_EQ(models.uma_direct_penalty(), 2.0);
    ASSERT_TRUE(models.set_uma_direct_penalty(1.25));
    EXPECT_DOUBLE_EQ(models.uma_direct_penalty(), 1.25);
}

TEST(TransferModel, RejectsInvalidSamplesPathsBucketsAndPenalties) {
    TransferModelSet models;
    EXPECT_EQ(models.observe(TransferPath::HostToDevice, 0, 10).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(models.observe(TransferPath::HostToDevice, 1, -1).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(models.observe(static_cast<TransferPath>(99), 1, 10).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(models.bucket_stats(TransferPath::HostToDevice, 64).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(models.set_uma_direct_penalty(0.9).error().code, ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace kora::detail
