#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <thread>
#include <vector>

#include "core/performance_db.hpp"

namespace kora::detail {
namespace {

TEST(PerformanceDatabase, ComputesEwmaVarianceAndWindowQuantiles) {
    PerformanceDatabase::Options options;
    options.ewma_alpha = 0.5;
    PerformanceDatabase database(options);
    ASSERT_TRUE(database.observe("kernel/a", 10, 1));
    ASSERT_TRUE(database.observe("kernel/a", 20, 2));
    auto stats = database.query("kernel/a");
    ASSERT_TRUE(stats);
    EXPECT_EQ(stats.value().count, 2U);
    EXPECT_DOUBLE_EQ(stats.value().ewma, 15.0);
    EXPECT_DOUBLE_EQ(stats.value().ew_variance, 25.0);
    EXPECT_DOUBLE_EQ(stats.value().median, 10.0);
    EXPECT_DOUBLE_EQ(stats.value().p95, 20.0);
    EXPECT_DOUBLE_EQ(stats.value().minimum, 10.0);
    EXPECT_DOUBLE_EQ(stats.value().maximum, 20.0);
}

TEST(PerformanceDatabase, RejectsInvalidObservationsAndUsesSafeInvalidOptionsDefaults) {
    PerformanceDatabase::Options invalid;
    invalid.ewma_alpha = 0;
    invalid.sample_window = 0;
    PerformanceDatabase database(invalid);
    EXPECT_EQ(database.observe("", 1, 1).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(database.observe("valid", -1, 1).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(database.observe("valid", std::numeric_limits<double>::infinity(), 1).error().code,
              ErrorCode::InvalidArgument);
    ASSERT_TRUE(database.observe("valid", 10, 1));
    ASSERT_TRUE(database.observe("valid", 20, 2));
    EXPECT_DOUBLE_EQ(database.query("valid").value().ewma, 12.0);  // Default alpha is 0.2.
}

TEST(PerformanceDatabase, RejectsMadOutliersWithoutPoisoningEstimate) {
    PerformanceDatabase database;
    for (std::uint64_t index = 0; index < 10; ++index)
        ASSERT_TRUE(database.observe("copy/64k", 100, index + 1));
    ASSERT_TRUE(database.observe("copy/64k", 10000, 11));
    const auto stats = database.query("copy/64k");
    ASSERT_TRUE(stats);
    EXPECT_EQ(stats.value().count, 10U);
    EXPECT_EQ(stats.value().rejected_outliers, 1U);
    EXPECT_DOUBLE_EQ(stats.value().ewma, 100.0);
}

TEST(PerformanceDatabase, BoundedWindowAndAgingResetStaleHistory) {
    PerformanceDatabase::Options options;
    options.sample_window = 3;
    options.maximum_age_ms = 10;
    PerformanceDatabase database(options);
    for (std::uint64_t index = 0; index < 4; ++index)
        ASSERT_TRUE(database.observe("x", 10.0 + static_cast<double>(index), index + 1));
    EXPECT_DOUBLE_EQ(database.query("x").value().median, 12.0);
    ASSERT_TRUE(database.observe("x", 50, 100));
    const auto stats = database.query("x");
    ASSERT_TRUE(stats);
    EXPECT_EQ(stats.value().count, 1U);
    EXPECT_DOUBLE_EQ(stats.value().ewma, 50.0);
}

TEST(PerformanceDatabase, RejectedOutlierDoesNotPreventStaleHistoryFromAgingOut) {
    PerformanceDatabase::Options options;
    options.maximum_age_ms = 10;
    PerformanceDatabase database(options);
    for (std::uint64_t index = 0; index < 5; ++index) ASSERT_TRUE(database.observe("regime", 100, index + 1));
    ASSERT_TRUE(database.observe("regime", 10000, 6));
    ASSERT_EQ(database.query("regime").value().rejected_outliers, 1U);
    ASSERT_TRUE(database.observe("regime", 50, 20));
    const auto stats = database.query("regime");
    ASSERT_TRUE(stats);
    EXPECT_EQ(stats.value().count, 1U);
    EXPECT_EQ(stats.value().rejected_outliers, 0U);
    EXPECT_DOUBLE_EQ(stats.value().ewma, 50.0);
}

TEST(PerformanceDatabase, PersistsAndTreatsCorruptionAsRecoverableLoadFailure) {
    const auto directory = std::filesystem::temp_directory_path() / "kora-performance-db-test";
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    const auto path = directory / "stats.json";
    PerformanceDatabase source;
    ASSERT_TRUE(source.observe("kernel/\"quoted", 12.1234567890123, 100));
    ASSERT_TRUE(source.observe("kernel/\"quoted", 13.5, 101));
    ASSERT_TRUE(source.save(path));
    ASSERT_TRUE(source.observe("kernel/\"quoted", 12.75, 102));
    ASSERT_TRUE(source.save(path));  // Replacing an existing snapshot is atomic on every platform.
    PerformanceDatabase loaded;
    auto count = loaded.load(path, 103);
    ASSERT_TRUE(count) << count.error().describe();
    EXPECT_EQ(count.value(), 1U);
    EXPECT_EQ(loaded.query("kernel/\"quoted").value().count, 3U);
    EXPECT_DOUBLE_EQ(loaded.query("kernel/\"quoted").value().ewma,
                     source.query("kernel/\"quoted").value().ewma);
    {
        std::ofstream corrupt(path, std::ios::binary | std::ios::trunc);
        corrupt << "{broken";
    }
    EXPECT_FALSE(loaded.load(path, 104));
    EXPECT_EQ(loaded.size(), 1U);  // Failed load leaves the existing database untouched.
    std::filesystem::remove_all(directory, ignored);
}

TEST(PerformanceDatabase, IgnoresExpiredPersistedRows) {
    const auto directory = std::filesystem::temp_directory_path() / "kora-performance-db-aging-test";
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    const auto path = directory / "stats.json";
    PerformanceDatabase source;
    ASSERT_TRUE(source.observe("old", 1, 100));
    ASSERT_TRUE(source.save(path));
    PerformanceDatabase::Options options;
    options.maximum_age_ms = 10;
    PerformanceDatabase loaded(options);
    auto count = loaded.load(path, 1000);
    ASSERT_TRUE(count);
    EXPECT_EQ(count.value(), 0U);
    std::filesystem::remove_all(directory, ignored);
}

TEST(PerformanceDatabase, ConcurrentKeysAndUpdatesRemainConsistent) {
    PerformanceDatabase database;
    std::vector<std::thread> threads;
    for (int worker = 0; worker < 8; ++worker) {
        threads.emplace_back([&, worker] {
            for (int sample = 0; sample < 500; ++sample)
                EXPECT_TRUE(database.observe("key/" + std::to_string(worker), sample + 1.0,
                                             static_cast<std::uint64_t>(sample + 1)));
        });
    }
    for (int worker = 0; worker < 8; ++worker) {
        threads.emplace_back([&, worker] {
            for (int sample = 0; sample < 500; ++sample)
                EXPECT_TRUE(
                    database.observe("shared", 1.0, static_cast<std::uint64_t>(worker * 500 + sample + 1)));
        });
    }
    for (auto& thread : threads) thread.join();
    EXPECT_EQ(database.size(), 9U);
    for (int worker = 0; worker < 8; ++worker)
        EXPECT_EQ(database.query("key/" + std::to_string(worker)).value().count, 500U);
    EXPECT_EQ(database.query("shared").value().count, 4000U);
}

}  // namespace
}  // namespace kora::detail
