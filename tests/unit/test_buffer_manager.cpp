#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <thread>
#include <vector>

#include "kora/buffer.hpp"

using kora::BufferId;
using kora::BufferManager;
using kora::BufferUsage;
using kora::ErrorCode;
using kora::MapAccess;

namespace {

TEST(BufferManager, CopyingHandlesRetainsOneLogicalAllocation) {
    BufferManager manager;
    auto result = manager.create(64, BufferUsage::Storage);
    ASSERT_TRUE(result);
    auto handle = std::move(result).value();
    const auto id = handle.id();
    auto copy = handle;

    auto info = handle.info();
    ASSERT_TRUE(info);
    EXPECT_EQ(info.value().reference_count, 2U);
    EXPECT_EQ(info.value().size_bytes, 64U);
    EXPECT_EQ(info.value().usage, BufferUsage::Storage);
    EXPECT_FALSE(info.value().pending_free);

    handle = {};
    EXPECT_EQ(manager.inspect(id).value().reference_count, 1U);
    copy = {};
    EXPECT_EQ(manager.deferred_count(), 0U);  // Timeline value defaults to completed zero.
    EXPECT_EQ(manager.inspect(id).error().code, ErrorCode::StaleHandle);
    EXPECT_EQ(manager.advance_completion(0).value(), 0U);
    EXPECT_EQ(manager.inspect(id).error().code, ErrorCode::StaleHandle);
}

TEST(BufferManager, DebugHostShadowCanariesDetectOutOfRangeMappedWrites) {
#if defined(NDEBUG)
    GTEST_SKIP() << "host-shadow canaries are disabled in Release builds";
#else
    BufferManager manager;
    auto created = manager.create(8);
    ASSERT_TRUE(created);
    auto mapped = created.value().map(MapAccess::Write);
    ASSERT_TRUE(mapped);
    mapped.value().data()[mapped.value().size()] = std::byte{0};
    const auto read = created.value().read_bytes();
    ASSERT_FALSE(read);
    EXPECT_EQ(read.error().code, ErrorCode::InternalInvariantViolation);
#endif
}

TEST(BufferManager, DefersReclamationUntilTheLastUseCompletes) {
    BufferManager manager;
    BufferId old_id;
    {
        auto handle = std::move(manager.create(32)).value();
        old_id = handle.id();
        ASSERT_TRUE(handle.mark_used(7));
    }
    EXPECT_EQ(manager.deferred_count(), 1U);
    auto pending = manager.inspect(old_id);
    ASSERT_TRUE(pending);
    EXPECT_TRUE(pending.value().pending_free);
    EXPECT_EQ(pending.value().reference_count, 0U);
    EXPECT_EQ(manager.advance_completion(6).value(), 0U);
    EXPECT_EQ(manager.deferred_count(), 1U);
    {
        auto premature_reuse = manager.create(16);
        ASSERT_TRUE(premature_reuse);
        EXPECT_NE(premature_reuse.value().id().slot, old_id.slot);
    }
    EXPECT_EQ(manager.advance_completion(7).value(), 1U);
    EXPECT_EQ(manager.deferred_count(), 0U);

    auto replacement = manager.create(16);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(replacement.value().id().slot, old_id.slot);
    EXPECT_EQ(replacement.value().id().generation, old_id.generation + 1);
    EXPECT_EQ(manager.inspect(old_id).error().code, ErrorCode::StaleHandle);
#ifndef NDEBUG
    EXPECT_DEATH(static_cast<void>(manager.mark_used(old_id, 8)), "stale buffer generation");
#else
    EXPECT_EQ(manager.mark_used(old_id, 8).error().code, ErrorCode::StaleHandle);
#endif
}

TEST(BufferManager, MappedPinPreventsReuseAndWritableMapsTrackDirtyRange) {
    BufferManager manager;
    kora::MappedBytes mapped;
    BufferId id;
    {
        auto handle = std::move(manager.create(128)).value();
        id = handle.id();
        auto mapped_result = handle.map(MapAccess::Write, 24, 32);
        ASSERT_TRUE(mapped_result);
        mapped = std::move(mapped_result).value();
        std::fill(mapped.bytes().begin(), mapped.bytes().end(), std::byte{0x2a});
        EXPECT_EQ(handle.info().value().pin_count, 1U);
        ASSERT_EQ(manager.dirty_ranges(id).value().size(), 1U);
        EXPECT_EQ(manager.dirty_ranges(id).value()[0].offset, 24U);
        EXPECT_EQ(manager.dirty_ranges(id).value()[0].size, 32U);
    }
    EXPECT_EQ(manager.deferred_count(), 1U);
    EXPECT_EQ(manager.advance_completion(0).value(), 0U);
    mapped = {};
    EXPECT_EQ(manager.deferred_count(), 0U);
    EXPECT_EQ(manager.inspect(id).error().code, ErrorCode::StaleHandle);
}

TEST(BufferManager, TypedBufferMapsElementsAndChecksByteSizeOverflow) {
    BufferManager manager;
    auto created = manager.create<std::uint32_t>(4, BufferUsage::Storage);
    ASSERT_TRUE(created);
    auto buffer = std::move(created).value();
    EXPECT_EQ(buffer.size(), 4U);
    EXPECT_EQ(buffer.size_bytes(), 4U * sizeof(std::uint32_t));
    {
        auto mapped = buffer.map(MapAccess::Write);
        ASSERT_TRUE(mapped);
        ASSERT_EQ(mapped.value().size(), 4U);
        mapped.value()[0] = 10;
        mapped.value()[1] = 20;
        mapped.value()[2] = 30;
        mapped.value()[3] = 40;
    }
    auto read = buffer.map(MapAccess::Read);
    ASSERT_TRUE(read);
    EXPECT_EQ(read.value()[0], 10U);
    EXPECT_EQ(read.value()[3], 40U);
    EXPECT_EQ(manager.dirty_ranges(buffer.id()).value().size(), 1U);
    EXPECT_EQ(manager.create(0).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(manager.create<std::uint64_t>(UINT64_MAX).error().code, ErrorCode::Overflow);
}

TEST(BufferManager, EnforcesMonotonicTimelineAndValidMapRanges) {
    BufferManager manager;
    auto handle = std::move(manager.create(64)).value();
    ASSERT_TRUE(handle.mark_used(5));
    EXPECT_EQ(handle.mark_used(4).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(handle.map(MapAccess::Read, 65).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(handle.map(MapAccess::Read, 48, 17).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(manager.advance_completion(5).value(), 0U);
    EXPECT_EQ(manager.advance_completion(4).error().code, ErrorCode::InvalidArgument);
}

TEST(BufferManager, HostContentVersionAdvancesForWritesAndWritableMappings) {
    BufferManager manager;
    auto handle = std::move(manager.create(16)).value();
    ASSERT_EQ(handle.host_version().value(), 1U);
    const std::array<std::byte, 16> bytes{};
    ASSERT_TRUE(handle.write_bytes(bytes));
    EXPECT_EQ(handle.host_version().value(), 2U);
    const std::array<std::byte, 16> stale_bytes{std::byte{0x55}};
    const auto stale_write = handle.write_bytes_if_version(stale_bytes, 1);
    ASSERT_TRUE(stale_write);
    EXPECT_FALSE(stale_write.value().written);
    EXPECT_EQ(handle.host_version().value(), 2U);
    const auto current_write = handle.write_bytes_if_version(stale_bytes, 2);
    ASSERT_TRUE(current_write);
    ASSERT_TRUE(current_write.value().written);
    EXPECT_EQ(current_write.value().host_version, 3U);
    {
        auto mapping = handle.map(MapAccess::Write, 4, 4);
        ASSERT_TRUE(mapping);
        EXPECT_EQ(handle.host_version().value(), 4U);
    }
    {
        auto mapping = handle.map(MapAccess::Read, 0, 1);
        ASSERT_TRUE(mapping);
        EXPECT_EQ(handle.host_version().value(), 4U);
    }
}

TEST(BufferManager, LostDeviceContentsRequireACompleteHostRewrite) {
    BufferManager manager;
    auto handle = std::move(manager.create(4)).value();
    ASSERT_TRUE(handle.mark_data_lost());
    EXPECT_EQ(handle.read_bytes().error().code, ErrorCode::DataLoss);
    EXPECT_EQ(handle.map(MapAccess::Read).error().code, ErrorCode::DataLoss);
    EXPECT_EQ(handle.map(MapAccess::Write, 0, 2).error().code, ErrorCode::DataLoss);

    const std::array<std::byte, 4> replacement{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    ASSERT_TRUE(handle.write_bytes(replacement));
    const auto restored = handle.read_bytes();
    ASSERT_TRUE(restored);
    EXPECT_EQ(restored.value(), std::vector<std::byte>(replacement.begin(), replacement.end()));
}

TEST(BufferManager, ConcurrentAllocateMapReleaseCyclesRemainConsistent) {
    BufferManager manager;
    constexpr std::size_t kThreads = 8;
    constexpr std::size_t kIterations = 250;
    std::array<std::thread, kThreads> workers;
    std::array<std::size_t, kThreads> failures{};
    for (std::size_t thread_index = 0; thread_index < kThreads; ++thread_index) {
        workers[thread_index] = std::thread([&, thread_index] {
            for (std::size_t iteration = 0; iteration < kIterations; ++iteration) {
                auto created = manager.create(64 + iteration, BufferUsage::TransferDestination);
                if (!created) {
                    ++failures[thread_index];
                    continue;
                }
                auto handle = std::move(created).value();
                auto mapping = handle.map(MapAccess::Write);
                if (!mapping) {
                    ++failures[thread_index];
                    continue;
                }
                mapping.value().data()[0] = static_cast<std::byte>(iteration & 0xffU);
                if (!handle.mark_used(3)) ++failures[thread_index];
            }
        });
    }
    for (auto& worker : workers) worker.join();
    for (const auto failure_count : failures) EXPECT_EQ(failure_count, 0U);
    EXPECT_EQ(manager.advance_completion(2).value(), 0U);
    EXPECT_EQ(manager.deferred_count(), kThreads * kIterations);
    EXPECT_EQ(manager.advance_completion(3).value(), kThreads * kIterations);
    EXPECT_EQ(manager.deferred_count(), 0U);
}

}  // namespace
