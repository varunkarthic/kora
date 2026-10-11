#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <thread>
#include <vector>

#include "core/event_state.hpp"

namespace kora {
namespace {

TEST(Event, TimeoutDoesNotChangePendingStateAndMultipleWaitersComplete) {
    detail::EventPromise promise;
    auto event = promise.event();
    auto timeout = event.wait_for(std::chrono::nanoseconds{0});
    ASSERT_FALSE(timeout);
    EXPECT_EQ(timeout.error().code, ErrorCode::Timeout);
    EXPECT_FALSE(event.done());

    std::array<Status, 8> results;
    std::array<std::thread, 8> waiters;
    for (std::size_t i = 0; i < waiters.size(); ++i)
        waiters[i] = std::thread([&, i] { results[i] = event.wait(); });
    promise.complete(ok_status());
    for (auto& waiter : waiters) waiter.join();
    EXPECT_TRUE(event.done());
    for (const auto& result : results) EXPECT_TRUE(result);
}

TEST(Event, ThenPreservesFailureAndRunsOnce) {
    detail::EventPromise promise;
    auto source = promise.event();
    std::atomic<int> calls{0};
    auto chained = source.then([&](Status status) {
        ++calls;
        return status;
    });
    ASSERT_TRUE(chained);
    promise.complete(Error(ErrorCode::Cancelled, "source failed"));
    auto result = chained.value().wait();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::Cancelled);
    EXPECT_EQ(calls.load(), 1);
}

TEST(Event, WhenAllWaitsForEveryInputAndReturnsFirstInputFailure) {
    detail::EventPromise first_promise;
    detail::EventPromise second_promise;
    const std::array events{first_promise.event(), second_promise.event()};
    auto combined = Event::when_all(events);
    ASSERT_TRUE(combined);
    first_promise.complete(Error(ErrorCode::Cancelled, "first failure"));
    EXPECT_FALSE(combined.value().done());
    second_promise.complete(Error(ErrorCode::Timeout, "second failure"));
    auto result = combined.value().wait();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::Cancelled);
}

TEST(Event, WhenAllEmptyInputCompletesImmediately) {
    auto combined = Event::when_all(std::span<const Event>{});
    ASSERT_TRUE(combined);
    EXPECT_TRUE(combined.value().done());
    EXPECT_TRUE(combined.value().wait());
}

}  // namespace
}  // namespace kora
