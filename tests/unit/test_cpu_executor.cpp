#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "kora/cpu_executor.hpp"

namespace {

kora::CpuExecutorOptions one_worker() {
    kora::CpuExecutorOptions options;
    options.worker_count = 1;
    return options;
}

TEST(CpuExecutor, DrainsOneMillionTasksAcrossStealingWorkers) {
    constexpr std::size_t task_count = 1'000'000;
    kora::CpuExecutorOptions options;
    options.worker_count = 4;
    auto created = kora::CpuExecutor::create(options);
    ASSERT_TRUE(created);
    auto executor = std::move(created).value();
    EXPECT_EQ(executor->worker_count(), 4U);

    std::atomic<std::size_t> completed{0};
    for (std::size_t index = 0; index < task_count; ++index) {
        auto task = executor->submit(
            [&completed] {
                completed.fetch_add(1, std::memory_order_relaxed);
                return kora::ok_status();
            },
            static_cast<kora::CpuTaskPriority>(index % 4));
        ASSERT_TRUE(task) << "submission=" << index << ": " << task.error().describe();
    }
    ASSERT_TRUE(executor->shutdown());
    EXPECT_EQ(completed.load(std::memory_order_relaxed), task_count);
    EXPECT_EQ(executor->submit([] { return kora::ok_status(); }).error().code, kora::ErrorCode::Cancelled);
}

TEST(CpuExecutor, HigherPriorityLaneRunsBeforeQueuedLowerPriorityWork) {
    auto created = kora::CpuExecutor::create(one_worker());
    ASSERT_TRUE(created);
    auto executor = std::move(created).value();

    std::mutex mutex;
    std::condition_variable changed;
    bool started = false;
    bool release = false;
    std::vector<int> order;
    auto blocker = executor->submit([&] {
        std::unique_lock lock(mutex);
        started = true;
        changed.notify_all();
        changed.wait(lock, [&] { return release; });
        return kora::ok_status();
    });
    ASSERT_TRUE(blocker);
    {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return started; });
    }
    auto low = executor->submit(
        [&] {
            order.push_back(0);
            return kora::ok_status();
        },
        kora::CpuTaskPriority::Low);
    auto high = executor->submit(
        [&] {
            order.push_back(1);
            return kora::ok_status();
        },
        kora::CpuTaskPriority::High);
    ASSERT_TRUE(low);
    ASSERT_TRUE(high);
    {
        std::lock_guard lock(mutex);
        release = true;
    }
    changed.notify_all();
    ASSERT_TRUE(executor->wait(blocker.value()));
    ASSERT_TRUE(executor->wait(low.value()));
    ASSERT_TRUE(executor->wait(high.value()));
    ASSERT_EQ(order.size(), 2U);
    EXPECT_EQ(order[0], 1);
    EXPECT_EQ(order[1], 0);
}

TEST(CpuExecutor, CooperativeWaitPreventsSingleWorkerRecursiveDeadlock) {
    auto created = kora::CpuExecutor::create(one_worker());
    ASSERT_TRUE(created);
    auto executor = std::move(created).value();
    constexpr int recursion_depth = 96;
    std::function<kora::Status(int)> recurse;
    recurse = [&](int depth) -> kora::Status {
        if (depth == 0) return {};
        auto child = executor->submit([&recurse, depth] { return recurse(depth - 1); });
        if (!child) return child.error();
        return executor->wait(child.value());
    };
    auto root = executor->submit([&recurse] { return recurse(recursion_depth); });
    ASSERT_TRUE(root);
    EXPECT_TRUE(executor->wait(root.value()));
    EXPECT_TRUE(executor->shutdown());
}

TEST(CpuExecutor, CapturesTaskExceptionsIntoEventStatus) {
    auto created = kora::CpuExecutor::create(one_worker());
    ASSERT_TRUE(created);
    auto executor = std::move(created).value();
    auto task = executor->submit([]() -> kora::Status { throw std::runtime_error("synthetic CPU failure"); });
    ASSERT_TRUE(task);
    const auto status = executor->wait(task.value());
    ASSERT_FALSE(status);
    EXPECT_EQ(status.error().code, kora::ErrorCode::InternalInvariantViolation);
    EXPECT_NE(status.error().message.find("synthetic CPU failure"), std::string::npos);
}

}  // namespace
