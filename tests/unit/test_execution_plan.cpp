#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <new>

#include "kora/execution_plan.hpp"

namespace {

std::atomic<bool> track_allocations{false};
std::atomic<std::size_t> allocation_count{0};

struct ReplayState {
    std::size_t calls = 0;
    bool order_valid = true;
};

struct PrepareState {
    std::size_t calls = 0;
};

kora::Status prepare_task(const kora::TaskNode&, void* opaque) {
    ++static_cast<PrepareState*>(opaque)->calls;
    return {};
}

kora::Status fail_preparing_third_task(const kora::TaskNode& task, void*) {
    if (task.id == 3)
        return kora::Error(kora::ErrorCode::PipelineCreationFailed, "synthetic prepare failure");
    return {};
}

kora::Status record_task(const kora::TaskNode& task, void* opaque) {
    auto& state = *static_cast<ReplayState*>(opaque);
    const auto expected = static_cast<kora::TaskId>((state.calls % 20) + 1);
    if (task.id != expected) state.order_valid = false;
    ++state.calls;
    return {};
}

kora::Status fail_on_third_task(const kora::TaskNode& task, void*) {
    if (task.id == 3) return kora::Error(kora::ErrorCode::TransferFailed, "synthetic task failure");
    return {};
}

kora::TaskMetadata empty_task() {
    return {};
}

}  // namespace

void* operator new(std::size_t size) {
    if (void* memory = std::malloc(size == 0 ? 1 : size)) {
        if (track_allocations.load(std::memory_order_relaxed))
            allocation_count.fetch_add(1, std::memory_order_relaxed);
        return memory;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}
void operator delete(void* memory) noexcept {
    std::free(memory);
}
void operator delete[](void* memory) noexcept {
    ::operator delete(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
    ::operator delete(memory);
}
void operator delete[](void* memory, std::size_t) noexcept {
    ::operator delete(memory);
}

TEST(ExecutionPlan, ReplaysTwentyNodeGraphOneThousandTimesWithoutAllocating) {
    kora::TaskGraph graph;
    for (std::uint32_t index = 0; index < 20; ++index) {
        auto metadata = empty_task();
        metadata.type = kora::TaskType::HostCallback;
        metadata.priority = static_cast<std::int32_t>(index % 4);
        ASSERT_TRUE(graph.add_task(std::move(metadata)));
    }
    PrepareState prepare_state;
    const auto plan = kora::ExecutionPlan::create(graph, prepare_task, &prepare_state);
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan.value().size(), 20U);
    ASSERT_EQ(prepare_state.calls, 20U);

    ReplayState state;
    bool replay_succeeded = true;
    allocation_count.store(0, std::memory_order_relaxed);
    track_allocations.store(true, std::memory_order_relaxed);
    for (int replay = 0; replay < 1000; ++replay) {
        const auto status = plan.value().replay(record_task, &state);
        if (!status) {
            replay_succeeded = false;
            break;
        }
    }
    track_allocations.store(false, std::memory_order_relaxed);

    EXPECT_TRUE(replay_succeeded);
    EXPECT_EQ(state.calls, 20'000U);
    EXPECT_TRUE(state.order_valid);
    EXPECT_EQ(allocation_count.load(std::memory_order_relaxed), 0U);
}

TEST(ExecutionPlan, StopsAtFirstFailureAndAttachesTaskIdentity) {
    kora::TaskGraph graph;
    for (int index = 0; index < 5; ++index) ASSERT_TRUE(graph.add_task(empty_task()));
    const auto plan = kora::ExecutionPlan::create(graph);
    ASSERT_TRUE(plan);
    const auto status = plan.value().replay(fail_on_third_task);
    ASSERT_FALSE(status);
    EXPECT_EQ(status.error().code, kora::ErrorCode::TransferFailed);
    EXPECT_EQ(status.error().task_id, 3U);
}

TEST(ExecutionPlan, PreparationRunsOnceAndPreparationFailurePreventsPlanCreation) {
    kora::TaskGraph graph;
    for (int index = 0; index < 5; ++index) ASSERT_TRUE(graph.add_task(empty_task()));

    const auto failed = kora::ExecutionPlan::create(graph, fail_preparing_third_task);
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error().code, kora::ErrorCode::PipelineCreationFailed);
    EXPECT_EQ(failed.error().task_id, 3U);

    PrepareState state;
    const auto plan = kora::ExecutionPlan::create(graph, prepare_task, &state);
    ASSERT_TRUE(plan);
    EXPECT_EQ(state.calls, 5U);
}

TEST(ExecutionPlan, RejectsInvalidExecutorAndCyclicGraph) {
    kora::TaskGraph graph;
    const auto first = graph.add_task(empty_task()).value();
    const auto second = graph.add_task(empty_task()).value();
    ASSERT_TRUE(graph.add_dependency(first, second));
    ASSERT_TRUE(graph.add_dependency(second, first));
    const auto plan = kora::ExecutionPlan::create(graph);
    ASSERT_FALSE(plan);
    EXPECT_EQ(plan.error().code, kora::ErrorCode::GraphCycle);

    kora::TaskGraph empty_graph;
    const auto empty_plan = kora::ExecutionPlan::create(empty_graph);
    ASSERT_TRUE(empty_plan);
    EXPECT_EQ(empty_plan.value().replay(nullptr).error().code, kora::ErrorCode::InvalidArgument);
}
