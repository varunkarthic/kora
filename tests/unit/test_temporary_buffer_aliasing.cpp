#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "kora/execution_plan.hpp"
#include "kora/temporary_buffer_aliasing.hpp"

namespace {

using kora::AccessMode;
using kora::BufferAccess;
using kora::BufferId;
using kora::TaskGraph;
using kora::TaskMetadata;
using kora::TemporaryBufferRequirements;

constexpr BufferId a{1, 1};
constexpr BufferId b{2, 1};
constexpr BufferId c{3, 1};

TemporaryBufferRequirements requirement(BufferId id, std::uint64_t size = 64) {
    return {id, size, 16, 7, 0b11, false};
}

TaskMetadata access(std::initializer_list<BufferAccess> accesses) {
    TaskMetadata metadata;
    metadata.accesses = accesses;
    return metadata;
}

kora::TaskGraph chain_graph() {
    TaskGraph graph;
    EXPECT_TRUE(graph.add_task(access({{a, AccessMode::Write}})));
    EXPECT_TRUE(graph.add_task(access({{a, AccessMode::Read}, {b, AccessMode::Write}})));
    EXPECT_TRUE(graph.add_task(access({{b, AccessMode::Read}, {c, AccessMode::Write}})));
    EXPECT_TRUE(graph.add_task(access({{c, AccessMode::Read}})));
    return graph;
}

std::uint32_t allocation_for(const kora::TemporaryBufferAliasingPlan& plan, BufferId id) {
    const auto found = std::find_if(plan.placements.begin(), plan.placements.end(),
                                    [id](const auto& placement) { return placement.buffer == id; });
    return found == plan.placements.end() ? UINT32_MAX : found->allocation_id;
}

TEST(TemporaryBufferAliasing, ColorsNonOverlappingChainLifetimes) {
    const auto task_graph = chain_graph();
    const auto graph = task_graph.finalize().value();
    const std::array buffers{requirement(a), requirement(b), requirement(c)};
    const auto plan = kora::plan_temporary_buffer_aliasing(graph, buffers);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan.value().allocations.size(), 2U);
    EXPECT_EQ(plan.value().unaliased_bytes, 192U);
    EXPECT_EQ(plan.value().planned_bytes, 128U);
    EXPECT_EQ(plan.value().planned_savings_bytes(), 64U);
    EXPECT_EQ(allocation_for(plan.value(), a), allocation_for(plan.value(), c));
    EXPECT_NE(allocation_for(plan.value(), a), allocation_for(plan.value(), b));
    EXPECT_NE(allocation_for(plan.value(), b), allocation_for(plan.value(), c));
}

TEST(TemporaryBufferAliasing, KeepsPersistentAndIncompatibleRequirementsSeparate) {
    TaskGraph graph;
    ASSERT_TRUE(graph.add_task(access({{a, AccessMode::Write}})));
    ASSERT_TRUE(graph.add_task(access({{a, AccessMode::Read}})));
    ASSERT_TRUE(graph.add_task(access({{b, AccessMode::Write}})));
    ASSERT_TRUE(graph.add_task(access({{b, AccessMode::Read}})));
    ASSERT_TRUE(graph.add_task(access({{c, AccessMode::Write}})));
    ASSERT_TRUE(graph.add_task(access({{c, AccessMode::Read}})));
    const auto finalized = graph.finalize();
    ASSERT_TRUE(finalized);

    auto persistent = requirement(a);
    persistent.persistent = true;
    auto different_usage = requirement(b);
    different_usage.usage_class++;
    auto different_memory = requirement(c);
    different_memory.memory_type_bits = 0b100;
    const std::array buffers{persistent, different_usage, different_memory};
    const auto plan = kora::plan_temporary_buffer_aliasing(finalized.value(), buffers);
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan.value().allocations.size(), 3U);
    EXPECT_TRUE(plan.value().allocations[allocation_for(plan.value(), a)].persistent_exclusive);
    EXPECT_NE(allocation_for(plan.value(), a), allocation_for(plan.value(), b));
    EXPECT_NE(allocation_for(plan.value(), b), allocation_for(plan.value(), c));
}

TEST(TemporaryBufferAliasing, DoesNotAliasIndependentConcurrentBranches) {
    TaskGraph graph;
    ASSERT_TRUE(graph.add_task(access({{a, AccessMode::Write}})));
    ASSERT_TRUE(graph.add_task(access({{b, AccessMode::Write}})));
    const auto finalized = graph.finalize();
    ASSERT_TRUE(finalized);
    const std::array buffers{requirement(a), requirement(b)};
    const auto plan = kora::plan_temporary_buffer_aliasing(finalized.value(), buffers);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan.value().allocations.size(), 2U);
    EXPECT_NE(allocation_for(plan.value(), a), allocation_for(plan.value(), b));
}

TEST(TemporaryBufferAliasing, RejectsInvalidAndUnreferencedRequirements) {
    const auto task_graph = chain_graph();
    const auto graph = task_graph.finalize().value();
    for (auto invalid : {requirement(a, 0), requirement(a)}) {
        if (invalid.size_bytes != 0) invalid.alignment = 3;
        const std::array buffers{invalid};
        EXPECT_FALSE(kora::plan_temporary_buffer_aliasing(graph, buffers));
    }
    auto no_memory = requirement(a);
    no_memory.memory_type_bits = 0;
    EXPECT_FALSE(kora::plan_temporary_buffer_aliasing(graph, std::span(&no_memory, 1)));

    const auto empty =
        kora::plan_temporary_buffer_aliasing(graph, std::span<const TemporaryBufferRequirements>{});
    ASSERT_TRUE(empty);
    EXPECT_TRUE(empty.value().allocations.empty());

    const std::array duplicate{requirement(a), requirement(a)};
    EXPECT_FALSE(kora::plan_temporary_buffer_aliasing(graph, duplicate));
    const std::array unused{requirement(BufferId{99, 1})};
    EXPECT_FALSE(kora::plan_temporary_buffer_aliasing(graph, unused));
}

struct Simulation {
    const kora::TemporaryBufferAliasingPlan* plan = nullptr;
    std::vector<std::vector<std::uint32_t>>* allocations = nullptr;
    bool output_matches = false;
};

std::vector<std::uint32_t>& storage(Simulation& simulation, BufferId id) {
    const auto allocation = allocation_for(*simulation.plan, id);
    return simulation.allocations->at(allocation);
}

kora::Status simulate_aliasing(const kora::TaskNode& task, void* opaque) {
    auto& simulation = *static_cast<Simulation*>(opaque);
    if (task.id == 1) {
        auto& output = storage(simulation, a);
        for (std::size_t index = 0; index < output.size(); ++index)
            output[index] = static_cast<std::uint32_t>(index * 3U + 1U);
    } else if (task.id == 2) {
        storage(simulation, b) = storage(simulation, a);
    } else if (task.id == 3) {
        storage(simulation, c) = storage(simulation, b);
    } else if (task.id == 4) {
        const auto& result = storage(simulation, c);
        simulation.output_matches = result.size() == 8;
        for (std::size_t index = 0; index < result.size(); ++index)
            simulation.output_matches &= result[index] == index * 3U + 1U;
    }
    return {};
}

TEST(TemporaryBufferAliasing, SimulatedExecutionPreservesResultsWithReducedPeakStorage) {
    const auto task_graph = chain_graph();
    const auto graph = task_graph.finalize().value();
    const auto execution = kora::ExecutionPlan::create(task_graph);
    ASSERT_TRUE(execution);
    const std::array buffers{requirement(a, 8 * sizeof(std::uint32_t)),
                             requirement(b, 8 * sizeof(std::uint32_t)),
                             requirement(c, 8 * sizeof(std::uint32_t))};
    const auto aliasing = kora::plan_temporary_buffer_aliasing(graph, buffers);
    ASSERT_TRUE(aliasing);
    std::vector<std::vector<std::uint32_t>> backing(aliasing.value().allocations.size(),
                                                    std::vector<std::uint32_t>(8));
    Simulation simulation{&aliasing.value(), &backing, false};
    ASSERT_TRUE(execution.value().replay(simulate_aliasing, &simulation));
    EXPECT_TRUE(simulation.output_matches);
    EXPECT_LT(aliasing.value().planned_bytes, aliasing.value().unaliased_bytes);
    EXPECT_EQ(aliasing.value().planned_bytes, 2 * 8 * sizeof(std::uint32_t));
}

}  // namespace
