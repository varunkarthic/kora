#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <map>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "kora/task_graph.hpp"

namespace {

using kora::AccessMode;
using kora::BufferAccess;
using kora::BufferId;
using kora::ErrorCode;
using kora::TaskGraph;
using kora::TaskMetadata;
using kora::TaskType;

TaskMetadata task(std::initializer_list<BufferAccess> accesses) {
    TaskMetadata metadata;
    metadata.accesses = accesses;
    return metadata;
}

std::set<kora::TaskId> dependencies_of(const kora::TaskGraphPlan& plan, kora::TaskId id) {
    const auto found =
        std::find_if(plan.tasks.begin(), plan.tasks.end(), [id](const auto& node) { return node.id == id; });
    if (found == plan.tasks.end()) return {};
    return {found->dependencies.begin(), found->dependencies.end()};
}

TEST(TaskGraph, InfersRawWarWawAndReadWriteEdges) {
    constexpr BufferId buffer{3, 17};
    TaskGraph graph;
    const auto writer = graph.add_task(task({{buffer, AccessMode::Write}})).value();
    const auto reader_a = graph.add_task(task({{buffer, AccessMode::Read}})).value();
    const auto reader_b = graph.add_task(task({{buffer, AccessMode::Read}})).value();
    const auto writer_after_readers = graph.add_task(task({{buffer, AccessMode::Write}})).value();
    const auto read_writer = graph.add_task(task({{buffer, AccessMode::ReadWrite}})).value();
    const auto reader_after_read_writer = graph.add_task(task({{buffer, AccessMode::Read}})).value();

    const auto plan = graph.finalize();
    ASSERT_TRUE(plan);
    EXPECT_EQ(dependencies_of(plan.value(), writer), std::set<kora::TaskId>{});
    EXPECT_EQ(dependencies_of(plan.value(), reader_a), (std::set<kora::TaskId>{writer}));
    EXPECT_EQ(dependencies_of(plan.value(), reader_b), (std::set<kora::TaskId>{writer}));
    EXPECT_EQ(dependencies_of(plan.value(), writer_after_readers),
              (std::set<kora::TaskId>{writer, reader_a, reader_b}));
    EXPECT_EQ(dependencies_of(plan.value(), read_writer), (std::set<kora::TaskId>{writer_after_readers}));
    EXPECT_EQ(dependencies_of(plan.value(), reader_after_read_writer), (std::set<kora::TaskId>{read_writer}));
    EXPECT_EQ(plan.value().topological_order,
              (std::vector<kora::TaskId>{writer, reader_a, reader_b, writer_after_readers, read_writer,
                                         reader_after_read_writer}));
}

TEST(TaskGraph, CombinesDuplicateAccessesAndDistinguishesBufferGenerations) {
    TaskGraph graph;
    const BufferId old_generation{0, 1};
    const BufferId recycled{0, 2};
    auto initial = graph.add_task(task({{old_generation, AccessMode::Write}}));
    ASSERT_TRUE(initial);
    auto combined =
        graph.add_task(task({{old_generation, AccessMode::Read}, {old_generation, AccessMode::Write}}));
    ASSERT_TRUE(combined);
    auto independent = graph.add_task(task({{recycled, AccessMode::Read}}));
    ASSERT_TRUE(independent);
    const auto plan = graph.finalize();
    ASSERT_TRUE(plan);
    EXPECT_EQ(dependencies_of(plan.value(), combined.value()), (std::set<kora::TaskId>{initial.value()}));
    EXPECT_TRUE(dependencies_of(plan.value(), independent.value()).empty());
}

TEST(TaskGraph, PreservesMetadataForEveryTaskType) {
    constexpr std::array types{TaskType::Upload,   TaskType::Download, TaskType::Copy,        TaskType::Fill,
                               TaskType::Dispatch, TaskType::CpuTask,  TaskType::HostCallback};
    TaskGraph graph;
    for (const auto type : types) {
        TaskMetadata metadata;
        metadata.type = type;
        metadata.kernel_id = "kernel";
        metadata.cpu_function_id = "cpu-function";
        metadata.problem_dimensions = {2, 3, 4};
        metadata.priority = 9;
        metadata.deadline_hint_ns = 1234;
        metadata.allowed_devices = {"device-a", "device-b"};
        metadata.preferred_device = "device-a";
        metadata.cpu_fallback_available = true;
        metadata.estimated_bytes_read = 64;
        metadata.estimated_bytes_written = 32;
        metadata.estimated_work_units = 10;
        ASSERT_TRUE(graph.add_task(std::move(metadata)));
    }
    const auto plan = graph.finalize();
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan.value().tasks.size(), types.size());
    for (std::size_t index = 0; index < types.size(); ++index) {
        const auto& metadata = plan.value().tasks[index].metadata;
        EXPECT_EQ(metadata.type, types[index]);
        EXPECT_EQ(metadata.kernel_id, "kernel");
        EXPECT_EQ(metadata.cpu_function_id, "cpu-function");
        EXPECT_EQ(metadata.problem_dimensions, (std::array<std::uint64_t, 3>{2, 3, 4}));
        EXPECT_EQ(metadata.priority, 9);
        EXPECT_EQ(metadata.deadline_hint_ns, 1234U);
        EXPECT_EQ(metadata.allowed_devices, (std::vector<std::string>{"device-a", "device-b"}));
        ASSERT_TRUE(metadata.preferred_device.has_value());
        EXPECT_EQ(*metadata.preferred_device, "device-a");
        EXPECT_TRUE(metadata.cpu_fallback_available);
        EXPECT_EQ(metadata.estimated_bytes_read, 64U);
        EXPECT_EQ(metadata.estimated_bytes_written, 32U);
        EXPECT_EQ(metadata.estimated_work_units, 10U);
    }
}

TEST(TaskGraph, ExplicitDependenciesAreDeduplicatedAndUnknownIDsRejected) {
    TaskGraph graph;
    const auto first = graph.add_task(task({})).value();
    const auto second = graph.add_task(task({})).value();
    ASSERT_TRUE(graph.add_dependency(first, second));
    ASSERT_TRUE(graph.add_dependency(first, second));
    EXPECT_FALSE(graph.add_dependency(0, second));
    EXPECT_FALSE(graph.add_dependency(first, second + 1));
    const auto plan = graph.finalize();
    ASSERT_TRUE(plan);
    EXPECT_EQ(dependencies_of(plan.value(), second), (std::set<kora::TaskId>{first}));
}

TEST(TaskGraph, ReportsExplicitSelfAndMultiTaskCycles) {
    for (const bool self_cycle : {true, false}) {
        TaskGraph graph;
        const auto first = graph.add_task(task({})).value();
        const auto second = graph.add_task(task({})).value();
        ASSERT_TRUE(graph.add_dependency(first, second));
        ASSERT_TRUE(graph.add_dependency(self_cycle ? first : second, first));
        const auto plan = graph.finalize();
        ASSERT_FALSE(plan);
        EXPECT_EQ(plan.error().code, ErrorCode::GraphCycle);
        EXPECT_NE(plan.error().task_id, 0U);
    }
}

TEST(TaskGraph, RandomAccessSequencesMatchIndependentHazardOracle) {
    constexpr std::uint32_t buffer_count = 7;
    constexpr int sequence_count = 80;
    constexpr int task_count = 96;
    constexpr std::array modes{AccessMode::Read, AccessMode::Write, AccessMode::ReadWrite};
    std::mt19937 random(0x4b4f5241U);

    for (int sequence = 0; sequence < sequence_count; ++sequence) {
        TaskGraph graph;
        std::vector<std::set<kora::TaskId>> expected(task_count);
        struct HazardState {
            std::optional<kora::TaskId> writer;
            std::set<kora::TaskId> readers;
        };
        std::map<std::uint32_t, HazardState> hazards;

        for (int index = 0; index < task_count; ++index) {
            TaskMetadata metadata;
            metadata.type = static_cast<TaskType>(random() % 7U);
            const auto access_count = 1U + random() % 4U;
            std::map<std::uint32_t, AccessMode> accesses;
            for (unsigned access_index = 0; access_index < access_count; ++access_index) {
                const auto buffer = random() % buffer_count;
                const auto mode = modes[random() % modes.size()];
                const auto [found, inserted] = accesses.emplace(buffer, mode);
                if (!inserted) {
                    const bool has_read = found->second != AccessMode::Write || mode != AccessMode::Write;
                    const bool has_write = found->second != AccessMode::Read || mode != AccessMode::Read;
                    found->second = has_read && has_write ? AccessMode::ReadWrite
                                    : has_write           ? AccessMode::Write
                                                          : AccessMode::Read;
                }
            }

            for (const auto& [buffer, mode] : accesses) {
                metadata.accesses.push_back({BufferId{buffer, 1}, mode});
                auto& state = hazards[buffer];
                const bool read = mode != AccessMode::Write;
                const bool write = mode != AccessMode::Read;
                if (read && state.writer) expected[index].insert(*state.writer);
                if (write) {
                    if (state.writer) expected[index].insert(*state.writer);
                    expected[index].insert(state.readers.begin(), state.readers.end());
                }
                if (write) {
                    state.writer = static_cast<kora::TaskId>(index + 1);
                    state.readers.clear();
                } else if (read) {
                    state.readers.insert(static_cast<kora::TaskId>(index + 1));
                }
            }
            ASSERT_TRUE(graph.add_task(std::move(metadata))) << "sequence=" << sequence << " task=" << index;
        }

        const auto plan = graph.finalize();
        ASSERT_TRUE(plan) << "sequence=" << sequence;
        for (int index = 0; index < task_count; ++index)
            EXPECT_EQ(dependencies_of(plan.value(), static_cast<kora::TaskId>(index + 1)), expected[index])
                << "sequence=" << sequence << " task=" << index;
    }
}

TEST(TaskGraph, RandomExplicitDagsRespectEveryOracleEdgeAndStayDeterministic) {
    constexpr int sequence_count = 64;
    constexpr int task_count = 72;
    std::mt19937 random(0x47524150U);
    for (int sequence = 0; sequence < sequence_count; ++sequence) {
        TaskGraph graph;
        for (int index = 0; index < task_count; ++index) ASSERT_TRUE(graph.add_task(task({})));
        std::vector<std::pair<kora::TaskId, kora::TaskId>> edges;
        for (kora::TaskId before = 1; before <= task_count; ++before) {
            for (kora::TaskId after = before + 1; after <= task_count; ++after) {
                if ((random() % 19U) == 0) {
                    ASSERT_TRUE(graph.add_dependency(before, after));
                    edges.emplace_back(before, after);
                }
            }
        }
        const auto plan = graph.finalize();
        ASSERT_TRUE(plan) << "sequence=" << sequence;
        ASSERT_EQ(plan.value().topological_order.size(), task_count);
        std::vector<std::size_t> position(task_count + 1);
        for (std::size_t index = 0; index < task_count; ++index)
            position[plan.value().topological_order[index]] = index;
        for (const auto& [before, after] : edges)
            EXPECT_LT(position[before], position[after]) << "sequence=" << sequence;
        EXPECT_EQ(graph.finalize().value().topological_order, plan.value().topological_order);
    }
}

TEST(TaskGraph, RejectsInvalidTaskTypesAccessesAndBufferGenerations) {
    TaskGraph graph;
    auto bad_type = task({});
    bad_type.type = static_cast<TaskType>(255);
    EXPECT_EQ(graph.add_task(std::move(bad_type)).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(graph.add_task(task({{BufferId{1, 0}, AccessMode::Read}})).error().code,
              ErrorCode::InvalidArgument);
    EXPECT_EQ(graph.add_task(task({{BufferId{1, 1}, static_cast<AccessMode>(255)}})).error().code,
              ErrorCode::InvalidArgument);
    EXPECT_TRUE(graph.empty());
}

}  // namespace
