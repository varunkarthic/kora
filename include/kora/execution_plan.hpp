#pragma once
/// @file execution_plan.hpp
/// Immutable, reusable task ordering with allocation-free callback replay.

#include <cstddef>
#include <utility>

#include "kora/task_graph.hpp"

namespace kora {

/// Executes one prepared metadata node. The callback and user state are borrowed for one replay.
/// Implementations should return expected failures as Status and must not retain the TaskNode ref.
using TaskExecutor = Status (*)(const TaskNode& task, void* user_data);
/// Resolves/caches resources for one node before the plan is returned. Any prepared resources remain
/// owned by user_data, which must outlive every replay that refers to them.
using TaskPreparer = Status (*)(const TaskNode& task, void* user_data);

/// Snapshot of a finalized task graph. Static dependency analysis is performed once at creation;
/// replay walks the stable topological order and invokes the caller's executor for each task.
class ExecutionPlan final {
public:
    static Result<ExecutionPlan> create(const TaskGraph& graph, TaskPreparer prepare = nullptr,
                                        void* user_data = nullptr);

    /// Replays each task in deterministic topological order without allocating plan storage.
    /// Stops at the first callback error and attaches the current task ID if none was provided.
    Status replay(TaskExecutor executor, void* user_data = nullptr) const;

    const TaskGraphPlan& graph_plan() const noexcept { return graph_plan_; }
    std::size_t size() const noexcept { return graph_plan_.tasks.size(); }

private:
    explicit ExecutionPlan(TaskGraphPlan graph_plan) : graph_plan_(std::move(graph_plan)) {}
    TaskGraphPlan graph_plan_;
};

}  // namespace kora
