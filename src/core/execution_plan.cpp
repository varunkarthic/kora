#include "kora/execution_plan.hpp"

namespace kora {

Result<ExecutionPlan> ExecutionPlan::create(const TaskGraph& graph, TaskPreparer prepare, void* user_data) {
    auto graph_plan = graph.finalize();
    if (!graph_plan) return std::move(graph_plan).error();
    if (prepare != nullptr) {
        for (const TaskId id : graph_plan.value().topological_order) {
            const auto index = static_cast<std::size_t>(id - 1);
            auto status = prepare(graph_plan.value().tasks[index], user_data);
            if (!status) {
                auto error = std::move(status).error();
                if (error.task_id == 0) error.with_task(id);
                return error;
            }
        }
    }
    return ExecutionPlan(std::move(graph_plan).value());
}

Status ExecutionPlan::replay(TaskExecutor executor, void* user_data) const {
    if (executor == nullptr)
        return Error(ErrorCode::InvalidArgument, "execution plan replay requires a task executor");

    for (const TaskId id : graph_plan_.topological_order) {
        const auto index = static_cast<std::size_t>(id - 1);
        const auto& task = graph_plan_.tasks[index];
        auto status = executor(task, user_data);
        if (!status) {
            auto error = std::move(status).error();
            if (error.task_id == 0) error.with_task(id);
            return error;
        }
    }
    return {};
}

}  // namespace kora
