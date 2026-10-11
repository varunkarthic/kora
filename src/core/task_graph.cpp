#include "kora/task_graph.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <queue>
#include <set>

namespace kora {
namespace {

bool valid_task_type(TaskType type) noexcept {
    return type >= TaskType::Upload && type <= TaskType::HostCallback;
}

bool valid_access_mode(AccessMode mode) noexcept {
    return mode >= AccessMode::Read && mode <= AccessMode::ReadWrite;
}

bool reads(AccessMode mode) noexcept {
    return mode == AccessMode::Read || mode == AccessMode::ReadWrite;
}

bool writes(AccessMode mode) noexcept {
    return mode == AccessMode::Write || mode == AccessMode::ReadWrite;
}

}  // namespace

Result<TaskId> TaskGraph::add_task(TaskMetadata metadata) {
    if (!valid_task_type(metadata.type))
        return Error(ErrorCode::InvalidArgument, "task type is not recognized");
    if (tasks_.size() >= std::numeric_limits<TaskId>::max())
        return Error(ErrorCode::Overflow, "task ID space is exhausted");

    // Fold repeated declarations for one buffer into a single access mode. For example, declaring
    // the same buffer as both read and write is equivalent to read-write.
    std::map<std::pair<std::uint32_t, std::uint64_t>, AccessMode> accesses;
    for (const auto& access : metadata.accesses) {
        if (access.buffer.generation == 0)
            return Error(ErrorCode::InvalidArgument, "task access has an invalid buffer generation");
        if (!valid_access_mode(access.mode))
            return Error(ErrorCode::InvalidArgument, "task access mode is not recognized");

        const auto key = std::pair{access.buffer.slot, access.buffer.generation};
        auto [it, inserted] = accesses.emplace(key, access.mode);
        if (!inserted) {
            const bool combined_reads = reads(it->second) || reads(access.mode);
            const bool combined_writes = writes(it->second) || writes(access.mode);
            it->second = combined_reads && combined_writes ? AccessMode::ReadWrite
                         : combined_writes                 ? AccessMode::Write
                                                           : AccessMode::Read;
        }
    }

    const TaskId id = static_cast<TaskId>(tasks_.size()) + 1;
    std::set<TaskId> dependencies;
    for (const auto& [key, mode] : accesses) {
        auto& state = buffer_states_[key];
        if (reads(mode) && state.last_writer) dependencies.insert(*state.last_writer);
        if (writes(mode)) {
            if (state.last_writer) dependencies.insert(*state.last_writer);
            dependencies.insert(state.active_readers.begin(), state.active_readers.end());
        }
    }
    dependencies.erase(id);

    TaskNode node;
    node.id = id;
    node.metadata = std::move(metadata);
    node.dependencies.assign(dependencies.begin(), dependencies.end());
    tasks_.push_back(std::move(node));

    for (const auto& [key, mode] : accesses) {
        auto& state = buffer_states_[key];
        if (writes(mode)) {
            state.last_writer = id;
            state.active_readers.clear();
        } else if (reads(mode)) {
            state.active_readers.push_back(id);
        }
    }
    return id;
}

Status TaskGraph::add_dependency(TaskId prerequisite, TaskId dependent) {
    if (prerequisite == 0 || dependent == 0 || prerequisite > tasks_.size() || dependent > tasks_.size())
        return Error(ErrorCode::InvalidArgument, "task dependency references an unknown task ID");
    if (std::find(explicit_dependencies_.begin(), explicit_dependencies_.end(),
                  std::pair{prerequisite, dependent}) == explicit_dependencies_.end())
        explicit_dependencies_.emplace_back(prerequisite, dependent);
    return {};
}

Result<TaskGraphPlan> TaskGraph::finalize() const {
    TaskGraphPlan plan;
    plan.tasks = tasks_;
    const auto count = tasks_.size();
    std::vector<std::vector<TaskId>> successors(count);
    std::vector<std::size_t> indegree(count, 0);
    std::vector<std::set<TaskId>> dependencies(count);

    for (const auto& task : tasks_) {
        auto& task_dependencies = dependencies[static_cast<std::size_t>(task.id - 1)];
        task_dependencies.insert(task.dependencies.begin(), task.dependencies.end());
    }
    for (const auto& [prerequisite, dependent] : explicit_dependencies_)
        dependencies[static_cast<std::size_t>(dependent - 1)].insert(prerequisite);

    for (TaskId dependent = 1; dependent <= count; ++dependent) {
        const auto index = static_cast<std::size_t>(dependent - 1);
        indegree[index] = dependencies[index].size();
        plan.tasks[index].dependencies.assign(dependencies[index].begin(), dependencies[index].end());
        for (const auto prerequisite : dependencies[index])
            successors[static_cast<std::size_t>(prerequisite - 1)].push_back(dependent);
    }

    std::priority_queue<TaskId, std::vector<TaskId>, std::greater<>> ready;
    for (TaskId id = 1; id <= count; ++id)
        if (indegree[static_cast<std::size_t>(id - 1)] == 0) ready.push(id);

    plan.topological_order.reserve(count);
    while (!ready.empty()) {
        const TaskId completed = ready.top();
        ready.pop();
        plan.topological_order.push_back(completed);
        for (const auto successor : successors[static_cast<std::size_t>(completed - 1)]) {
            auto& degree = indegree[static_cast<std::size_t>(successor - 1)];
            if (--degree == 0) ready.push(successor);
        }
    }

    if (plan.topological_order.size() != count) {
        const auto cycle_member = static_cast<TaskId>(
            std::distance(indegree.begin(), std::find_if(indegree.begin(), indegree.end(),
                                                         [](std::size_t degree) { return degree != 0; })) +
            1);
        return Error(ErrorCode::GraphCycle, "task graph contains a dependency cycle").with_task(cycle_member);
    }
    return plan;
}

}  // namespace kora
