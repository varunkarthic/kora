#include "kora/temporary_buffer_aliasing.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <utility>

namespace kora {
namespace {

using BufferKey = std::pair<std::uint32_t, std::uint64_t>;
BufferKey buffer_key(BufferId id) {
    return {id.slot, id.generation};
}

struct Lifetime {
    TemporaryBufferRequirements requirements;
    std::vector<std::size_t> uses;
};

bool path_exists(std::size_t from, std::size_t to, const std::vector<std::vector<std::size_t>>& successors) {
    if (from == to) return false;
    std::vector<std::size_t> pending{from};
    std::vector<bool> visited(successors.size(), false);
    visited[from] = true;
    while (!pending.empty()) {
        const auto task = pending.back();
        pending.pop_back();
        for (const auto next : successors[task]) {
            if (next == to) return true;
            if (!visited[next]) {
                visited[next] = true;
                pending.push_back(next);
            }
        }
    }
    return false;
}

bool wholly_before(const Lifetime& left, const Lifetime& right,
                   const std::vector<std::vector<std::size_t>>& successors) {
    if (left.uses.empty() || right.uses.empty()) return false;
    for (const auto left_use : left.uses)
        for (const auto right_use : right.uses)
            if (!path_exists(left_use, right_use, successors)) return false;
    return true;
}

bool lifetimes_disjoint(const Lifetime& left, const Lifetime& right,
                        const std::vector<std::vector<std::size_t>>& successors) {
    return wholly_before(left, right, successors) || wholly_before(right, left, successors);
}

bool valid_requirements(const TemporaryBufferRequirements& requirements) {
    return requirements.buffer.generation != 0 && requirements.size_bytes != 0 &&
           requirements.alignment != 0 && std::has_single_bit(requirements.alignment) &&
           requirements.memory_type_bits != 0;
}

}  // namespace

Result<TemporaryBufferAliasingPlan> plan_temporary_buffer_aliasing(
    const TaskGraphPlan& graph, std::span<const TemporaryBufferRequirements> buffers) {
    const auto task_count = graph.tasks.size();
    if (graph.topological_order.size() != task_count)
        return Error(ErrorCode::InvalidArgument, "task graph plan has an incomplete topological order");

    std::vector<std::size_t> topological_position(task_count, task_count);
    std::vector<std::vector<std::size_t>> successors(task_count);
    for (std::size_t position = 0; position < graph.topological_order.size(); ++position) {
        const auto id = graph.topological_order[position];
        if (id == 0 || id > task_count ||
            topological_position[static_cast<std::size_t>(id - 1)] != task_count)
            return Error(ErrorCode::InvalidArgument, "task graph plan has an invalid topological task ID");
        const auto index = static_cast<std::size_t>(id - 1);
        if (graph.tasks[index].id != id)
            return Error(ErrorCode::InvalidArgument, "task graph plan task IDs are not contiguous");
        topological_position[index] = position;
    }

    for (const auto& task : graph.tasks) {
        if (task.id == 0 || task.id > task_count ||
            graph.tasks[static_cast<std::size_t>(task.id - 1)].id != task.id)
            return Error(ErrorCode::InvalidArgument, "task graph plan contains an invalid task ID");
        const auto task_index = static_cast<std::size_t>(task.id - 1);
        for (const auto dependency : task.dependencies) {
            if (dependency == 0 || dependency > task_count ||
                topological_position[static_cast<std::size_t>(dependency - 1)] >=
                    topological_position[task_index])
                return Error(ErrorCode::InvalidArgument, "task graph plan has an invalid dependency order");
            successors[static_cast<std::size_t>(dependency - 1)].push_back(task_index);
        }
    }

    std::map<BufferKey, Lifetime> lifetimes;
    for (const auto& requirements : buffers) {
        if (!valid_requirements(requirements))
            return Error(ErrorCode::InvalidArgument, "temporary buffer requirements are invalid");
        auto [unused, inserted] =
            lifetimes.emplace(buffer_key(requirements.buffer), Lifetime{requirements, {}});
        if (!inserted)
            return Error(ErrorCode::InvalidArgument, "temporary buffer requirements contain duplicate IDs");
        (void)unused;
    }

    for (std::size_t task_index = 0; task_index < task_count; ++task_index) {
        for (const auto& access : graph.tasks[task_index].metadata.accesses) {
            if (access.buffer.generation == 0)
                return Error(ErrorCode::InvalidArgument,
                             "task graph plan contains an invalid buffer generation");
            const auto found = lifetimes.find(buffer_key(access.buffer));
            if (found != lifetimes.end()) found->second.uses.push_back(task_index);
        }
    }
    for (const auto& [key, lifetime] : lifetimes) {
        (void)key;
        if (lifetime.uses.empty())
            return Error(ErrorCode::InvalidArgument, "temporary buffer is not accessed by the task graph");
    }

    std::vector<Lifetime*> ordered;
    ordered.reserve(lifetimes.size());
    for (auto& [key, lifetime] : lifetimes) {
        (void)key;
        ordered.push_back(&lifetime);
    }
    std::sort(ordered.begin(), ordered.end(), [](const Lifetime* left, const Lifetime* right) {
        if (left->requirements.persistent != right->requirements.persistent)
            return left->requirements.persistent;
        if (left->requirements.size_bytes != right->requirements.size_bytes)
            return left->requirements.size_bytes > right->requirements.size_bytes;
        return buffer_key(left->requirements.buffer) < buffer_key(right->requirements.buffer);
    });

    TemporaryBufferAliasingPlan plan;
    std::map<BufferKey, std::uint32_t> placement_by_buffer;
    for (const auto* lifetime : ordered) {
        const auto& requirement = lifetime->requirements;
        if (plan.unaliased_bytes > std::numeric_limits<std::uint64_t>::max() - requirement.size_bytes)
            return Error(ErrorCode::Overflow, "unaliased temporary memory total overflows");
        plan.unaliased_bytes += requirement.size_bytes;

        std::optional<std::size_t> selected;
        if (!requirement.persistent) {
            for (std::size_t allocation_index = 0; allocation_index < plan.allocations.size();
                 ++allocation_index) {
                const auto& allocation = plan.allocations[allocation_index];
                if (allocation.persistent_exclusive || allocation.usage_class != requirement.usage_class ||
                    allocation.size_bytes < requirement.size_bytes ||
                    allocation.alignment < requirement.alignment ||
                    (allocation.memory_type_bits & requirement.memory_type_bits) == 0)
                    continue;
                const bool all_disjoint =
                    std::all_of(ordered.begin(), ordered.end(), [&](const Lifetime* other) {
                        const auto existing =
                            placement_by_buffer.find(buffer_key(other->requirements.buffer));
                        return existing == placement_by_buffer.end() || existing->second != allocation.id ||
                               lifetimes_disjoint(*lifetime, *other, successors);
                    });
                if (all_disjoint &&
                    (!selected || allocation.size_bytes < plan.allocations[*selected].size_bytes))
                    selected = allocation_index;
            }
        }

        if (selected) {
            auto& allocation = plan.allocations[*selected];
            allocation.alignment = std::max(allocation.alignment, requirement.alignment);
            allocation.memory_type_bits &= requirement.memory_type_bits;
            placement_by_buffer[buffer_key(requirement.buffer)] = allocation.id;
        } else {
            if (plan.allocations.size() >= std::numeric_limits<std::uint32_t>::max())
                return Error(ErrorCode::Overflow, "temporary allocation ID space is exhausted");
            const auto id = static_cast<std::uint32_t>(plan.allocations.size());
            plan.allocations.push_back({id, requirement.size_bytes, requirement.alignment,
                                        requirement.usage_class, requirement.memory_type_bits,
                                        requirement.persistent});
            placement_by_buffer[buffer_key(requirement.buffer)] = id;
        }
    }

    for (const auto& allocation : plan.allocations) {
        if (plan.planned_bytes > std::numeric_limits<std::uint64_t>::max() - allocation.size_bytes)
            return Error(ErrorCode::Overflow, "planned temporary memory total overflows");
        plan.planned_bytes += allocation.size_bytes;
    }
    plan.placements.reserve(buffers.size());
    for (const auto& requirements : buffers)
        plan.placements.push_back(
            {requirements.buffer, placement_by_buffer.at(buffer_key(requirements.buffer)), 0});
    return plan;
}

}  // namespace kora
