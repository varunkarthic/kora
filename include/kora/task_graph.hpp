#pragma once
/// @file task_graph.hpp
/// Task metadata and deterministic dependency inference for reusable task graphs.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "kora/buffer.hpp"
#include "kora/errors.hpp"

namespace kora {

using TaskId = std::uint64_t;

enum class TaskType : std::uint8_t {
    Upload,
    Download,
    Copy,
    Fill,
    Dispatch,
    CpuTask,
    HostCallback,
};

enum class AccessMode : std::uint8_t { Read, Write, ReadWrite };

struct BufferAccess {
    BufferId buffer;
    AccessMode mode = AccessMode::Read;
    friend bool operator==(const BufferAccess&, const BufferAccess&) = default;
};

/// Scheduler-facing information for one operation. The graph only stores metadata; execution and
/// device selection are performed by later runtime layers.
struct TaskMetadata {
    TaskType type = TaskType::HostCallback;
    std::string kernel_id;
    std::string cpu_function_id;
    std::vector<BufferAccess> accesses;
    std::array<std::uint64_t, 3> problem_dimensions{};
    std::int32_t priority = 0;
    std::optional<std::uint64_t> deadline_hint_ns;
    std::vector<std::string> allowed_devices;
    std::optional<std::string> preferred_device;
    bool cpu_fallback_available = false;
    std::uint64_t estimated_bytes_read = 0;
    std::uint64_t estimated_bytes_written = 0;
    std::uint64_t estimated_work_units = 0;
};

struct TaskNode {
    TaskId id = 0;
    TaskMetadata metadata;
    /// Prerequisite task IDs, sorted and unique.
    std::vector<TaskId> dependencies;
};

struct TaskGraphPlan {
    std::vector<TaskNode> tasks;
    /// Deterministic topological order; ties are resolved by task ID.
    std::vector<TaskId> topological_order;
};

/// Mutable graph builder with automatic RAW/WAR/WAW edges and explicit dependency support.
class TaskGraph final {
public:
    /// Adds a task and infers dependencies from its declared buffer accesses.
    Result<TaskId> add_task(TaskMetadata metadata);
    /// Adds an ordering edge prerequisite -> dependent. Cycles are reported by finalize().
    Status add_dependency(TaskId prerequisite, TaskId dependent);
    /// Validates acyclicity and returns immutable graph data in deterministic topological order.
    Result<TaskGraphPlan> finalize() const;
    std::size_t size() const noexcept { return tasks_.size(); }
    bool empty() const noexcept { return tasks_.empty(); }

private:
    struct BufferState {
        std::optional<TaskId> last_writer;
        std::vector<TaskId> active_readers;
    };

    std::vector<TaskNode> tasks_;
    std::vector<std::pair<TaskId, TaskId>> explicit_dependencies_;
    std::map<std::pair<std::uint32_t, std::uint64_t>, BufferState> buffer_states_;
};

}  // namespace kora
