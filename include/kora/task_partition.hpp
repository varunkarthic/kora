#pragma once
/// @file task_partition.hpp
/// Explicit contiguous work partitioning for caller-partitionable tasks.

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kora/device_selection.hpp"

namespace kora {

struct PartitionDevice {
    DeviceId id;
    /// Available bytes for this task (normally a fresh Runtime heap budget snapshot).
    std::uint64_t available_memory_bytes = 0;
    /// Cold-start throughput estimate in work units per nanosecond.
    double initial_throughput = 0;
};

struct PartitionRequest {
    std::uint64_t total_work_units = 0;
    std::uint64_t bytes_per_work_unit = 0;
    std::uint64_t granularity = 1;
    std::uint64_t minimum_chunk_units = 1;
    double estimated_unsplit_duration_ns = 0;
    double partition_overhead_ns = 0;
    /// Decline a split unless it predicts at least this fractional improvement.
    double minimum_gain_fraction = 0.05;
};

struct PartitionSlice {
    DeviceId device;
    std::uint64_t offset = 0;
    std::uint64_t work_units = 0;
    double weight = 0;
};

struct PartitionPlan {
    std::vector<PartitionSlice> slices;
    double estimated_makespan_ns = 0;
    bool split = false;
};

/// Uses observed per-device throughput to create non-overlapping contiguous slices.
/// It is advisory: callers must make their task interface range-aware and merge outputs as required.
class DynamicPartitioner final {
public:
    explicit DynamicPartitioner(double observation_alpha = 0.25);
    Status observe(const DeviceId& device, std::uint64_t work_units, double duration_ns);
    Result<PartitionPlan> partition(const PartitionRequest& request,
                                    const std::vector<PartitionDevice>& devices) const;

private:
    struct DeviceKeyHash {
        std::size_t operator()(const DeviceId& device) const noexcept;
    };
    double alpha_;
    std::unordered_map<DeviceId, double, DeviceKeyHash> throughput_;
};

}  // namespace kora
