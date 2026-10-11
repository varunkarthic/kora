#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/device_caps.hpp"
#include "kora/memory_stats.hpp"
#include "memory/memory_pool.hpp"
#include "vulkan/api.hpp"

namespace kora::detail {

struct MemoryTypeRequirements {
    bool device_local = false;
    bool host_visible = false;
    bool host_coherent = false;
    bool host_cached = false;
};

struct HeapBudget {
    std::uint64_t limit_bytes = 0;
    std::uint64_t externally_used_bytes = 0;
    bool queried = false;
    std::uint64_t heap_size_bytes = 0;
};

struct HeapBudgetStats {
    std::uint64_t limit_bytes = 0;
    std::uint64_t reserved_bytes = 0;
    std::uint64_t available_bytes = 0;
    std::uint64_t externally_used_bytes = 0;
    bool queried = false;
};

/// Underflow-safe accounting for KORA reservations against one heap's usable budget.
class BudgetLedger final {
public:
    explicit BudgetLedger(std::vector<HeapBudget> heaps);

    Status reserve(std::uint32_t heap_index, std::uint64_t bytes);
    Status release(std::uint32_t heap_index, std::uint64_t bytes);
    Status refresh(std::vector<HeapBudget> heaps);
    Status record_allocation_failure(std::uint32_t heap_index, double shrink_factor = 0.8);
    Result<HeapBudgetStats> stats(std::uint32_t heap_index) const;
    std::size_t heap_count() const noexcept { return heaps_.size(); }

private:
    std::vector<HeapBudget> heaps_;
    std::vector<std::uint64_t> reserved_bytes_;
    std::vector<double> shrink_factors_;
};

/// Selects legal memory types in policy order and tracks per-heap budget reservations.
class MemoryManager final {
public:
    static Result<MemoryManager> create(const vk::PhysicalDeviceInfo& physical_device,
                                        const DeviceInfo& device_info, double fallback_heap_fraction = 0.8);

    static Result<std::uint32_t> select_memory_type(const std::vector<DeviceInfo::MemoryType>& memory_types,
                                                    const DeviceInfo::MemoryTypePreferences& preferences,
                                                    MemoryPolicy policy, std::uint32_t compatible_type_bits,
                                                    const MemoryTypeRequirements& required = {});

    Result<std::uint32_t> select_memory_type(MemoryPolicy policy, std::uint32_t compatible_type_bits,
                                             const MemoryTypeRequirements& required = {}) const;
    Status reserve(std::uint32_t heap_index, std::uint64_t bytes);
    Status release(std::uint32_t heap_index, std::uint64_t bytes);
    Status refresh_budget(const vk::PhysicalDeviceInfo& physical_device, double fallback_heap_fraction = 0.8);
    Status record_allocation_failure(std::uint32_t heap_index, double shrink_factor = 0.8);
    Result<HeapBudgetStats> budget(std::uint32_t heap_index) const;
    static Result<kora::MemoryStats> aggregate_stats(std::span<const MemoryPoolStats> pools);

    const DeviceInfo::MemoryTypePreferences& preferences() const noexcept { return preferences_; }
    MemoryTopology topology() const noexcept { return topology_; }

private:
    MemoryManager(std::vector<DeviceInfo::MemoryType> memory_types,
                  DeviceInfo::MemoryTypePreferences preferences, MemoryTopology topology,
                  BudgetLedger ledger);

    std::vector<DeviceInfo::MemoryType> memory_types_;
    DeviceInfo::MemoryTypePreferences preferences_;
    MemoryTopology topology_ = MemoryTopology::Unknown;
    BudgetLedger ledger_;
};

}  // namespace kora::detail
