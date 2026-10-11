#include "memory/memory_manager.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <string_view>
#include <utility>

#include "kora/detail/checked.hpp"

namespace kora::detail {
namespace {

const std::vector<std::uint32_t>& preference_order(const DeviceInfo::MemoryTypePreferences& preferences,
                                                   MemoryPolicy policy) {
    switch (policy) {
        case MemoryPolicy::Auto:
            return preferences.automatic;
        case MemoryPolicy::DevicePreferred:
            return preferences.device_preferred;
        case MemoryPolicy::HostVisible:
            return preferences.host_visible;
        case MemoryPolicy::ReadMostly:
            return preferences.read_mostly;
        case MemoryPolicy::WriteMostly:
            return preferences.write_mostly;
        case MemoryPolicy::Streaming:
            return preferences.streaming;
        case MemoryPolicy::Readback:
            return preferences.readback;
    }
    return preferences.automatic;
}

bool satisfies(const DeviceInfo::MemoryType& type, const MemoryTypeRequirements& required) {
    return (!required.device_local || type.device_local) && (!required.host_visible || type.host_visible) &&
           (!required.host_coherent || type.host_coherent) && (!required.host_cached || type.host_cached);
}

}  // namespace

BudgetLedger::BudgetLedger(std::vector<HeapBudget> heaps)
    : heaps_(std::move(heaps)), reserved_bytes_(heaps_.size(), 0), shrink_factors_(heaps_.size(), 1.0) {
    for (auto& heap : heaps_) {
        if (heap.heap_size_bytes != 0) heap.limit_bytes = std::min(heap.limit_bytes, heap.heap_size_bytes);
    }
}

Status BudgetLedger::reserve(std::uint32_t heap_index, std::uint64_t bytes) {
    if (heap_index >= heaps_.size()) {
        return Error(ErrorCode::InvalidArgument, "memory budget heap index is out of range");
    }
    const auto updated = checked::add(reserved_bytes_[heap_index], bytes);
    if (!updated) return Error(ErrorCode::Overflow, "memory budget reservation arithmetic overflow");
    if (*updated > heaps_[heap_index].limit_bytes) {
        return Error(ErrorCode::OutOfMemoryBudget, "memory budget reservation exceeds usable heap budget");
    }
    reserved_bytes_[heap_index] = *updated;
#ifndef NDEBUG
    const auto snapshot = stats(heap_index);
    assert(snapshot && snapshot.value().reserved_bytes <= snapshot.value().limit_bytes);
#endif
    return ok_status();
}

Status BudgetLedger::release(std::uint32_t heap_index, std::uint64_t bytes) {
    if (heap_index >= heaps_.size()) {
        return Error(ErrorCode::InvalidArgument, "memory budget heap index is out of range");
    }
    if (bytes > reserved_bytes_[heap_index]) {
        return Error(ErrorCode::InternalInvariantViolation,
                     "memory budget release would underflow the heap reservation");
    }
    reserved_bytes_[heap_index] -= bytes;
#ifndef NDEBUG
    const auto snapshot = stats(heap_index);
    assert(snapshot && snapshot.value().reserved_bytes <= snapshot.value().limit_bytes);
#endif
    return ok_status();
}

Result<HeapBudgetStats> BudgetLedger::stats(std::uint32_t heap_index) const {
    if (heap_index >= heaps_.size()) {
        return Error(ErrorCode::InvalidArgument, "memory budget heap index is out of range");
    }
    const auto& heap = heaps_[heap_index];
    const auto reserved = reserved_bytes_[heap_index];
    const auto effective_limit = std::max(heap.limit_bytes, reserved);
    return HeapBudgetStats{effective_limit, reserved, effective_limit - reserved, heap.externally_used_bytes,
                           heap.queried};
}

Status BudgetLedger::refresh(std::vector<HeapBudget> heaps) {
    if (heaps.size() != heaps_.size())
        return Error(ErrorCode::InvalidArgument, "refreshed memory budget heap count changed");
    for (std::size_t index = 0; index < heaps.size(); ++index) {
        const auto heap_size =
            heaps[index].heap_size_bytes == 0 ? heaps_[index].heap_size_bytes : heaps[index].heap_size_bytes;
        if (heap_size != 0 && reserved_bytes_[index] > heap_size)
            return Error(ErrorCode::InternalInvariantViolation,
                         "existing memory reservations exceed the physical heap size");
    }
    for (std::size_t index = 0; index < heaps.size(); ++index) {
        auto& heap = heaps[index];
        const auto prior = heaps_[index];
        if (heap.heap_size_bytes == 0) heap.heap_size_bytes = prior.heap_size_bytes;
        // Preserve allocation-failure feedback even when live extension data is refreshed.
        heap.limit_bytes = std::min(
            heap.limit_bytes,
            static_cast<std::uint64_t>(static_cast<long double>(heap.limit_bytes) * shrink_factors_[index]));
        if (!heap.queried && !prior.queried) heap.limit_bytes = std::min(heap.limit_bytes, prior.limit_bytes);
        if (heap.heap_size_bytes != 0) heap.limit_bytes = std::min(heap.limit_bytes, heap.heap_size_bytes);
        // Do not let an external-usage refresh make existing KORA allocations appear over budget.
        heap.limit_bytes = std::max(heap.limit_bytes, reserved_bytes_[index]);
        heaps_[index] = heap;
    }
    return ok_status();
}

Status BudgetLedger::record_allocation_failure(std::uint32_t heap_index, double shrink_factor) {
    if (heap_index >= heaps_.size())
        return Error(ErrorCode::InvalidArgument, "memory budget heap index is out of range");
    if (!std::isfinite(shrink_factor) || shrink_factor < 0.0 || shrink_factor > 1.0)
        return Error(ErrorCode::InvalidArgument, "memory budget shrink factor must be between zero and one");
    auto& heap = heaps_[heap_index];
    shrink_factors_[heap_index] *= shrink_factor;
    const auto shrunk =
        static_cast<std::uint64_t>(static_cast<long double>(heap.limit_bytes) * shrink_factor);
    heap.limit_bytes = std::max(shrunk, reserved_bytes_[heap_index]);
    return ok_status();
}

MemoryManager::MemoryManager(std::vector<DeviceInfo::MemoryType> memory_types,
                             DeviceInfo::MemoryTypePreferences preferences, MemoryTopology topology,
                             BudgetLedger ledger)
    : memory_types_(std::move(memory_types)),
      preferences_(std::move(preferences)),
      topology_(topology),
      ledger_(std::move(ledger)) {}

Result<MemoryManager> MemoryManager::create(const vk::PhysicalDeviceInfo& physical_device,
                                            const DeviceInfo& device_info, double fallback_heap_fraction) {
    if (!std::isfinite(fallback_heap_fraction) || fallback_heap_fraction < 0.0 ||
        fallback_heap_fraction > 1.0) {
        return Error(ErrorCode::InvalidArgument, "fallback heap fraction must be between zero and one");
    }
    const auto heap_count =
        std::min<std::uint32_t>(physical_device.memory_properties.memoryHeapCount, VK_MAX_MEMORY_HEAPS);
    std::vector<HeapBudget> heaps;
    heaps.reserve(heap_count);
    const bool queried = std::any_of(
        physical_device.extensions.begin(), physical_device.extensions.end(),
        [&physical_device](const VkExtensionProperties& extension) {
            return std::string_view(extension.extensionName) == VK_EXT_MEMORY_BUDGET_EXTENSION_NAME &&
                   physical_device.memory_budget_properties.sType ==
                       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        });
    for (std::uint32_t index = 0; index < heap_count; ++index) {
        const auto heap_size = physical_device.memory_properties.memoryHeaps[index].size;
        if (queried) {
            const auto reported_budget = physical_device.memory_budget_properties.heapBudget[index];
            const auto usage = physical_device.memory_budget_properties.heapUsage[index];
            const auto available = std::min(reported_budget, heap_size) > usage
                                       ? std::min(reported_budget, heap_size) - usage
                                       : 0;
            const auto soft_limit =
                static_cast<std::uint64_t>(static_cast<long double>(available) * fallback_heap_fraction);
            heaps.push_back(HeapBudget{std::min(soft_limit, heap_size), usage, true, heap_size});
        } else {
            const auto fallback =
                static_cast<std::uint64_t>(static_cast<long double>(heap_size) * fallback_heap_fraction);
            heaps.push_back(HeapBudget{std::min(fallback, heap_size), 0, false, heap_size});
        }
    }
    return MemoryManager(device_info.memory_types(), device_info.memory_type_preferences(),
                         device_info.memory_topology(), BudgetLedger(std::move(heaps)));
}

Result<std::uint32_t> MemoryManager::select_memory_type(
    const std::vector<DeviceInfo::MemoryType>& memory_types,
    const DeviceInfo::MemoryTypePreferences& preferences, MemoryPolicy policy,
    std::uint32_t compatible_type_bits, const MemoryTypeRequirements& required) {
    if (compatible_type_bits == 0) {
        return Error(ErrorCode::InvalidArgument, "Vulkan memory type compatibility mask is empty");
    }
    const auto& order = preference_order(preferences, policy);
    for (const auto index : order) {
        if (index >= memory_types.size() || index >= 32 ||
            (compatible_type_bits & (std::uint32_t{1} << index)) == 0) {
            continue;
        }
        if (satisfies(memory_types[index], required)) return index;
    }
    return Error(ErrorCode::UnsupportedFeature,
                 "no compatible Vulkan memory type satisfies the selected memory policy and requirements");
}

Result<std::uint32_t> MemoryManager::select_memory_type(MemoryPolicy policy,
                                                        std::uint32_t compatible_type_bits,
                                                        const MemoryTypeRequirements& required) const {
    return select_memory_type(memory_types_, preferences_, policy, compatible_type_bits, required);
}

Status MemoryManager::reserve(std::uint32_t heap_index, std::uint64_t bytes) {
    return ledger_.reserve(heap_index, bytes);
}

Status MemoryManager::release(std::uint32_t heap_index, std::uint64_t bytes) {
    return ledger_.release(heap_index, bytes);
}

Status MemoryManager::refresh_budget(const vk::PhysicalDeviceInfo& physical_device,
                                     double fallback_heap_fraction) {
    if (!std::isfinite(fallback_heap_fraction) || fallback_heap_fraction < 0.0 ||
        fallback_heap_fraction > 1.0)
        return Error(ErrorCode::InvalidArgument, "fallback heap fraction must be between zero and one");
    const auto heap_count =
        std::min<std::uint32_t>(physical_device.memory_properties.memoryHeapCount, VK_MAX_MEMORY_HEAPS);
    std::vector<HeapBudget> heaps;
    heaps.reserve(heap_count);
    const bool queried = std::any_of(
        physical_device.extensions.begin(), physical_device.extensions.end(),
        [&physical_device](const VkExtensionProperties& extension) {
            return std::string_view(extension.extensionName) == VK_EXT_MEMORY_BUDGET_EXTENSION_NAME &&
                   physical_device.memory_budget_properties.sType ==
                       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        });
    for (std::uint32_t index = 0; index < heap_count; ++index) {
        const auto heap_size = physical_device.memory_properties.memoryHeaps[index].size;
        if (queried) {
            const auto reported_budget =
                std::min(physical_device.memory_budget_properties.heapBudget[index], heap_size);
            const auto usage = physical_device.memory_budget_properties.heapUsage[index];
            const auto reserved = ledger_.stats(index);
            if (!reserved) return Status(std::move(reserved).error());
            const auto external_usage =
                usage > reserved.value().reserved_bytes ? usage - reserved.value().reserved_bytes : 0;
            const auto available = reported_budget > external_usage ? reported_budget - external_usage : 0;
            const auto soft_limit =
                static_cast<std::uint64_t>(static_cast<long double>(available) * fallback_heap_fraction);
            heaps.push_back(HeapBudget{std::min(soft_limit, heap_size), external_usage, true, heap_size});
        } else {
            const auto fallback =
                static_cast<std::uint64_t>(static_cast<long double>(heap_size) * fallback_heap_fraction);
            heaps.push_back(HeapBudget{std::min(fallback, heap_size), 0, false, heap_size});
        }
    }
    return ledger_.refresh(std::move(heaps));
}

Status MemoryManager::record_allocation_failure(std::uint32_t heap_index, double shrink_factor) {
    return ledger_.record_allocation_failure(heap_index, shrink_factor);
}

Result<HeapBudgetStats> MemoryManager::budget(std::uint32_t heap_index) const {
    return ledger_.stats(heap_index);
}

Result<kora::MemoryStats> MemoryManager::aggregate_stats(std::span<const MemoryPoolStats> pools) {
    kora::MemoryStats total{};
    for (const auto& pool : pools) {
        const auto add = [](std::uint64_t& destination, std::uint64_t value) {
            const auto sum = checked::add(destination, value);
            if (!sum) return false;
            destination = *sum;
            return true;
        };
        if (!add(total.reserved_bytes, pool.reserved_bytes) ||
            !add(total.allocated_bytes, pool.allocated_bytes) ||
            !add(total.requested_bytes, pool.requested_bytes) || !add(total.free_bytes, pool.free_bytes) ||
            !add(total.internal_fragmentation_bytes, pool.internal_fragmentation_bytes) ||
            !add(total.external_fragmentation_bytes, pool.external_fragmentation_bytes) ||
            !add(total.high_water_mark, pool.high_water_mark) ||
            pool.blocks > std::numeric_limits<std::size_t>::max() - total.blocks ||
            pool.dedicated_allocations >
                std::numeric_limits<std::size_t>::max() - total.dedicated_allocations ||
            pool.live_allocations > std::numeric_limits<std::size_t>::max() - total.live_allocations ||
            !add(total.device_allocation_calls, pool.device_allocation_calls)) {
            return Error(ErrorCode::Overflow, "memory pool statistics aggregation overflow");
        }
        total.largest_free_block = std::max(total.largest_free_block, pool.largest_free_block);
        ++total.pools;
        total.blocks += pool.blocks;
        total.dedicated_allocations += pool.dedicated_allocations;
        total.live_allocations += pool.live_allocations;
    }
    return total;
}

}  // namespace kora::detail
