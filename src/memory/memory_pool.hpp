#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "memory/suballocator.hpp"
#include "vulkan/api.hpp"

namespace kora::detail {

enum class MemoryUsageClass : std::uint8_t { General, Upload, Readback };

struct MemoryPoolConfig {
    std::uint32_t memory_type_index = 0;
    MemoryUsageClass usage = MemoryUsageClass::General;
    bool host_visible = false;
    bool host_coherent = false;
    bool device_address = false;
    std::uint64_t heap_size = 0;
    std::uint64_t budget = 0;
    std::uint64_t block_size_hint = 0;
};

struct PoolAllocationRequest {
    std::uint64_t size = 0;
    std::uint64_t alignment = 1;
    bool dedicated_required = false;
    bool dedicated_preferred = false;
};

struct PoolAllocation {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::uint64_t offset = 0;
    std::uint64_t requested_size = 0;
    std::uint64_t allocated_size = 0;
    void* mapped_data = nullptr;
    std::size_t block_index = 0;
    std::uint64_t token = 0;
    bool dedicated = false;
};

struct MemoryPoolStats {
    std::uint64_t reserved_bytes = 0;
    std::uint64_t allocated_bytes = 0;
    std::uint64_t requested_bytes = 0;
    std::uint64_t free_bytes = 0;
    std::uint64_t internal_fragmentation_bytes = 0;
    std::uint64_t external_fragmentation_bytes = 0;
    std::uint64_t largest_free_block = 0;
    std::uint64_t high_water_mark = 0;
    std::size_t blocks = 0;
    std::size_t dedicated_allocations = 0;
    std::size_t live_allocations = 0;
    std::uint64_t device_allocation_calls = 0;
};

/// Owns VkDeviceMemory blocks and suballocates buffer-compatible byte ranges from them.
class MemoryPool final {
public:
    static Result<std::unique_ptr<MemoryPool>> create(const vk::Device& device, MemoryPoolConfig config);
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    Result<PoolAllocation> allocate(const PoolAllocationRequest& request);
    Status deallocate(const PoolAllocation& allocation);
    MemoryPoolStats stats() const noexcept;
    std::uint64_t block_size() const noexcept { return block_size_; }

private:
    struct Block;
    struct Dedicated;

    MemoryPool(const vk::Device& device, MemoryPoolConfig config, std::uint64_t block_size);
    Result<std::unique_ptr<Block>> create_block(std::uint64_t size);
    Result<PoolAllocation> allocate_dedicated(const PoolAllocationRequest& request);
    Result<VkDeviceMemory> allocate_memory(std::uint64_t size);
    void destroy_memory(VkDeviceMemory memory, void* mapped) noexcept;

    const vk::Device* device_ = nullptr;
    MemoryPoolConfig config_;
    std::uint64_t block_size_ = 0;
    std::uint64_t high_water_mark_ = 0;
    std::uint64_t next_token_ = 1;
    std::uint64_t device_allocation_calls_ = 0;
    std::vector<std::unique_ptr<Block>> blocks_;
    std::unordered_map<std::uint64_t, Dedicated> dedicated_;
};

}  // namespace kora::detail
