#include "memory/memory_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "kora/detail/checked.hpp"

namespace kora::detail {
namespace {

constexpr std::uint64_t kMiB = 1024U * 1024U;

std::uint64_t choose_block_size(const MemoryPoolConfig& config) {
    if (config.block_size_hint != 0) return config.block_size_hint;
    const auto available = config.budget == 0 ? config.heap_size : std::min(config.heap_size, config.budget);
    const auto divisor = config.host_visible ? 128U : 64U;
    const auto minimum = (config.host_visible ? 8U : 16U) * kMiB;
    const auto maximum = (config.host_visible ? 64U : 256U) * kMiB;
    return std::clamp(available / divisor, minimum, maximum);
}

}  // namespace

struct MemoryPool::Block {
    Block(VkDeviceMemory memory_value, void* mapped_value, Suballocator allocator_value)
        : memory(memory_value), mapped(mapped_value), allocator(std::move(allocator_value)) {}
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    Suballocator allocator;
};

struct MemoryPool::Dedicated {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    std::uint64_t size = 0;
};

Result<std::unique_ptr<MemoryPool>> MemoryPool::create(const vk::Device& device, MemoryPoolConfig config) {
    if (device.handle() == VK_NULL_HANDLE || !device.api().allocate_memory || !device.api().free_memory) {
        return Error(ErrorCode::InvalidArgument, "memory pool requires a live Vulkan device");
    }
    if (config.heap_size == 0) {
        return Error(ErrorCode::InvalidArgument, "memory pool heap size must be greater than zero");
    }
    const auto size = choose_block_size(config);
    if (size == 0) return Error(ErrorCode::InvalidArgument, "memory pool block size must be nonzero");
    return std::unique_ptr<MemoryPool>(new MemoryPool(device, config, size));
}

MemoryPool::MemoryPool(const vk::Device& device, MemoryPoolConfig config, std::uint64_t block_size)
    : device_(&device), config_(config), block_size_(block_size) {}

MemoryPool::~MemoryPool() {
    for (auto& [token, allocation] : dedicated_) {
        static_cast<void>(token);
        destroy_memory(allocation.memory, allocation.mapped);
    }
    for (auto& block : blocks_) destroy_memory(block->memory, block->mapped);
}

Result<VkDeviceMemory> MemoryPool::allocate_memory(std::uint64_t size) {
    VkMemoryAllocateFlagsInfo flags{};
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    if (config_.device_address) flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.pNext = config_.device_address ? &flags : nullptr;
    info.allocationSize = size;
    info.memoryTypeIndex = config_.memory_type_index;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    ++device_allocation_calls_;
    const auto result = device_->api().allocate_memory(device_->handle(), &info, nullptr, &memory);
    if (result != VK_SUCCESS) {
        block_size_ = std::max<std::uint64_t>(block_size_ / 2, 1U * kMiB);
        return Error(ErrorCode::AllocationFailed, "vkAllocateMemory failed for memory-pool block")
            .with_vk(result);
    }
    return memory;
}

Result<std::unique_ptr<MemoryPool::Block>> MemoryPool::create_block(std::uint64_t size) {
    KORA_TRY(memory, allocate_memory(size));
    void* mapped = nullptr;
    if (config_.host_visible) {
        const auto result = device_->api().map_memory(device_->handle(), memory, 0, size, 0, &mapped);
        if (result != VK_SUCCESS) {
            device_->api().free_memory(device_->handle(), memory, nullptr);
            return Error(ErrorCode::AllocationFailed, "vkMapMemory failed for persistently mapped block")
                .with_vk(result);
        }
    }
    auto allocator = Suballocator::create(size);
    if (!allocator) {
        destroy_memory(memory, mapped);
        return std::move(allocator).error();
    }
    return std::make_unique<Block>(memory, mapped, std::move(allocator).value());
}

Result<PoolAllocation> MemoryPool::allocate_dedicated(const PoolAllocationRequest& request) {
    KORA_TRY(memory, allocate_memory(request.size));
    void* mapped = nullptr;
    if (config_.host_visible) {
        const auto result = device_->api().map_memory(device_->handle(), memory, 0, request.size, 0, &mapped);
        if (result != VK_SUCCESS) {
            device_->api().free_memory(device_->handle(), memory, nullptr);
            return Error(ErrorCode::AllocationFailed, "vkMapMemory failed for dedicated memory allocation")
                .with_vk(result);
        }
    }
    const auto token = next_token_++;
    dedicated_.emplace(token, Dedicated{memory, mapped, request.size});
    return PoolAllocation{memory, 0, request.size, request.size, mapped, 0, token, true};
}

Result<PoolAllocation> MemoryPool::allocate(const PoolAllocationRequest& request) {
    if (request.size == 0 || !checked::is_pow2(request.alignment)) {
        return Error(ErrorCode::InvalidArgument,
                     "pool allocation requires nonzero size and power-of-two alignment");
    }
    if (request.dedicated_required || request.dedicated_preferred || request.size > block_size_ / 2) {
        return allocate_dedicated(request);
    }
    for (std::size_t index = 0; index < blocks_.size(); ++index) {
        auto allocation = blocks_[index]->allocator.allocate(request.size, request.alignment);
        if (allocation) {
            const auto suballocation = allocation.value();
            auto* mapped = static_cast<std::byte*>(blocks_[index]->mapped);
            return PoolAllocation{blocks_[index]->memory,
                                  suballocation.offset,
                                  suballocation.requested_size,
                                  suballocation.allocated_size,
                                  mapped == nullptr ? nullptr : mapped + suballocation.offset,
                                  index,
                                  0,
                                  false};
        }
        if (allocation.error().code != ErrorCode::AllocationFailed) return allocation.error();
    }
    KORA_TRY(block, create_block(block_size_));
    blocks_.push_back(std::move(block));
    auto allocation = blocks_.back()->allocator.allocate(request.size, request.alignment);
    if (!allocation) return std::move(allocation).error();
    const auto suballocation = allocation.value();
    auto* mapped = static_cast<std::byte*>(blocks_.back()->mapped);
    return PoolAllocation{blocks_.back()->memory,
                          suballocation.offset,
                          suballocation.requested_size,
                          suballocation.allocated_size,
                          mapped == nullptr ? nullptr : mapped + suballocation.offset,
                          blocks_.size() - 1,
                          0,
                          false};
}

Status MemoryPool::deallocate(const PoolAllocation& allocation) {
    if (allocation.dedicated) {
        const auto entry = dedicated_.find(allocation.token);
        if (entry == dedicated_.end() || entry->second.memory != allocation.memory ||
            entry->second.size != allocation.requested_size) {
            return Error(ErrorCode::InvalidArgument, "dedicated pool allocation is unknown or invalid");
        }
        destroy_memory(entry->second.memory, entry->second.mapped);
        dedicated_.erase(entry);
        return ok_status();
    }
    if (allocation.block_index >= blocks_.size() ||
        blocks_[allocation.block_index]->memory != allocation.memory) {
        return Error(ErrorCode::InvalidArgument, "pooled allocation references an invalid block");
    }
    return blocks_[allocation.block_index]->allocator.deallocate(
        Suballocation{allocation.offset, allocation.requested_size, allocation.allocated_size});
}

void MemoryPool::destroy_memory(VkDeviceMemory memory, void* mapped) noexcept {
    if (mapped != nullptr) device_->api().unmap_memory(device_->handle(), memory);
    if (memory != VK_NULL_HANDLE) device_->api().free_memory(device_->handle(), memory, nullptr);
}

MemoryPoolStats MemoryPool::stats() const noexcept {
    MemoryPoolStats result{};
    result.blocks = blocks_.size();
    result.dedicated_allocations = dedicated_.size();
    result.device_allocation_calls = device_allocation_calls_;
    for (const auto& block : blocks_) {
        const auto block_stats = block->allocator.stats();
        result.reserved_bytes += block_stats.total_bytes;
        result.allocated_bytes += block_stats.allocated_bytes;
        result.requested_bytes += block_stats.requested_bytes;
        result.live_allocations += block_stats.live_allocations;
    }
    for (const auto& [token, allocation] : dedicated_) {
        static_cast<void>(token);
        result.reserved_bytes += allocation.size;
        result.allocated_bytes += allocation.size;
        result.requested_bytes += allocation.size;
        ++result.live_allocations;
    }
    return result;
}

}  // namespace kora::detail
