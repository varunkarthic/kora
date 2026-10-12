#include "memory/staging_ring.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

#include "kora/detail/checked.hpp"

namespace kora::detail {
namespace {

Error vk_error(const char* operation, VkResult result) {
    return Error(ErrorCode::AllocationFailed, operation).with_vk(result);
}

}  // namespace

struct StagingRing::Segment {
    std::uint64_t id = 0;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    VkDeviceSize reserved_size = 0;
    std::optional<std::uint64_t> release_timeline;
};

struct StagingRing::Block {
    VkBuffer buffer = VK_NULL_HANDLE;
    PoolAllocation allocation;
    VkDeviceSize capacity = 0;
    VkDeviceSize cursor = 0;
    // Kept in offset order so reserve() can scan live ranges without per-call scratch allocation/sort.
    std::vector<Segment> segments;
};

StagingRing::StagingRing(const vk::Device& device, MemoryPool& host_pool, Options options)
    : device_(&device), host_pool_(&host_pool), options_(options) {}

Result<std::unique_ptr<StagingRing>> StagingRing::create(const vk::Device& device, MemoryPool& host_pool,
                                                         Options options) {
    const auto& api = device.api();
    if (device.handle() == VK_NULL_HANDLE || !host_pool.host_visible() || !api.create_buffer ||
        !api.destroy_buffer || !api.get_buffer_memory_requirements || !api.bind_buffer_memory) {
        return Error(ErrorCode::InvalidArgument,
                     "StagingRing requires a live device, host-visible pool, and buffer dispatch");
    }
    if (options.initial_capacity == 0 || options.maximum_capacity < options.initial_capacity ||
        !checked::is_pow2(options.non_coherent_atom_size)) {
        return Error(ErrorCode::InvalidArgument, "StagingRing capacity limits are invalid");
    }
    if (!should_use_staging(options.topology, options.measured_staging_benefit)) {
        return Error(ErrorCode::UnsupportedFeature,
                     "transfer policy selected direct host-visible access instead of staging");
    }
    auto ring = std::unique_ptr<StagingRing>(new StagingRing(device, host_pool, options));
    auto block = ring->add_block(options.initial_capacity);
    if (!block) return std::move(block).error();
    return ring;
}

StagingRing::~StagingRing() {
    for (const auto& block : blocks_) {
        if (block->buffer != VK_NULL_HANDLE)
            device_->api().destroy_buffer(device_->handle(), block->buffer, nullptr);
        if (block->allocation.memory != VK_NULL_HANDLE) (void)host_pool_->deallocate(block->allocation);
    }
}

Result<void> StagingRing::add_block(VkDeviceSize capacity) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = capacity;
    buffer_info.usage = options_.additional_usage |
                        (options_.direction == StagingDirection::Upload ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                                                        : VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    auto& api = device_->api();
    auto result = api.create_buffer(device_->handle(), &buffer_info, nullptr, &buffer);
    if (result != VK_SUCCESS) return vk_error("vkCreateBuffer failed for staging ring", result);

    VkMemoryRequirements requirements{};
    api.get_buffer_memory_requirements(device_->handle(), buffer, &requirements);
    if (requirements.size == 0 || !checked::is_pow2(requirements.alignment) ||
        requirements.memoryTypeBits == 0) {
        api.destroy_buffer(device_->handle(), buffer, nullptr);
        return Error(ErrorCode::UnsupportedFeature, "Vulkan buffer memory requirements are unavailable");
    }
    if (host_pool_->memory_type_index() >= 32 ||
        (requirements.memoryTypeBits & (std::uint32_t{1} << host_pool_->memory_type_index())) == 0) {
        api.destroy_buffer(device_->handle(), buffer, nullptr);
        return Error(ErrorCode::UnsupportedFeature,
                     "StagingRing host-visible pool memory type is incompatible with its Vulkan buffer");
    }
    const auto storage_size = checked::align_up(requirements.size, options_.non_coherent_atom_size);
    if (!storage_size) {
        api.destroy_buffer(device_->handle(), buffer, nullptr);
        return Error(ErrorCode::Overflow, "StagingRing memory requirement alignment overflowed");
    }
    const auto memory_alignment =
        std::max<VkDeviceSize>(requirements.alignment, options_.non_coherent_atom_size);
    auto allocation = host_pool_->allocate({*storage_size, memory_alignment, false, false});
    if (!allocation) {
        api.destroy_buffer(device_->handle(), buffer, nullptr);
        return std::move(allocation).error();
    }
    result = api.bind_buffer_memory(device_->handle(), buffer, allocation.value().memory,
                                    allocation.value().offset);
    if (result != VK_SUCCESS) {
        api.destroy_buffer(device_->handle(), buffer, nullptr);
        (void)host_pool_->deallocate(allocation.value());
        return vk_error("vkBindBufferMemory failed for staging ring", result);
    }
    auto block = std::make_unique<Block>();
    block->buffer = buffer;
    block->allocation = allocation.value();
    block->capacity = capacity;
    blocks_.push_back(std::move(block));
    capacity_ += capacity;
    return {};
}

void StagingRing::reclaim(std::uint64_t completed_timeline) {
    for (auto& block : blocks_) {
        std::erase_if(block->segments, [completed_timeline](const Segment& segment) {
            return segment.release_timeline && *segment.release_timeline <= completed_timeline;
        });
    }
}

Status StagingRing::reclaim_completed(std::uint64_t completed_timeline) {
    std::lock_guard lock(mutex_);
    reclaim(completed_timeline);
    return ok_status();
}

Result<StagingSlice> StagingRing::reserve(VkDeviceSize size, VkDeviceSize alignment,
                                          std::uint64_t completed_timeline, const WaitForTimeline& wait_for) {
    if (size == 0 || !checked::is_pow2(alignment)) {
        return Error(ErrorCode::InvalidArgument,
                     "staging reservation requires nonzero size and power-of-two alignment");
    }
    const auto reserved_size = checked::align_up(size, options_.non_coherent_atom_size);
    if (!reserved_size) return Error(ErrorCode::Overflow, "staging reservation alignment overflowed");
    alignment = std::max(alignment, options_.non_coherent_atom_size);
    std::unique_lock lock(mutex_);
    auto completed = completed_timeline;
    for (;;) {
        reclaim(completed);
        for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
            auto& block = *blocks_[block_index];
            if (*reserved_size > block.capacity) continue;
            auto try_gap = [&](VkDeviceSize begin) -> std::optional<VkDeviceSize> {
                auto aligned = checked::align_up(begin, alignment);
                if (!aligned || *aligned > block.capacity || *reserved_size > block.capacity - *aligned)
                    return std::nullopt;
                auto candidate = *aligned;
                const auto first =
                    std::lower_bound(block.segments.begin(), block.segments.end(), candidate,
                                     [](const Segment& segment, VkDeviceSize offset) {
                                         return segment.offset + segment.reserved_size <= offset;
                                     });
                for (auto segment = first; segment != block.segments.end(); ++segment) {
                    const auto start = segment->offset;
                    const auto end = start + segment->reserved_size;
                    if (candidate <= start && *reserved_size <= start - candidate) return candidate;
                    if (candidate < end) {
                        auto next = checked::align_up(end, alignment);
                        if (!next) return std::nullopt;
                        candidate = *next;
                    }
                }
                if (candidate <= block.capacity && *reserved_size <= block.capacity - candidate)
                    return candidate;
                return std::nullopt;
            };
            auto offset = try_gap(block.cursor);
            if (!offset && block.cursor != 0) offset = try_gap(0);
            if (offset) {
                if (next_reservation_ == std::numeric_limits<std::uint64_t>::max())
                    return Error(ErrorCode::Overflow, "staging reservation ID space is exhausted");
                const auto id = next_reservation_++;
                const Segment reservation{id, *offset, size, *reserved_size, std::nullopt};
                const auto insertion = std::lower_bound(
                    block.segments.begin(), block.segments.end(), *offset,
                    [](const Segment& segment, VkDeviceSize offset) { return segment.offset < offset; });
                block.segments.insert(insertion, reservation);
                block.cursor = (*offset + *reserved_size) % block.capacity;
                auto* mapped = static_cast<std::byte*>(block.allocation.mapped_data);
                if (mapped == nullptr)
                    return Error(ErrorCode::InternalInvariantViolation,
                                 "host-visible staging allocation is not mapped");
                return StagingSlice{
                    id,   block.buffer,   block.allocation.memory,           *offset,
                    size, *reserved_size, block.allocation.offset + *offset, mapped + *offset};
            }
        }

        if (size <= options_.maximum_capacity - capacity_) {
            const auto growth =
                std::min(std::max(options_.initial_capacity, size), options_.maximum_capacity - capacity_);
            auto added = add_block(growth);
            if (added) continue;
            if (added.error().code != ErrorCode::AllocationFailed &&
                added.error().code != ErrorCode::OutOfMemoryBudget)
                return std::move(added).error();
        }

        std::optional<std::uint64_t> oldest;
        for (const auto& block : blocks_)
            for (const auto& segment : block->segments)
                if (segment.release_timeline && (!oldest || *segment.release_timeline < *oldest))
                    oldest = segment.release_timeline;
        if (!oldest || !wait_for)
            return Error(ErrorCode::AllocationFailed,
                         "staging ring is full with unsubmitted reservations or has no wait callback");
        lock.unlock();
        auto waited = wait_for(*oldest);
        if (!waited) return std::move(waited).error();
        lock.lock();
        completed = std::max(completed, waited.value());
        reclaim(completed);
    }
}

Status StagingRing::retire(const StagingSlice& slice, std::uint64_t release_timeline) {
    if (release_timeline == 0)
        return Error(ErrorCode::InvalidArgument, "release timeline value must be nonzero");
    std::lock_guard lock(mutex_);
    for (auto& block : blocks_) {
        if (block->buffer != slice.buffer) continue;
        for (auto& segment : block->segments) {
            if (segment.id != slice.reservation || segment.offset != slice.offset ||
                segment.size != slice.size)
                continue;
            if (segment.release_timeline)
                return Error(ErrorCode::InvalidArgument, "staging reservation was already retired");
            segment.release_timeline = release_timeline;
            return ok_status();
        }
    }
    return Error(ErrorCode::InvalidArgument, "staging reservation is unknown or was already reclaimed");
}

Status StagingRing::cancel(const StagingSlice& slice) {
    std::lock_guard lock(mutex_);
    for (auto& block : blocks_) {
        if (block->buffer != slice.buffer) continue;
        const auto found =
            std::find_if(block->segments.begin(), block->segments.end(), [&](const Segment& segment) {
                return segment.id == slice.reservation && segment.offset == slice.offset &&
                       segment.size == slice.size;
            });
        if (found == block->segments.end()) break;
        if (found->release_timeline)
            return Error(ErrorCode::InvalidArgument, "submitted staging reservation cannot be cancelled");
        block->segments.erase(found);
        return ok_status();
    }
    return Error(ErrorCode::InvalidArgument, "staging reservation is unknown or already released");
}

Status StagingRing::flush(const StagingSlice& slice) const {
    std::lock_guard lock(mutex_);
    for (const auto& block : blocks_) {
        if (block->buffer != slice.buffer || block->allocation.memory != slice.memory) continue;
        const auto found =
            std::find_if(block->segments.begin(), block->segments.end(), [&](const Segment& segment) {
                return segment.id == slice.reservation && segment.offset == slice.offset &&
                       segment.size == slice.size && segment.reserved_size == slice.reserved_size &&
                       !segment.release_timeline;
            });
        if (found == block->segments.end()) break;
        if (host_pool_->host_coherent()) return ok_status();
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = slice.memory;
        range.offset = slice.memory_offset;
        range.size = slice.reserved_size;
        const auto result = device_->api().flush_mapped_memory_ranges(device_->handle(), 1, &range);
        if (result != VK_SUCCESS)
            return Error(ErrorCode::TransferFailed, "Flushing staging upload bytes failed").with_vk(result);
        return ok_status();
    }
    return Error(ErrorCode::InvalidArgument, "staging upload slice is unknown or already submitted");
}

Status StagingRing::invalidate(const StagingSlice& slice) const {
    std::lock_guard lock(mutex_);
    for (const auto& block : blocks_) {
        if (block->buffer != slice.buffer || block->allocation.memory != slice.memory) continue;
        const auto found =
            std::find_if(block->segments.begin(), block->segments.end(), [&](const Segment& segment) {
                return segment.id == slice.reservation && segment.offset == slice.offset &&
                       segment.size == slice.size && segment.reserved_size == slice.reserved_size &&
                       segment.release_timeline.has_value();
            });
        if (found == block->segments.end()) break;
        if (host_pool_->host_coherent()) return ok_status();
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = slice.memory;
        range.offset = slice.memory_offset;
        range.size = slice.reserved_size;
        const auto result = device_->api().invalidate_mapped_memory_ranges(device_->handle(), 1, &range);
        if (result != VK_SUCCESS)
            return Error(ErrorCode::TransferFailed, "Invalidating staging readback bytes failed")
                .with_vk(result);
        return ok_status();
    }
    return Error(ErrorCode::InvalidArgument, "staging readback slice is unknown or not submitted");
}

bool StagingRing::should_use_staging(MemoryTopology topology, bool measured_staging_benefit) noexcept {
    if (measured_staging_benefit) return true;
    switch (topology) {
        case MemoryTopology::Discrete:
            return true;
        case MemoryTopology::Uma:
            return false;
        case MemoryTopology::DiscreteWithBar:
        case MemoryTopology::Unknown:
            return false;
    }
    return false;
}

VkDeviceSize StagingRing::capacity() const noexcept {
    std::lock_guard lock(mutex_);
    return capacity_;
}

}  // namespace kora::detail
