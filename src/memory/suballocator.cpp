#include "memory/suballocator.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <string>

#include "kora/detail/checked.hpp"

namespace kora::detail {

Result<Suballocator> Suballocator::create(std::uint64_t capacity) {
    if (capacity == 0) {
        return Error(ErrorCode::InvalidArgument, "suballocator capacity must be greater than zero");
    }
    return Suballocator(capacity);
}

Suballocator::Suballocator(std::uint64_t capacity) : capacity_(capacity) {
    free_heads_.fill(kInvalid);
    regions_.push_back(Region{.offset = 0, .size = capacity, .free = true, .active = true});
    first_region_ = 0;
    insert_free(0);
    debug_validate();
}

std::pair<std::size_t, std::size_t> Suballocator::size_class(std::uint64_t size) noexcept {
    assert(size != 0);
    const auto first = static_cast<std::size_t>(std::bit_width(size) - 1U);
    const std::uint64_t base = std::uint64_t{1} << first;
    const std::uint64_t step = std::max<std::uint64_t>(base / kSecondLevelCount, 1);
    const auto second = static_cast<std::size_t>(std::min<std::uint64_t>((size - base) / step, 3));
    return {first, second};
}

std::size_t Suballocator::bin_index(std::size_t first, std::size_t second) noexcept {
    return first * kSecondLevelCount + second;
}

std::size_t Suballocator::create_region(Region region) {
    if (!recycled_regions_.empty()) {
        const auto index = recycled_regions_.back();
        recycled_regions_.pop_back();
        regions_[index] = region;
        return index;
    }
    regions_.push_back(region);
    return regions_.size() - 1;
}

void Suballocator::retire_region(std::size_t index) noexcept {
    regions_[index] = Region{};
    recycled_regions_.push_back(index);
}

void Suballocator::insert_free(std::size_t index) noexcept {
    auto& region = regions_[index];
    assert(region.active && region.free && region.size != 0);
    const auto [first, second] = size_class(region.size);
    const auto bin = bin_index(first, second);
    region.previous_free = kInvalid;
    region.next_free = free_heads_[bin];
    if (region.next_free != kInvalid) regions_[region.next_free].previous_free = index;
    free_heads_[bin] = index;
    second_level_bitmaps_[first] |= static_cast<std::uint8_t>(1U << second);
    first_level_bitmap_ |= std::uint64_t{1} << first;
}

void Suballocator::remove_free(std::size_t index) noexcept {
    auto& region = regions_[index];
    assert(region.active && region.free);
    const auto [first, second] = size_class(region.size);
    const auto bin = bin_index(first, second);
    if (region.previous_free == kInvalid) {
        assert(free_heads_[bin] == index);
        free_heads_[bin] = region.next_free;
    } else {
        regions_[region.previous_free].next_free = region.next_free;
    }
    if (region.next_free != kInvalid) regions_[region.next_free].previous_free = region.previous_free;
    region.previous_free = kInvalid;
    region.next_free = kInvalid;
    if (free_heads_[bin] == kInvalid) {
        second_level_bitmaps_[first] &= static_cast<std::uint8_t>(~(1U << second));
        if (second_level_bitmaps_[first] == 0) first_level_bitmap_ &= ~(std::uint64_t{1} << first);
    }
}

std::pair<std::size_t, std::size_t> Suballocator::next_nonempty_bin(std::size_t first,
                                                                    std::size_t second) const noexcept {
    const auto remaining_second =
        static_cast<std::uint8_t>(second_level_bitmaps_[first] & static_cast<std::uint8_t>(0xFU << second));
    if (remaining_second != 0) {
        return {first, static_cast<std::size_t>(std::countr_zero(remaining_second))};
    }
    if (first == kFirstLevelCount - 1) return {kInvalid, kInvalid};
    const auto remaining_first = first_level_bitmap_ & (~std::uint64_t{0} << (first + 1));
    if (remaining_first == 0) return {kInvalid, kInvalid};
    const auto next_first = static_cast<std::size_t>(std::countr_zero(remaining_first));
    return {next_first, static_cast<std::size_t>(std::countr_zero(second_level_bitmaps_[next_first]))};
}

std::pair<std::size_t, std::uint64_t> Suballocator::find_fit(std::uint64_t size,
                                                             std::uint64_t alignment) const noexcept {
    auto [first, second] = size_class(size);
    while (first != kInvalid) {
        auto index = free_heads_[bin_index(first, second)];
        while (index != kInvalid) {
            const auto& region = regions_[index];
            const auto aligned = checked::align_up(region.offset, alignment);
            if (aligned) {
                const auto end = checked::add(*aligned, size);
                const auto region_end = checked::add(region.offset, region.size);
                if (end && region_end && *end <= *region_end) return {index, *aligned};
            }
            index = region.next_free;
        }
        std::tie(first, second) = next_nonempty_bin(first, second + 1);
    }
    return {kInvalid, 0};
}

Result<Suballocation> Suballocator::allocate(std::uint64_t size, std::uint64_t alignment) {
    if (size == 0) return Error(ErrorCode::InvalidArgument, "suballocation size must be greater than zero");
    if (!checked::is_pow2(alignment)) {
        return Error(ErrorCode::InvalidArgument, "suballocation alignment must be a non-zero power of two");
    }
    const auto rounded = checked::align_up(size, kAllocationQuantum);
    if (!rounded) return Error(ErrorCode::Overflow, "suballocation size rounding overflow");

    const auto [free_index, allocation_offset] = find_fit(*rounded, alignment);
    if (free_index == kInvalid) {
        return Error(ErrorCode::AllocationFailed,
                     "no free suballocation region satisfies the requested size and alignment");
    }

    const auto original_offset = regions_[free_index].offset;
    const auto original_size = regions_[free_index].size;
    const auto original_previous = regions_[free_index].previous;
    const auto original_next = regions_[free_index].next;
    const auto prefix_size = allocation_offset - original_offset;
    const auto allocation_end = *checked::add(allocation_offset, *rounded);
    const auto original_end = *checked::add(original_offset, original_size);
    const auto suffix_size = original_end - allocation_end;
    remove_free(free_index);

    std::size_t allocation_index = free_index;
    if (prefix_size != 0) {
        regions_[free_index] = Region{.offset = original_offset,
                                      .size = prefix_size,
                                      .previous = original_previous,
                                      .free = true,
                                      .active = true};
        allocation_index = create_region(Region{.offset = allocation_offset,
                                                .size = *rounded,
                                                .requested_size = size,
                                                .previous = free_index,
                                                .next = original_next,
                                                .active = true});
        regions_[free_index].next = allocation_index;
        insert_free(free_index);
    } else {
        regions_[free_index] = Region{.offset = allocation_offset,
                                      .size = *rounded,
                                      .requested_size = size,
                                      .previous = original_previous,
                                      .next = original_next,
                                      .active = true};
    }

    if (suffix_size != 0) {
        const auto suffix_index = create_region(Region{.offset = allocation_end,
                                                       .size = suffix_size,
                                                       .previous = allocation_index,
                                                       .next = original_next,
                                                       .free = true,
                                                       .active = true});
        regions_[allocation_index].next = suffix_index;
        if (original_next != kInvalid) regions_[original_next].previous = suffix_index;
        insert_free(suffix_index);
    } else if (original_next != kInvalid) {
        regions_[original_next].previous = allocation_index;
    }

    live_allocations_.emplace(allocation_offset, allocation_index);
    const auto current = stats().allocated_bytes;
    high_water_mark_ = std::max(high_water_mark_, current);
    debug_validate();
    return Suballocation{allocation_offset, size, *rounded};
}

Status Suballocator::deallocate(const Suballocation& allocation) {
    const auto live = live_allocations_.find(allocation.offset);
    if (live == live_allocations_.end()) {
        return Error(ErrorCode::InvalidArgument, "suballocation is unknown or has already been freed");
    }
    std::size_t index = live->second;
    auto& allocated = regions_[index];
    if (allocated.requested_size != allocation.requested_size ||
        allocated.size != allocation.allocated_size) {
        return Error(ErrorCode::InvalidArgument, "suballocation handle does not match the live allocation");
    }
    live_allocations_.erase(live);
    allocated.requested_size = 0;
    allocated.free = true;

    if (allocated.previous != kInvalid && regions_[allocated.previous].free) {
        const auto previous = allocated.previous;
        remove_free(previous);
        regions_[previous].size += allocated.size;
        regions_[previous].next = allocated.next;
        if (allocated.next != kInvalid) regions_[allocated.next].previous = previous;
        retire_region(index);
        index = previous;
    }
    if (regions_[index].next != kInvalid && regions_[regions_[index].next].free) {
        const auto next = regions_[index].next;
        remove_free(next);
        regions_[index].size += regions_[next].size;
        regions_[index].next = regions_[next].next;
        if (regions_[next].next != kInvalid) regions_[regions_[next].next].previous = index;
        retire_region(next);
    }
    insert_free(index);
    debug_validate();
    return ok_status();
}

SuballocatorStats Suballocator::stats() const noexcept {
    SuballocatorStats result{};
    result.total_bytes = capacity_;
    result.high_water_mark = high_water_mark_;
    result.live_allocations = live_allocations_.size();
    for (auto index = first_region_; index != kInvalid; index = regions_[index].next) {
        const auto& region = regions_[index];
        if (region.free) {
            result.free_bytes += region.size;
            result.largest_free_block = std::max(result.largest_free_block, region.size);
        } else {
            result.allocated_bytes += region.size;
            result.requested_bytes += region.requested_size;
        }
    }
    result.internal_fragmentation_bytes = result.allocated_bytes - result.requested_bytes;
    result.external_fragmentation_bytes = result.free_bytes - result.largest_free_block;
    return result;
}

bool Suballocator::validate_invariants() const noexcept {
    if (capacity_ == 0 || first_region_ == kInvalid || regions_.empty()) return false;
    std::vector<unsigned> free_membership(regions_.size(), 0);
    for (std::size_t first = 0; first < kFirstLevelCount; ++first) {
        const bool first_marked = (first_level_bitmap_ & (std::uint64_t{1} << first)) != 0;
        if (first_marked != (second_level_bitmaps_[first] != 0)) return false;
        for (std::size_t second = 0; second < kSecondLevelCount; ++second) {
            const auto head = free_heads_[bin_index(first, second)];
            const bool second_marked = (second_level_bitmaps_[first] & (1U << second)) != 0;
            if (second_marked != (head != kInvalid)) return false;
            std::size_t previous = kInvalid;
            std::size_t visited = 0;
            for (auto index = head; index != kInvalid; index = regions_[index].next_free) {
                if (++visited > regions_.size()) return false;
                if (index >= regions_.size()) return false;
                const auto& region = regions_[index];
                if (!region.active || !region.free || region.previous_free != previous) return false;
                if (size_class(region.size) != std::pair{first, second}) return false;
                if (++free_membership[index] != 1) return false;
                previous = index;
            }
        }
    }

    std::uint64_t expected_offset = 0;
    std::uint64_t total = 0;
    std::size_t previous = kInvalid;
    std::size_t live_count = 0;
    std::size_t visited = 0;
    for (auto index = first_region_; index != kInvalid; index = regions_[index].next) {
        if (++visited > regions_.size()) return false;
        if (index >= regions_.size()) return false;
        const auto& region = regions_[index];
        if (!region.active || region.size == 0 || region.previous != previous ||
            region.offset != expected_offset) {
            return false;
        }
        if (region.free != (free_membership[index] == 1)) return false;
        if (region.free && region.next != kInvalid && regions_[region.next].free) return false;
        const auto end = checked::add(region.offset, region.size);
        const auto sum = checked::add(total, region.size);
        if (!end || !sum) return false;
        expected_offset = *end;
        total = *sum;
        if (!region.free) {
            const auto live = live_allocations_.find(region.offset);
            if (live == live_allocations_.end() || live->second != index || region.requested_size == 0 ||
                region.requested_size > region.size) {
                return false;
            }
            ++live_count;
        }
        previous = index;
    }
    return total == capacity_ && expected_offset == capacity_ && live_count == live_allocations_.size();
}

void Suballocator::debug_validate() const noexcept {
#ifndef NDEBUG
    assert(validate_invariants());
#endif
}

}  // namespace kora::detail
