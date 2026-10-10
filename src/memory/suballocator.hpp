#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kora/errors.hpp"

namespace kora::detail {

/// A byte range returned by Suballocator. The allocated size includes allocator rounding.
struct Suballocation {
    std::uint64_t offset = 0;
    std::uint64_t requested_size = 0;
    std::uint64_t allocated_size = 0;
};

struct SuballocatorStats {
    std::uint64_t total_bytes = 0;
    std::uint64_t allocated_bytes = 0;
    std::uint64_t requested_bytes = 0;
    std::uint64_t free_bytes = 0;
    std::uint64_t internal_fragmentation_bytes = 0;
    std::uint64_t external_fragmentation_bytes = 0;
    std::uint64_t largest_free_block = 0;
    std::uint64_t high_water_mark = 0;
    std::size_t live_allocations = 0;
};

/// CPU-only aligned segregated-fit allocator for offsets within one memory block.
class Suballocator final {
public:
    static Result<Suballocator> create(std::uint64_t capacity);

    Result<Suballocation> allocate(std::uint64_t size, std::uint64_t alignment);
    Status deallocate(const Suballocation& allocation);

    SuballocatorStats stats() const noexcept;
    bool validate_invariants() const noexcept;
    std::uint64_t capacity() const noexcept { return capacity_; }

private:
    static constexpr std::size_t kFirstLevelCount = 64;
    static constexpr std::size_t kSecondLevelCount = 4;
    static constexpr std::size_t kBinCount = kFirstLevelCount * kSecondLevelCount;
    static constexpr std::size_t kInvalid = std::numeric_limits<std::size_t>::max();
    static constexpr std::uint64_t kAllocationQuantum = 16;

    struct Region {
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
        std::uint64_t requested_size = 0;
        std::size_t previous = kInvalid;
        std::size_t next = kInvalid;
        std::size_t previous_free = kInvalid;
        std::size_t next_free = kInvalid;
        bool free = false;
        bool active = false;
    };

    explicit Suballocator(std::uint64_t capacity);

    static std::pair<std::size_t, std::size_t> size_class(std::uint64_t size) noexcept;
    static std::size_t bin_index(std::size_t first, std::size_t second) noexcept;
    std::size_t create_region(Region region);
    void retire_region(std::size_t index) noexcept;
    void insert_free(std::size_t index) noexcept;
    void remove_free(std::size_t index) noexcept;
    std::pair<std::size_t, std::uint64_t> find_fit(std::uint64_t size,
                                                   std::uint64_t alignment) const noexcept;
    std::pair<std::size_t, std::size_t> next_nonempty_bin(std::size_t first,
                                                          std::size_t second) const noexcept;
    void debug_validate() const noexcept;

    std::uint64_t capacity_ = 0;
    std::uint64_t high_water_mark_ = 0;
    std::size_t first_region_ = kInvalid;
    std::vector<Region> regions_;
    std::vector<std::size_t> recycled_regions_;
    std::array<std::size_t, kBinCount> free_heads_{};
    std::uint64_t first_level_bitmap_ = 0;
    std::array<std::uint8_t, kFirstLevelCount> second_level_bitmaps_{};
    std::unordered_map<std::uint64_t, std::size_t> live_allocations_;
};

}  // namespace kora::detail
