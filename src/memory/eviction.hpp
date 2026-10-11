#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "kora/buffer.hpp"

namespace kora::detail {

/// Fixed-size LRU-K history. A single recent access is deliberately insufficient for a recency bonus.
class AccessHistory final {
public:
    static constexpr std::size_t kDistance = 2;

    Status record(std::uint64_t tick);
    std::size_t size() const noexcept { return size_; }
    std::optional<std::uint64_t> most_recent_tick() const noexcept;
    std::optional<std::uint64_t> kth_recent_age(std::uint64_t now_tick) const noexcept;

private:
    std::array<std::uint64_t, kDistance> ticks_{};
    std::size_t next_ = 0;
    std::size_t size_ = 0;
};

struct EvictionPolicyConfig {
    double recent_access_weight = 1.0;
    double idle_age_penalty_per_tick = 0.01;
};

struct EvictionCandidate {
    BufferId id{};
    std::uint64_t resident_bytes = 0;
    double reuse_probability = 0.0;
    double estimated_reload_cost_ms = 0.0;
    double user_priority_bonus = 0.0;
    double expensive_recreation_bonus = 0.0;
    double estimated_writeback_cost_ms = 0.0;
    std::uint64_t last_access_tick = 0;
    std::uint32_t pin_count = 0;
    bool in_flight = false;
    bool being_submitted = false;
    bool device_newer = false;
    bool disposable = false;
    bool alternate_current_copy = false;
};

enum class EvictionAction : std::uint8_t { Evict, WritebackThenEvict };

struct EvictionDecision {
    BufferId id{};
    std::uint64_t resident_bytes = 0;
    double keep_value = 0.0;
    EvictionAction action = EvictionAction::Evict;
};

struct EvictionPlan {
    std::uint64_t bytes_to_reclaim = 0;
    std::vector<EvictionDecision> decisions;
};

/// Deterministic policy planner. It does not perform I/O or free allocations; callers must execute
/// writeback decisions and revalidate buffer lifetime before applying an eviction.
class EvictionPolicy final {
public:
    explicit EvictionPolicy(EvictionPolicyConfig config = {}) : config_(config) {}

    Status record_access(BufferId id, std::uint64_t tick);
    void forget(BufferId id) noexcept;
    Result<double> keep_value(const EvictionCandidate& candidate, std::uint64_t now_tick) const;
    Result<EvictionPlan> plan(std::span<const EvictionCandidate> candidates, std::uint64_t available_bytes,
                              std::uint64_t requested_bytes, std::uint64_t now_tick) const;

private:
    struct BufferIdHash {
        std::size_t operator()(BufferId id) const noexcept {
            return static_cast<std::size_t>(id.generation ^ (static_cast<std::uint64_t>(id.slot) << 32U));
        }
    };

    EvictionPolicyConfig config_;
    std::unordered_map<BufferId, AccessHistory, BufferIdHash> histories_;
};

}  // namespace kora::detail
