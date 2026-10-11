#include "memory/eviction.hpp"

#include <algorithm>
#include <cmath>
#include <new>
#include <stdexcept>

#include "kora/detail/checked.hpp"

namespace kora::detail {

Status AccessHistory::record(std::uint64_t tick) {
    if (const auto recent = most_recent_tick(); recent && tick < *recent)
        return Error(ErrorCode::InvalidArgument, "buffer access ticks must be monotonic");
    ticks_[next_] = tick;
    next_ = (next_ + 1) % kDistance;
    size_ = std::min(size_ + 1, kDistance);
    return ok_status();
}

std::optional<std::uint64_t> AccessHistory::most_recent_tick() const noexcept {
    if (size_ == 0) return std::nullopt;
    const auto index = (next_ + kDistance - 1) % kDistance;
    return ticks_[index];
}

std::optional<std::uint64_t> AccessHistory::kth_recent_age(std::uint64_t now_tick) const noexcept {
    if (size_ < kDistance) return std::nullopt;
    const auto oldest = ticks_[next_];
    if (oldest > now_tick) return std::nullopt;
    return now_tick - oldest;
}

Status EvictionPolicy::record_access(BufferId id, std::uint64_t tick) {
    try {
        auto [entry, inserted] = histories_.try_emplace(id);
        static_cast<void>(inserted);
        return entry->second.record(tick);
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to record buffer eviction history");
    } catch (const std::length_error&) {
        return Error(ErrorCode::Overflow, "buffer eviction history exceeds container capacity");
    }
}

void EvictionPolicy::forget(BufferId id) noexcept {
    histories_.erase(id);
}

Result<double> EvictionPolicy::keep_value(const EvictionCandidate& candidate, std::uint64_t now_tick) const {
    const auto finite_nonnegative = [](double value) { return std::isfinite(value) && value >= 0.0; };
    if (!std::isfinite(config_.recent_access_weight) || config_.recent_access_weight < 0.0 ||
        !std::isfinite(config_.idle_age_penalty_per_tick) || config_.idle_age_penalty_per_tick < 0.0 ||
        !std::isfinite(candidate.reuse_probability) || candidate.reuse_probability < 0.0 ||
        candidate.reuse_probability > 1.0 || !finite_nonnegative(candidate.estimated_reload_cost_ms) ||
        !finite_nonnegative(candidate.user_priority_bonus) ||
        !finite_nonnegative(candidate.expensive_recreation_bonus) ||
        !finite_nonnegative(candidate.estimated_writeback_cost_ms)) {
        return Error(ErrorCode::InvalidArgument, "eviction keep-value inputs must be finite and in range");
    }
    auto last_access = candidate.last_access_tick;
    double recent_bonus = 0.0;
    const auto history = histories_.find(candidate.id);
    if (history != histories_.end()) {
        if (const auto recent = history->second.most_recent_tick()) last_access = *recent;
        if (const auto age = history->second.kth_recent_age(now_tick))
            recent_bonus = config_.recent_access_weight / (1.0 + static_cast<double>(*age));
    }
    if (last_access > now_tick)
        return Error(ErrorCode::InvalidArgument, "buffer last-access tick is later than the planning tick");
    const auto idle_age = now_tick - last_access;
    const auto idle_penalty = config_.idle_age_penalty_per_tick * static_cast<double>(idle_age);
    const auto writeback_penalty =
        candidate.device_newer && !candidate.disposable && !candidate.alternate_current_copy
            ? candidate.estimated_writeback_cost_ms
            : 0.0;
    const auto value = candidate.reuse_probability * candidate.estimated_reload_cost_ms + recent_bonus +
                       candidate.user_priority_bonus + candidate.expensive_recreation_bonus +
                       writeback_penalty - idle_penalty;
    if (!std::isfinite(value)) return Error(ErrorCode::Overflow, "eviction keep-value arithmetic overflow");
    return value;
}

Result<EvictionPlan> EvictionPolicy::plan(std::span<const EvictionCandidate> candidates,
                                          std::uint64_t available_bytes, std::uint64_t requested_bytes,
                                          std::uint64_t now_tick) const {
    EvictionPlan plan;
    if (requested_bytes <= available_bytes) return plan;
    const auto bytes_needed = requested_bytes - available_bytes;
    std::vector<EvictionDecision> eligible;
    try {
        eligible.reserve(candidates.size());
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const auto& candidate = candidates[index];
            if (candidate.resident_bytes == 0)
                return Error(ErrorCode::InvalidArgument, "eviction candidates must own resident bytes");
            for (std::size_t previous = 0; previous < index; ++previous) {
                if (candidates[previous].id == candidate.id)
                    return Error(ErrorCode::InvalidArgument, "eviction candidate IDs must be unique");
            }
            if (candidate.pin_count != 0 || candidate.in_flight || candidate.being_submitted) continue;
            KORA_TRY(value, keep_value(candidate, now_tick));
            const auto action =
                candidate.device_newer && !candidate.disposable && !candidate.alternate_current_copy
                    ? EvictionAction::WritebackThenEvict
                    : EvictionAction::Evict;
            eligible.push_back({candidate.id, candidate.resident_bytes, value, action});
        }
        std::sort(eligible.begin(), eligible.end(), [](const auto& left, const auto& right) {
            if (left.keep_value != right.keep_value) return left.keep_value < right.keep_value;
            if (left.id.slot != right.id.slot) return left.id.slot < right.id.slot;
            return left.id.generation < right.id.generation;
        });
        for (const auto& candidate : eligible) {
            plan.decisions.push_back(candidate);
            const auto total = checked::add(plan.bytes_to_reclaim, candidate.resident_bytes);
            if (!total) return Error(ErrorCode::Overflow, "eviction plan byte count overflow");
            plan.bytes_to_reclaim = *total;
            if (plan.bytes_to_reclaim >= bytes_needed) return plan;
        }
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to allocate eviction plan");
    } catch (const std::length_error&) {
        return Error(ErrorCode::Overflow, "eviction plan exceeds container capacity");
    }
    return Error(ErrorCode::OutOfMemoryBudget, "eligible buffers cannot satisfy the requested memory budget");
}

}  // namespace kora::detail
