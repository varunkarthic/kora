#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/performance_db.hpp"
#include "kora/device_info.hpp"

namespace kora::detail::kernel {

struct WorkgroupCandidate {
    std::array<std::uint32_t, 3> local_size{1, 1, 1};
    friend bool operator==(const WorkgroupCandidate&, const WorkgroupCandidate&) = default;
};

Result<std::vector<WorkgroupCandidate>> generate_workgroup_candidates(
    const std::array<std::uint32_t, 3>& base_local_size, bool local_size_id_based,
    const DeviceInfo::Limits& limits, const DeviceInfo::Subgroup& subgroup,
    std::size_t maximum_candidates = 8);

std::string workgroup_performance_key(std::string_view device_key, std::uint32_t driver_version,
                                      std::string_view kernel_id, std::string_view entry_point,
                                      const WorkgroupCandidate& candidate);

class WorkgroupAutotuner final {
public:
    struct Trial {
        std::size_t candidate_index = 0;
        bool warmup = false;
        bool tracked = false;
    };

    WorkgroupAutotuner(std::vector<WorkgroupCandidate> candidates, std::size_t base_candidate,
                       std::uint32_t warmup_count = 2, std::uint32_t measurement_count = 5,
                       double maximum_dispersion = 0.5);

    Trial reserve();
    void cancel(const Trial& trial) noexcept;
    Status observe(const Trial& trial, double duration_nanoseconds);
    Status restore(std::size_t candidate_index, const PerformanceStats& stats);
    std::array<std::uint32_t, 3> selected_local_size() const noexcept;
    std::optional<std::size_t> selected_index() const noexcept;
    bool ready() const noexcept;
    const std::vector<WorkgroupCandidate>& candidates() const noexcept { return candidates_; }

private:
    struct CandidateState {
        std::uint32_t warmups = 0;
        std::uint32_t pending_warmups = 0;
        std::uint32_t pending_measurements = 0;
        std::vector<double> measurements;
        bool evaluated = false;
        bool stable = false;
        double score = 0.0;
    };

    void evaluate_locked();

    std::vector<WorkgroupCandidate> candidates_;
    std::vector<CandidateState> states_;
    std::size_t base_candidate_ = 0;
    std::uint32_t warmup_count_ = 2;
    std::uint32_t measurement_count_ = 5;
    double maximum_dispersion_ = 0.5;
    mutable std::mutex mutex_;
    std::optional<std::size_t> selected_;
    std::size_t cursor_ = 0;
};

}  // namespace kora::detail::kernel
