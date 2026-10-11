#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <thread>
#include <vector>

#include "kernel/autotuner.hpp"

namespace {
using kora::DeviceInfo;
using kora::detail::PerformanceStats;
using kora::detail::kernel::generate_workgroup_candidates;
using kora::detail::kernel::workgroup_performance_key;
using kora::detail::kernel::WorkgroupAutotuner;
using kora::detail::kernel::WorkgroupCandidate;

TEST(WorkgroupCandidates, FiltersByDimensionsInvocationCountAndSubgroupWidth) {
    DeviceInfo::Limits limits;
    limits.max_compute_work_group_size = {128, 4, 4};
    limits.max_compute_work_group_invocations = 128;
    DeviceInfo::Subgroup subgroup;
    subgroup.supported = true;
    subgroup.size = 32;

    const auto candidates = generate_workgroup_candidates({64, 1, 1}, true, limits, subgroup, 8);
    ASSERT_TRUE(candidates);
    EXPECT_EQ(candidates.value(),
              (std::vector<WorkgroupCandidate>{{{64, 1, 1}}, {{32, 1, 1}}, {{128, 1, 1}}}));
    const auto fixed = generate_workgroup_candidates({64, 1, 1}, false, limits, subgroup);
    ASSERT_TRUE(fixed);
    EXPECT_EQ(fixed.value().size(), 1U);
    EXPECT_FALSE(generate_workgroup_candidates({256, 1, 1}, true, limits, subgroup));
}

TEST(WorkgroupAutotuner, UsesTwoWarmupsFiveMeasurementsAndChoosesFastestStableCandidate) {
    const std::vector<WorkgroupCandidate> candidates{{{32, 1, 1}}, {{64, 1, 1}}, {{128, 1, 1}}};
    WorkgroupAutotuner tuner(candidates, 1);
    std::array<std::uint32_t, 3> warmups{};
    std::array<std::uint32_t, 3> measurements{};
    for (std::uint32_t sample = 0; sample < 64 && !tuner.ready(); ++sample) {
        const auto trial = tuner.reserve();
        ASSERT_TRUE(trial.tracked);
        if (trial.warmup)
            ++warmups[trial.candidate_index];
        else
            ++measurements[trial.candidate_index];
        EXPECT_EQ(tuner.selected_local_size(), candidates[1].local_size);
        const double measurement = trial.candidate_index == 0   ? 300.0
                                   : trial.candidate_index == 1 ? 200.0
                                                                : 100.0;
        ASSERT_TRUE(tuner.observe(trial, measurement));
    }
    ASSERT_TRUE(tuner.ready());
    EXPECT_EQ(warmups, (std::array<std::uint32_t, 3>{2, 2, 2}));
    EXPECT_EQ(measurements, (std::array<std::uint32_t, 3>{5, 5, 5}));
    EXPECT_EQ(tuner.selected_index(), 2U);
    EXPECT_EQ(tuner.selected_local_size(), candidates[2].local_size);
}

TEST(WorkgroupAutotuner, RejectsUnstableCandidatesAndRestoresOnlyMatureRecords) {
    const std::vector<WorkgroupCandidate> candidates{{{32, 1, 1}}, {{64, 1, 1}}};
    WorkgroupAutotuner tuner(candidates, 0);
    PerformanceStats immature;
    immature.count = 4;
    immature.minimum = 80;
    immature.median = 80;
    immature.maximum = 80;
    immature.p95 = 80;
    ASSERT_TRUE(tuner.restore(0, immature));
    EXPECT_FALSE(tuner.ready());
    PerformanceStats unstable;
    unstable.count = 5;
    unstable.minimum = 50;
    unstable.median = 80;
    unstable.maximum = 160;
    ASSERT_TRUE(tuner.restore(0, unstable));
    PerformanceStats stable;
    stable.count = 5;
    stable.minimum = 118;
    stable.median = 120;
    stable.maximum = 125;
    ASSERT_TRUE(tuner.restore(1, stable));
    EXPECT_TRUE(tuner.ready());
    EXPECT_EQ(tuner.selected_index(), 1U);
    EXPECT_NE(workgroup_performance_key("device", 1, "kernel", "main", candidates[0]),
              workgroup_performance_key("device", 2, "kernel", "main", candidates[0]));
}

TEST(WorkgroupAutotuner, ConcurrentReservationsDoNotOverbookWarmupOrMeasurementTrials) {
    const std::vector<WorkgroupCandidate> candidates{{{32, 1, 1}}, {{64, 1, 1}}};
    WorkgroupAutotuner tuner(candidates, 0);
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 6; ++worker) {
        workers.emplace_back([&] {
            while (!tuner.ready()) {
                const auto trial = tuner.reserve();
                if (!trial.tracked) {
                    std::this_thread::yield();
                    continue;
                }
                const double latency = trial.candidate_index == 0 ? 200.0 : 100.0;
                EXPECT_TRUE(tuner.observe(trial, latency));
            }
        });
    }
    for (auto& worker : workers) worker.join();
    EXPECT_EQ(tuner.selected_index(), 1U);
}
}  // namespace
