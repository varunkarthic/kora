#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kora/errors.hpp"

namespace kora::detail {

enum class PerformanceTimingDomain : std::uint8_t {
    DeviceExecution = 0,
    HostSubmission = 1,
    HostTransfer = 2,
    QueueCompletion = 3,
};

struct PerformanceObservationKey {
    std::string device_uuid;
    std::uint32_t driver_version = 0;
    std::uint32_t runtime_schema = 0;
    std::uint64_t kernel_identity = 0;
    std::uint8_t workload_class = 0;
    std::uint8_t workload_bucket = 0;
    PerformanceTimingDomain timing_domain = PerformanceTimingDomain::DeviceExecution;
};

Result<std::string> performance_observation_key(const PerformanceObservationKey& key);

struct PerformanceStats {
    std::uint64_t count = 0;
    std::uint64_t rejected_outliers = 0;
    double ewma = 0;
    double ew_variance = 0;
    double minimum = 0;
    double maximum = 0;
    double median = 0;
    double p95 = 0;
    std::uint64_t last_observed_unix_ms = 0;
};

class PerformanceDatabase final {
public:
    struct Options {
        double ewma_alpha = 0.2;
        std::size_t sample_window = 256;
        std::uint64_t maximum_age_ms = 30ULL * 24ULL * 60ULL * 60ULL * 1000ULL;
        std::size_t maximum_entries = 256;
    };

    PerformanceDatabase();
    explicit PerformanceDatabase(Options options);
    Status observe(std::string key, double sample, std::uint64_t now_unix_ms);
    Result<PerformanceStats> query(const std::string& key) const;
    std::optional<PerformanceStats> query_confident(const std::string& key, std::uint64_t now_unix_ms,
                                                    std::uint64_t minimum_samples = 5) const;
    std::size_t size() const;
    Status save(const std::filesystem::path& path) const;
    /// A malformed, incompatible, or unreadable file returns an error; startup callers may ignore it.
    Result<std::size_t> load(const std::filesystem::path& path, std::uint64_t now_unix_ms);

private:
    struct Entry;
    Options options_;
    mutable std::shared_mutex entries_mutex_;
    std::unordered_map<std::string, std::shared_ptr<Entry>> entries_;
};

}  // namespace kora::detail
