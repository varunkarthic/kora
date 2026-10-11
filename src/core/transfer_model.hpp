#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "kora/errors.hpp"

namespace kora::detail {

inline constexpr std::size_t kTransferSizeBucketCount = 64;

struct TransferSizeBucket {
    std::uint8_t index = 0;
    std::uint64_t lower_bytes = 0;
    std::uint64_t upper_bytes = 1;
};

/// Logarithmic power-of-two bucket using floor(log2(bytes)); zero and one share bucket zero.
TransferSizeBucket transfer_size_bucket(std::uint64_t bytes) noexcept;

enum class TransferPath : std::uint8_t {
    HostToDevice,
    DeviceToHost,
    DeviceCopy,
    CrossDeviceViaHost,
    UmaDirect
};

struct TransferBucketStats {
    std::uint64_t observations = 0;
    double ewma_bytes = 0;
    double ewma_duration_ns = 0;
};

struct TransferCostModel {
    std::uint64_t observations = 0;
    double latency_ns = 0;
    double nanoseconds_per_byte = 0;
};

/// Direction-specific latency + bandwidth estimates. One entry is maintained per logarithmic size
/// bucket; fitting across bucket means produces one monotone linear cost curve per transfer path.
class TransferModelSet final {
public:
    struct Options {
        double ewma_alpha = 0.2;
        double uma_direct_penalty = 1.0;
    };

    TransferModelSet();
    explicit TransferModelSet(Options options);

    Status observe(TransferPath path, std::uint64_t bytes, double duration_ns);
    Result<double> estimate_ns(TransferPath path, std::uint64_t bytes) const;
    Result<TransferCostModel> model(TransferPath path) const;
    Result<TransferBucketStats> bucket_stats(TransferPath path, std::uint8_t bucket) const;
    Status set_uma_direct_penalty(double penalty);
    double uma_direct_penalty() const;

private:
    struct PathState {
        std::array<TransferBucketStats, kTransferSizeBucketCount> buckets{};
        TransferCostModel model;
        bool initialized = false;
    };
    static constexpr std::size_t kPathCount = 5;
    static std::size_t path_index(TransferPath path) noexcept;
    void refit(PathState& state);

    Options options_;
    mutable std::mutex mutex_;
    std::array<PathState, kPathCount> paths_{};
};

}  // namespace kora::detail
