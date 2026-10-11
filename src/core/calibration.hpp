#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "kora/errors.hpp"

namespace kora::detail::calibration {

inline constexpr std::uint32_t kSchemaVersion = 1;

struct Record {
    std::uint32_t schema = kSchemaVersion;
    std::string device_uuid;
    std::uint32_t driver_version = 0;
    std::uint32_t kora_schema = 1;
    std::uint64_t submit_overhead_ns = 0;
    std::uint64_t tiny_dispatch_ns = 0;
    std::uint64_t h2d_bytes_per_second = 0;
    std::uint64_t d2h_bytes_per_second = 0;
    std::uint64_t device_copy_bytes_per_second = 0;
    std::uint64_t streaming_bytes_per_second = 0;
    std::uint64_t alu_operations_per_second = 0;
};

std::vector<std::uint8_t> encode(const Record& record);
Result<Record> decode(const std::vector<std::uint8_t>& bytes, const std::string& expected_uuid,
                      std::uint32_t expected_driver);
bool read(const std::filesystem::path& path, const std::string& expected_uuid,
          std::uint32_t expected_driver, Record& record) noexcept;
Status write_atomic(const std::filesystem::path& path, const Record& record);

}  // namespace kora::detail::calibration
