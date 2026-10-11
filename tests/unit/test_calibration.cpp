#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "core/calibration.hpp"

namespace kora::detail::calibration {
namespace {

Record sample() {
    Record record;
    record.device_uuid = "00112233445566778899aabbccddeeff";
    record.driver_version = 17;
    record.submit_overhead_ns = 1234;
    record.tiny_dispatch_ns = 4567;
    record.h2d_bytes_per_second = 1'000'000;
    record.d2h_bytes_per_second = 2'000'000;
    record.device_copy_bytes_per_second = 3'000'000;
    record.streaming_bytes_per_second = 4'000'000;
    record.alu_operations_per_second = 5'000'000;
    return record;
}

TEST(CalibrationRecord, RoundTripsAllMeasurements) {
    const auto record = sample();
    const auto bytes = encode(record);
    auto decoded = decode(bytes, record.device_uuid, record.driver_version);
    ASSERT_TRUE(decoded) << decoded.error().describe();
    EXPECT_EQ(decoded.value().device_uuid, record.device_uuid);
    EXPECT_EQ(decoded.value().driver_version, record.driver_version);
    EXPECT_EQ(decoded.value().submit_overhead_ns, record.submit_overhead_ns);
    EXPECT_EQ(decoded.value().tiny_dispatch_ns, record.tiny_dispatch_ns);
    EXPECT_EQ(decoded.value().h2d_bytes_per_second, record.h2d_bytes_per_second);
    EXPECT_EQ(decoded.value().d2h_bytes_per_second, record.d2h_bytes_per_second);
    EXPECT_EQ(decoded.value().device_copy_bytes_per_second, record.device_copy_bytes_per_second);
    EXPECT_EQ(decoded.value().streaming_bytes_per_second, record.streaming_bytes_per_second);
    EXPECT_EQ(decoded.value().alu_operations_per_second, record.alu_operations_per_second);
}

TEST(CalibrationRecord, RejectsCorruptionAndIdentityOrSchemaMismatch) {
    const auto record = sample();
    auto corrupt = std::vector<std::uint8_t>{'{', '!'};
    EXPECT_FALSE(decode(corrupt, record.device_uuid, record.driver_version));
    const auto bytes = encode(record);
    EXPECT_FALSE(decode(bytes, "ffeeddccbbaa99887766554433221100", record.driver_version));
    EXPECT_FALSE(decode(bytes, record.device_uuid, record.driver_version + 1));
    auto incompatible = bytes;
    const std::string newer_schema = "\"schema\":2";
    const auto old_schema = std::string("\"schema\":1");
    std::string text(incompatible.begin(), incompatible.end());
    text.replace(text.find(old_schema), old_schema.size(), newer_schema);
    incompatible.assign(text.begin(), text.end());
    EXPECT_FALSE(decode(incompatible, record.device_uuid, record.driver_version));
}

TEST(CalibrationRecord, AtomicFileRoundTripAndCorruptFileAreCacheMisses) {
    const auto directory = std::filesystem::temp_directory_path() / "kora-calibration-unit-test";
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    const auto path = directory / "device.json";
    const auto record = sample();
    ASSERT_TRUE(write_atomic(path, record));
    Record loaded;
    EXPECT_TRUE(read(path, record.device_uuid, record.driver_version, loaded));
    EXPECT_EQ(loaded.streaming_bytes_per_second, record.streaming_bytes_per_second);
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output << "not json";
    }
    EXPECT_FALSE(read(path, record.device_uuid, record.driver_version, loaded));
    std::filesystem::remove_all(directory, ignored);
}

}  // namespace
}  // namespace kora::detail::calibration
