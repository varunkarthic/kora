#include "core/calibration.hpp"

#include <atomic>
#include <charconv>
#include <fstream>
#include <iterator>
#include <system_error>
#include <string_view>

#include "core/json.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace kora::detail::calibration {
namespace {

constexpr std::size_t kMaximumRecordBytes = 64U * 1024U;

Error invalid(std::string message) {
    return Error(ErrorCode::InvalidArgument, std::move(message));
}

std::string number(std::uint64_t value) { return std::to_string(value); }

bool get_u64(const json::Value* value, std::uint64_t& output) {
    if (!value) return false;
    const auto* item = std::get_if<json::Value::Number>(&value->storage);
    if (!item) return false;
    const auto [end, error] = std::from_chars(item->text.data(), item->text.data() + item->text.size(), output);
    return error == std::errc{} && end == item->text.data() + item->text.size();
}

bool get_string(const json::Value* value, std::string& output) {
    if (!value) return false;
    const auto* item = std::get_if<std::string>(&value->storage);
    if (!item) return false;
    output = *item;
    return true;
}

bool flush_file(const std::filesystem::path& path) {
#if defined(_WIN32)
    const auto handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) return false;
    const bool result = FlushFileBuffers(handle) != 0;
    CloseHandle(handle);
    return result;
#else
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) return false;
    const bool result = ::fsync(descriptor) == 0;
    (void)::close(descriptor);
    return result;
#endif
}

}  // namespace

std::vector<std::uint8_t> encode(const Record& record) {
    const std::string text =
        "{\"schema\":" + number(record.schema) + ",\"device_uuid\":\"" + record.device_uuid +
        "\",\"driver_version\":" + number(record.driver_version) + ",\"kora_schema\":" +
        number(record.kora_schema) + ",\"submit_overhead_ns\":" + number(record.submit_overhead_ns) +
        ",\"tiny_dispatch_ns\":" + number(record.tiny_dispatch_ns) + ",\"h2d_bytes_per_second\":" +
        number(record.h2d_bytes_per_second) + ",\"d2h_bytes_per_second\":" +
        number(record.d2h_bytes_per_second) + ",\"device_copy_bytes_per_second\":" +
        number(record.device_copy_bytes_per_second) + ",\"streaming_bytes_per_second\":" +
        number(record.streaming_bytes_per_second) + ",\"alu_operations_per_second\":" +
        number(record.alu_operations_per_second) + "}";
    return {text.begin(), text.end()};
}

Result<Record> decode(const std::vector<std::uint8_t>& bytes, const std::string& expected_uuid,
                      std::uint32_t expected_driver) {
    if (bytes.size() > kMaximumRecordBytes) return invalid("calibration record exceeds size limit");
    const std::string_view text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    auto parsed = json::parse(text, {.maximum_bytes = kMaximumRecordBytes});
    if (!parsed) return invalid("calibration record is malformed");
    Record record;
    std::uint64_t schema = 0, driver = 0, kora_schema = 0;
    if (!get_u64(parsed.value().find("schema"), schema) ||
        !get_string(parsed.value().find("device_uuid"), record.device_uuid) ||
        !get_u64(parsed.value().find("driver_version"), driver) ||
        !get_u64(parsed.value().find("kora_schema"), kora_schema) ||
        schema != kSchemaVersion || driver > UINT32_MAX || kora_schema != 1 ||
        record.device_uuid != expected_uuid || driver != expected_driver)
        return invalid("calibration record identity or schema mismatch");
    record.schema = static_cast<std::uint32_t>(schema);
    record.driver_version = static_cast<std::uint32_t>(driver);
    record.kora_schema = static_cast<std::uint32_t>(kora_schema);
    if (!get_u64(parsed.value().find("submit_overhead_ns"), record.submit_overhead_ns) ||
        !get_u64(parsed.value().find("tiny_dispatch_ns"), record.tiny_dispatch_ns) ||
        !get_u64(parsed.value().find("h2d_bytes_per_second"), record.h2d_bytes_per_second) ||
        !get_u64(parsed.value().find("d2h_bytes_per_second"), record.d2h_bytes_per_second) ||
        !get_u64(parsed.value().find("device_copy_bytes_per_second"), record.device_copy_bytes_per_second) ||
        !get_u64(parsed.value().find("streaming_bytes_per_second"), record.streaming_bytes_per_second) ||
        !get_u64(parsed.value().find("alu_operations_per_second"), record.alu_operations_per_second))
        return invalid("calibration record is missing a measurement");
    return record;
}

bool read(const std::filesystem::path& path, const std::string& expected_uuid,
          std::uint32_t expected_driver, Record& record) noexcept {
    try {
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error || size == 0 || size > kMaximumRecordBytes) return false;
        std::ifstream input(path, std::ios::binary);
        if (!input) return false;
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!input) return false;
        auto decoded = decode(bytes, expected_uuid, expected_driver);
        if (!decoded) return false;
        record = std::move(decoded).value();
        return true;
    } catch (...) {
        return false;
    }
}

Status write_atomic(const std::filesystem::path& path, const Record& record) {
    if (path.empty()) return {};
    std::error_code error;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) return Error(ErrorCode::IoError, "cannot create calibration directory: " + error.message());
    }
    static std::atomic<std::uint64_t> sequence{1};
    auto temporary = path;
    temporary += ".tmp." + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
    const auto bytes = encode(record);
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) return Error(ErrorCode::IoError, "cannot open temporary calibration record");
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        output.flush();
        if (!output) {
            output.close();
            std::filesystem::remove(temporary, error);
            return Error(ErrorCode::IoError, "cannot write temporary calibration record");
        }
    }
    if (!flush_file(temporary)) {
        std::filesystem::remove(temporary, error);
        return Error(ErrorCode::IoError, "cannot flush temporary calibration record");
    }
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::error_code remove_error;
        std::filesystem::remove(temporary, remove_error);
        return Error(ErrorCode::IoError, "cannot publish calibration record: " + error.message());
    }
#if !defined(_WIN32)
    const auto parent = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    const int directory_descriptor = ::open(parent.c_str(), O_RDONLY);
    if (directory_descriptor < 0) return Error(ErrorCode::IoError, "cannot open calibration directory");
    const bool directory_flushed = ::fsync(directory_descriptor) == 0;
    (void)::close(directory_descriptor);
    if (!directory_flushed) return Error(ErrorCode::IoError, "cannot flush calibration directory");
#endif
    return {};
}

}  // namespace kora::detail::calibration
