#include "core/performance_db.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <locale>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string_view>
#include <type_traits>

#include "core/json.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace kora::detail {
namespace {

constexpr std::uint64_t kSchema = 1;
constexpr std::size_t kMaximumFileBytes = 4U * 1024U * 1024U;

std::string escape_json(std::string_view input) {
    std::string output;
    output.reserve(input.size() + 2);
    output.push_back('"');
    for (const unsigned char c : input) {
        switch (c) {
            case '"':
                output += "\\\"";
                break;
            case '\\':
                output += "\\\\";
                break;
            case '\b':
                output += "\\b";
                break;
            case '\f':
                output += "\\f";
                break;
            case '\n':
                output += "\\n";
                break;
            case '\r':
                output += "\\r";
                break;
            case '\t':
                output += "\\t";
                break;
            default:
                if (c < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    output += "\\u00";
                    output.push_back(hex[c >> 4U]);
                    output.push_back(hex[c & 0x0fU]);
                } else {
                    output.push_back(static_cast<char>(c));
                }
        }
    }
    output.push_back('"');
    return output;
}

std::string real(double value) {
    char buffer[64];
    const auto [end, error] =
        std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general,
                      std::numeric_limits<double>::max_digits10);
    if (error != std::errc{}) return "0";
    return {buffer, end};
}

template <class T>
bool number(const json::Value* value, T& result) {
    if (!value) return false;
    const auto* item = std::get_if<json::Value::Number>(&value->storage);
    if (!item) return false;
    if constexpr (std::is_floating_point_v<T>) {
        std::istringstream stream(item->text);
        stream.imbue(std::locale::classic());
        stream >> std::noskipws >> result;
        return !stream.fail() && stream.peek() == std::char_traits<char>::eof() && std::isfinite(result);
    } else {
        const auto [end, error] =
            std::from_chars(item->text.data(), item->text.data() + item->text.size(), result);
        return error == std::errc{} && end == item->text.data() + item->text.size();
    }
}

bool string_value(const json::Value* value, std::string& output) {
    if (!value) return false;
    const auto* string = std::get_if<std::string>(&value->storage);
    if (!string) return false;
    output = *string;
    return true;
}

double quantile(std::vector<double> values, double q) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(std::ceil(q * static_cast<double>(values.size()))) - 1;
    return values[std::min(index, values.size() - 1)];
}

std::string encode_stats(const std::unordered_map<std::string, PerformanceStats>& snapshot,
                         const std::unordered_map<std::string, std::vector<double>>& samples) {
    std::vector<std::string> keys;
    keys.reserve(snapshot.size());
    for (const auto& [key, stats] : snapshot) {
        (void)stats;
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());
    std::string output = "{\"schema\":1,\"entries\":[";
    bool first = true;
    for (const auto& key : keys) {
        const auto& stats = snapshot.at(key);
        if (!first) output.push_back(',');
        first = false;
        output += "{\"key\":" + escape_json(key) + ",\"count\":" + std::to_string(stats.count) +
                  ",\"rejected\":" + std::to_string(stats.rejected_outliers) +
                  ",\"ewma\":" + real(stats.ewma) + ",\"variance\":" + real(stats.ew_variance) +
                  ",\"min\":" + real(stats.minimum) + ",\"max\":" + real(stats.maximum) +
                  ",\"last\":" + std::to_string(stats.last_observed_unix_ms) + ",\"samples\":[";
        const auto found = samples.find(key);
        if (found != samples.end()) {
            for (std::size_t i = 0; i < found->second.size(); ++i) {
                if (i) output.push_back(',');
                output += real(found->second[i]);
            }
        }
        output += "]}";
    }
    output += "]}";
    return output;
}

Status write_atomic(const std::filesystem::path& path, std::string_view contents) {
    std::error_code error;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), error);
    if (error) return Error(ErrorCode::IoError, "cannot create performance database directory");
    static std::atomic<std::uint64_t> sequence{1};
    auto temporary = path;
#if defined(_WIN32)
    const auto process = static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    const auto process = static_cast<std::uint64_t>(::getpid());
#endif
    temporary += ".tmp." + std::to_string(process) + "." +
                 std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) return Error(ErrorCode::IoError, "cannot open temporary performance database");
        output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        output.flush();
        if (!output) {
            output.close();
            std::filesystem::remove(temporary, error);
            return Error(ErrorCode::IoError, "cannot write temporary performance database");
        }
    }
#if defined(_WIN32)
    if (MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ==
        0) {
        std::filesystem::remove(temporary, error);
        return Error(ErrorCode::IoError, "cannot atomically publish performance database");
    }
#else
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return Error(ErrorCode::IoError, "cannot atomically publish performance database");
    }
    const auto parent = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    const int directory = ::open(parent.c_str(), O_RDONLY);
    if (directory >= 0) {
        (void)::fsync(directory);
        (void)::close(directory);
    }
#endif
    return {};
}

}  // namespace

struct PerformanceDatabase::Entry {
    mutable std::mutex mutex;
    PerformanceStats stats;
    std::vector<double> samples;
};

PerformanceDatabase::PerformanceDatabase()
    : PerformanceDatabase(Options{0.2, 256, 30ULL * 24 * 60 * 60 * 1000}) {}

PerformanceDatabase::PerformanceDatabase(Options options) : options_(options) {
    if (!(options_.ewma_alpha > 0 && options_.ewma_alpha <= 1) || options_.sample_window == 0)
        options_ = Options{0.2, 256, 30ULL * 24 * 60 * 60 * 1000};
}

Status PerformanceDatabase::observe(std::string key, double sample, std::uint64_t now_unix_ms) {
    if (key.empty() || !std::isfinite(sample) || sample < 0)
        return Error(ErrorCode::InvalidArgument,
                     "performance observation needs a key and finite nonnegative sample");
    std::shared_ptr<Entry> entry;
    {
        std::shared_lock lock(entries_mutex_);
        const auto found = entries_.find(key);
        if (found != entries_.end()) entry = found->second;
    }
    if (!entry) {
        std::unique_lock lock(entries_mutex_);
        auto [found, inserted] = entries_.try_emplace(std::move(key), std::make_shared<Entry>());
        (void)inserted;
        entry = found->second;
    }
    std::lock_guard lock(entry->mutex);
    if (entry->stats.count != 0 && now_unix_ms > entry->stats.last_observed_unix_ms &&
        now_unix_ms - entry->stats.last_observed_unix_ms > options_.maximum_age_ms) {
        entry->stats = {};
        entry->samples.clear();
    }
    if (entry->samples.size() >= 5) {
        const auto center = quantile(entry->samples, 0.5);
        std::vector<double> deviations;
        deviations.reserve(entry->samples.size());
        for (const auto value : entry->samples) deviations.push_back(std::abs(value - center));
        const auto mad = quantile(std::move(deviations), 0.5);
        const auto scale = std::max(1.4826 * mad, std::max(1.0, std::abs(center) * 0.01));
        if (std::abs(sample - center) > 6.0 * scale) {
            ++entry->stats.rejected_outliers;
            return {};
        }
    }
    auto& stats = entry->stats;
    if (stats.count == 0) {
        stats.ewma = sample;
        stats.minimum = sample;
        stats.maximum = sample;
    } else {
        const auto delta = sample - stats.ewma;
        stats.ewma += options_.ewma_alpha * delta;
        stats.ew_variance =
            (1 - options_.ewma_alpha) * (stats.ew_variance + options_.ewma_alpha * delta * delta);
        stats.minimum = std::min(stats.minimum, sample);
        stats.maximum = std::max(stats.maximum, sample);
    }
    ++stats.count;
    stats.last_observed_unix_ms = now_unix_ms;
    if (entry->samples.size() == options_.sample_window) entry->samples.erase(entry->samples.begin());
    entry->samples.push_back(sample);
    stats.median = quantile(entry->samples, 0.5);
    stats.p95 = quantile(entry->samples, 0.95);
    return {};
}

Result<PerformanceStats> PerformanceDatabase::query(const std::string& key) const {
    std::shared_ptr<Entry> entry;
    {
        std::shared_lock lock(entries_mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end()) return Error(ErrorCode::InvalidArgument, "unknown performance key");
        entry = found->second;
    }
    std::lock_guard lock(entry->mutex);
    return entry->stats;
}

std::size_t PerformanceDatabase::size() const {
    std::shared_lock lock(entries_mutex_);
    return entries_.size();
}

Status PerformanceDatabase::save(const std::filesystem::path& path) const {
    if (path.empty()) return Error(ErrorCode::InvalidArgument, "performance database path is empty");
    std::unordered_map<std::string, PerformanceStats> snapshot;
    std::unordered_map<std::string, std::vector<double>> sample_snapshot;
    {
        std::shared_lock lock(entries_mutex_);
        for (const auto& [key, entry] : entries_) {
            std::lock_guard entry_lock(entry->mutex);
            snapshot.emplace(key, entry->stats);
            sample_snapshot.emplace(key, entry->samples);
        }
    }
    const auto encoded = encode_stats(snapshot, sample_snapshot);
    if (encoded.size() > kMaximumFileBytes)
        return Error(ErrorCode::IoError, "performance database exceeds size limit");
    return write_atomic(path, encoded);
}

Result<std::size_t> PerformanceDatabase::load(const std::filesystem::path& path, std::uint64_t now_unix_ms) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size == 0 || size > kMaximumFileBytes)
        return Error(ErrorCode::IoError, "performance database is missing, empty, or oversized");
    std::ifstream input(path, std::ios::binary);
    if (!input) return Error(ErrorCode::IoError, "cannot open performance database");
    const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    auto parsed = json::parse(contents, {.maximum_bytes = kMaximumFileBytes});
    if (!parsed) return Error(ErrorCode::IoError, "performance database JSON is corrupt");
    std::uint64_t schema = 0;
    if (!number(parsed.value().find("schema"), schema) || schema != kSchema)
        return Error(ErrorCode::IoError, "performance database schema is incompatible");
    const auto* entries_value = parsed.value().find("entries");
    if (!entries_value) return Error(ErrorCode::IoError, "performance database entries are missing");
    const auto* entries = std::get_if<json::Value::Array>(&entries_value->storage);
    if (!entries) return Error(ErrorCode::IoError, "performance database entries are invalid");
    std::unordered_map<std::string, std::shared_ptr<Entry>> loaded;
    for (const auto& value : *entries) {
        std::string key;
        std::uint64_t count = 0, rejected = 0, last = 0;
        double ewma = 0, variance = 0, minimum = 0, maximum = 0;
        if (!string_value(value.find("key"), key) || key.empty() || !number(value.find("count"), count) ||
            !number(value.find("rejected"), rejected) || !number(value.find("last"), last) ||
            !number(value.find("ewma"), ewma) || !number(value.find("variance"), variance) ||
            !number(value.find("min"), minimum) || !number(value.find("max"), maximum))
            return Error(ErrorCode::IoError, "performance database entry is malformed");
        if (count == 0 || ewma < 0 || variance < 0 || minimum < 0 || maximum < minimum)
            return Error(ErrorCode::IoError, "performance database statistics are out of range");
        if (now_unix_ms > last && now_unix_ms - last > options_.maximum_age_ms) continue;
        const auto* sample_value = value.find("samples");
        const auto* samples =
            sample_value ? std::get_if<json::Value::Array>(&sample_value->storage) : nullptr;
        if (!samples || samples->size() > options_.sample_window)
            return Error(ErrorCode::IoError, "performance database sample window is invalid");
        if (samples->size() > count)
            return Error(ErrorCode::IoError, "performance database has more samples than observations");
        auto entry = std::make_shared<Entry>();
        entry->stats = {count, rejected, ewma, variance, minimum, maximum, 0, 0, last};
        for (const auto& sample : *samples) {
            double parsed_sample = 0;
            if (!number(&sample, parsed_sample) || parsed_sample < 0)
                return Error(ErrorCode::IoError, "performance database contains invalid sample");
            entry->samples.push_back(parsed_sample);
        }
        entry->stats.median = quantile(entry->samples, 0.5);
        entry->stats.p95 = quantile(entry->samples, 0.95);
        if (!loaded.emplace(std::move(key), std::move(entry)).second)
            return Error(ErrorCode::IoError, "performance database contains duplicate keys");
    }
    const auto loaded_count = loaded.size();
    {
        std::unique_lock lock(entries_mutex_);
        entries_ = std::move(loaded);
    }
    return loaded_count;
}

}  // namespace kora::detail
