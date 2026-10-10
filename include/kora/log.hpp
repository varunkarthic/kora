#pragma once
/// @file log.hpp
/// Structured logging with categories and levels. Disabled statements never evaluate their arguments.
#include <array>
#include <atomic>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#if __has_include(<format>)
#include <format>
#endif

namespace kora::log {

enum class Level : int { Trace = 0, Debug, Info, Warn, Error, Off };
enum class Category : int {
    Runtime = 0,
    Device,
    Memory,
    Scheduler,
    Queue,
    Sync,
    Kernel,
    Profiling,
    Recovery,
    Python,
    kCount_
};

const char* to_string(Level) noexcept;
const char* to_string(Category) noexcept;
/// Parses "trace|debug|info|warn|error|off" (case-insensitive). Returns false if unrecognized.
bool parse_level(std::string_view s, Level& out) noexcept;
/// Re-read KORA_LOG_LEVEL and apply it to every category. Returns false if absent or invalid.
bool configure_from_environment() noexcept;

using Sink = std::function<void(Level, Category, std::string_view msg)>;

/// Replace the sink (default writes "[CAT] msg" to stderr). Pass nullptr to restore the default.
void set_sink(Sink s);
void set_level(Level l) noexcept;  ///< Global default for all categories.
void set_category_level(Category c, Level l) noexcept;
Level level(Category c) noexcept;
inline bool enabled(Category c, Level l) noexcept {
    const auto threshold = level(c);
    return l != Level::Off && threshold != Level::Off && l >= threshold;
}
void write(Level l, Category c, std::string_view msg);

namespace detail {

template <class T>
std::string format_value(T&& value) {
    std::ostringstream out;
    out << std::forward<T>(value);
    return out.str();
}

/// Minimal fallback for toolchains without C++20 <format>: supports {}, {{, and }}.
template <class... Args>
std::string fallback_format(std::string_view pattern, Args&&... args) {
    std::array<std::string, sizeof...(Args)> values{format_value(std::forward<Args>(args))...};
    std::string result;
    result.reserve(pattern.size());
    std::size_t argument = 0;
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        if (pattern[i] == '{' && i + 1 < pattern.size()) {
            if (pattern[i + 1] == '{') {
                result.push_back('{');
                ++i;
            } else if (pattern[i + 1] == '}') {
                if (argument < values.size()) result += values[argument++];
                ++i;
            } else {
                result.push_back(pattern[i]);
            }
        } else if (pattern[i] == '}' && i + 1 < pattern.size() && pattern[i + 1] == '}') {
            result.push_back('}');
            ++i;
        } else {
            result.push_back(pattern[i]);
        }
    }
    return result;
}

#if defined(__cpp_lib_format) && __cpp_lib_format >= 201907L
template <class... Args>
std::string format(std::format_string<Args...> pattern, Args&&... args) {
    return std::format(pattern, std::forward<Args>(args)...);
}
#else
template <class... Args>
std::string format(std::string_view pattern, Args&&... args) {
    return fallback_format(pattern, std::forward<Args>(args)...);
}
#endif

}  // namespace detail

}  // namespace kora::log

#define KORA_LOG(cat, lvl, ...)                                                        \
    do {                                                                               \
        if (::kora::log::enabled(::kora::log::Category::cat, ::kora::log::Level::lvl)) \
            ::kora::log::write(::kora::log::Level::lvl, ::kora::log::Category::cat,    \
                               ::kora::log::detail::format(__VA_ARGS__));              \
    } while (0)
