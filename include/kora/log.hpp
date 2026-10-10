#pragma once
/// @file log.hpp
/// Structured logging with categories and levels. Disabled statements never evaluate their arguments.
#include <atomic>
#include <format>
#include <functional>
#include <string>
#include <string_view>

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

using Sink = std::function<void(Level, Category, std::string_view msg)>;

/// Replace the sink (default writes "[CAT] msg" to stderr). Pass nullptr to restore the default.
void set_sink(Sink s);
void set_level(Level l) noexcept;  ///< Global default for all categories.
void set_category_level(Category c, Level l) noexcept;
Level level(Category c) noexcept;
inline bool enabled(Category c, Level l) noexcept {
    return l >= level(c);
}
void write(Level l, Category c, std::string_view msg);

}  // namespace kora::log

#define KORA_LOG(cat, lvl, ...)                                                        \
    do {                                                                               \
        if (::kora::log::enabled(::kora::log::Category::cat, ::kora::log::Level::lvl)) \
            ::kora::log::write(::kora::log::Level::lvl, ::kora::log::Category::cat,    \
                               std::format(__VA_ARGS__));                              \
    } while (0)
