#include "kora/log.hpp"

#include <array>
#include <cctype>
#include <cstdio>
#include <mutex>

namespace kora::log {
namespace {

constexpr int kCats = static_cast<int>(Category::kCount_);
std::array<std::atomic<int>, kCats>& levels() {
    static std::array<std::atomic<int>, kCats> l{};
    static const bool init = [] {
        for (auto& x : l) x.store(static_cast<int>(Level::Warn));
        return true;
    }();
    (void)init;
    return l;
}
std::mutex& sink_mutex() {
    static std::mutex m;
    return m;
}
Sink& sink() {
    static Sink s;
    return s;
}

}  // namespace

const char* to_string(Level l) noexcept {
    switch (l) {
        case Level::Trace:
            return "TRACE";
        case Level::Debug:
            return "DEBUG";
        case Level::Info:
            return "INFO";
        case Level::Warn:
            return "WARN";
        case Level::Error:
            return "ERROR";
        case Level::Off:
            return "OFF";
    }
    return "?";
}

const char* to_string(Category c) noexcept {
    static constexpr const char* n[] = {"RUNTIME", "DEVICE", "MEM",  "SCHED",    "QUEUE",
                                        "SYNC",    "KERNEL", "PROF", "RECOVERY", "PYTHON"};
    int i = static_cast<int>(c);
    return (i >= 0 && i < kCats) ? n[i] : "?";
}

bool parse_level(std::string_view s, Level& out) noexcept {
    std::string t;
    for (char c : s) t += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    static constexpr std::pair<const char*, Level> m[] = {{"trace", Level::Trace}, {"debug", Level::Debug},
                                                          {"info", Level::Info},   {"warn", Level::Warn},
                                                          {"error", Level::Error}, {"off", Level::Off}};
    for (auto& [k, v] : m)
        if (t == k) {
            out = v;
            return true;
        }
    return false;
}

void set_sink(Sink s) {
    std::lock_guard lk(sink_mutex());
    sink() = std::move(s);
}
void set_level(Level l) noexcept {
    for (auto& x : levels()) x.store(static_cast<int>(l));
}
void set_category_level(Category c, Level l) noexcept {
    levels()[static_cast<int>(c)].store(static_cast<int>(l));
}
Level level(Category c) noexcept {
    return static_cast<Level>(levels()[static_cast<int>(c)].load(std::memory_order_relaxed));
}

void write(Level l, Category c, std::string_view msg) {
    std::lock_guard lk(sink_mutex());
    if (sink()) {
        sink()(l, c, msg);
        return;
    }
    std::fprintf(stderr, "[%s] %.*s\n", to_string(c), static_cast<int>(msg.size()), msg.data());
}

}  // namespace kora::log
