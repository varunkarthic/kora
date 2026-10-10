#include "kora/log.hpp"

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace kora::log {
namespace {

constexpr int kCats = static_cast<int>(Category::kCount_);
std::array<std::atomic<int>, kCats>& levels() {
    static std::array<std::atomic<int>, kCats> l{};
    static const bool init = [] {
        Level initial = Level::Warn;
        if (const char* configured = std::getenv("KORA_LOG_LEVEL")) {
            Level parsed{};
            if (parse_level(configured, parsed)) initial = parsed;
        }
        for (auto& x : l) x.store(static_cast<int>(initial));
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

bool configure_from_environment() noexcept {
    const char* configured = std::getenv("KORA_LOG_LEVEL");
    Level parsed{};
    if (!configured || !parse_level(configured, parsed)) return false;
    set_level(parsed);
    return true;
}

void set_sink(Sink s) {
    std::lock_guard lk(sink_mutex());
    sink() = std::move(s);
}
void set_level(Level l) noexcept {
    for (auto& x : levels()) x.store(static_cast<int>(l));
}
void set_category_level(Category c, Level l) noexcept {
    const auto index = static_cast<int>(c);
    if (index >= 0 && index < kCats) levels()[index].store(static_cast<int>(l));
}
Level level(Category c) noexcept {
    const auto index = static_cast<int>(c);
    if (index < 0 || index >= kCats) return Level::Off;
    return static_cast<Level>(levels()[index].load(std::memory_order_relaxed));
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
