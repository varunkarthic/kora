#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "kora/log.hpp"

using namespace kora::log;

namespace {
void set_log_environment(const char* value) {
#if defined(_WIN32)
    _putenv_s("KORA_LOG_LEVEL", value);
#else
    setenv("KORA_LOG_LEVEL", value, 1);
#endif
}

struct Capture {
    std::vector<std::string> lines;
    Capture() {
        set_level(Level::Info);
        set_sink([this](Level, Category c, std::string_view m) {
            lines.push_back(std::string(to_string(c)) + ":" + std::string(m));
        });
    }
    ~Capture() {
        set_sink(nullptr);
        set_level(Level::Warn);
    }
};
}  // namespace

TEST(Log, EnvironmentConfiguresThresholds) {
    set_log_environment("DeBuG");
    EXPECT_TRUE(configure_from_environment());
    EXPECT_EQ(level(Category::Runtime), Level::Debug);
    set_log_environment("");
    EXPECT_FALSE(configure_from_environment());
    set_level(Level::Warn);
}

TEST(Log, LevelFiltering) {
    Capture cap;
    KORA_LOG(Runtime, Debug, "hidden");
    KORA_LOG(Runtime, Info, "shown {}", 1);
    ASSERT_EQ(cap.lines.size(), 1u);
    EXPECT_EQ(cap.lines[0], "RUNTIME:shown 1");
}

TEST(Log, DisabledDoesNotEvaluateArgs) {
    Capture cap;
    int evaluated = 0;
    auto f = [&] {
        ++evaluated;
        return 1;
    };
    KORA_LOG(Memory, Trace, "x {}", f());
    EXPECT_EQ(evaluated, 0);
    KORA_LOG(Memory, Error, "x {}", f());
    EXPECT_EQ(evaluated, 1);
}

TEST(Log, CategoryLevel) {
    Capture cap;
    set_category_level(Category::Sync, Level::Trace);
    KORA_LOG(Sync, Trace, "t");
    KORA_LOG(Queue, Trace, "q");
    ASSERT_EQ(cap.lines.size(), 1u);
    EXPECT_EQ(cap.lines[0], "SYNC:t");
}

TEST(Log, OffDisablesEverySeverity) {
    Capture cap;
    set_level(Level::Off);
    KORA_LOG(Runtime, Error, "hidden");
    KORA_LOG(Runtime, Off, "also hidden");
    EXPECT_TRUE(cap.lines.empty());
}

TEST(Log, FallbackFormatting) {
    EXPECT_EQ(detail::fallback_format("escaped {{}}: {} {}", 42, "ready"), "escaped {}: 42 ready");
}

TEST(Log, ParseLevel) {
    Level l;
    EXPECT_TRUE(parse_level("WaRn", l));
    EXPECT_EQ(l, Level::Warn);
    EXPECT_FALSE(parse_level("nope", l));
}

TEST(Log, InvalidCategoryIsSafelyDisabled) {
    const auto invalid = static_cast<Category>(999);
    set_category_level(invalid, Level::Trace);
    EXPECT_EQ(level(invalid), Level::Off);
    EXPECT_FALSE(enabled(invalid, Level::Error));
}

TEST(Log, ConcurrentWritesAreSerialized) {
    Capture cap;
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t)
        ts.emplace_back([] {
            for (int i = 0; i < 200; ++i) KORA_LOG(Runtime, Info, "m");
        });
    for (auto& t : ts) t.join();
    EXPECT_EQ(cap.lines.size(), 1600u);
}
