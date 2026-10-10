#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "kora/log.hpp"

using namespace kora::log;

namespace {
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

TEST(Log, ParseLevel) {
    Level l;
    EXPECT_TRUE(parse_level("WaRn", l));
    EXPECT_EQ(l, Level::Warn);
    EXPECT_FALSE(parse_level("nope", l));
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
