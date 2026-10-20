// kora-top: live terminal view of a kora::Cluster. Attaches to the status stream an application
// opens with ClusterOptions::status_listen (or KORA_STATUS_LISTEN), or runs a self-contained demo.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN  // Keeps out rpcndr.h, which defines `small` as a macro.
#endif
#include <io.h>
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include "core/json.hpp"
#include "demo_saxpy_spv.hpp"
#include "kora/cluster.hpp"
#include "kora/worker.hpp"
#include "remote/socket.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using kora::detail::json::Value;
namespace remote = kora::detail::remote;

constexpr const char* kDefaultEndpoint = "127.0.0.1:7979";

struct Args {
    std::string endpoint = kDefaultEndpoint;
    bool demo = false;
    int frames = 0;  // 0 = run until interrupted.
    bool color = true;
};

// --- JSON access --------------------------------------------------------------------------------

std::uint64_t number(const Value& value, const char* key) {
    const auto* found = value.find(key);
    const auto* item = found ? std::get_if<Value::Number>(&found->storage) : nullptr;
    return item ? std::strtoull(item->text.c_str(), nullptr, 10) : 0;
}

std::string text(const Value& value, const char* key) {
    const auto* found = value.find(key);
    const auto* item = found ? std::get_if<std::string>(&found->storage) : nullptr;
    return item ? *item : std::string();
}

bool flag(const Value& value, const char* key) {
    const auto* found = value.find(key);
    const auto* item = found ? std::get_if<bool>(&found->storage) : nullptr;
    return item && *item;
}

const Value::Array& items(const Value& value, const char* key) {
    static const Value::Array empty;
    const auto* found = value.find(key);
    const auto* item = found ? std::get_if<Value::Array>(&found->storage) : nullptr;
    return item ? *item : empty;
}

// --- Formatting ---------------------------------------------------------------------------------

std::string bytes_text(std::uint64_t bytes) {
    static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), unit == 0 ? "%.0f %s" : "%.1f %s", value, units[unit]);
    return buffer;
}

std::string grouped(std::uint64_t value) {
    auto digits = std::to_string(value);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3)
        digits.insert(static_cast<std::size_t>(i), ",");
    return digits;
}

std::string fit(std::string value, std::size_t width) {
    if (value.size() > width) value = value.substr(0, width - 1) + "~";
    value.resize(width, ' ');
    return value;
}

int terminal_width() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info))
        return info.srWindow.Right - info.srWindow.Left + 1;
#else
    winsize size{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 0) return size.ws_col;
#endif
    if (const char* columns = std::getenv("COLUMNS"); columns && std::atoi(columns) > 0)
        return std::atoi(columns);
    return 100;
}

bool stdout_is_terminal() {
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(STDOUT_FILENO) != 0;
#endif
}

struct Style {
    bool on = true;
    std::string operator()(const char* code, const std::string& value) const {
        return on ? std::string("\x1b[") + code + "m" + value + "\x1b[0m" : value;
    }
};

std::string bar(double fraction, int width, const Style& style, const char* color) {
    fraction = std::clamp(fraction, 0.0, 1.0);
    const int filled = static_cast<int>(fraction * width + 0.5);
    std::string full, empty;
    for (int i = 0; i < width; ++i)
        (i < filled ? full : empty) += i < filled ? "\xE2\x96\x88" : "\xE2\x96\x91";
    if (!style.on) return full + empty;
    return style(color, full) + style("2", empty);
}

// --- Rendering ----------------------------------------------------------------------------------

// Remembers recent work counts per target so the view can show where work is going right now.
class Activity {
public:
    // Returns each target's share of the work completed during roughly the last second.
    std::map<std::string, double> update(std::uint64_t now_ms,
                                         const std::map<std::string, std::uint64_t>& totals) {
        history_.push_back({now_ms, totals});
        while (history_.size() > 2 && now_ms - history_[1].first >= 1'000) history_.pop_front();
        const auto& [then_ms, before] = history_.front();
        std::map<std::string, double> share;
        double sum = 0;
        for (const auto& [key, total] : totals) {
            const auto found = before.find(key);
            const auto previous = found == before.end() ? 0 : found->second;
            const auto delta = static_cast<double>(total >= previous ? total - previous : total);
            share[key] = delta;
            sum += delta;
        }
        for (auto& [key, value] : share) value = sum > 0 ? value / sum : 0;
        return share;
    }

private:
    std::deque<std::pair<std::uint64_t, std::map<std::string, std::uint64_t>>> history_;
};

std::string render(const Value& snapshot, Activity& activity, const std::string& endpoint, const Style& style,
                   const std::string& footer) {
    const int width = std::max(terminal_width(), 72);
    const auto stats_it = snapshot.find("stats");
    const Value stats = stats_it ? *stats_it : Value{};
    std::map<std::string, std::uint64_t> totals;
    for (const auto& device : items(snapshot, "devices"))
        totals["local:" + std::to_string(number(device, "index"))] =
            number(device, "dispatches") + number(device, "slices");
    for (const auto& worker : items(snapshot, "workers"))
        totals["worker:" + text(worker, "endpoint")] = number(worker, "dispatches");
    const auto share = activity.update(number(snapshot, "uptime_ms"), totals);

    std::string out;
    const auto line = [&](const std::string& value) { out += value + (style.on ? "\x1b[K\n" : "\n"); };
    char uptime[32];
    std::snprintf(uptime, sizeof(uptime), "%.1f s",
                  static_cast<double>(number(snapshot, "uptime_ms")) / 1000.0);
    line(style("1", "kora-top") + "  " + endpoint + "  up " + uptime + "  policy " +
         text(snapshot, "policy"));
    line("dispatches: local " + grouped(number(stats, "local_dispatches")) + ", remote " +
         grouped(number(stats, "remote_dispatches")) + ", split " +
         grouped(number(stats, "split_dispatches")) + " (" + grouped(number(stats, "slices")) +
         " slices)   failovers " + grouped(number(stats, "failovers")) + "   device rescans " +
         grouped(number(stats, "device_rescans")) + "   recoveries " +
         grouped(number(stats, "local_recoveries")));
    line("");

    const std::size_t name_width = static_cast<std::size_t>(std::clamp(width - 68, 24, 44));
    const int bar_width = 16;
    line(style("1;36", fit("  TARGET", 12 + name_width) + fit("WORK NOW", bar_width + 7) + fit("TOTAL", 10) +
                           fit("RUN", 5) + "MEMORY"));
    for (const auto& device : items(snapshot, "devices")) {
        const auto key = "local:" + std::to_string(number(device, "index"));
        const auto fraction = share.count(key) ? share.at(key) : 0.0;
        char percent[8];
        std::snprintf(percent, sizeof(percent), "%4.0f%%", fraction * 100);
        const auto allocated = number(device, "allocated_bytes");
        const auto budget = number(device, "budget_bytes");
        std::string memory = bytes_text(allocated);
        if (budget > 0)
            memory = bar(static_cast<double>(allocated) / static_cast<double>(budget), 8, style, "35") + " " +
                     memory + " of " + bytes_text(budget);
        line(style("32", "* ") + fit("local " + std::to_string(number(device, "index")), 10) +
             fit(text(device, "name"), name_width - 1) + " " + bar(fraction, bar_width, style, "32") + " " +
             percent + "  " + fit(grouped(number(device, "dispatches") + number(device, "slices")), 10) +
             fit(std::to_string(number(device, "inflight")), 5) + memory);
    }
    for (const auto& worker : items(snapshot, "workers")) {
        const auto key = "worker:" + text(worker, "endpoint");
        const bool connected = flag(worker, "connected");
        const auto fraction = share.count(key) ? share.at(key) : 0.0;
        char percent[8];
        std::snprintf(percent, sizeof(percent), "%4.0f%%", fraction * 100);
        const auto label = text(worker, "endpoint");
        const auto device = connected ? text(worker, "device") : std::string("unplugged");
        const auto status = connected ? "cached " + bytes_text(number(worker, "cached_bytes"))
                                      : std::to_string(number(worker, "connects")) + " connects, " +
                                            std::to_string(number(worker, "failovers")) + " failovers";
        line((connected ? style("33", "* ") : style("2", "o ")) + fit("worker", 10) +
             fit(label + "  " + device, name_width - 1) + " " + bar(fraction, bar_width, style, "33") + " " +
             percent + "  " + fit(grouped(number(worker, "dispatches")), 10) +
             fit(std::to_string(number(worker, "inflight")), 5) + status);
    }
    if (items(snapshot, "workers").empty()) line(style("2", "  no remote workers configured"));
    line("");
    line(style("1;36", "  EVENTS"));
    const auto& events = items(snapshot, "events");
    const std::size_t shown = std::min<std::size_t>(events.size(), 10);
    for (std::size_t i = events.size() - shown; i < events.size(); ++i) {
        char at[24];
        std::snprintf(at, sizeof(at), "%8.1fs  ", static_cast<double>(number(events[i], "at_ms")) / 1000.0);
        auto message = text(events[i], "message");
        if (const auto repeats = number(events[i], "repeats"); repeats > 1)
            message += "  (x" + grouped(repeats) + ")";
        line(style("2", at) + fit(message, static_cast<std::size_t>(width - 12)));
    }
    if (events.empty()) line(style("2", "  none yet"));
    if (!footer.empty()) {
        line("");
        line(footer);
    }
    return out;
}

// Reads newline-delimited snapshots from the status endpoint and redraws, reconnecting as needed.
int attach(
    const Args& args, const std::function<std::string()>& footer = {},
    const std::function<bool()>& keep_running = [] { return true; }) {
    const bool terminal = stdout_is_terminal() && args.frames == 0;
    const Style style{args.color && stdout_is_terminal()};
    auto endpoint = remote::parse_endpoint(args.endpoint);
    if (!endpoint) {
        std::cerr << "kora-top: " << endpoint.error().describe() << '\n';
        return 2;
    }
    Activity activity;
    int frames = 0;
    bool waiting_shown = false;
    while (keep_running()) {
        auto socket = remote::Socket::connect(endpoint.value(), std::chrono::milliseconds(500));
        if (!socket) {
            if (!waiting_shown) {
                std::cout << (terminal ? "\x1b[H\x1b[2J" : "")
                          << "kora-top: waiting for a KORA application at " << args.endpoint
                          << "\n  start it with KORA_STATUS_LISTEN=" << args.endpoint
                          << ", or run kora-top --demo\n"
                          << std::flush;
                waiting_shown = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        waiting_shown = false;
        std::string pending;
        while (keep_running()) {
            auto ready = socket.value().readable(std::chrono::milliseconds(1'000));
            if (!ready) break;
            if (!ready.value()) continue;
            char c = 0;
            if (!socket.value().receive_all(std::as_writable_bytes(std::span(&c, 1)),
                                            std::chrono::seconds(2)))
                break;
            if (c != '\n') {
                pending.push_back(c);
                if (pending.size() > (1U << 20U)) break;  // Not a status stream.
                continue;
            }
            auto snapshot = kora::detail::json::parse(pending);
            pending.clear();
            if (!snapshot) continue;
            const auto frame =
                render(snapshot.value(), activity, args.endpoint, style, footer ? footer() : "");
            std::cout << (terminal ? "\x1b[H" : "") << frame << (terminal ? "\x1b[J" : "\n") << std::flush;
            if (args.frames > 0 && ++frames >= args.frames) return 0;
        }
        socket.value().close();
    }
    return 0;
}

// --- Demo ---------------------------------------------------------------------------------------

// Starts a loopback worker, retrying briefly while a previous instance releases the port.
kora::Result<std::unique_ptr<kora::WorkerServer>> start_worker(std::uint16_t port) {
    kora::WorkerOptions options;
    options.listen = "127.0.0.1:" + std::to_string(port);
    options.runtime.log_level = kora::log::Level::Off;
    for (int attempt = 0;; ++attempt) {
        auto server = kora::WorkerServer::start(options);
        if (server || attempt == 99 || server.error().code == kora::ErrorCode::NoVulkanImplementation ||
            server.error().code == kora::ErrorCode::NoEligibleDevice)
            return server;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

int exit_code_for(const kora::Error& error) {
    // 77 tells CTest the demo was skipped because this machine has no usable Vulkan device.
    return error.code == kora::ErrorCode::NoVulkanImplementation ||
                   error.code == kora::ErrorCode::NoEligibleDevice
               ? 77
               : 1;
}

struct DemoParams {
    float scale;
    std::uint32_t count;
    std::uint32_t base;
};

int run_demo(Args args) {
    // Two loopback workers stand in for attached devices; the control thread unplugs and replugs
    // them while a workload runs, and every result is checked against the expected value.
    std::unique_ptr<kora::WorkerServer> workers[2];
    for (auto& worker : workers) {
        auto started = start_worker(0);
        if (!started) {
            std::cerr << "kora-top --demo: could not start a loopback worker: " << started.error().describe()
                      << '\n';
            return exit_code_for(started.error());
        }
        worker = std::move(started).value();
    }
    const std::uint16_t ports[2] = {workers[0]->port(), workers[1]->port()};
    workers[1].reset();  // Worker B starts unplugged.

    kora::ClusterOptions options;
    options.local.log_level = kora::log::Level::Off;
    options.workers = {"127.0.0.1:" + std::to_string(ports[0]), "127.0.0.1:" + std::to_string(ports[1])};
    options.policy = kora::OffloadPolicy::PreferRemote;
    options.status_listen = "127.0.0.1:0";
    options.connect_timeout = std::chrono::milliseconds(200);
    options.reconnect_interval = std::chrono::milliseconds(200);
    options.heartbeat_interval = std::chrono::milliseconds(250);
    auto cluster = kora::Cluster::create(options);
    if (!cluster) {
        std::cerr << "kora-top --demo: " << cluster.error().describe() << '\n';
        return exit_code_for(cluster.error());
    }
    kora::KernelMetadata metadata;
    metadata.kernel_id = "demo.saxpy";
    metadata.local_size = {64, 1, 1};
    metadata.push_constant_size = 12;
    metadata.push_constants = {{"scale", "f32", 0, 4}, {"count", "u32", 4, 4}, {"base", "u32", 8, 4}};
    metadata.bindings = {
        {0, 0, "input_x", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Read},
        {0, 1, "input_y", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::ReadWrite}};
    auto kernel = cluster.value().load_kernel(std::span<const std::uint32_t>(kora_top::kDemoSaxpySpv), "main",
                                              metadata);
    if (!kernel) {
        std::cerr << "kora-top --demo: " << kernel.error().describe() << '\n';
        return 1;
    }

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> verified{0}, mismatches{0};
    std::mutex schedule_mutex;
    std::string next_change;
    std::thread workload([&] {
        constexpr std::uint32_t small_count = 4'096, large_count = 1U << 20U;
        auto x_small = cluster.value().create_buffer<float>(small_count);
        auto y_small = cluster.value().create_buffer<float>(small_count);
        auto x_large = cluster.value().create_buffer<float>(large_count);
        auto y_large = cluster.value().create_buffer<float>(large_count);
        if (!x_small || !y_small || !x_large || !y_large ||
            !x_small.value().write(std::vector<float>(small_count, 1.0F)) ||
            !y_small.value().write(std::vector<float>(small_count, 0.0F)) ||
            !x_large.value().write(std::vector<float>(large_count, 1.0F)) ||
            !y_large.value().write(std::vector<float>(large_count, 0.0F))) {
            mismatches.fetch_add(1);
            return;
        }
        float expected = 0.0F;
        const auto check = [&](auto& buffer) {
            auto values = buffer.read();
            const bool exact = values && std::all_of(values.value().begin(), values.value().end(),
                                                     [&](float value) { return value == expected; });
            (exact ? verified : mismatches).fetch_add(1);
        };
        for (std::uint64_t round = 0; !stop.load(); ++round) {
            auto first = cluster.value()
                             .dispatch(kernel.value())
                             .global(small_count)
                             .read(x_small.value())
                             .read_write(y_small.value())
                             .push(DemoParams{2.0F, small_count, 0})
                             .submit();
            auto second = cluster.value()
                              .dispatch(kernel.value())
                              .global(large_count)
                              .read(x_large.value())
                              .read_write(y_large.value())
                              .push(DemoParams{2.0F, large_count, 0})
                              .split(sizeof(float))
                              .submit();
            if (!first || !second || !first.value().wait() || !second.value().wait()) {
                mismatches.fetch_add(1);
                continue;
            }
            expected += 2.0F;
            if (round % 8 == 0) {
                check(y_small.value());
                check(y_large.value());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
    });

    // Repeating 16-second script: plug B in, unplug A, plug A back, unplug B.
    std::thread control([&] {
        struct Step {
            int at_s;
            int worker;
            bool plug;
        };
        static constexpr Step steps[] = {{3, 1, true}, {7, 0, false}, {11, 0, true}, {15, 1, false}};
        const auto start = Clock::now();
        for (std::uint64_t cycle = 0; !stop.load(); ++cycle) {
            for (const auto& step : steps) {
                const auto when =
                    start + std::chrono::seconds(16 * cycle + static_cast<std::uint64_t>(step.at_s));
                {
                    std::lock_guard lock(schedule_mutex);
                    next_change = std::string(step.plug ? "plug in" : "unplug") + " worker " +
                                  (step.worker == 0 ? "A" : "B") +
                                  " (127.0.0.1:" + std::to_string(ports[step.worker]) + ")";
                }
                while (!stop.load() && Clock::now() < when)
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                if (stop.load()) return;
                if (step.plug) {
                    if (auto started = start_worker(ports[step.worker]))
                        workers[step.worker] = std::move(started).value();
                } else {
                    workers[step.worker].reset();
                }
            }
        }
    });

    const auto start = Clock::now();
    args.endpoint = "127.0.0.1:" + std::to_string(cluster.value().status_port());
    const Style style{args.color && stdout_is_terminal()};
    const auto footer = [&] {
        const auto seconds_to_next = [&] {
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start).count() % 16;
            for (const int at : {3, 7, 11, 15})
                if (elapsed < at) return at - elapsed;
            return 19 - elapsed;
        }();
        std::lock_guard lock(schedule_mutex);
        const auto bad = mismatches.load();
        return style("1", "demo") + "  results verified " + grouped(verified.load()) + ", " +
               (bad == 0 ? style("32", "0 mismatches") : style("31", grouped(bad) + " mismatches")) +
               "   next: " + next_change + " in " + std::to_string(seconds_to_next) + " s   (Ctrl-C to quit)";
    };
    const auto status = attach(args, footer);
    stop.store(true);
    workload.join();
    control.join();
    if (mismatches.load() != 0) {
        std::cerr << "kora-top --demo: " << mismatches.load() << " results did not match\n";
        return 1;
    }
    return status;
}

void usage() {
    std::cout << "usage: kora-top [host:port] [--demo] [--frames N] [--no-color]\n\n"
                 "Shows a running kora::Cluster: local devices and remote workers, where work is going\n"
                 "now, device memory, and recent connect, disconnect, failover, and routing events.\n"
                 "The application opts in with ClusterOptions::status_listen or\n"
                 "KORA_STATUS_LISTEN=127.0.0.1:7979 (the default kora-top endpoint).\n\n"
                 "  --demo       run a built-in Cluster with two loopback workers that are\n"
                 "               unplugged and replugged while verified work runs\n"
                 "  --frames N   draw N snapshots and exit (no screen control)\n"
                 "  --no-color   plain text (also when NO_COLOR is set)\n";
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (std::getenv("NO_COLOR")) args.color = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--demo") {
            args.demo = true;
        } else if (argument == "--frames" && i + 1 < argc) {
            args.frames = std::max(1, std::atoi(argv[++i]));
        } else if (argument == "--no-color") {
            args.color = false;
        } else if (argument == "--help" || argument == "-h") {
            usage();
            return 0;
        } else if (!argument.starts_with("-")) {
            args.endpoint = argument;
        } else {
            usage();
            return 2;
        }
    }
#ifdef _WIN32
    if (HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE); out != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if (GetConsoleMode(out, &mode)) SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
    SetConsoleOutputCP(CP_UTF8);
#endif
    return args.demo ? run_demo(args) : attach(args);
}
