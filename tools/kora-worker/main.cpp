// kora-worker: runs KORA dispatches for a remote kora::Cluster.
//
//   kora-worker [--listen HOST:PORT] [--device INDEX] [--cache-mb N] [--name NAME]
//
// Set KORA_WORKER_TOKEN to accept authenticated non-loopback clients. Without it, only clients on
// this machine can connect. Traffic is not encrypted: use a trusted link or an SSH/VPN tunnel.
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>

#include "kora/worker.hpp"

namespace {
std::atomic<bool> g_stop{false};
void handle_signal(int) {
    g_stop.store(true);
}
}  // namespace

int main(int argc, char** argv) {
    kora::WorkerOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg = argv[index];
        const auto value = [&]() -> const char* { return index + 1 < argc ? argv[++index] : nullptr; };
        const char* next = nullptr;
        if (arg == "--listen" && (next = value())) {
            options.listen = next;
        } else if (arg == "--device" && (next = value())) {
            options.runtime.device_policy = kora::DevicePolicy::ByIndex;
            options.runtime.device_index = static_cast<std::uint32_t>(std::strtoul(next, nullptr, 10));
        } else if (arg == "--cache-mb" && (next = value())) {
            options.max_cache_bytes = std::strtoull(next, nullptr, 10) * 1024ULL * 1024ULL;
        } else if (arg == "--name" && (next = value())) {
            options.name = next;
        } else {
            std::cerr
                << "usage: kora-worker [--listen HOST:PORT] [--device INDEX] [--cache-mb N] [--name NAME]\n";
            return arg == "--help" ? 0 : 2;
        }
    }
    auto server = kora::WorkerServer::start(options);
    if (!server) {
        std::cerr << "kora-worker: " << server.error().describe() << '\n';
        return 1;
    }
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::cout << "kora-worker listening on port " << server.value()->port() << std::endl;
    while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    server.value()->stop();
    return 0;
}
