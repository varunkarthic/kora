#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include "kora/cluster.hpp"
#include "kora/worker.hpp"
#include "support/vulkan_test.hpp"

namespace {
#ifdef KORA_SAXPY_FIXTURE_PATH
kora::KernelMetadata saxpy_metadata() {
    kora::KernelMetadata metadata;
    metadata.kernel_id = "integration.cluster.saxpy";
    metadata.local_size = {64, 1, 1};
    metadata.push_constant_size = 8;
    metadata.push_constants = {{"scale", "f32", 0, 4}, {"count", "u32", 4, 4}};
    metadata.bindings = {
        {0, 0, "input_x", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Read},
        {0, 1, "input_y", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::ReadWrite}};
    return metadata;
}

struct SaxpyParams {
    float scale;
    std::uint32_t count;
};

constexpr std::uint32_t kCount = 4099;

// Starts a loopback worker, retrying briefly while a previous instance releases the port.
std::unique_ptr<kora::WorkerServer> start_worker(std::uint16_t port, std::string token = {}) {
    kora::WorkerOptions options;
    options.listen = "127.0.0.1:" + std::to_string(port);
    options.token = std::move(token);
    options.io_timeout = std::chrono::milliseconds(2'000);
    for (int attempt = 0; attempt < 50; ++attempt) {
        auto server = kora::WorkerServer::start(options);
        if (server) return std::move(server).value();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return nullptr;
}

kora::ClusterOptions fast_options(std::uint16_t port, kora::OffloadPolicy policy) {
    kora::ClusterOptions options;
    options.workers = {"127.0.0.1:" + std::to_string(port)};
    options.policy = policy;
    options.connect_timeout = std::chrono::milliseconds(200);
    options.reconnect_interval = std::chrono::milliseconds(20);
    options.request_timeout = std::chrono::milliseconds(3'000);
    options.heartbeat_interval = std::chrono::milliseconds(100);
    return options;
}

bool wait_until(const std::function<bool()>& predicate,
                std::chrono::milliseconds limit = std::chrono::seconds(10)) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return predicate();
}

bool any_connected(const kora::Cluster& cluster) {
    for (const auto& worker : cluster.workers())
        if (worker.connected) return true;
    return false;
}

// Runs `jobs` chained y += x dispatches; every job depends on the previous one through y.
void run_chain(kora::Cluster& cluster, const kora::ClusterKernel& kernel, kora::Buffer<float>& x,
               kora::Buffer<float>& y, int jobs, const std::function<void(int)>& between = {}) {
    std::vector<kora::Event> events;
    for (int job = 0; job < jobs; ++job) {
        auto event = cluster.dispatch(kernel)
                         .global(kCount)
                         .read(x)
                         .read_write(y)
                         .push(SaxpyParams{1.0F, kCount})
                         .submit();
        ASSERT_TRUE(event) << event.error().describe();
        events.push_back(std::move(event).value());
        if (between) between(job);
    }
    for (auto& event : events) ASSERT_TRUE(event.wait());
}

void expect_value(kora::Buffer<float>& y, float expected) {
    auto values = y.read();
    ASSERT_TRUE(values) << values.error().describe();
    for (std::uint32_t i = 0; i < kCount; ++i) ASSERT_EQ(values.value()[i], expected) << "index " << i;
}

struct Fixture {
    kora::Cluster cluster;
    kora::ClusterKernel kernel;
    kora::Buffer<float> x;
    kora::Buffer<float> y;
};

std::optional<Fixture> make_fixture(kora::ClusterOptions options) {
    auto cluster = kora::Cluster::create(std::move(options));
    if (!cluster) return std::nullopt;
    auto kernel = cluster.value().load_kernel(KORA_SAXPY_FIXTURE_PATH, "main", saxpy_metadata());
    auto x = cluster.value().create_buffer<float>(kCount);
    auto y = cluster.value().create_buffer<float>(kCount);
    if (!kernel || !x || !y) return std::nullopt;
    if (!x.value().write(std::vector<float>(kCount, 1.0F)) ||
        !y.value().write(std::vector<float>(kCount, 0.0F)))
        return std::nullopt;
    return Fixture{std::move(cluster).value(), std::move(kernel).value(), std::move(x).value(),
                   std::move(y).value()};
}
#endif
}  // namespace

#ifdef KORA_SAXPY_FIXTURE_PATH
TEST(Cluster, OffloadsToAConnectedWorkerWithIdenticalResults) {
    auto worker = start_worker(0);
    if (!worker) KORA_REQUIRE_VULKAN(false);
    auto fixture = make_fixture(fast_options(worker->port(), kora::OffloadPolicy::PreferRemote));
    ASSERT_TRUE(fixture);
    ASSERT_TRUE(wait_until([&] { return any_connected(fixture->cluster); }));
    run_chain(fixture->cluster, fixture->kernel, fixture->x, fixture->y, 20);
    expect_value(fixture->y, 20.0F);
    const auto stats = fixture->cluster.stats();
    EXPECT_EQ(stats.remote_dispatches, 20U);
    EXPECT_EQ(stats.failovers, 0U);
    EXPECT_EQ(worker->stats().dispatches, 20U);
    EXPECT_EQ(worker->stats().kernels_loaded, 1U);
}

TEST(Cluster, PicksUpAWorkerThatStartsAfterTheCluster) {
    auto probe = start_worker(0);
    if (!probe) KORA_REQUIRE_VULKAN(false);
    const auto port = probe->port();
    probe.reset();
    auto fixture = make_fixture(fast_options(port, kora::OffloadPolicy::PreferRemote));
    ASSERT_TRUE(fixture);
    run_chain(fixture->cluster, fixture->kernel, fixture->x, fixture->y, 5);  // No worker yet: local.
    EXPECT_EQ(fixture->cluster.stats().remote_dispatches, 0U);
    auto worker = start_worker(port);
    ASSERT_TRUE(worker);
    ASSERT_TRUE(wait_until([&] { return any_connected(fixture->cluster); }));
    run_chain(fixture->cluster, fixture->kernel, fixture->x, fixture->y, 5);
    expect_value(fixture->y, 10.0F);
    EXPECT_GT(fixture->cluster.stats().remote_dispatches, 0U);
}

TEST(Cluster, SurvivesWorkerStopRestartAndDetachMidStream) {
    auto worker = start_worker(0);
    if (!worker) KORA_REQUIRE_VULKAN(false);
    const auto port = worker->port();
    auto fixture = make_fixture(fast_options(port, kora::OffloadPolicy::PreferRemote));
    ASSERT_TRUE(fixture);
    ASSERT_TRUE(wait_until([&] { return any_connected(fixture->cluster); }));
    run_chain(fixture->cluster, fixture->kernel, fixture->x, fixture->y, 60, [&](int job) {
        if (job == 20) worker.reset();  // Unplug while work is queued.
        if (job == 40) {
            worker = start_worker(port);  // Plug back in.
            ASSERT_TRUE(worker);
            ASSERT_TRUE(wait_until([&] { return any_connected(fixture->cluster); }));
        }
    });
    expect_value(fixture->y, 60.0F);
    auto status = fixture->cluster.workers();
    ASSERT_EQ(status.size(), 1U);
    EXPECT_GE(status[0].connects, 2U);
    EXPECT_GE(status[0].disconnects, 1U);
    const auto stats = fixture->cluster.stats();
    EXPECT_EQ(stats.local_dispatches + stats.remote_dispatches, 60U);

    ASSERT_TRUE(fixture->cluster.detach("127.0.0.1:" + std::to_string(port)));
    run_chain(fixture->cluster, fixture->kernel, fixture->x, fixture->y, 5);
    expect_value(fixture->y, 65.0F);
    EXPECT_TRUE(fixture->cluster.workers().empty());
}

TEST(Cluster, RandomDisconnectsNeverCorruptOrLoseWork) {
    auto worker = start_worker(0);
    if (!worker) KORA_REQUIRE_VULKAN(false);
    const auto port = worker->port();
    auto fixture = make_fixture(fast_options(port, kora::OffloadPolicy::PreferRemote));
    ASSERT_TRUE(fixture);
    std::atomic<bool> done{false};
    std::thread chaos([&] {
        std::mt19937 random(1234);
        std::uniform_int_distribution<int> delay(5, 60);
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay(random)));
            if (worker)
                worker.reset();
            else
                worker = start_worker(port);
        }
    });
    // Keep submitting paced batches until the worker has been unplugged and replugged several times
    // (slower under sanitizers), with at least 300 dependent dispatches overall.
    int jobs = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while ((jobs < 300 || fixture->cluster.workers().at(0).connects < 3) &&
           std::chrono::steady_clock::now() < deadline) {
        run_chain(fixture->cluster, fixture->kernel, fixture->x, fixture->y, 25, [](int job) {
            if (job % 5 == 4) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        jobs += 25;
    }
    done.store(true);
    chaos.join();
    expect_value(fixture->y, static_cast<float>(jobs));
    const auto stats = fixture->cluster.stats();
    EXPECT_EQ(stats.local_dispatches + stats.remote_dispatches, static_cast<std::uint64_t>(jobs));
    EXPECT_GT(stats.remote_dispatches, 0U);
    EXPECT_GE(fixture->cluster.workers().at(0).connects, 3U);
}

TEST(Cluster, RejectsWorkersWithTheWrongTokenAndRunsLocally) {
    auto worker = start_worker(0, "correct horse battery staple");
    if (!worker) KORA_REQUIRE_VULKAN(false);
    auto options = fast_options(worker->port(), kora::OffloadPolicy::PreferRemote);
    options.token = "wrong token";
    auto fixture = make_fixture(options);
    ASSERT_TRUE(fixture);
    run_chain(fixture->cluster, fixture->kernel, fixture->x, fixture->y, 5);
    expect_value(fixture->y, 5.0F);
    EXPECT_EQ(fixture->cluster.stats().remote_dispatches, 0U);
    EXPECT_TRUE(wait_until([&] { return worker->stats().rejected_connections > 0; }));
    EXPECT_FALSE(any_connected(fixture->cluster));
}
#endif

#ifdef KORA_SPLIT_SAXPY_FIXTURE_PATH
namespace {
kora::KernelMetadata split_saxpy_metadata() {
    auto metadata = saxpy_metadata();
    metadata.kernel_id = "integration.cluster.split_saxpy";
    metadata.push_constant_size = 12;
    metadata.push_constants.push_back({"base", "u32", 8, 4});
    return metadata;
}

struct SplitParams {
    float scale;
    std::uint32_t count;
    std::uint32_t base;
};

kora::ClusterOptions split_options() {
    kora::ClusterOptions options;
    options.local.device_policy = kora::DevicePolicy::AllEligible;
    options.policy = kora::OffloadPolicy::LocalOnly;
    options.split_min_elements = 1024;
    options.split_overhead = std::chrono::microseconds(0);
    options.device_rescan_interval = std::chrono::milliseconds(0);
    return options;
}
}  // namespace

TEST(Cluster, RoutesAcrossEveryLocalDevice) {
    auto cluster = kora::Cluster::create(split_options());
    if (!cluster) KORA_REQUIRE_VULKAN(false);
    auto kernel = cluster.value().load_kernel(KORA_SAXPY_FIXTURE_PATH, "main", saxpy_metadata());
    ASSERT_TRUE(kernel) << kernel.error().describe();
    auto x = cluster.value().create_buffer<float>(kCount);
    auto y = cluster.value().create_buffer<float>(kCount);
    ASSERT_TRUE(x && y);
    ASSERT_TRUE(x.value().write(std::vector<float>(kCount, 1.0F)) &&
                y.value().write(std::vector<float>(kCount, 0.0F)));
    const auto devices = cluster.value().local_devices();
    const int jobs = static_cast<int>(devices.size()) * 4;
    run_chain(cluster.value(), kernel.value(), x.value(), y.value(), jobs);
    expect_value(y.value(), static_cast<float>(jobs));
    std::uint64_t total = 0;
    std::size_t used = 0;
    for (const auto& device : cluster.value().local_devices()) {
        total += device.dispatches;
        used += device.dispatches > 0 ? 1U : 0U;
    }
    EXPECT_EQ(total, static_cast<std::uint64_t>(jobs));
    EXPECT_EQ(used, devices.size()) << "Auto measures every local device before choosing";
}

TEST(Cluster, SplitDispatchMatchesCpuReferenceAcrossDevices) {
    auto cluster = kora::Cluster::create(split_options());
    if (!cluster) KORA_REQUIRE_VULKAN(false);
    if (cluster.value().local_devices().size() < 2) GTEST_SKIP() << "needs at least two local devices";
    auto kernel = cluster.value().load_kernel(KORA_SPLIT_SAXPY_FIXTURE_PATH, "main", split_saxpy_metadata());
    ASSERT_TRUE(kernel) << kernel.error().describe();
    constexpr std::uint32_t count = 200'003;  // Not a multiple of the 64-wide workgroup.
    std::vector<float> x_values(count), y_values(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        x_values[i] = static_cast<float>(i % 97);
        y_values[i] = static_cast<float>(i % 13);
    }
    auto x = cluster.value().create_buffer<float>(count);
    auto y = cluster.value().create_buffer<float>(count);
    ASSERT_TRUE(x && y && x.value().write(x_values) && y.value().write(y_values));
    for (int round = 0; round < 6; ++round) {
        auto event = cluster.value()
                         .dispatch(kernel.value())
                         .global(count)
                         .read(x.value())
                         .read_write(y.value())
                         .push(SplitParams{2.0F, count, 0})
                         .split(sizeof(float))
                         .submit();
        ASSERT_TRUE(event) << event.error().describe();
        ASSERT_TRUE(event.value().wait());
        for (std::uint32_t i = 0; i < count; ++i) y_values[i] += 2.0F * x_values[i];
    }
    auto result = y.value().read();
    ASSERT_TRUE(result);
    for (std::uint32_t i = 0; i < count; ++i) ASSERT_EQ(result.value()[i], y_values[i]) << "index " << i;
    const auto stats = cluster.value().stats();
    EXPECT_GT(stats.split_dispatches, 0U);
    EXPECT_GE(stats.slices, 2U * stats.split_dispatches);
}

TEST(Cluster, SplitSurvivesWorkerChaos) {
    auto worker = start_worker(0);
    if (!worker) KORA_REQUIRE_VULKAN(false);
    const auto port = worker->port();
    auto options = fast_options(port, kora::OffloadPolicy::PreferRemote);
    options.local.device_policy = kora::DevicePolicy::AllEligible;
    options.split_min_elements = 1024;
    options.split_overhead = std::chrono::microseconds(0);
    options.device_rescan_interval = std::chrono::milliseconds(0);
    auto cluster = kora::Cluster::create(options);
    ASSERT_TRUE(cluster) << cluster.error().describe();
    auto kernel = cluster.value().load_kernel(KORA_SPLIT_SAXPY_FIXTURE_PATH, "main", split_saxpy_metadata());
    ASSERT_TRUE(kernel) << kernel.error().describe();
    constexpr std::uint32_t count = 65'537;
    auto x = cluster.value().create_buffer<float>(count);
    auto y = cluster.value().create_buffer<float>(count);
    ASSERT_TRUE(x && y && x.value().write(std::vector<float>(count, 1.0F)) &&
                y.value().write(std::vector<float>(count, 0.0F)));
    std::atomic<bool> done{false};
    std::thread chaos([&] {
        std::mt19937 random(99);
        std::uniform_int_distribution<int> delay(5, 40);
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay(random)));
            if (worker)
                worker.reset();
            else
                worker = start_worker(port);
        }
    });
    constexpr int rounds = 120;
    std::vector<kora::Event> events;
    for (int round = 0; round < rounds; ++round) {
        auto event = cluster.value()
                         .dispatch(kernel.value())
                         .global(count)
                         .read(x.value())
                         .read_write(y.value())
                         .push(SplitParams{1.0F, count, 0})
                         .split(sizeof(float))
                         .submit();
        ASSERT_TRUE(event) << event.error().describe();
        events.push_back(std::move(event).value());
        if (round % 4 == 3) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    for (auto& event : events) ASSERT_TRUE(event.wait());
    done.store(true);
    chaos.join();
    auto result = y.value().read();
    ASSERT_TRUE(result);
    for (std::uint32_t i = 0; i < count; ++i) ASSERT_EQ(result.value()[i], static_cast<float>(rounds)) << i;
    EXPECT_GT(cluster.value().stats().split_dispatches, 0U);
}

TEST(Cluster, SplitRequiresABaseField) {
    auto cluster = kora::Cluster::create(split_options());
    if (!cluster) KORA_REQUIRE_VULKAN(false);
    auto kernel = cluster.value().load_kernel(KORA_SAXPY_FIXTURE_PATH, "main", saxpy_metadata());
    ASSERT_TRUE(kernel);
    auto x = cluster.value().create_buffer<float>(kCount);
    auto y = cluster.value().create_buffer<float>(kCount);
    ASSERT_TRUE(x && y);
    auto event = cluster.value()
                     .dispatch(kernel.value())
                     .global(kCount)
                     .read(x.value())
                     .read_write(y.value())
                     .push(SaxpyParams{1.0F, kCount})
                     .split(sizeof(float))
                     .submit();
    ASSERT_FALSE(event);
    EXPECT_EQ(event.error().code, kora::ErrorCode::InvalidArgument);
}
#endif
