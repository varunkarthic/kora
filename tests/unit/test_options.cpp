#include <gtest/gtest.h>

#include <map>

#include "kora/options.hpp"

using namespace kora;

namespace {
EnvLookup fake(std::map<std::string, std::string> m) {
    return [m](const char* k) -> std::optional<std::string> {
        auto it = m.find(k);
        if (it == m.end()) return std::nullopt;
        return it->second;
    };
}
}  // namespace

TEST(Options, Defaults) {
    auto r = resolve({}, fake({{"HOME", "/h"}}));
    EXPECT_FALSE(r.enable_validation);
    EXPECT_EQ(r.device_policy, DevicePolicy::Auto);
    EXPECT_DOUBLE_EQ(r.memory_budget_fraction, 0.80);
    EXPECT_EQ(r.telemetry, TelemetryLevel::Basic);
    EXPECT_TRUE(r.warnings.empty());
    EXPECT_FALSE(r.cache_dir.empty());
    EXPECT_TRUE(r.enable_startup_calibration);
    EXPECT_EQ(r.calibration_budget_ms, 250U);
}

TEST(Options, StartupCalibrationCanBeDisabledAndBudgetConfigured) {
    auto from_environment = resolve({}, fake({{"KORA_STARTUP_CALIBRATION", "false"},
                                               {"KORA_CALIBRATION_BUDGET_MS", "500"}}));
    EXPECT_FALSE(from_environment.enable_startup_calibration);
    EXPECT_EQ(from_environment.calibration_budget_ms, 500U);
    auto explicit_options = RuntimeOptions{};
    explicit_options.enable_startup_calibration = true;
    explicit_options.calibration_budget_ms = 75;
    auto from_api = resolve(explicit_options,
                            fake({{"KORA_STARTUP_CALIBRATION", "false"},
                                  {"KORA_CALIBRATION_BUDGET_MS", "900"}}));
    EXPECT_TRUE(from_api.enable_startup_calibration);
    EXPECT_EQ(from_api.calibration_budget_ms, 75U);
}

TEST(Options, EnvApplies) {
    auto r = resolve({}, fake({{"KORA_VALIDATION", "1"},
                               {"KORA_DEVICE", "2"},
                               {"KORA_LOG_LEVEL", "DEBUG"},
                               {"KORA_MEMORY_BUDGET_FRACTION", "0.5"},
                               {"KORA_PROFILE", "trace"},
                               {"KORA_CACHE_DIR", "/c"},
                               {"KORA_DISABLE_AUTOTUNE", "true"}}));
    EXPECT_TRUE(r.enable_validation);
    EXPECT_EQ(r.device_policy, DevicePolicy::ByIndex);
    EXPECT_EQ(r.device_index, 2u);
    EXPECT_EQ(r.log_level, log::Level::Debug);
    EXPECT_DOUBLE_EQ(r.memory_budget_fraction, 0.5);
    EXPECT_EQ(r.telemetry, TelemetryLevel::Trace);
    EXPECT_EQ(r.cache_dir, "/c");
    EXPECT_TRUE(r.disable_autotune);
}

TEST(Options, ApiBeatsEnv) {
    RuntimeOptions o;
    o.enable_validation = false;
    o.memory_budget_fraction = 0.9;
    o.device_policy = DevicePolicy::PreferIntegrated;
    auto r = resolve(o, fake({{"KORA_VALIDATION", "1"},
                              {"KORA_MEMORY_BUDGET_FRACTION", "0.3"},
                              {"KORA_DEVICE", "discrete"}}));
    EXPECT_FALSE(r.enable_validation);
    EXPECT_DOUBLE_EQ(r.memory_budget_fraction, 0.9);
    EXPECT_EQ(r.device_policy, DevicePolicy::PreferIntegrated);
}

TEST(Options, BadEnvWarnsNeverThrows) {
    auto r = resolve({}, fake({{"KORA_VALIDATION", "maybe"},
                               {"KORA_MEMORY_BUDGET_FRACTION", "7"},
                               {"KORA_DEVICE", "???"},
                               {"KORA_LOG_LEVEL", "loud"},
                               {"KORA_PROFILE", "x"}}));
    EXPECT_EQ(r.warnings.size(), 5u);
    EXPECT_FALSE(r.enable_validation);
    EXPECT_DOUBLE_EQ(r.memory_budget_fraction, 0.80);
}

TEST(Options, OutOfRangeApiFractionReset) {
    RuntimeOptions o;
    o.memory_budget_fraction = 2.0;
    auto r = resolve(o, fake({}));
    EXPECT_DOUBLE_EQ(r.memory_budget_fraction, 0.80);
    EXPECT_EQ(r.warnings.size(), 1u);
}

TEST(Options, UuidDevice) {
    auto r = resolve({}, fake({{"KORA_DEVICE", "00112233445566778899aabbccddeeff"}}));
    EXPECT_EQ(r.device_policy, DevicePolicy::ByUuid);
    EXPECT_EQ(r.device_uuid.size(), 32u);
}

TEST(Options, OperationalEnvironmentKnobs) {
    auto r = resolve({}, fake({{"KORA_DETERMINISTIC", "true"},
                               {"KORA_DISABLE_MULTI_QUEUE", "1"},
                               {"KORA_DISABLE_CPU_BACKEND", "yes"},
                               {"KORA_FORCE_LEGACY_SYNC", "on"},
                               {"KORA_SHUTDOWN_MODE", "cancel_pending"},
                               {"KORA_TRACE", "true"}}));
    EXPECT_TRUE(r.deterministic);
    EXPECT_TRUE(r.disable_multi_queue);
    EXPECT_TRUE(r.disable_cpu_backend);
    EXPECT_TRUE(r.force_legacy_sync);
    EXPECT_EQ(r.shutdown_mode, ShutdownMode::CancelPending);
    EXPECT_EQ(r.telemetry, TelemetryLevel::Trace);
}

TEST(Options, ExplicitOperationalKnobsBeatEnvironment) {
    RuntimeOptions o;
    o.deterministic = false;
    o.force_legacy_sync = false;
    o.shutdown_mode = ShutdownMode::Graceful;
    auto r = resolve(o, fake({{"KORA_DETERMINISTIC", "true"},
                              {"KORA_FORCE_LEGACY_SYNC", "true"},
                              {"KORA_SHUTDOWN_MODE", "cancel_pending"}}));
    EXPECT_FALSE(r.deterministic);
    EXPECT_FALSE(r.force_legacy_sync);
    EXPECT_EQ(r.shutdown_mode, ShutdownMode::Graceful);
}

TEST(Options, BadShutdownModeWarns) {
    auto r = resolve({}, fake({{"KORA_SHUTDOWN_MODE", "immediate"}}));
    EXPECT_EQ(r.shutdown_mode, ShutdownMode::Graceful);
    ASSERT_EQ(r.warnings.size(), 1u);
    EXPECT_NE(r.warnings[0].find("KORA_SHUTDOWN_MODE"), std::string::npos);
}
