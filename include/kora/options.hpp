#pragma once
/// @file options.hpp
/// RuntimeOptions: every field is optional. Unset fields fall back to KORA_* environment variables and
/// then to built-in defaults (API values always win over the environment).
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "kora/log.hpp"

namespace kora {

enum class DevicePolicy { Auto, PreferDiscrete, PreferIntegrated, ByIndex, ByUuid, AllEligible };
enum class TelemetryLevel { Off, Basic, Profile, Trace };
enum class ShutdownMode { Graceful, CancelPending };

struct RuntimeOptions {
    std::optional<bool> enable_validation;
    std::optional<DevicePolicy> device_policy;
    std::optional<std::uint32_t> device_index;     ///< For DevicePolicy::ByIndex.
    std::optional<std::string> device_uuid;        ///< 32 hex chars, for DevicePolicy::ByUuid.
    std::optional<double> memory_budget_fraction;  ///< (0,1]; default 0.80.
    std::optional<TelemetryLevel> telemetry;
    std::optional<log::Level> log_level;
    std::optional<std::string> cache_dir;
    std::optional<bool> deterministic;
    std::optional<ShutdownMode> shutdown_mode;
    std::optional<bool> disable_autotune;
    std::optional<bool> disable_multi_queue;
    std::optional<bool> disable_cpu_backend;
    std::optional<bool> force_legacy_sync;  ///< Test hook: skip timeline semaphores / sync2.
    std::optional<bool> trace;
};

/// Fully concrete options after merging API values, environment and defaults.
struct ResolvedOptions {
    bool enable_validation = false;
    DevicePolicy device_policy = DevicePolicy::Auto;
    std::uint32_t device_index = 0;
    std::string device_uuid;
    double memory_budget_fraction = 0.80;
    TelemetryLevel telemetry = TelemetryLevel::Basic;
    log::Level log_level = log::Level::Warn;
    std::string cache_dir;
    bool deterministic = false;
    ShutdownMode shutdown_mode = ShutdownMode::Graceful;
    bool disable_autotune = false;
    bool disable_multi_queue = false;
    bool disable_cpu_backend = false;
    bool force_legacy_sync = false;
    std::vector<std::string> warnings;  ///< Problems found while parsing the environment.
};

/// Environment lookup, injectable for tests. Returns nullopt when the variable is unset.
using EnvLookup = std::function<std::optional<std::string>(const char*)>;
EnvLookup system_env();

/// Merge `opts` over the environment over defaults. Bad environment values add a warning, never throw.
ResolvedOptions resolve(const RuntimeOptions& opts, const EnvLookup& env = system_env());

}  // namespace kora
