#include "kora/options.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace kora {
namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool parse_bool(const std::string& v, bool& out) {
    auto s = lower(v);
    if (s == "1" || s == "true" || s == "on" || s == "yes") {
        out = true;
        return true;
    }
    if (s == "0" || s == "false" || s == "off" || s == "no" || s.empty()) {
        out = false;
        return true;
    }
    return false;
}

bool is_uuid_hex(const std::string& s) {
    return s.size() == 32 &&
           std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isxdigit(c); });
}

std::string default_cache_dir(const EnvLookup& env) {
#if defined(_WIN32)
    if (auto v = env("LOCALAPPDATA")) return *v + "\\kora";
    return ".kora-cache";
#elif defined(__APPLE__)
    if (auto h = env("HOME")) return *h + "/Library/Caches/kora";
    return ".kora-cache";
#else
    if (auto x = env("XDG_CACHE_HOME"); x && !x->empty()) return *x + "/kora";
    if (auto h = env("HOME")) return *h + "/.cache/kora";
    return ".kora-cache";
#endif
}

}  // namespace

EnvLookup system_env() {
    return [](const char* name) -> std::optional<std::string> {
        const char* v = std::getenv(name);
        if (!v) return std::nullopt;
        return std::string(v);
    };
}

ResolvedOptions resolve(const RuntimeOptions& o, const EnvLookup& env) {
    ResolvedOptions r;
    auto warn = [&](std::string m) { r.warnings.push_back(std::move(m)); };

    auto env_bool = [&](const char* name, std::optional<bool> api, bool def) {
        if (api) return *api;
        if (auto v = env(name)) {
            bool b;
            if (parse_bool(*v, b)) return b;
            warn(std::string(name) + ": invalid boolean '" + *v + "', ignored");
        }
        return def;
    };

    r.enable_validation = env_bool("KORA_VALIDATION", o.enable_validation, false);
    r.disable_autotune = env_bool("KORA_DISABLE_AUTOTUNE", o.disable_autotune, false);
    r.disable_multi_queue = env_bool("KORA_DISABLE_MULTI_QUEUE", o.disable_multi_queue, false);
    r.disable_cpu_backend = env_bool("KORA_DISABLE_CPU_BACKEND", o.disable_cpu_backend, false);
    r.deterministic = env_bool("KORA_DETERMINISTIC", o.deterministic, false);
    r.force_legacy_sync = env_bool("KORA_FORCE_LEGACY_SYNC", o.force_legacy_sync, false);
    if (o.shutdown_mode) {
        r.shutdown_mode = *o.shutdown_mode;
    } else if (auto v = env("KORA_SHUTDOWN_MODE")) {
        auto s = lower(*v);
        if (s == "graceful")
            r.shutdown_mode = ShutdownMode::Graceful;
        else if (s == "cancel_pending" || s == "cancel-pending")
            r.shutdown_mode = ShutdownMode::CancelPending;
        else
            warn("KORA_SHUTDOWN_MODE: expected graceful or cancel_pending, got '" + *v + "', ignored");
    }

    // Device policy: API > KORA_DEVICE ("auto|discrete|integrated|all|<index>|<uuid>").
    if (o.device_policy) {
        r.device_policy = *o.device_policy;
        r.device_index = o.device_index.value_or(0);
        r.device_uuid = o.device_uuid.value_or("");
    } else if (auto v = env("KORA_DEVICE")) {
        auto s = lower(*v);
        if (s == "auto" || s.empty())
            r.device_policy = DevicePolicy::Auto;
        else if (s == "discrete")
            r.device_policy = DevicePolicy::PreferDiscrete;
        else if (s == "integrated")
            r.device_policy = DevicePolicy::PreferIntegrated;
        else if (s == "all")
            r.device_policy = DevicePolicy::AllEligible;
        else if (is_uuid_hex(s)) {
            r.device_policy = DevicePolicy::ByUuid;
            r.device_uuid = s;
        } else if (std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); }) &&
                   s.size() < 9) {
            r.device_policy = DevicePolicy::ByIndex;
            r.device_index = static_cast<std::uint32_t>(std::stoul(s));
        } else
            warn("KORA_DEVICE: unrecognized value '" + *v + "', using auto");
    }

    if (o.memory_budget_fraction)
        r.memory_budget_fraction = *o.memory_budget_fraction;
    else if (auto v = env("KORA_MEMORY_BUDGET_FRACTION")) {
        char* end = nullptr;
        double d = std::strtod(v->c_str(), &end);
        if (end != v->c_str() && *end == '\0' && d > 0.0 && d <= 1.0)
            r.memory_budget_fraction = d;
        else
            warn("KORA_MEMORY_BUDGET_FRACTION: expected (0,1], got '" + *v + "', ignored");
    }
    if (!(r.memory_budget_fraction > 0.0 && r.memory_budget_fraction <= 1.0)) {
        warn("memory_budget_fraction out of range, reset to 0.80");
        r.memory_budget_fraction = 0.80;
    }

    if (o.telemetry)
        r.telemetry = *o.telemetry;
    else if (auto v = env("KORA_PROFILE")) {
        auto s = lower(*v);
        if (s == "off")
            r.telemetry = TelemetryLevel::Off;
        else if (s == "basic")
            r.telemetry = TelemetryLevel::Basic;
        else if (s == "profile" || s == "1")
            r.telemetry = TelemetryLevel::Profile;
        else if (s == "trace")
            r.telemetry = TelemetryLevel::Trace;
        else
            warn("KORA_PROFILE: unrecognized value '" + *v + "', ignored");
    }
    if (env_bool("KORA_TRACE", o.trace, false)) r.telemetry = TelemetryLevel::Trace;

    if (o.log_level)
        r.log_level = *o.log_level;
    else if (auto v = env("KORA_LOG_LEVEL")) {
        if (!log::parse_level(*v, r.log_level))
            warn("KORA_LOG_LEVEL: unrecognized value '" + *v + "', ignored");
    }

    r.cache_dir = o.cache_dir ? *o.cache_dir : env("KORA_CACHE_DIR").value_or(default_cache_dir(env));
    return r;
}

}  // namespace kora
