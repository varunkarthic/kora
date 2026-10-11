#pragma once
/// @file runtime.hpp
/// Owning Vulkan runtime and selected-device diagnostics.

#include <memory>
#include <vector>

#include "kora/device_info.hpp"
#include "kora/device_selection.hpp"
#include "kora/memory_stats.hpp"
#include "kora/options.hpp"

namespace kora {

/// Owns the Vulkan loader, instance, selected logical devices, and their memory managers.
class Runtime final {
public:
    static Result<Runtime> create(const RuntimeOptions& options = {});

    Runtime();
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;

    /// IDs are stable for this Runtime's lifetime and retain Vulkan enumeration indices.
    std::vector<DeviceId> devices() const;
    Result<DeviceInfo> device_info(const DeviceId& device) const;
    /// Returns pool totals and per-heap budget data; an unused device has zero pool counters.
    Result<MemoryStats> memory_stats(const DeviceId& device) const;

private:
    struct Impl;
    explicit Runtime(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace kora
