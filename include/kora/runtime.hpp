#pragma once
/// @file runtime.hpp
/// Owning Vulkan runtime and selected-device diagnostics.

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "kora/buffer.hpp"
#include "kora/cpu_executor.hpp"
#include "kora/device_info.hpp"
#include "kora/device_selection.hpp"
#include "kora/event.hpp"
#include "kora/kernel_metadata.hpp"
#include "kora/memory_stats.hpp"
#include "kora/options.hpp"

namespace kora {

namespace detail {
struct KernelState;
struct DispatchContext;
}  // namespace detail

class Kernel final {
public:
    Kernel() = default;
    explicit operator bool() const noexcept { return state_ != nullptr; }
    const KernelMetadata& metadata() const noexcept;
    const std::string& entry_point() const noexcept;

private:
    friend class Runtime;
    friend class DispatchBuilder;
    explicit Kernel(std::shared_ptr<const detail::KernelState> state) : state_(std::move(state)) {}
    std::shared_ptr<const detail::KernelState> state_;
};

/// Host-visible mappings for a native CPU implementation. Writable mappings advance host versions
/// on release, causing later GPU dispatches to upload the modified bytes.
class CpuTaskContext final {
public:
    CpuTaskContext() = default;
    std::size_t buffer_count() const noexcept { return buffers_.size(); }
    Result<std::span<const std::byte>> read(std::size_t index) const;
    Result<std::span<std::byte>> write(std::size_t index);
    Result<std::span<std::byte>> read_write(std::size_t index);
    std::span<const std::byte> push_constants() const noexcept { return push_constants_; }

private:
    friend class Runtime;
    std::vector<BufferHandle> buffers_;
    std::vector<KernelAccess> accesses_;
    std::vector<MappedBytes> mappings_;
    std::span<const std::byte> push_constants_;
};

using CpuFallbackFunction = std::function<Status(CpuTaskContext&)>;

enum class DispatchPriority : std::uint8_t { Low, Normal, High, RealtimeHint };

class DispatchBuilder final {
public:
    DispatchBuilder() = default;
    DispatchBuilder& global(std::uint64_t x);
    DispatchBuilder& global(std::uint64_t x, std::uint64_t y, std::uint64_t z);
    DispatchBuilder& read(const BufferHandle& buffer);
    DispatchBuilder& write(const BufferHandle& buffer);
    DispatchBuilder& read_write(const BufferHandle& buffer);
    /// Keep device-written outputs resident instead of copying them to the host on completion.
    /// Use Runtime::readback() when host access is subsequently required.
    DispatchBuilder& defer_readback();
    /// Priority is a scheduler hint; High and RealtimeHint dispatches bypass coalescing delay.
    DispatchBuilder& priority(DispatchPriority priority);
    template <class T>
    DispatchBuilder& read(const Buffer<T>& buffer) {
        return read(buffer.handle());
    }
    template <class T>
    DispatchBuilder& write(const Buffer<T>& buffer) {
        return write(buffer.handle());
    }
    template <class T>
    DispatchBuilder& read_write(const Buffer<T>& buffer) {
        return read_write(buffer.handle());
    }
    DispatchBuilder& push(std::span<const std::byte> bytes);
    template <class T>
    DispatchBuilder& push(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        return push(std::as_bytes(std::span<const T>(&value, 1)));
    }
    Result<Event> submit();

private:
    friend class Runtime;
    DispatchBuilder(std::shared_ptr<detail::DispatchContext> context,
                    std::shared_ptr<const detail::KernelState> kernel);
    std::shared_ptr<detail::DispatchContext> context_;
    std::shared_ptr<const detail::KernelState> kernel_;
    std::array<std::uint64_t, 3> global_size_{};
    std::vector<BufferHandle> buffers_;
    std::vector<KernelAccess> accesses_;
    std::vector<std::byte> push_constants_;
    bool defer_readback_ = false;
    DispatchPriority priority_ = DispatchPriority::Normal;
    friend class Runtime;
};

struct RuntimeTransferStats {
    std::uint64_t uploads = 0;
    std::uint64_t downloads = 0;
};

/// Nonblocking GPU dispatch timing collected from completed queue submissions.
struct RuntimeProfilingStats {
    std::uint64_t samples_collected = 0;
    std::uint64_t dropped_samples = 0;
    std::uint64_t last_submission_id = 0;
    std::uint64_t last_duration_nanoseconds = 0;
    std::uint64_t last_host_timestamp_nanoseconds = 0;
    std::uint64_t last_calibration_deviation_nanoseconds = 0;
};

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
    /// Transfer operations issued by this Runtime since creation.
    Result<RuntimeTransferStats> transfer_stats() const;
    /// Returns completed GPU timestamp samples observed so far; does not wait for pending work.
    Result<RuntimeProfilingStats> profiling_stats(const DeviceId& device) const;
    Result<BufferHandle> create_buffer(std::uint64_t size_bytes, BufferUsage usage = BufferUsage::Storage);
    template <class T>
    Result<Buffer<T>> create_buffer(std::uint64_t count, BufferUsage usage = BufferUsage::Storage) {
        if (!impl_) return Error(ErrorCode::InvalidArgument, "runtime is not initialized");
        auto buffer = buffers_.create<T>(count, usage);
        if (!buffer) return std::move(buffer).error();
        return std::move(buffer).value();
    }
    Result<Kernel> load_kernel(const std::filesystem::path& spirv, std::string entry_point,
                               KernelMetadata metadata);
    /// Loads and validates a kernel for a specific device owned by this Runtime.
    Result<Kernel> load_kernel(const std::filesystem::path& spirv, std::string entry_point,
                               KernelMetadata metadata, const DeviceId& device);
    DispatchBuilder dispatch(const Kernel& kernel) const;
    /// Schedule a host readback for a device-resident buffer and complete when host bytes are fresh.
    Result<Event> readback(const BufferHandle& buffer) const;
    /// Read a device-resident version through host memory for CPU-visible or cross-device staging.
    Result<Event> readback(const BufferHandle& buffer, const DeviceId& device) const;
    /// Register/replace a CPU implementation under a stable semantic key.
    Status register_cpu_fallback(std::string key, CpuFallbackFunction function);
    /// Execute a registered CPU implementation after requested device outputs are visible on host.
    Result<Event> run_cpu_fallback(std::string_view key, std::span<const BufferHandle> buffers,
                                   std::span<const KernelAccess> accesses,
                                   std::span<const std::byte> push_constants = {});
    template <class T>
    Result<Event> readback(const Buffer<T>& buffer) const {
        return readback(buffer.handle());
    }
    template <class T>
    Result<Event> readback(const Buffer<T>& buffer, const DeviceId& device) const {
        return readback(buffer.handle(), device);
    }

private:
    struct Impl;
    explicit Runtime(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    BufferManager buffers_;
};

}  // namespace kora
