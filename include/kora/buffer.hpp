#pragma once
/// @file buffer.hpp
/// Logical, host-shadow-backed buffers and typed mapped views.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kora/detail/checked.hpp"
#include "kora/errors.hpp"

namespace kora {

namespace detail {
struct BufferManagerState;
}

enum class BufferUsage : std::uint8_t { Generic, Storage, Uniform, TransferSource, TransferDestination };
enum class BufferResidency : std::uint8_t { HostOnly, DeviceResident, Mirrored };
enum class MapAccess : std::uint8_t { Read, Write, ReadWrite };

struct BufferId {
    std::uint32_t slot = 0;
    std::uint64_t generation = 0;
    friend bool operator==(const BufferId&, const BufferId&) = default;
};

struct DirtyRange {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

struct BufferInfo {
    std::uint64_t size_bytes = 0;
    BufferUsage usage = BufferUsage::Generic;
    BufferResidency residency = BufferResidency::HostOnly;
    std::uint32_t pin_count = 0;
    std::uint64_t generation = 0;
    std::uint64_t last_use_timeline_value = 0;
    std::uint32_t reference_count = 0;
    bool pending_free = false;
};

struct ConditionalBufferWrite {
    bool written = false;
    std::uint64_t host_version = 0;
};

class MappedBytes final {
public:
    MappedBytes() = default;
    ~MappedBytes();
    MappedBytes(const MappedBytes&) = delete;
    MappedBytes& operator=(const MappedBytes&) = delete;
    MappedBytes(MappedBytes&& other) noexcept;
    MappedBytes& operator=(MappedBytes&& other) noexcept;

    std::byte* data() noexcept { return data_; }
    const std::byte* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    std::span<std::byte> bytes() noexcept { return {data_, size_}; }
    explicit operator bool() const noexcept { return data_ != nullptr; }

private:
    friend class BufferHandle;
    MappedBytes(std::shared_ptr<detail::BufferManagerState> state, BufferId id, std::byte* data,
                std::size_t size);
    void reset() noexcept;

    std::shared_ptr<detail::BufferManagerState> state_;
    BufferId id_{};
    std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};

/// Copyable, reference-counted logical allocation handle.
class BufferHandle final {
public:
    BufferHandle() = default;
    ~BufferHandle();
    BufferHandle(const BufferHandle& other);
    BufferHandle& operator=(const BufferHandle& other);
    BufferHandle(BufferHandle&& other) noexcept;
    BufferHandle& operator=(BufferHandle&& other) noexcept;

    BufferId id() const noexcept { return id_; }
    explicit operator bool() const noexcept { return state_ != nullptr; }
    Result<BufferInfo> info() const;
    std::uint64_t size_bytes() const noexcept;
    /// Monotonic host-shadow content version; changes on writes and writable mappings.
    Result<std::uint64_t> host_version() const;
    Result<std::vector<std::byte>> read_bytes() const;
    Status write_bytes(std::span<const std::byte> bytes) const;
    /// Write downloaded bytes only if the host shadow still has the captured version.
    Result<ConditionalBufferWrite> write_bytes_if_version(std::span<const std::byte> bytes,
                                                          std::uint64_t expected_version) const;
    Result<MappedBytes> map(MapAccess access = MapAccess::ReadWrite, std::uint64_t offset = 0,
                            std::uint64_t size = std::numeric_limits<std::uint64_t>::max()) const;
    Status mark_used(std::uint64_t timeline_value) const;

private:
    friend class BufferManager;
    BufferHandle(std::shared_ptr<detail::BufferManagerState> state, BufferId id);
    void reset() noexcept;

    std::shared_ptr<detail::BufferManagerState> state_;
    BufferId id_{};
};

template <class T>
class Buffer;

template <class T>
class MappedBuffer final {
public:
    MappedBuffer() = default;
    std::span<T> span() noexcept {
        return {reinterpret_cast<T*>(mapped_.data()), static_cast<std::size_t>(count_)};
    }
    T* data() noexcept { return reinterpret_cast<T*>(mapped_.data()); }
    std::size_t size() const noexcept { return static_cast<std::size_t>(count_); }
    T& operator[](std::size_t index) noexcept { return span()[index]; }

private:
    friend class Buffer<T>;
    MappedBuffer(MappedBytes mapped, std::uint64_t count) : mapped_(std::move(mapped)), count_(count) {}
    MappedBytes mapped_;
    std::uint64_t count_ = 0;
};

template <class T>
class Buffer final {
    static_assert(std::is_trivially_copyable_v<T>, "Buffer<T> requires trivially copyable elements");
    static_assert(alignof(T) <= alignof(std::max_align_t),
                  "Buffer<T> does not support over-aligned elements");

public:
    Buffer() = default;
    std::uint64_t size() const noexcept { return count_; }
    std::uint64_t size_bytes() const noexcept { return count_ * sizeof(T); }
    const BufferHandle& handle() const noexcept { return handle_; }
    BufferId id() const noexcept { return handle_.id(); }
    explicit operator bool() const noexcept { return static_cast<bool>(handle_); }
    Result<BufferInfo> info() const { return handle_.info(); }
    Status write(std::span<const T> values) const {
        if (values.size() != count_)
            return Error(ErrorCode::InvalidArgument,
                         "buffer write element count does not match its capacity");
        return handle_.write_bytes(std::as_bytes(values));
    }
    Status read(std::span<T> values) const {
        if (values.size() != count_)
            return Error(ErrorCode::InvalidArgument, "buffer read element count does not match its capacity");
        auto bytes = handle_.read_bytes();
        if (!bytes) return std::move(bytes).error();
        std::memcpy(values.data(), bytes.value().data(), bytes.value().size());
        return {};
    }
    Result<std::vector<T>> read() const {
        std::vector<T> values(static_cast<std::size_t>(count_));
        auto status = read(std::span<T>(values));
        if (!status) return std::move(status).error();
        return values;
    }

    Result<MappedBuffer<T>> map(MapAccess access = MapAccess::ReadWrite) {
        auto mapped = handle_.map(access);
        if (!mapped) return mapped.error();
        return MappedBuffer<T>(std::move(mapped).value(), count_);
    }
    Status mark_used(std::uint64_t timeline_value) const { return handle_.mark_used(timeline_value); }

private:
    friend class BufferManager;
    Buffer(BufferHandle handle, std::uint64_t count) : handle_(std::move(handle)), count_(count) {}

    BufferHandle handle_;
    std::uint64_t count_ = 0;
};

/// Thread-safe logical-buffer registry. Completion values are supplied by the queue/completion layer.
class BufferManager final {
public:
    BufferManager();
    ~BufferManager();
    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;
    BufferManager(BufferManager&&) noexcept;
    BufferManager& operator=(BufferManager&&) noexcept;

    Result<BufferHandle> create(std::uint64_t size_bytes, BufferUsage usage = BufferUsage::Generic);
    template <class T>
    Result<Buffer<T>> create(std::uint64_t element_count, BufferUsage usage = BufferUsage::Storage) {
        auto bytes = checked::byte_size<T>(element_count, "typed buffer size");
        if (!bytes) return bytes.error();
        auto handle = create(bytes.value(), usage);
        if (!handle) return handle.error();
        return Buffer<T>(std::move(handle).value(), element_count);
    }

    Result<BufferInfo> inspect(BufferId id) const;
    Result<std::vector<DirtyRange>> dirty_ranges(BufferId id) const;
    Status mark_used(BufferId id, std::uint64_t timeline_value);
    Result<std::size_t> advance_completion(std::uint64_t completed_timeline_value);
    std::size_t deferred_count() const noexcept;

private:
    std::shared_ptr<detail::BufferManagerState> state_;
};

}  // namespace kora
