#include <cassert>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>

#include "kora/buffer.hpp"

namespace kora::detail {

struct BufferRecord {
    std::vector<std::byte> host_shadow;
    std::vector<DirtyRange> dirty_ranges;
    std::uint64_t size_bytes = 0;
    std::uint64_t last_use_timeline_value = 0;
    std::uint32_t reference_count = 0;
    std::uint32_t pin_count = 0;
    BufferUsage usage = BufferUsage::Generic;
    BufferResidency residency = BufferResidency::HostOnly;
    bool pending_free = false;
};

struct BufferSlot {
    std::uint64_t generation = 1;
    bool retired = false;
    std::unique_ptr<BufferRecord> record;
};

struct BufferManagerState {
    mutable std::mutex mutex;
    std::vector<BufferSlot> slots;
    std::vector<std::uint32_t> available_slots;
    std::vector<BufferId> deferred;
    std::uint64_t completed_timeline_value = 0;

    BufferRecord* find(BufferId id) noexcept {
        if (id.slot >= slots.size()) return nullptr;
        auto& slot = slots[id.slot];
        if (slot.retired || slot.generation != id.generation || slot.record == nullptr) return nullptr;
        return slot.record.get();
    }

    const BufferRecord* find(BufferId id) const noexcept {
        if (id.slot >= slots.size()) return nullptr;
        const auto& slot = slots[id.slot];
        if (slot.retired || slot.generation != id.generation || slot.record == nullptr) return nullptr;
        return slot.record.get();
    }

    void collect_ready() noexcept {
        auto it = deferred.begin();
        while (it != deferred.end()) {
            auto* record = find(*it);
            if (record == nullptr) {
                it = deferred.erase(it);
                continue;
            }
            if (!record->pending_free || record->reference_count != 0 || record->pin_count != 0 ||
                record->last_use_timeline_value > completed_timeline_value) {
                ++it;
                continue;
            }
            auto& slot = slots[it->slot];
            slot.record.reset();
            if (slot.generation == std::numeric_limits<std::uint64_t>::max()) {
                slot.retired = true;
            } else {
                ++slot.generation;
                available_slots.push_back(it->slot);
            }
            it = deferred.erase(it);
        }
    }

    bool retain(BufferId id) noexcept {
        std::lock_guard lock(mutex);
        auto* record = find(id);
        if (record == nullptr || record->pending_free || record->reference_count == UINT32_MAX) return false;
        ++record->reference_count;
        return true;
    }

    void release(BufferId id) noexcept {
        std::lock_guard lock(mutex);
        auto* record = find(id);
        if (record == nullptr || record->reference_count == 0) {
            assert(false && "buffer reference count invariant violated");
            return;
        }
        --record->reference_count;
        if (record->reference_count == 0) {
            record->pending_free = true;
            deferred.push_back(id);
            collect_ready();
        }
    }

    Status mark_used(BufferId id, std::uint64_t value) {
        std::lock_guard lock(mutex);
        auto* record = find(id);
        if (record == nullptr || record->pending_free) {
            assert(false && "stale buffer generation used by a command");
            return Error(ErrorCode::StaleHandle, "buffer generation is stale or pending reclamation");
        }
        if (value < record->last_use_timeline_value) {
            return Error(ErrorCode::InvalidArgument, "buffer last-use timeline value must be monotonic");
        }
        record->last_use_timeline_value = value;
        return ok_status();
    }

    void unmap(BufferId id) noexcept {
        std::lock_guard lock(mutex);
        auto* record = find(id);
        if (record == nullptr || record->pin_count == 0) {
            assert(false && "mapped buffer pin invariant violated");
            return;
        }
        --record->pin_count;
        collect_ready();
    }
};

}  // namespace kora::detail

namespace kora {
namespace {

Result<BufferInfo> inspect_locked(const detail::BufferManagerState& state, BufferId id) {
    const auto* record = state.find(id);
    if (record == nullptr) return Error(ErrorCode::StaleHandle, "buffer generation is stale");
    return BufferInfo{record->size_bytes,      record->usage,       record->residency,
                      record->pin_count,       id.generation,       record->last_use_timeline_value,
                      record->reference_count, record->pending_free};
}

}  // namespace

MappedBytes::MappedBytes(std::shared_ptr<detail::BufferManagerState> state, BufferId id, std::byte* data,
                         std::size_t size)
    : state_(std::move(state)), id_(id), data_(data), size_(size) {}

MappedBytes::~MappedBytes() {
    reset();
}

MappedBytes::MappedBytes(MappedBytes&& other) noexcept
    : state_(std::move(other.state_)),
      id_(other.id_),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedBytes& MappedBytes::operator=(MappedBytes&& other) noexcept {
    if (this == &other) return *this;
    reset();
    state_ = std::move(other.state_);
    id_ = other.id_;
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    return *this;
}

void MappedBytes::reset() noexcept {
    if (state_ != nullptr && data_ != nullptr) state_->unmap(id_);
    state_.reset();
    data_ = nullptr;
    size_ = 0;
}

BufferHandle::BufferHandle(std::shared_ptr<detail::BufferManagerState> state, BufferId id)
    : state_(std::move(state)), id_(id) {}

BufferHandle::~BufferHandle() {
    reset();
}

BufferHandle::BufferHandle(const BufferHandle& other) : state_(other.state_), id_(other.id_) {
    if (state_ != nullptr && !state_->retain(id_)) {
        assert(false && "copying a stale buffer handle");
        state_.reset();
    }
}

BufferHandle& BufferHandle::operator=(const BufferHandle& other) {
    if (this == &other) return *this;
    reset();
    state_ = other.state_;
    id_ = other.id_;
    if (state_ != nullptr && !state_->retain(id_)) {
        assert(false && "copying a stale buffer handle");
        state_.reset();
    }
    return *this;
}

BufferHandle::BufferHandle(BufferHandle&& other) noexcept : state_(std::move(other.state_)), id_(other.id_) {}

BufferHandle& BufferHandle::operator=(BufferHandle&& other) noexcept {
    if (this == &other) return *this;
    reset();
    state_ = std::move(other.state_);
    id_ = other.id_;
    return *this;
}

void BufferHandle::reset() noexcept {
    if (state_ != nullptr) state_->release(id_);
    state_.reset();
}

Result<BufferInfo> BufferHandle::info() const {
    if (state_ == nullptr) return Error(ErrorCode::StaleHandle, "buffer handle is empty");
    std::lock_guard lock(state_->mutex);
    return inspect_locked(*state_, id_);
}

Result<MappedBytes> BufferHandle::map(MapAccess access, std::uint64_t offset, std::uint64_t size) const {
    if (state_ == nullptr) return Error(ErrorCode::StaleHandle, "buffer handle is empty");
    std::lock_guard lock(state_->mutex);
    auto* record = state_->find(id_);
    if (record == nullptr || record->pending_free) {
        return Error(ErrorCode::StaleHandle, "buffer generation is stale or pending reclamation");
    }
    if (offset > record->size_bytes)
        return Error(ErrorCode::InvalidArgument, "buffer map offset exceeds size");
    if (size == std::numeric_limits<std::uint64_t>::max()) size = record->size_bytes - offset;
    if (!checked::in_range(offset, size, record->size_bytes)) {
        return Error(ErrorCode::InvalidArgument, "buffer map range exceeds buffer size");
    }
    if (offset > std::numeric_limits<std::size_t>::max() || size > std::numeric_limits<std::size_t>::max()) {
        return Error(ErrorCode::Overflow, "buffer map range exceeds host addressable size");
    }
    if (record->pin_count == UINT32_MAX) return Error(ErrorCode::Overflow, "buffer pin count overflow");
    if (access != MapAccess::Read && size != 0) {
        try {
            record->dirty_ranges.push_back({offset, size});
        } catch (const std::bad_alloc&) {
            return Error(ErrorCode::AllocationFailed, "unable to record the mapped buffer dirty range");
        } catch (const std::length_error&) {
            return Error(ErrorCode::Overflow, "buffer dirty-range metadata exceeds container capacity");
        }
    }
    ++record->pin_count;
    auto* data = record->host_shadow.data() + static_cast<std::size_t>(offset);
    return MappedBytes(state_, id_, data, static_cast<std::size_t>(size));
}

Status BufferHandle::mark_used(std::uint64_t timeline_value) const {
    if (state_ == nullptr) return Error(ErrorCode::StaleHandle, "buffer handle is empty");
    return state_->mark_used(id_, timeline_value);
}

BufferManager::BufferManager() : state_(std::make_shared<detail::BufferManagerState>()) {}
BufferManager::~BufferManager() = default;
BufferManager::BufferManager(BufferManager&&) noexcept = default;
BufferManager& BufferManager::operator=(BufferManager&&) noexcept = default;

Result<BufferHandle> BufferManager::create(std::uint64_t size_bytes, BufferUsage usage) {
    if (state_ == nullptr) return Error(ErrorCode::InvalidArgument, "buffer manager is moved from");
    if (size_bytes == 0) return Error(ErrorCode::InvalidArgument, "buffer size must be greater than zero");
    if (size_bytes > std::numeric_limits<std::size_t>::max()) {
        return Error(ErrorCode::Overflow, "buffer size exceeds host addressable size");
    }

    std::unique_ptr<detail::BufferRecord> record;
    try {
        record = std::make_unique<detail::BufferRecord>();
        record->host_shadow.resize(static_cast<std::size_t>(size_bytes));
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to allocate buffer host shadow");
    } catch (const std::length_error&) {
        return Error(ErrorCode::Overflow, "buffer host shadow exceeds container maximum size");
    }
    record->size_bytes = size_bytes;
    record->usage = usage;
    record->reference_count = 1;

    std::lock_guard lock(state_->mutex);
    std::uint32_t index = 0;
    if (!state_->available_slots.empty()) {
        index = state_->available_slots.back();
        state_->available_slots.pop_back();
    } else {
        if (state_->slots.size() >= UINT32_MAX)
            return Error(ErrorCode::Overflow, "buffer slot index overflow");
        index = static_cast<std::uint32_t>(state_->slots.size());
        try {
            state_->available_slots.reserve(state_->slots.size() + 1);
            state_->deferred.reserve(state_->slots.size() + 1);
            state_->slots.emplace_back();
        } catch (const std::bad_alloc&) {
            return Error(ErrorCode::AllocationFailed, "unable to allocate buffer registry metadata");
        } catch (const std::length_error&) {
            return Error(ErrorCode::Overflow, "buffer registry exceeds container maximum size");
        }
    }
    auto& slot = state_->slots[index];
    assert(slot.record == nullptr && !slot.retired);
    const BufferId id{index, slot.generation};
    slot.record = std::move(record);
    return BufferHandle(state_, id);
}

Result<BufferInfo> BufferManager::inspect(BufferId id) const {
    if (state_ == nullptr) return Error(ErrorCode::InvalidArgument, "buffer manager is moved from");
    std::lock_guard lock(state_->mutex);
    return inspect_locked(*state_, id);
}

Result<std::vector<DirtyRange>> BufferManager::dirty_ranges(BufferId id) const {
    if (state_ == nullptr) return Error(ErrorCode::InvalidArgument, "buffer manager is moved from");
    std::lock_guard lock(state_->mutex);
    const auto* record = state_->find(id);
    if (record == nullptr || record->pending_free) {
        return Error(ErrorCode::StaleHandle, "buffer generation is stale or pending reclamation");
    }
    try {
        return record->dirty_ranges;
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::AllocationFailed, "unable to copy buffer dirty-range metadata");
    }
}

Status BufferManager::mark_used(BufferId id, std::uint64_t timeline_value) {
    if (state_ == nullptr) return Error(ErrorCode::InvalidArgument, "buffer manager is moved from");
    return state_->mark_used(id, timeline_value);
}

Result<std::size_t> BufferManager::advance_completion(std::uint64_t completed_timeline_value) {
    if (state_ == nullptr) return Error(ErrorCode::InvalidArgument, "buffer manager is moved from");
    std::lock_guard lock(state_->mutex);
    if (completed_timeline_value < state_->completed_timeline_value) {
        return Error(ErrorCode::InvalidArgument, "completed timeline value must be monotonic");
    }
    const auto before = state_->deferred.size();
    state_->completed_timeline_value = completed_timeline_value;
    state_->collect_ready();
    return before - state_->deferred.size();
}

std::size_t BufferManager::deferred_count() const noexcept {
    if (state_ == nullptr) return 0;
    std::lock_guard lock(state_->mutex);
    return state_->deferred.size();
}

}  // namespace kora
