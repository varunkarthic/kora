#include "kora/event.hpp"

#include "core/event_state.hpp"


namespace kora {

bool Event::done() const noexcept {
    if (!state_) return false;
    std::lock_guard lock(state_->mutex);
    return state_->completed;
}

Status Event::wait() const {
    if (!state_) return Error(ErrorCode::InvalidArgument, "Cannot wait on an empty Event");
    std::unique_lock lock(state_->mutex);
    state_->changed.wait(lock, [this] { return state_->completed; });
    return state_->completion_status;
}

Status Event::wait_for(std::chrono::nanoseconds timeout) const {
    if (!state_) return Error(ErrorCode::InvalidArgument, "Cannot wait on an empty Event");
    std::unique_lock lock(state_->mutex);
    if (!state_->changed.wait_for(lock, timeout, [this] { return state_->completed; }))
        return Error(ErrorCode::Timeout, "Event wait timed out; the operation continues");
    return state_->completion_status;
}

Status Event::status() const {
    if (!state_) return Error(ErrorCode::InvalidArgument, "Empty Event has no completion status");
    std::lock_guard lock(state_->mutex);
    if (!state_->completed) return Error(ErrorCode::Timeout, "Event is pending");
    return state_->completion_status;
}

}  // namespace kora

namespace kora::detail {

EventPromise::EventPromise() : state_(std::make_shared<EventState>()) {}

Event EventPromise::event() const {
    return Event(state_);
}

void EventPromise::complete(Status status) const {
    {
        std::lock_guard lock(state_->mutex);
        if (state_->completed) return;
        state_->completion_status = std::move(status);
        state_->completed = true;
    }
    state_->changed.notify_all();
}

}  // namespace kora::detail
