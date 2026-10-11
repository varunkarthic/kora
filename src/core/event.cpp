#include "kora/event.hpp"

#include <optional>
#include <vector>

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

Status Event::on_complete(std::function<void(Status)> callback) const {
    if (!state_) return Error(ErrorCode::InvalidArgument, "Cannot register on an empty Event");
    if (!callback) return Error(ErrorCode::InvalidArgument, "Event callback must be callable");
    std::optional<Status> ready;
    {
        std::lock_guard lock(state_->mutex);
        if (state_->completed)
            ready = state_->completion_status;
        else
            state_->callbacks.push_back(std::move(callback));
    }
    if (ready) {
        try {
            callback(std::move(*ready));
        } catch (...) {
            return Error(ErrorCode::InternalInvariantViolation, "Event completion callback threw");
        }
    }
    return ok_status();
}

Result<Event> Event::then(std::function<Status(Status)> callback) const {
    if (!state_) return Error(ErrorCode::InvalidArgument, "Cannot chain from an empty Event");
    if (!callback) return Error(ErrorCode::InvalidArgument, "Event continuation must be callable");
    auto promise = std::make_shared<detail::EventPromise>();
    auto chained = promise->event();
    auto registered = on_complete([promise, callback = std::move(callback)](Status previous) mutable {
        try {
            promise->complete(callback(std::move(previous)));
        } catch (...) {
            promise->complete(Error(ErrorCode::InternalInvariantViolation, "Event continuation threw"));
        }
    });
    if (!registered) return std::move(registered).error();
    return chained;
}

Result<Event> Event::when_all(std::span<const Event> events) {
    for (const auto& event : events)
        if (!event.valid()) return Error(ErrorCode::InvalidArgument, "when_all received an empty Event");
    auto promise = std::make_shared<detail::EventPromise>();
    auto combined = promise->event();
    struct Aggregate {
        std::mutex mutex;
        std::size_t remaining = 0;
        std::vector<Status> statuses;
    };
    auto aggregate = std::make_shared<Aggregate>();
    aggregate->remaining = events.size();
    aggregate->statuses.resize(events.size());
    if (events.empty()) {
        promise->complete(ok_status());
        return combined;
    }
    for (std::size_t index = 0; index < events.size(); ++index) {
        auto registered = events[index].on_complete([aggregate, promise, index](Status status) {
            bool complete = false;
            {
                std::lock_guard lock(aggregate->mutex);
                aggregate->statuses[index] = std::move(status);
                complete = --aggregate->remaining == 0;
            }
            if (!complete) return;
            for (const auto& result : aggregate->statuses) {
                if (!result) {
                    promise->complete(result);
                    return;
                }
            }
            promise->complete(ok_status());
        });
        if (!registered) return std::move(registered).error();
    }
    return combined;
}

}  // namespace kora

namespace kora::detail {

EventPromise::EventPromise() : state_(std::make_shared<EventState>()) {}

Event EventPromise::event() const {
    return Event(state_);
}

void EventPromise::complete(Status status) const {
    std::vector<std::function<void(Status)>> callbacks;
    Status callback_status;
    {
        std::lock_guard lock(state_->mutex);
        if (state_->completed) return;
        state_->completion_status = status;
        callback_status = status;
        state_->completed = true;
        callbacks.swap(state_->callbacks);
    }
    state_->changed.notify_all();
    for (auto& callback : callbacks) {
        try {
            callback(callback_status);
        } catch (...) {
            // Completion callbacks are isolated so one consumer cannot prevent other waiters.
        }
    }
}

}  // namespace kora::detail
