#pragma once
/// @file event.hpp
/// Copyable completion handle for asynchronous runtime work.

#include <chrono>
#include <memory>
#include <span>
#include <utility>

#include "kora/errors.hpp"

namespace kora {
namespace detail {
class EventPromise;
class EventState;
}  // namespace detail

/// Shared asynchronous completion state. A timed-out wait does not cancel or mutate the operation.
class Event final {
public:
    Event() = default;

    bool valid() const noexcept { return state_ != nullptr; }
    bool done() const noexcept;
    /// Blocks until completion and returns its success or error status.
    Status wait() const;
    /// Returns Timeout if the deadline expires; the work continues unchanged.
    Status wait_for(std::chrono::nanoseconds timeout) const;
    /// Returns Timeout while pending, otherwise the operation's completion status.
    Status status() const;

private:
    friend class detail::EventPromise;
    explicit Event(std::shared_ptr<detail::EventState> state) : state_(std::move(state)) {}
    std::shared_ptr<detail::EventState> state_;
};

}  // namespace kora
