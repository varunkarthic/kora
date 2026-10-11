#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "kora/event.hpp"

namespace kora::detail {

class EventState final {
public:
    mutable std::mutex mutex;
    mutable std::condition_variable changed;
    bool completed = false;
    Status completion_status = Error(ErrorCode::Timeout, "Event is pending");
    std::vector<std::function<void(Status)>> callbacks;
};

class EventPromise final {
public:
    EventPromise();
    Event event() const;
    void complete(Status status) const;

private:
    std::shared_ptr<EventState> state_;
};

}  // namespace kora::detail
