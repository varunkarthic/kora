#pragma once
/// @file errors.hpp
/// Structured error model. Expected failures use Result<T>/Status; the throwing API layer and the
/// Python bindings wrap an Error in kora::Exception.
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace kora {

enum class ErrorCode : int {
    Ok = 0,
    NoVulkanImplementation,
    NoEligibleDevice,
    NoComputeQueue,
    UnsupportedFeature,
    InvalidKernel,
    InvalidKernelBinding,
    InvalidDispatch,
    AllocationFailed,
    OutOfMemoryBudget,
    TransferFailed,
    Timeout,
    DeviceLost,
    PipelineCreationFailed,
    SynchronizationError,
    ValidationError,
    PythonBindingError,
    InternalInvariantViolation,
    // Additions beyond the original list:
    InvalidArgument,  ///< Bad user input (null, out-of-range, mismatched sizes).
    Overflow,         ///< Checked arithmetic overflow.
    GraphCycle,       ///< Task graph contains a cycle.
    Cancelled,        ///< Task cancelled by shutdown or user.
    StaleHandle,      ///< Buffer generation mismatch (use after free).
    DataLoss,         ///< Authoritative data existed only on a lost device.
    IoError,          ///< Persistent cache / file failure.
    kCount_
};

/// Stable name such as "DeviceLost".
const char* to_string(ErrorCode c) noexcept;

struct Error {
    ErrorCode code = ErrorCode::Ok;
    std::string message;
    std::int32_t vk_result = 0;  ///< VkResult when relevant (0 == VK_SUCCESS).
    std::string device;          ///< Device identity (name/uuid) when relevant.
    std::uint64_t task_id = 0;   ///< 0 == none.
    std::string kernel_id;
    std::string recovery_action;  ///< Recovery attempted, if any.
    std::shared_ptr<const Error> cause;

    Error() = default;
    Error(ErrorCode c, std::string msg) : code(c), message(std::move(msg)) {}

    Error& with_vk(std::int32_t r) {
        vk_result = r;
        return *this;
    }
    Error& with_device(std::string d) {
        device = std::move(d);
        return *this;
    }
    Error& with_task(std::uint64_t t) {
        task_id = t;
        return *this;
    }
    Error& with_kernel(std::string k) {
        kernel_id = std::move(k);
        return *this;
    }
    Error& with_recovery(std::string r) {
        recovery_action = std::move(r);
        return *this;
    }
    Error& with_cause(Error c) {
        cause = std::make_shared<const Error>(std::move(c));
        return *this;
    }

    /// "Code: message [vk=..] [device=..] ... <- cause" (single line).
    std::string describe() const;
};

/// Exception carrying an Error; thrown only by the throwing API layer.
class Exception : public std::runtime_error {
public:
    explicit Exception(Error e) : std::runtime_error(e.describe()), error_(std::move(e)) {}
    const Error& error() const noexcept { return error_; }

private:
    Error error_;
};

/// Value-or-Error. Expected failures never throw.
template <class T>
class [[nodiscard]] Result {
public:
    Result(T v) : v_(std::move(v)) {}      // NOLINT: implicit by design
    Result(Error e) : v_(std::move(e)) {}  // NOLINT: implicit by design

    bool ok() const noexcept { return v_.index() == 0; }
    explicit operator bool() const noexcept { return ok(); }

    T& value() & { return std::get<0>(v_); }
    const T& value() const& { return std::get<0>(v_); }
    T&& value() && { return std::get<0>(std::move(v_)); }
    const Error& error() const& { return std::get<1>(v_); }
    Error&& error() && { return std::get<1>(std::move(v_)); }

    /// Returns the value or throws Exception.
    T& value_or_throw() & {
        if (!ok()) throw Exception(error());
        return value();
    }
    T value_or_throw() && {
        if (!ok()) throw Exception(std::move(*this).error());
        return std::move(*this).value();
    }

    template <class F>
    auto map(F&& f) && -> Result<decltype(f(std::declval<T>()))> {
        if (!ok()) return std::move(*this).error();
        return f(std::move(*this).value());
    }
    template <class F>
    auto and_then(F&& f) && -> decltype(f(std::declval<T>())) {
        if (!ok()) return std::move(*this).error();
        return f(std::move(*this).value());
    }

private:
    std::variant<T, Error> v_;
};

/// Result<void>.
template <>
class [[nodiscard]] Result<void> {
public:
    Result() = default;
    Result(Error e) : err_(std::move(e)) {}  // NOLINT: implicit by design
    bool ok() const noexcept { return err_.code == ErrorCode::Ok; }
    explicit operator bool() const noexcept { return ok(); }
    const Error& error() const& { return err_; }
    Error&& error() && { return std::move(err_); }
    void value_or_throw() const {
        if (!ok()) throw Exception(err_);
    }

private:
    Error err_;
};
using Status = Result<void>;

inline Status ok_status() {
    return Status{};
}

}  // namespace kora

/// Propagate an error: `KORA_TRY(x, compute());` declares `x` or returns the error.
#define KORA_TRY(var, expr)                                           \
    auto var##_result_ = (expr);                                      \
    if (!var##_result_.ok()) return std::move(var##_result_).error(); \
    auto var = std::move(var##_result_).value()

/// Propagate a Status error.
#define KORA_TRY_STATUS(expr)                       \
    do {                                            \
        auto s_ = (expr);                           \
        if (!s_.ok()) return std::move(s_).error(); \
    } while (0)
