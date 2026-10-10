# Errors and results

KORA represents expected runtime failures as values. Native APIs return `Result<T>` when successful
operations produce a value and `Status` when they do not. Exceptions are reserved for explicitly
throwing API boundaries and language bindings.

## Error codes

`ErrorCode` is the stable machine-readable category. It distinguishes environment and capability
problems, invalid inputs and kernels, memory and transfer failures, synchronization and device
loss, cancellation, data loss, I/O failures, and internal invariant violations. Call
`to_string(code)` for the stable enumerator name used by diagnostics.

`ErrorCode::Ok` represents success only inside a default `Status`. Functions returning
`Result<T>` construct success from a `T`, not from an `Error` with code `Ok`.

## Error context

`Error` contains a code and human-readable message plus optional Vulkan result, device identity,
task ID, kernel ID, recovery action, and nested cause. The `with_vk`, `with_device`, `with_task`,
`with_kernel`, `with_recovery`, and `with_cause` methods add context and return the same object for
chaining.

`describe()` produces a single-line diagnostic. Context fields appear in brackets and nested causes
follow `<-`, preserving both the high-level failure and its underlying cause. Do not parse this text
for control flow; inspect `Error::code` and structured fields instead.

## `Result<T>`

A `Result<T>` contains either a value or an `Error`:

```cpp
kora::Result<int> result = compute();
if (!result) {
    return result.error();
}
int value = result.value();
```

`ok()` and explicit Boolean conversion report the active alternative. Call `value()` only after a
successful check and `error()` only after a failed check. These accessors follow `std::variant`
semantics when used on the wrong alternative.

`map(function)` transforms a successful value and propagates an error unchanged. `and_then(function)`
chains another result-returning operation. Both consume the source result so move-only values and
errors can propagate without an extra copy.

`value_or_throw()` returns the value or throws `kora::Exception`. Use it only at a deliberately
throwing API boundary, not inside ordinary runtime control flow.

## `Status`

`Status` is `Result<void>`. A default-constructed status and `ok_status()` mean success; constructing
from `Error` means failure. `throw_if_error(status)` returns normally for success and throws
`kora::Exception` containing the original error for failure.

## Propagation macros

`KORA_TRY(name, expression)` evaluates a `Result<T>`, returns its error from the current function on
failure, and declares `name` with the successful value. `KORA_TRY_STATUS(expression)` performs the
same propagation for `Status`.

Expressions passed to either macro are evaluated exactly once. Use each macro as a standalone
statement, and do not reuse its generated name in the same scope.

## Exceptions

`kora::Exception` derives from `std::runtime_error`, owns the structured `Error`, and exposes it
through `error()`. Its `what()` string is the error's `describe()` output. Catch `Exception` by
reference when a throwing boundary must recover structured context.

## Invariants

- Expected failures travel through `Result<T>` or `Status`; internal code does not use exceptions
  for normal error handling.
- Every failure has a non-`Ok` code and a useful message.
- Backend failures preserve their original Vulkan result when one exists.
- Recovery layers add context or a nested cause instead of replacing the original failure.
- `ErrorCode` names remain stable because logs, tests, and bindings expose them.
- `kCount_` is an internal sentinel and is never returned to users.
