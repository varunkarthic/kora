# Duplicate-result comparators

`kora/verification.hpp` provides `VerificationPolicy` for comparing floating-point result spans. Use
`DispatchBuilder::verify_with()` to opt into duplicate execution and output comparison across all
initialized, non-quarantined devices in the Runtime. A custom comparator can compare arbitrary output
bytes, including integer buffers.

```cpp
#include <array>
#include <kora/verification.hpp>

const std::array<float, 3> first{1.0F, 2.0F, 3.0F};
const std::array<float, 3> second{1.0F, 2.0001F, 3.0F};
kora::VerificationPolicy policy;
policy.mode = kora::VerificationMode::AbsoluteTolerance;
policy.tolerance = 0.001;
auto matches = policy.compare<float>(first, second);
if (!matches) {
    // Invalid policy configuration; inspect matches.error().
} else if (!matches.value()) {
    // The duplicate result differs under this policy.
}
```

The available policies are exact numeric equality, absolute error, relative error, ULP distance for
IEEE-754 `float`/`double`, or a custom callback. Relative error uses
`abs(expected - actual) <= tolerance * max(abs(expected), abs(actual))`; the absolute and relative
tolerance must be finite and nonnegative. Unequal non-finite values do not match under relative mode.
ULP comparison treats equal values (including signed zero) as equal and rejects NaNs. Different span
lengths compare unequal. A custom comparator receives each complete span as bytes and must be set when
`mode == Custom`. A callback exception becomes `InternalInvariantViolation`; invalid comparator
configuration returns `InvalidArgument`.

For built-in modes, `verify_with(policy, VerificationDataType::Float32)` is the default; select
`Float64` for double-precision output. The output byte size must be a multiple of the selected type.
Custom mode is type-agnostic. The Runtime runs the same kernel, inputs, push constants, and grid on every
eligible initialized device, keeps output snapshots private until all devices finish, and compares
each writable buffer against the first participant. Matching results publish the first snapshot to the
host shadow. A mismatch returns `ValidationError`, marks writable outputs `DataLoss`, and quarantines
all participating devices because KORA cannot infer which device is correct. Quarantine lasts for that
Runtime's lifetime; create a new Runtime to retry on those devices. At least two eligible devices must
support the kernel or the verified submission fails without dispatch.
