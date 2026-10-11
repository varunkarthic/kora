# Duplicate-result comparators

`kora/verification.hpp` provides `VerificationPolicy` for comparing floating-point result spans. It is
the comparison layer used by duplicate-result verification; it does not dispatch kernels itself.
Duplicate execution and device quarantine remain a Runtime policy and are opt-in separately.

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
tolerance must be finite and nonnegative. ULP comparison treats equal values (including signed zero)
as equal and rejects NaNs. Different span lengths compare unequal. A custom comparator receives each
complete span as bytes and must be set when `mode == Custom`. A callback exception becomes
`InternalInvariantViolation`; invalid comparator configuration returns `InvalidArgument`.
