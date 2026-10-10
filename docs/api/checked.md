# Checked arithmetic

Header: `include/kora/detail/checked.hpp`

Checked arithmetic protects byte sizes, offsets, alignments, staging ranges, and allocation
calculations from unsigned overflow.

`add(a, b)` and `mul(a, b)` return `std::optional<T>`. An empty result means overflow.
`is_pow2(value)` recognizes non-zero powers of two. `align_up(value, alignment)` returns empty for a
zero/non-power-of-two alignment or overflow. `align_down` is valid only after the caller establishes
a non-zero power-of-two alignment. `in_range(offset, size, total)` checks a half-open range without
overflowing `offset + size`.

`add_r` and `mul_r` adapt failures to `Result<std::uint64_t>` with `ErrorCode::Overflow` and caller
context.

Higher-level helpers cover the size calculations used at API boundaries:

- `byte_size(count, element_size, context)` and `byte_size<T>(count, context)`;
- `volume(width, height, depth, context)`;
- `range_end(offset, size, capacity, context)`;
- `align_up_r(value, alignment, context)`.

See {doc}`../internals/checked-arithmetic` for their exact failure semantics and usage rules.

This header is under `detail` and is not a stable public application API. It is documented because
its invariants protect security- and correctness-sensitive internal calculations.
