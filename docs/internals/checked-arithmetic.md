# Checked arithmetic internals

Memory sizes and offsets often originate in application input but eventually reach allocators,
copies, and Vulkan structures. Unsigned wraparound can turn an oversized request into a small
allocation followed by an out-of-bounds access. KORA therefore treats size arithmetic as a checked
operation and reports failure before a backend call.

The implementation is header-only in `include/kora/detail/checked.hpp`. It is an internal contract,
not a stable application-facing API.

## Core invariants

- Arithmetic never relies on a wrapped result.
- Zero-sized ranges and allocations are valid mathematical inputs.
- Alignments are non-zero powers of two.
- Ranges are half-open: `[offset, offset + size)`.
- Overflow is distinct from a representable range that exceeds its capacity.
- Public API boundaries use `Result` helpers so errors retain operation context.

## Primitive operations

### `add(a, b)`

Returns `std::optional<T>` for unsigned integral `T` other than `bool`. An empty result means the
sum is greater than `numeric_limits<T>::max()`. GCC and Clang builds use overflow builtins at runtime;
constant evaluation and other compilers use the portable `max - a` test.

### `mul(a, b)`

Returns an optional product under the same type restriction. Zero multiplied by any representable
value is zero. Runtime GCC and Clang builds use an overflow builtin; the portable path checks
`max / a` only when `a` is non-zero.

### `is_pow2`, `align_up`, and `align_down`

`is_pow2` rejects zero. Both alignment functions reject zero and non-power-of-two alignments.
`align_up` also rejects overflow while adding `alignment - 1`. The optional-returning primitives
are suitable for internal composition; API boundaries should use `align_up_r`.

### `in_range`

Checks that `offset + size` is representable and no greater than `capacity`. An empty range at the
end of a capacity is valid.

## Result-returning helpers

| Helper | Successful value | `Overflow` | `InvalidArgument` |
| --- | --- | --- | --- |
| `add_r` | `a + b` | Sum is not representable | Never |
| `mul_r` | `a * b` | Product is not representable | Never |
| `byte_size` | `count * element_size` | Product is not representable | Never |
| `volume` | `width * height * depth` | Either multiplication overflows | Never |
| `range_end` | `offset + size` | Addition overflows | End exceeds capacity |
| `align_up_r` | Next aligned value | Rounding addition overflows | Alignment is invalid |

Every helper accepts a short context string. Errors include it so callers can distinguish, for
example, an upload range from an image extent without reconstructing unchecked operands in a log.
Do not put credentials, private paths, or buffer contents in the context.

## Usage patterns

Allocate storage for typed elements:

```cpp
auto bytes = kora::checked::byte_size<float>(element_count, "tensor elements");
if (!bytes) return bytes.error();
```

Validate a copy before passing it to a backend:

```cpp
auto end = kora::checked::range_end(offset, size, allocation_size, "upload range");
if (!end) return end.error();
```

Compute a three-dimensional element count and then its byte size as separate checked steps:

```cpp
auto elements = kora::checked::volume(width, height, depth, "image extent");
if (!elements) return elements.error();
auto bytes = kora::checked::byte_size<std::uint32_t>(elements.value(), "image bytes");
if (!bytes) return bytes.error();
```

## Review checklist

When adding code that handles sizes, review every addition, multiplication, alignment, and range
boundary. Route any value derived from input, device limits, or allocation metadata through these
helpers. Plain arithmetic remains appropriate for loop counters or strings only when it cannot
influence a memory, dispatch, transfer, or backend size.

Tests must cover zero, exact maximum values, one-past-maximum failures, invalid alignments, an empty
range at capacity, and both stages of chained multiplication. Keep compile-time assertions for the
portable constexpr path and runtime tests for the compiler-builtin path.
