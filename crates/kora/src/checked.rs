//! Overflow-safe size arithmetic.
//!
//! Every public size calculation (`count * size`, `w * h * d`, `offset + size`, alignment padding)
//! goes through these helpers so overflow becomes an [`ErrorCode::Overflow`] error before any
//! Vulkan call.

use crate::error::{Error, ErrorCode, Result};

/// `true` when `x` is a non-zero power of two.
pub const fn is_pow2(x: u64) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

/// Rounds `value` up to a multiple of the power-of-two `align`. `None` on overflow or bad alignment.
pub const fn align_up(value: u64, align: u64) -> Option<u64> {
    if !is_pow2(align) {
        return None;
    }
    match value.checked_add(align - 1) {
        Some(sum) => Some(sum & !(align - 1)),
        None => None,
    }
}

/// Rounds `value` down to a multiple of `align`. `None` for zero or non-power-of-two alignment.
pub const fn align_down(value: u64, align: u64) -> Option<u64> {
    if !is_pow2(align) {
        return None;
    }
    Some(value & !(align - 1))
}

/// `true` iff `[offset, offset + size)` lies inside `[0, capacity)` without overflow.
pub const fn in_range(offset: u64, size: u64, capacity: u64) -> bool {
    match offset.checked_add(size) {
        Some(end) => end <= capacity,
        None => false,
    }
}

fn overflow(what: &str) -> Error {
    Error::new(ErrorCode::Overflow, format!("size overflow computing {what}"))
}

/// `a * b` with a contextual overflow error.
pub fn mul(a: u64, b: u64, what: &str) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| overflow(what))
}

/// `a + b` with a contextual overflow error.
pub fn add(a: u64, b: u64, what: &str) -> Result<u64> {
    a.checked_add(b).ok_or_else(|| overflow(what))
}

/// `count * element_size` with a contextual overflow error.
pub fn byte_size(count: u64, element_size: u64, what: &str) -> Result<u64> {
    mul(count, element_size, what)
}

/// `count * size_of::<T>()` with a contextual overflow error.
pub fn byte_size_of<T>(count: u64, what: &str) -> Result<u64> {
    byte_size(count, std::mem::size_of::<T>() as u64, what)
}

/// `width * height * depth`, checking overflow after each multiplication.
pub fn volume(width: u64, height: u64, depth: u64, what: &str) -> Result<u64> {
    let area = width.checked_mul(height).ok_or_else(|| overflow(what))?;
    mul(area, depth, what)
}

/// Returns `offset + size` when the half-open range fits inside `capacity`.
pub fn range_end(offset: u64, size: u64, capacity: u64, what: &str) -> Result<u64> {
    let end = offset.checked_add(size).ok_or_else(|| overflow(what))?;
    if end > capacity {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            format!("range exceeds capacity computing {what}"),
        ));
    }
    Ok(end)
}

/// Result-returning alignment for API boundaries.
pub fn align_up_checked(value: u64, alignment: u64, what: &str) -> Result<u64> {
    if !is_pow2(alignment) {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            format!("alignment must be a non-zero power of two computing {what}"),
        ));
    }
    align_up(value, alignment).ok_or_else(|| overflow(what))
}

/// Converts a `u64` byte count to `usize`, failing on 32-bit hosts when it does not fit.
pub fn to_usize(value: u64, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| overflow(what))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const MAX: u64 = u64::MAX;

    #[test]
    fn add_and_mul_boundaries() {
        assert_eq!(add(1, 2, "x").unwrap(), 3);
        assert_eq!(add(MAX, 0, "x").unwrap(), MAX);
        assert_eq!(add(MAX, 1, "off+size").unwrap_err().code(), ErrorCode::Overflow);
        assert!(add(MAX / 2 + 1, MAX / 2 + 1, "x").is_err());
        assert_eq!(mul(0, MAX, "x").unwrap(), 0);
        assert_eq!(mul(MAX, 1, "x").unwrap(), MAX);
        assert_eq!(mul(MAX, 2, "count*size").unwrap_err().code(), ErrorCode::Overflow);
        assert!(mul(1 << 32, 1 << 32, "x").is_err());
        assert_eq!(mul(1 << 31, 1 << 32, "x").unwrap(), 1 << 63);
    }

    #[test]
    fn align_up_cases() {
        assert_eq!(align_up(0, 16), Some(0));
        assert_eq!(align_up(1, 16), Some(16));
        assert_eq!(align_up(16, 16), Some(16));
        assert_eq!(align_up(17, 16), Some(32));
        assert_eq!(align_up(1, 0), None);
        assert_eq!(align_up(1, 12), None);
        assert_eq!(align_up(MAX, 16), None);
        assert_eq!(align_up(MAX - 15, 16), Some(MAX - 15));
    }

    #[test]
    fn align_down_and_pow2() {
        assert_eq!(align_down(31, 16), Some(16));
        assert_eq!(align_down(31, 0), None);
        assert_eq!(align_down(31, 6), None);
        assert!(is_pow2(1));
        assert!(!is_pow2(0));
        assert!(!is_pow2(6));
    }

    #[test]
    fn in_range_cases() {
        assert!(in_range(0, 10, 10));
        assert!(!in_range(1, 10, 10));
        assert!(!in_range(MAX, 2, MAX));
        assert!(in_range(10, 0, 10));
    }

    #[test]
    fn byte_size_and_volume() {
        assert_eq!(byte_size(4, 8, "elements").unwrap(), 32);
        assert_eq!(byte_size_of::<u32>(4, "u32 elements").unwrap(), 16);
        assert_eq!(byte_size(0, MAX, "empty").unwrap(), 0);
        assert_eq!(
            byte_size(MAX, 2, "elements").unwrap_err().code(),
            ErrorCode::Overflow
        );
        assert_eq!(volume(2, 3, 4, "extent").unwrap(), 24);
        assert_eq!(volume(0, MAX, MAX, "empty").unwrap(), 0);
        assert_eq!(volume(MAX, 2, 1, "w*h").unwrap_err().code(), ErrorCode::Overflow);
        assert_eq!(
            volume(MAX / 2, 2, 2, "area*depth").unwrap_err().code(),
            ErrorCode::Overflow
        );
    }

    #[test]
    fn range_end_cases() {
        assert_eq!(range_end(4, 6, 10, "upload").unwrap(), 10);
        assert_eq!(range_end(10, 0, 10, "empty tail").unwrap(), 10);
        assert_eq!(
            range_end(MAX, 1, MAX, "overflow").unwrap_err().code(),
            ErrorCode::Overflow
        );
        assert_eq!(
            range_end(4, 7, 10, "outside").unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn align_up_checked_cases() {
        assert_eq!(align_up_checked(17, 16, "offset").unwrap(), 32);
        assert_eq!(
            align_up_checked(17, 0, "offset").unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            align_up_checked(17, 12, "offset").unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            align_up_checked(MAX, 16, "offset").unwrap_err().code(),
            ErrorCode::Overflow
        );
    }

    #[test]
    fn error_message_names_the_computation() {
        let e = mul(MAX, 2, "typed buffer size").unwrap_err();
        assert_eq!(e.message(), "size overflow computing typed buffer size");
    }

    const _: () = assert!(is_pow2(64));
    const _: () = assert!(matches!(align_up(3, 4), Some(4)));

    proptest! {
        #[test]
        fn align_up_is_the_smallest_aligned_value_not_below_input(value in 0u64..u64::MAX / 2, shift in 0u32..32) {
            let align = 1u64 << shift;
            let aligned = align_up(value, align).unwrap();
            prop_assert!(aligned >= value);
            prop_assert_eq!(aligned % align, 0);
            prop_assert!(aligned - value < align);
            prop_assert_eq!(align_down(aligned, align), Some(aligned));
        }

        #[test]
        fn checked_mul_agrees_with_u128(a in any::<u64>(), b in any::<u64>()) {
            let wide = u128::from(a) * u128::from(b);
            match mul(a, b, "p") {
                Ok(v) => prop_assert_eq!(u128::from(v), wide),
                Err(_) => prop_assert!(wide > u128::from(u64::MAX)),
            }
        }

        #[test]
        fn in_range_agrees_with_u128(offset in any::<u64>(), size in any::<u64>(), capacity in any::<u64>()) {
            let end = u128::from(offset) + u128::from(size);
            prop_assert_eq!(in_range(offset, size, capacity), end <= u128::from(capacity));
        }
    }
}
