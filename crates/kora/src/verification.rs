//! Comparator policies for checking duplicate floating-point results.

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::error::{Error, ErrorCode, Result, Status};

/// Element type of verified outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VerificationDataType {
    /// `f32` elements.
    #[default]
    Float32,
    /// `f64` elements.
    Float64,
}

/// A custom comparator over the complete expected/actual byte spans.
pub type CustomComparator = Arc<dyn Fn(&[u8], &[u8]) -> bool + Send + Sync>;

/// How duplicate numeric outputs are compared.
#[derive(Clone, Default)]
pub enum VerificationPolicy {
    /// Bitwise-equal values (`+0 == -0`; `NaN` never matches).
    #[default]
    Exact,
    /// `|a - b| <= tolerance`.
    AbsoluteTolerance(f64),
    /// `|a - b| <= tolerance * max(|a|, |b|)`; non-finite values must be equal.
    RelativeTolerance(f64),
    /// At most this many representable values apart; `NaN` never matches.
    Ulp(u64),
    /// Caller-defined comparison of complete byte spans. A panic becomes an error.
    Custom(CustomComparator),
}

impl fmt::Debug for VerificationPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact => f.write_str("Exact"),
            Self::AbsoluteTolerance(t) => write!(f, "AbsoluteTolerance({t})"),
            Self::RelativeTolerance(t) => write!(f, "RelativeTolerance({t})"),
            Self::Ulp(n) => write!(f, "Ulp({n})"),
            Self::Custom(_) => f.write_str("Custom(..)"),
        }
    }
}

/// Floating-point element types supported by verification.
pub trait VerifyFloat: Copy + PartialEq + Send + Sync + 'static {
    /// Size in bytes.
    const SIZE: usize;
    /// Widening conversion.
    fn to_f64(self) -> f64;
    /// `true` for finite values.
    fn is_finite(self) -> bool;
    /// `true` for NaN.
    fn is_nan(self) -> bool;
    /// Monotone integer ordering of the IEEE representation.
    fn ordered_bits(self) -> u64;
    /// Decodes one native-endian element.
    fn from_ne(bytes: &[u8]) -> Self;
}

impl VerifyFloat for f32 {
    const SIZE: usize = 4;
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    fn ordered_bits(self) -> u64 {
        let bits = self.to_bits();
        u64::from(if bits & 0x8000_0000 != 0 {
            (!bits).wrapping_add(1)
        } else {
            bits | 0x8000_0000
        })
    }
    fn from_ne(bytes: &[u8]) -> Self {
        f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl VerifyFloat for f64 {
    const SIZE: usize = 8;
    fn to_f64(self) -> f64 {
        self
    }
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    fn ordered_bits(self) -> u64 {
        let bits = self.to_bits();
        if bits & (1 << 63) != 0 {
            (!bits).wrapping_add(1)
        } else {
            bits | (1 << 63)
        }
    }
    fn from_ne(bytes: &[u8]) -> Self {
        let mut array = [0u8; 8];
        array.copy_from_slice(&bytes[..8]);
        f64::from_ne_bytes(array)
    }
}

fn within_ulps<T: VerifyFloat>(left: T, right: T, limit: u64) -> bool {
    if left.is_nan() || right.is_nan() {
        return false;
    }
    if left == right {
        return true;
    }
    left.ordered_bits().abs_diff(right.ordered_bits()) <= limit
}

fn as_bytes<T: VerifyFloat>(values: &[T]) -> &[u8] {
    // SAFETY: `T` is `f32` or `f64` (sealed by the trait's implementations): plain-old-data with no
    // padding, so viewing `size_of_val(values)` initialized bytes of the same allocation is valid.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

impl VerificationPolicy {
    /// Checks tolerances are finite and nonnegative.
    pub fn validate(&self) -> Status {
        match self {
            Self::AbsoluteTolerance(t) | Self::RelativeTolerance(t) if !(t.is_finite() && *t >= 0.0) => {
                Err(Error::new(
                    ErrorCode::InvalidArgument,
                    "verification tolerance must be finite and nonnegative",
                ))
            }
            _ => Ok(()),
        }
    }

    fn run_custom(custom: &CustomComparator, expected: &[u8], actual: &[u8]) -> Result<bool> {
        catch_unwind(AssertUnwindSafe(|| custom(expected, actual))).map_err(|_| {
            Error::new(
                ErrorCode::InternalInvariantViolation,
                "custom verification comparator panicked",
            )
        })
    }

    fn element_matches<T: VerifyFloat>(&self, left: T, right: T) -> bool {
        if left == right {
            return true;
        }
        match self {
            Self::Exact | Self::Custom(_) => false,
            Self::AbsoluteTolerance(t) => (left.to_f64() - right.to_f64()).abs() <= *t,
            Self::RelativeTolerance(t) => {
                if !left.is_finite() || !right.is_finite() {
                    return false;
                }
                let (l, r) = (left.to_f64(), right.to_f64());
                (l - r).abs() <= t * l.abs().max(r.abs())
            }
            Self::Ulp(limit) => within_ulps(left, right, *limit),
        }
    }

    /// Compares typed values. Unequal lengths compare unequal.
    pub fn compare<T: VerifyFloat>(&self, expected: &[T], actual: &[T]) -> Result<bool> {
        if expected.len() != actual.len() {
            return Ok(false);
        }
        self.validate()?;
        if let Self::Custom(custom) = self {
            return Self::run_custom(custom, as_bytes(expected), as_bytes(actual));
        }
        Ok(expected
            .iter()
            .zip(actual)
            .all(|(&l, &r)| self.element_matches(l, r)))
    }

    /// Compares raw native-endian byte spans interpreted as `data_type` elements.
    pub fn compare_bytes(
        &self,
        expected: &[u8],
        actual: &[u8],
        data_type: VerificationDataType,
    ) -> Result<bool> {
        if expected.len() != actual.len() {
            return Ok(false);
        }
        self.validate()?;
        if let Self::Custom(custom) = self {
            return Self::run_custom(custom, expected, actual);
        }
        match data_type {
            VerificationDataType::Float32 => self.compare_chunks::<f32>(expected, actual, "float32"),
            VerificationDataType::Float64 => self.compare_chunks::<f64>(expected, actual, "float64"),
        }
    }

    fn compare_chunks<T: VerifyFloat>(&self, expected: &[u8], actual: &[u8], name: &str) -> Result<bool> {
        if !expected.len().is_multiple_of(T::SIZE) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                format!("verification output size is not a {name} multiple"),
            ));
        }
        Ok(expected
            .chunks_exact(T::SIZE)
            .zip(actual.chunks_exact(T::SIZE))
            .all(|(l, r)| self.element_matches(T::from_ne(l), T::from_ne(r))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_up(x: f32) -> f32 {
        f32::from_bits(x.to_bits() + 1)
    }

    #[test]
    fn exact_and_tolerance_comparators_handle_floating_point_values() {
        let expected = [1.0f32, -2.0];
        let exact = [1.0f32, -2.0];
        let close = [1.0001f32, -2.0001];
        assert!(VerificationPolicy::Exact.compare(&expected, &exact).unwrap());
        assert!(!VerificationPolicy::Exact.compare(&expected, &close).unwrap());
        assert!(
            VerificationPolicy::AbsoluteTolerance(0.001)
                .compare(&expected, &close)
                .unwrap()
        );
        assert!(
            VerificationPolicy::RelativeTolerance(0.0002)
                .compare(&expected, &close)
                .unwrap()
        );
        assert!(VerificationPolicy::Exact.compare(&[0.0f32], &[-0.0f32]).unwrap());
    }

    #[test]
    fn ulp_comparator_counts_adjacent_values_and_rejects_nan() {
        let policy = VerificationPolicy::Ulp(1);
        assert!(policy.compare(&[1.0f32], &[next_up(1.0)]).unwrap());
        assert!(!policy.compare(&[1.0f32], &[2.0]).unwrap());
        assert!(!policy.compare(&[f32::NAN], &[f32::NAN]).unwrap());
        // Crossing zero: the smallest positive and negative subnormals are two steps apart.
        let tiny = f32::from_bits(1);
        assert!(!policy.compare(&[tiny], &[-tiny]).unwrap());
        assert!(VerificationPolicy::Ulp(2).compare(&[tiny], &[-tiny]).unwrap());
        assert!(
            policy
                .compare(&[1.0f64], &[f64::from_bits(1.0f64.to_bits() + 1)])
                .unwrap()
        );
    }

    #[test]
    fn custom_comparator_receives_whole_byte_spans_and_panics_become_errors() {
        let expected = [1.0f32, 2.0];
        let actual = [1.0f32, 3.0];
        let policy = VerificationPolicy::Custom(Arc::new(|left, right| {
            left.len() == right.len() && left.len() == 8 && left[0] == right[0]
        }));
        assert!(policy.compare(&expected, &actual).unwrap());
        let invalid = VerificationPolicy::AbsoluteTolerance(-1.0)
            .compare(&expected, &actual)
            .unwrap_err();
        assert_eq!(invalid.code(), ErrorCode::InvalidArgument);
        assert!(
            VerificationPolicy::RelativeTolerance(f64::INFINITY)
                .validate()
                .is_err()
        );
        let panicking = VerificationPolicy::Custom(Arc::new(|_, _| panic!("comparator failure")));
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let thrown = panicking.compare(&expected, &actual);
        std::panic::set_hook(default_hook);
        assert_eq!(thrown.unwrap_err().code(), ErrorCode::InternalInvariantViolation);
    }

    #[test]
    fn size_mismatch_is_not_equal() {
        assert!(!VerificationPolicy::Exact.compare(&[1.0f32], &[1.0, 1.0]).unwrap());
        assert!(
            !VerificationPolicy::Exact
                .compare_bytes(&[0; 4], &[0; 8], VerificationDataType::Float32)
                .unwrap()
        );
    }

    #[test]
    fn compares_byte_spans_and_rejects_unequal_infinities() {
        let first = [1.0f32, f32::INFINITY];
        let second = [next_up(1.0), f32::INFINITY];
        let policy = VerificationPolicy::Ulp(1);
        assert!(
            policy
                .compare_bytes(as_bytes(&first), as_bytes(&second), VerificationDataType::Float32)
                .unwrap()
        );
        let doubles = [2.0f64];
        assert!(
            policy
                .compare_bytes(
                    as_bytes(&doubles),
                    as_bytes(&doubles),
                    VerificationDataType::Float64
                )
                .unwrap()
        );
        let relative = VerificationPolicy::RelativeTolerance(1.0);
        assert!(!relative.compare(&[f32::INFINITY], &[f32::MAX]).unwrap());
        let ragged = policy
            .compare_bytes(&[0; 6], &[0; 6], VerificationDataType::Float32)
            .unwrap_err();
        assert_eq!(ragged.code(), ErrorCode::InvalidArgument);
        assert!(
            policy
                .compare_bytes(&[], &[], VerificationDataType::Float64)
                .unwrap()
        );
    }

    proptest::proptest! {
        #[test]
        fn ulp_distance_is_symmetric(a in proptest::prelude::any::<f32>(), b in proptest::prelude::any::<f32>()) {
            let policy = VerificationPolicy::Ulp(3);
            proptest::prop_assert_eq!(policy.compare(&[a], &[b]).unwrap(), policy.compare(&[b], &[a]).unwrap());
        }

        #[test]
        fn exact_matches_itself_unless_nan(a in proptest::prelude::any::<f64>()) {
            proptest::prop_assert_eq!(VerificationPolicy::Exact.compare(&[a], &[a]).unwrap(), !a.is_nan());
        }
    }
}
