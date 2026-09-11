//! Backend-neutral checked arithmetic and atomic accounting primitives.

use crate::BackendError;

pub use crate::accounting_atomic::*;

/// Add two `u64` values without wraparound.
pub fn checked_add_u64_value<E>(lhs: u64, rhs: u64, error: E) -> Result<u64, E> {
    lhs.checked_add(rhs).ok_or(error)
}

/// Add two `u64` values without wraparound, constructing the error lazily.
///
/// # Errors
///
/// Returns `E` from `error` when the addition would overflow.
pub fn checked_add_u64_lazy<E>(lhs: u64, rhs: u64, error: impl FnOnce() -> E) -> Result<u64, E> {
    lhs.checked_add(rhs).ok_or_else(error)
}

pub(crate) fn sum_optional_timing(
    accumulator: Option<u64>,
    next: Option<u64>,
    field: &str,
    scope: &str,
    split_unit: &str,
) -> Result<Option<u64>, BackendError> {
    match (accumulator, next) {
        (Some(left), Some(right)) => {
            checked_add_u64_lazy(left, right, || BackendError::InvalidProgram {
                fix: format!(
                    "Fix: {scope} {field} overflowed u64 nanoseconds. Split telemetry windows or report {split_unit} timing instead of silently clamping."
                ),
            })
            .map(Some)
        }
        _ => Ok(None),
    }
}

/// Multiply two `u64` values without wraparound.
pub fn checked_mul_u64_value<E>(lhs: u64, rhs: u64, error: E) -> Result<u64, E> {
    lhs.checked_mul(rhs).ok_or(error)
}

/// Multiply two `u64` values without wraparound, constructing the error lazily.
///
/// # Errors
///
/// Returns `E` from `error` when the multiplication would overflow.
pub fn checked_mul_u64_lazy<E>(lhs: u64, rhs: u64, error: impl FnOnce() -> E) -> Result<u64, E> {
    lhs.checked_mul(rhs).ok_or_else(error)
}

/// Subtract two `u64` values without underflow.
pub fn checked_sub_u64_value<E>(lhs: u64, rhs: u64, error: E) -> Result<u64, E> {
    lhs.checked_sub(rhs).ok_or(error)
}

/// Subtract two `u64` values without underflow, constructing the error lazily.
///
/// # Errors
///
/// Returns `E` from `error` when the subtraction would underflow.
pub fn checked_sub_u64_lazy<E>(lhs: u64, rhs: u64, error: impl FnOnce() -> E) -> Result<u64, E> {
    lhs.checked_sub(rhs).ok_or_else(error)
}

/// Subtract two `usize` values without underflow, constructing the error lazily.
///
/// # Errors
///
/// Returns `E` from `error` when the subtraction would underflow.
pub fn checked_sub_usize_lazy<E>(
    lhs: usize,
    rhs: usize,
    error: impl FnOnce() -> E,
) -> Result<usize, E> {
    lhs.checked_sub(rhs).ok_or_else(error)
}

/// Add two `usize` values without wraparound.
pub fn checked_add_usize_value<E>(lhs: usize, rhs: usize, error: E) -> Result<usize, E> {
    lhs.checked_add(rhs).ok_or(error)
}

/// Add two `usize` values without wraparound, constructing the error lazily.
///
/// # Errors
///
/// Returns `E` from `error` when the addition would overflow.
pub fn checked_add_usize_lazy<E>(
    lhs: usize,
    rhs: usize,
    error: impl FnOnce() -> E,
) -> Result<usize, E> {
    lhs.checked_add(rhs).ok_or_else(error)
}

/// Multiply two `usize` values without wraparound, constructing the error lazily.
///
/// # Errors
///
/// Returns `E` from `error` when the multiplication would overflow.
pub fn checked_mul_usize_lazy<E>(
    lhs: usize,
    rhs: usize,
    error: impl FnOnce() -> E,
) -> Result<usize, E> {
    lhs.checked_mul(rhs).ok_or_else(error)
}

/// Convert `usize` to `u64`, constructing the error lazily on overflow.
///
/// # Errors
///
/// Returns `E` from `error` when `value` cannot fit in `u64`.
pub fn checked_usize_to_u64_lazy<E>(value: usize, error: impl FnOnce() -> E) -> Result<u64, E> {
    u64::try_from(value).map_err(|_| error())
}

/// Validate a `usize` byte range and return its exclusive end.
///
/// # Errors
///
/// Returns `E` from `overflow_error` when `start + len` would overflow, or
/// `E` from `out_of_bounds_error` when the range end exceeds `limit`.
pub fn checked_usize_byte_range_end_lazy<E>(
    start: usize,
    len: usize,
    limit: usize,
    overflow_error: impl FnOnce() -> E,
    out_of_bounds_error: impl FnOnce(usize) -> E,
) -> Result<usize, E> {
    let end = start.checked_add(len).ok_or_else(overflow_error)?;
    if end > limit {
        return Err(out_of_bounds_error(end));
    }
    Ok(end)
}

/// Add a `usize` byte offset to a `u64` base pointer/counter without wraparound.
///
/// # Errors
///
/// Returns `E` from `conversion_error` when `offset` cannot be represented as
/// `u64`, or `E` from `overflow_error` when `base + offset` would overflow.
pub fn checked_add_u64_usize_offset_lazy<E>(
    base: u64,
    offset: usize,
    conversion_error: impl FnOnce() -> E,
    overflow_error: impl FnOnce() -> E,
) -> Result<u64, E> {
    let offset = u64::try_from(offset).map_err(|_| conversion_error())?;
    base.checked_add(offset).ok_or_else(overflow_error)
}

/// Add two `u32` values without wraparound.
pub fn checked_add_u32_value<E>(lhs: u32, rhs: u32, error: E) -> Result<u32, E> {
    lhs.checked_add(rhs).ok_or(error)
}

/// Multiply two `u32` values without wraparound.
pub fn checked_mul_u32_value<E>(lhs: u32, rhs: u32, error: E) -> Result<u32, E> {
    lhs.checked_mul(rhs).ok_or(error)
}

/// Domain error adapter for planner-specific arithmetic overflow fields.
pub trait ArithmeticOverflow: Sized {
    /// Build the planner-specific overflow error for `field`.
    fn arithmetic_overflow(field: &'static str) -> Self;
}

/// Add two `u64` counters and map overflow into the caller domain.
///
/// # Errors
///
/// Returns `E` when the addition would overflow.
pub fn checked_add_u64_count<E>(lhs: u64, rhs: u64, field: &'static str) -> Result<u64, E>
where
    E: ArithmeticOverflow,
{
    checked_add_u64_value(lhs, rhs, E::arithmetic_overflow(field))
}

/// Multiply two `u64` counters and map overflow into the caller domain.
///
/// # Errors
///
/// Returns `E` when the multiplication would overflow.
pub fn checked_mul_u64_count<E>(lhs: u64, rhs: u64, field: &'static str) -> Result<u64, E>
where
    E: ArithmeticOverflow,
{
    checked_mul_u64_value(lhs, rhs, E::arithmetic_overflow(field))
}

/// Subtract two `u64` counters and map underflow into the caller domain.
///
/// # Errors
///
/// Returns `E` when the subtraction would underflow.
pub fn checked_sub_u64_count<E>(lhs: u64, rhs: u64, field: &'static str) -> Result<u64, E>
where
    E: ArithmeticOverflow,
{
    checked_sub_u64_value(lhs, rhs, E::arithmetic_overflow(field))
}

/// Add two `usize` counters and map overflow into the caller domain.
///
/// # Errors
///
/// Returns `E` when the addition would overflow.
pub fn checked_add_usize_count<E>(lhs: usize, rhs: usize, field: &'static str) -> Result<usize, E>
where
    E: ArithmeticOverflow,
{
    checked_add_usize_value(lhs, rhs, E::arithmetic_overflow(field))
}

/// Add two `u32` counters and map overflow into the caller domain.
///
/// # Errors
///
/// Returns `E` when the addition would overflow.
pub fn checked_add_u32_count<E>(lhs: u32, rhs: u32, field: &'static str) -> Result<u32, E>
where
    E: ArithmeticOverflow,
{
    checked_add_u32_value(lhs, rhs, E::arithmetic_overflow(field))
}

/// Increment a `u64` scalar counter, pinning it at `u64::MAX` instead of wrapping.
///
/// Returns `true` when the counter was incremented and `false` when it was
/// already pinned. `on_pinned` is called exactly once on the pinned path.
pub fn pinning_increment_u64(counter: &mut u64, on_pinned: impl FnOnce()) -> bool {
    match counter.checked_add(1) {
        Some(next) => {
            *counter = next;
            true
        }
        None => {
            on_pinned();
            *counter = u64::MAX;
            false
        }
    }
}

// Inline: the suite grades against `crate::BackendError`, which is gated on `null` and so is absent
// from an integration test build.
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Eq, PartialEq)]
    enum ArithmeticError {
        Overflow(&'static str),
    }

    impl ArithmeticOverflow for ArithmeticError {
        fn arithmetic_overflow(field: &'static str) -> Self {
            Self::Overflow(field)
        }
    }

    #[test]
    fn checked_value_helpers_preserve_domain_errors() {
        assert_eq!(checked_add_u64_value(2, 3, "overflow"), Ok(5));
        assert_eq!(checked_mul_u64_value(2, 3, "overflow"), Ok(6));
        assert_eq!(checked_sub_u64_value(5, 3, "underflow"), Ok(2));
        assert_eq!(checked_add_usize_value(2, 3, "overflow"), Ok(5));
        assert_eq!(checked_add_u32_value(2, 3, "overflow"), Ok(5));

        assert_eq!(
            checked_add_u64_value(u64::MAX, 1, "overflow"),
            Err("overflow")
        );
        assert_eq!(
            checked_mul_u64_value(u64::MAX, 2, "overflow"),
            Err("overflow")
        );
        assert_eq!(checked_sub_u64_value(0, 1, "underflow"), Err("underflow"));
        assert_eq!(
            checked_add_usize_value(usize::MAX, 1, "overflow"),
            Err("overflow")
        );
        assert_eq!(
            checked_add_u32_value(u32::MAX, 1, "overflow"),
            Err("overflow")
        );
    }

    #[test]
    fn checked_add_usize_lazy_does_not_build_success_error() {
        let mut constructed = false;

        assert_eq!(
            checked_add_usize_lazy(2, 3, || {
                constructed = true;
                "overflow"
            }),
            Ok(5)
        );
        assert!(
            !constructed,
            "Fix: hot-path checked usize accounting must not construct error strings on success."
        );
        assert_eq!(
            checked_add_usize_lazy(usize::MAX, 1, || "overflow"),
            Err("overflow")
        );
    }

    #[test]
    fn checked_add_u64_lazy_does_not_build_success_error() {
        let mut constructed = false;

        assert_eq!(
            checked_add_u64_lazy(2, 3, || {
                constructed = true;
                "overflow"
            }),
            Ok(5)
        );
        assert!(
            !constructed,
            "Fix: hot-path checked u64 accounting must not construct error strings on success."
        );
        assert_eq!(
            checked_add_u64_lazy(u64::MAX, 1, || "overflow"),
            Err("overflow")
        );
    }

    #[test]
    fn checked_mul_usize_lazy_does_not_build_success_error() {
        let mut constructed = false;

        assert_eq!(
            checked_mul_usize_lazy(2, 3, || {
                constructed = true;
                "overflow"
            }),
            Ok(6)
        );
        assert!(
            !constructed,
            "Fix: hot-path checked usize multiplication must not construct error strings on success."
        );
        assert_eq!(
            checked_mul_usize_lazy(usize::MAX, 2, || "overflow"),
            Err("overflow")
        );
    }

    #[test]
    fn typed_checked_arithmetic_helpers_preserve_domain_error_fields() {
        assert_eq!(
            checked_add_u64_count::<ArithmeticError>(u64::MAX, 1, "u64 add"),
            Err(ArithmeticError::Overflow("u64 add"))
        );
        assert_eq!(
            checked_mul_u64_count::<ArithmeticError>(u64::MAX, 2, "u64 mul"),
            Err(ArithmeticError::Overflow("u64 mul"))
        );
        assert_eq!(
            checked_sub_u64_count::<ArithmeticError>(0, 1, "u64 sub"),
            Err(ArithmeticError::Overflow("u64 sub"))
        );
        assert_eq!(
            checked_add_usize_count::<ArithmeticError>(usize::MAX, 1, "usize add"),
            Err(ArithmeticError::Overflow("usize add"))
        );
        assert_eq!(
            checked_add_u32_count::<ArithmeticError>(u32::MAX, 1, "u32 add"),
            Err(ArithmeticError::Overflow("u32 add"))
        );
    }

    #[test]
    fn generated_checked_arithmetic_matrix_matches_primitive_semantics() {
        const VALUES: [u64; 12] = [
            0,
            1,
            2,
            3,
            7,
            31,
            255,
            1024,
            u32::MAX as u64,
            u64::MAX / 2,
            u64::MAX - 1,
            u64::MAX,
        ];

        for lhs in VALUES {
            for rhs in VALUES {
                assert_eq!(
                    checked_add_u64_value(lhs, rhs, "overflow").ok(),
                    lhs.checked_add(rhs)
                );
                assert_eq!(
                    checked_mul_u64_value(lhs, rhs, "overflow").ok(),
                    lhs.checked_mul(rhs)
                );
                assert_eq!(
                    checked_sub_u64_value(lhs, rhs, "underflow").ok(),
                    lhs.checked_sub(rhs)
                );
            }
        }
    }

    #[test]
    fn scalar_pinning_increment_never_wraps() {
        let mut scalar_counter = u64::MAX - 1;
        assert!(pinning_increment_u64(&mut scalar_counter, || {
            unreachable!("first scalar increment should fit")
        }));
        assert_eq!(scalar_counter, u64::MAX);
        let mut scalar_pinned = false;
        assert!(!pinning_increment_u64(&mut scalar_counter, || {
            scalar_pinned = true;
        }));
        assert!(scalar_pinned);
        assert_eq!(scalar_counter, u64::MAX);
    }
}
