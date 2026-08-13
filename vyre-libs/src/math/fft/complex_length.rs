//! Interleaved complex FFT length validation.

/// Validate an interleaved complex FFT length and return the scalar element
/// count (`2 * n`).
///
/// # Errors
///
/// Returns an actionable error when `n < 2`, `n` is not a power of two, or the
/// interleaved scalar length overflows `u32`.
pub(super) fn validate_complex_len(n: u32, op: &str) -> Result<u32, String> {
    if n < 2 {
        return Err(format!("Fix: {op} requires n >= 2; got n={n}."));
    }
    if !n.is_power_of_two() {
        return Err(format!("Fix: {op} requires n a power of two; got n={n}."));
    }
    n.checked_mul(2)
        .ok_or_else(|| format!("Fix: {op} 2*n overflows; reduce n."))
}

#[cfg(test)]
mod tests {
    use super::validate_complex_len;

    #[test]
    fn validate_complex_len_rejects_invalid_shapes() {
        assert!(validate_complex_len(0, "generated_fft").is_err());
        assert!(validate_complex_len(1, "generated_fft").is_err());
        assert!(validate_complex_len(6, "generated_fft")
            .expect_err("non-power-of-two must fail")
            .contains("power of two"));
        assert!(validate_complex_len(1_u32 << 31, "generated_fft")
            .expect_err("overflowing interleaved length must fail")
            .contains("2*n overflows"));
    }
}
