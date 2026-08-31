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
pub(crate) fn naive_dft(input: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; 2 * n];
    for k in 0..n {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for nn in 0..n {
            let xr = input[2 * nn];
            let xi = input[2 * nn + 1];
            let theta = -2.0_f32 * std::f32::consts::PI * (nn as f32) * (k as f32) / (n as f32);
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            re += xr * cos_t - xi * sin_t;
            im += xr * sin_t + xi * cos_t;
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
    out
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
