//! 4-point complex radix-2 FFT.
//!
//! For complex input `x[0..4]` (interleaved re/im in a length-8 F32 buffer),
//! compute `X[k] = sum_n x[n] * exp(-2πi·n·k/4)` for `k ∈ {0,1,2,3}`.
//!
//! This is the N=4 case of the radix-2 transform and nothing else. It used to
//! carry a hand-expanded copy of the four butterflies, which meant the 4-point
//! DFT had two implementations and no test that they agreed. The transform now
//! has one owner in `fft_radix2`, and this file is the fixed-size entry point
//! over it: same algorithm, its own operation identity.
//!
//! Every twiddle at N=4 is exactly `1`, `-1`, `i` or `-i`, so the butterflies
//! multiply by exact values and the bins are exact whenever the inputs are.

use vyre_foundation::ir::Program;

const OP_ID: &str = "vyre-libs::math::fft::fft4_complex";

/// Build a Program that computes a 4-point complex DFT.
/// `input` is a length-8 F32 buffer holding 4 complex values as
/// `[re0, im0, re1, im1, re2, im2, re3, im3]`. `output` has the
/// same shape and holds the 4 frequency bins in the same layout.
#[must_use]
pub fn fft4_complex(input: &str, output: &str) -> Program {
    super::fft_radix2::radix2_program(input, output, 4, OP_ID)
        .unwrap_or_else(|error| unreachable!("Fix: 4 is a valid radix-2 FFT size: {error}"))
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || fft4_complex("input", "output"),
        Some(|| {
            // Real-valued sequence [1, 0, 0, 0] (impulse): all bins = 1+0i
            let input = crate::fixture_bytes::f32_bytes(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            vec![vec![input]]
        }),
        Some(|| {
            // FFT of impulse = uniform [1, 1, 1, 1] across all bins.
            vec![vec![vec![
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, // 1.0, 0.0
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, // 1.0, 0.0
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, // 1.0, 0.0
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, // 1.0, 0.0
            ]]]
        }),
    )
    .with_category("math")
}

#[cfg(test)]
mod tests {
    use super::super::complex_length::naive_dft;
    use super::*;
    use crate::fixture_bytes::eval_f32;

    fn bins(input: &[f32]) -> Vec<f32> {
        eval_f32(
            "fft4_complex",
            &fft4_complex("input", "output"),
            &[input],
            8,
        )
    }

    /// WHY: the two registered operations are the same transform at N=4, and
    /// the reason there were two implementations for so long is that nothing
    /// ever compared them. Byte equality, not a tolerance: every twiddle at
    /// N=4 is exactly ±1 or ±i, so any difference is a difference in the
    /// program, never in the arithmetic. Delegation makes this pass trivially
    /// today; it stops being trivial the moment either entry point grows a
    /// body of its own again.
    #[test]
    fn the_fixed_size_entry_point_is_the_radix_2_transform_at_four_points() {
        for input in [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.5, -1.5, 2.25, 3.0, -4.75, 0.125, 6.0, -7.5],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ] {
            let wide = super::super::fft_radix2::fft_radix2_complex("input", "output", 4)
                .expect("Fix: 4 is a valid radix-2 FFT size.");
            assert_eq!(
                bins(&input),
                eval_f32("fft_radix2_complex", &wide, &[&input], 8),
                "the two 4-point entry points disagree on {input:?}"
            );
        }
    }

    /// Impulse response: FFT of [1, 0, 0, 0] is [1, 1, 1, 1] across
    /// all bins (each bin sums one term, x[0] = 1).
    #[test]
    fn fft4_impulse_yields_uniform_bins() {
        let input = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let actual = bins(&input);
        let expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= 1.0e-5, "{a} != {e}");
        }
    }

    /// DC signal: FFT of [1, 1, 1, 1] is [4, 0, 0, 0] (all energy
    /// in the DC bin).
    #[test]
    fn fft4_dc_signal_concentrates_in_dc_bin() {
        let input = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let actual = bins(&input);
        let expected = [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= 1.0e-5, "{a} != {e}");
        }
    }

    /// Frequency bin 1: FFT of [cos(2π·n/4)] for n=0..3 (real-axis
    /// alternating cosine) puts energy in bin 1 (and its conjugate
    /// bin 3 by Hermitian symmetry).
    #[test]
    fn fft4_freq1_cosine_concentrates_in_bins_1_and_3() {
        // cos(2π·n/4) for n = 0, 1, 2, 3 = [1, 0, -1, 0]
        let input = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0];
        let actual = bins(&input);
        // X[0] = 0, X[1] = 2, X[2] = 0, X[3] = 2 (real-only output
        // because the input is real and even-symmetric).
        let expected = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= 1.0e-5, "{a} != {e}");
        }
    }

    /// Random fuzz: 50 random length-4 complex sequences, agree
    /// with the naive DFT formula within 1.0e-4 absolute tolerance.
    #[test]
    fn fft4_matches_naive_dft_on_random_fuzz() {
        let mut state = 0xCAFEBABE_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
        };
        for _ in 0..50 {
            let input: Vec<f32> = (0..8).map(|_| next()).collect();
            let actual = bins(&input);
            let expected = naive_dft(&input, 4);
            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - e).abs() <= 1.0e-4,
                    "lane {i}: fft={a} naive={e} diff={}",
                    (a - e).abs()
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Adversarial fixtures exposing real gaps
    // ------------------------------------------------------------------

    /// NaN in the real part of x[0] propagates to the real part of every
    /// output bin (the imaginary parts remain finite because x0i=0).
    #[test]
    fn fft4_nan_input_propagates_to_real_parts() {
        let input = [f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let actual = bins(&input);
        for k in 0..4 {
            assert!(
                actual[2 * k].is_nan(),
                "FFT bin {k} real part must be NaN when x0r is NaN"
            );
        }
    }

    /// NaN in both re and im of x[0] must make every output component NaN.
    #[test]
    fn fft4_nan_both_components_propagates_everywhere() {
        let input = [f32::NAN, f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let actual = bins(&input);
        for (i, &v) in actual.iter().enumerate() {
            assert!(
                v.is_nan(),
                "FFT lane {i} must be NaN when both re/im inputs are NaN"
            );
        }
    }

    /// Inf in the real part of x[0] propagates to the real part of every
    /// output bin.
    #[test]
    fn fft4_inf_input_propagates_to_real_parts() {
        let input = [f32::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let actual = bins(&input);
        for k in 0..4 {
            assert!(
                actual[2 * k].is_infinite(),
                "FFT bin {k} real part must be Inf when x0r is Inf"
            );
        }
    }
}
