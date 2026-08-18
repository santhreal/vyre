//! Matrix multiplication  -  row-major 2D `u32` multiply with atomic
//! accumulation into an output matrix.
//!
//! Category A composition. Wraps the inner loop in a `Node::Region`
//! so the optimizer treats it as opaque unless an inline pass
//! explicitly unrolls.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use crate::builder::gemm::ContractionComposer;
use crate::builder::BuildOptions;
use crate::plumbing::operand::tensor_ref::{TensorRef, TensorRefError};

const OP_ID: &str = "vyre-libs::math::matmul";
const OP_ID_BIAS: &str = "vyre-libs::math::matmul_bias";

/// Typed Cat-A builder for [`matmul`].
#[derive(Debug, Clone)]
pub struct Matmul {
    a: TensorRef,
    b: TensorRef,
    out: TensorRef,
    options: BuildOptions,
}

impl Matmul {
    /// Start a builder. Shapes must be `a: [m, k]`, `b: [k, n]`,
    /// `out: [m, n]` with matching `k` dim.
    #[must_use]
    pub fn new(a: TensorRef, b: TensorRef, out: TensorRef) -> Self {
        Self {
            a,
            b,
            out,
            options: BuildOptions::default(),
        }
    }

    /// Validate + materialize.
    ///
    /// # Errors
    ///
    /// Standard [`TensorRefError`] set plus shape-coherence checks:
    /// `a.shape[1] == b.shape[0]` (shared k dim),
    /// `out.shape == [a.shape[0], b.shape[1]]`.
    pub fn build(self) -> Result<Program, TensorRefError> {
        let (m, k, n) = super::matmul_2d_dims(&self.a, &self.b);
        let composer = ContractionComposer::matmul_2d(OP_ID, self.a, self.b, self.out, m, k, n);
        super::apply_contraction_options(composer, &self.options).build()
    }
}

crate::builder::impl_cat_a_builder_options!(Matmul);

/// Typed Cat-A builder for [`matmul_bias`].
#[derive(Debug, Clone)]
pub struct MatmulBias {
    a: TensorRef,
    b: TensorRef,
    bias: TensorRef,
    out: TensorRef,
    options: BuildOptions,
}

impl MatmulBias {
    /// Start a builder. Shapes must be `a: [m, k]`, `b: [k, n]`,
    /// `bias: [n]`, `out: [m, n]` with matching `k` and `n` dims.
    #[must_use]
    pub fn new(a: TensorRef, b: TensorRef, bias: TensorRef, out: TensorRef) -> Self {
        Self {
            a,
            b,
            bias,
            out,
            options: BuildOptions::default(),
        }
    }

    /// Validate + materialize.
    ///
    /// # Errors
    ///
    /// Standard [`TensorRefError`] set plus shape-coherence checks:
    /// `a.shape[1] == b.shape[0]` (shared k dim),
    /// `bias.shape == [n]`,
    /// `out.shape == [a.shape[0], b.shape[1]]`.
    pub fn build(self) -> Result<Program, TensorRefError> {
        let (m, k, n) = super::matmul_2d_dims(&self.a, &self.b);
        let composer = ContractionComposer::matmul_bias_2d(
            OP_ID_BIAS, self.a, self.b, self.bias, self.out, m, k, n,
        );
        super::apply_contraction_options(composer, &self.options).build()
    }
}

crate::builder::impl_cat_a_builder_options!(MatmulBias);

/// Build a Program that computes `out = a @ b` where `a` is `m x k`,
/// `b` is `k x n`, and `out` is `m x n`. The caller supplies buffer
/// names + dimensions via buffer `count()` on the BufferDecls.
///
/// Each invocation computes one `out[i, j]` by iterating the `k`
/// dimension in a local loop. Workgroup size is `[256, 1, 1]` because
/// the non-tiled kernel maps row-major output cells onto a 1-D dispatch.
/// Callers with known-large matrices should use
/// `vyre-libs::math::matmul_tiled`.
#[must_use]
pub fn matmul(a: &str, b: &str, out: &str, m: u32, k: u32, n: u32) -> Program {
    Matmul::new(
        TensorRef::u32_2d(a, m, k),
        TensorRef::u32_2d(b, k, n),
        TensorRef::u32_2d(out, m, n),
    )
    .build()
    .unwrap_or_else(|err| trap_program(OP_ID, Some((out, DataType::U32)), format!("Fix: {err}")))
}

/// Build a Program that computes `out[i, j] = sum_k a[i, k] * b[k, j] + bias[j]`.
#[must_use]
pub fn matmul_bias(a: &str, b: &str, bias: &str, out: &str, m: u32, k: u32, n: u32) -> Program {
    MatmulBias::new(
        TensorRef::u32_2d(a, m, k),
        TensorRef::u32_2d(b, k, n),
        TensorRef::u32_1d(bias, n),
        TensorRef::u32_2d(out, m, n),
    )
    .build()
    .unwrap_or_else(|err| {
        trap_program(
            OP_ID_BIAS,
            Some((out, DataType::U32)),
            format!("Fix: {err}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::bytes_to_u32 as decode_u32_words;
    use vyre_reference::value::Value;

    fn next_u32(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }

    fn random_vec(size: usize, state: &mut u32) -> Vec<u32> {
        (0..size).map(|_| next_u32(state)).collect()
    }

    fn run_u32_output(program: &Program, inputs: Vec<Vec<u32>>, out_bytes: usize) -> Vec<u32> {
        let packed_inputs = inputs
            .into_iter()
            .map(|bytes| Value::from(vyre_primitives::wire::pack_u32_slice(&bytes)))
            .collect::<Vec<_>>();
        let outputs = vyre_reference::reference_eval(program, &packed_inputs)
            .expect("Fix: program must execute; restore this invariant before continuing.");
        let bytes = outputs[0].to_bytes();
        let mut result = decode_u32_words(&bytes);
        assert_eq!(result.len(), out_bytes);
        result.truncate(out_bytes);
        result
    }

    #[test]
    fn matmul_bias_matches_matmul_plus_bias_on_reference_sizes() {
        let sizes = [
            (4u32, 4u32, 4u32),
            (16, 16, 16),
            (32, 64, 32),
            (64, 32, 32),
            (128, 64, 64),
        ];

        for &(m, k, n) in &sizes {
            let mut seed = m ^ (k << 8) ^ (n << 16);
            let a = random_vec((m * k) as usize, &mut seed);
            let b = random_vec((k * n) as usize, &mut seed);
            let bias = random_vec(n as usize, &mut seed);
            let out_len = (m * n) as usize;

            let fused = matmul_bias("a", "b", "bias", "out_fused", m, k, n);
            let fused_out = run_u32_output(
                &fused,
                vec![a.clone(), b.clone(), bias.clone(), vec![0u32; out_len]],
                out_len,
            );

            let plain = matmul("a", "b", "out_plain", m, k, n);
            let plain_out = run_u32_output(
                &plain,
                vec![a.clone(), b.clone(), vec![0u32; out_len]],
                out_len,
            );

            let expected: Vec<u32> = plain_out
                .iter()
                .zip(bias.iter().copied().cycle())
                .map(|(value, b)| value.wrapping_add(b))
                .collect();

            assert_eq!(
                fused_out, expected,
                "fused matmul_bias diverged for shape ({m}, {k}, {n})"
            );
        }
    }

    // ------------------------------------------------------------------
    // Adversarial fixtures exposing real gaps
    // ------------------------------------------------------------------

    #[test]
    fn matmul_scalar_1x1x1() {
        let a = vec![7u32];
        let b = vec![3u32];
        let program = matmul("a", "b", "out", 1, 1, 1);
        let out = run_u32_output(&program, vec![a, b, vec![0u32; 1]], 1);
        assert_eq!(out[0], 21u32, "1x1x1 scalar matmul: 7*3 = 21");
    }

    #[test]
    fn matmul_bias_scalar_1x1x1() {
        let a = vec![7u32];
        let b = vec![3u32];
        let bias = vec![5u32];
        let program = matmul_bias("a", "b", "bias", "out", 1, 1, 1);
        let out = run_u32_output(&program, vec![a, b, bias, vec![0u32; 1]], 1);
        assert_eq!(out[0], 26u32, "1x1x1 bias matmul: 7*3+5 = 26");
    }

    #[test]
    fn matmul_builder_rejects_zero_m() {
        let error = Matmul::new(
            TensorRef::u32_2d("a", 0, 4),
            TensorRef::u32_2d("b", 4, 4),
            TensorRef::u32_2d("out", 0, 4),
        )
        .build()
        .expect_err("Matmul builder must reject M=0");
        assert!(
            matches!(error, TensorRefError::ShapeMismatch { .. }),
            "unexpected matmul zero-M error: {error:?}"
        );
        let msg = format!("{error:?}");
        assert!(
            msg.contains('0'),
            "zero-M error must mention zero dimension: {msg}"
        );
    }

    #[test]
    fn matmul_bias_builder_rejects_zero_m() {
        let error = MatmulBias::new(
            TensorRef::u32_2d("a", 0, 4),
            TensorRef::u32_2d("b", 4, 4),
            TensorRef::u32_1d("bias", 4),
            TensorRef::u32_2d("out", 0, 4),
        )
        .build()
        .expect_err("MatmulBias builder must reject M=0");
        assert!(
            matches!(error, TensorRefError::ShapeMismatch { .. }),
            "unexpected matmul-bias zero-M error: {error:?}"
        );
        let msg = format!("{error:?}");
        assert!(
            msg.contains('0'),
            "zero-M bias error must mention zero dimension: {msg}"
        );
    }

    #[test]
    fn matmul_zero_k_traps() {
        let a: Vec<u32> = vec![];
        let b: Vec<u32> = vec![];
        let program = matmul("a", "b", "out", 2, 0, 3);
        let error = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(vyre_primitives::wire::pack_u32_slice(&a)),
                Value::from(vyre_primitives::wire::pack_u32_slice(&b)),
                Value::from(vec![0u8; 6 * 4]),
            ],
        )
        .expect_err("zero-K matmul must trap");
        let msg = error.to_string();
        assert!(
            msg.contains("trap") || msg.contains("Fix:"),
            "zero-K matmul error must be actionable: {msg}"
        );
    }

    /// u32 wrapping overflow must be preserved.
    #[test]
    fn matmul_u32_max_values_wrap() {
        let a = vec![u32::MAX];
        let b = vec![2u32];
        let program = matmul("a", "b", "out", 1, 1, 1);
        let out = run_u32_output(&program, vec![a, b, vec![0u32; 1]], 1);
        assert_eq!(
            out[0],
            u32::MAX.wrapping_mul(2),
            "u32 matmul must wrap on overflow"
        );
    }

    #[test]
    fn matmul_bias_u32_max_values_wrap() {
        let a = vec![u32::MAX];
        let b = vec![2u32];
        let bias = vec![1u32];
        let program = matmul_bias("a", "b", "bias", "out", 1, 1, 1);
        let out = run_u32_output(&program, vec![a, b, bias, vec![0u32; 1]], 1);
        assert_eq!(
            out[0],
            u32::MAX.wrapping_mul(2).wrapping_add(1),
            "u32 matmul_bias must wrap on overflow"
        );
    }

    // ------------------------------------------------------------------
    // Proptest: random small dimensions with random u32 values.
    // ------------------------------------------------------------------
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn matmul_proptest_random_small_dims(
            m in 1u32..8u32,
            k in 1u32..8u32,
            n in 1u32..8u32,
            seed in any::<u32>(),
        ) {
            let mut state = seed;
            let a = random_vec((m * k) as usize, &mut state);
            let b = random_vec((k * n) as usize, &mut state);
            let out_len = (m * n) as usize;

            let program = matmul("a", "b", "out", m, k, n);
            let actual = run_u32_output(
                &program,
                vec![a.clone(), b.clone(), vec![0u32; out_len]],
                out_len,
            );

            // CPU reference using wrapping u32 arithmetic.
            let mut expected = vec![0u32; out_len];
            for i in 0..m as usize {
                for j in 0..n as usize {
                    let mut acc: u32 = 0;
                    for kk in 0..k as usize {
                        acc = acc.wrapping_add(
                            a[i * k as usize + kk]
                                .wrapping_mul(b[kk * n as usize + j]),
                        );
                    }
                    expected[i * n as usize + j] = acc;
                }
            }
            prop_assert_eq!(
                actual, expected,
                "matmul mismatch for ({},{},{}) seed={}", m, k, n, seed
            );
        }

        #[test]
        fn matmul_bias_proptest_random_small_dims(
            m in 1u32..8u32,
            k in 1u32..8u32,
            n in 1u32..8u32,
            seed in any::<u32>(),
        ) {
            let mut state = seed;
            let a = random_vec((m * k) as usize, &mut state);
            let b = random_vec((k * n) as usize, &mut state);
            let bias = random_vec(n as usize, &mut state);
            let out_len = (m * n) as usize;

            let program = matmul_bias("a", "b", "bias", "out", m, k, n);
            let actual = run_u32_output(
                &program,
                vec![a.clone(), b.clone(), bias.clone(), vec![0u32; out_len]],
                out_len,
            );

            let mut expected = vec![0u32; out_len];
            for i in 0..m as usize {
                for j in 0..n as usize {
                    let mut acc: u32 = 0;
                    for kk in 0..k as usize {
                        acc = acc.wrapping_add(
                            a[i * k as usize + kk]
                                .wrapping_mul(b[kk * n as usize + j]),
                        );
                    }
                    expected[i * n as usize + j] = acc.wrapping_add(bias[j]);
                }
            }
            prop_assert_eq!(
                actual, expected,
                "matmul_bias mismatch for ({},{},{}) seed={}", m, k, n, seed
            );
        }
    }
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::math::matmul",
        || matmul("a", "b", "out", 4, 4, 4),
        Some(|| {
            let a: Vec<u32> = (0..16).collect();
            let b: Vec<u32> = (0..16).map(|i| i + 1).collect();

            vec![vec![
                crate::fixture_bytes::u32_bytes(&a),
                crate::fixture_bytes::u32_bytes(&b),
            ]]
        }),
        Some(|| {
            // 4x4 matmul over u32: a[i,j] = i*4+j, b[i,j] = i*4+j+1.
            // out[i,j] = Σ_k a[i,k] * b[k,j]. Computed row-major.
            vec![vec![vec![
                0x3e, 0x00, 0x00, 0x00, // 62
                0x44, 0x00, 0x00, 0x00, // 68
                0x4a, 0x00, 0x00, 0x00, // 74
                0x50, 0x00, 0x00, 0x00, // 80
                0xae, 0x00, 0x00, 0x00, // 174
                0xc4, 0x00, 0x00, 0x00, // 196
                0xda, 0x00, 0x00, 0x00, // 218
                0xf0, 0x00, 0x00, 0x00, // 240
                0x1e, 0x01, 0x00, 0x00, // 286
                0x44, 0x01, 0x00, 0x00, // 324
                0x6a, 0x01, 0x00, 0x00, // 362
                0x90, 0x01, 0x00, 0x00, // 400
                0x8e, 0x01, 0x00, 0x00, // 398
                0xc4, 0x01, 0x00, 0x00, // 452
                0xfa, 0x01, 0x00, 0x00, // 506
                0x30, 0x02, 0x00, 0x00, // 560
            ]]]
        }),
    )
    .with_category("math")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID_BIAS,
        || matmul_bias("a", "b", "bias", "out", 2, 2, 2),
        Some(super::matmul_bias_2x2_fixture_inputs),
        Some(super::matmul_bias_2x2_fixture_expected),
    )
    .with_category("math")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::math::matmul_bias::scalar",
        || matmul_bias("a", "b", "bias", "out", 1, 1, 1),
        Some(|| {
            vec![vec![
                crate::fixture_bytes::u32_bytes(&[2]),
                crate::fixture_bytes::u32_bytes(&[3]),
                crate::fixture_bytes::u32_bytes(&[5]),
            ]]
        }),
        Some(|| {
            vec![vec![vec![
                0x0b, 0x00, 0x00, 0x00, // 11
            ]]]
        }),
    )
    .with_category("math")
}
