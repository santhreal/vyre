//! Flash-attention tiled fusion  -  `softmax(Q·Kᵀ / √d) · V` computed
//! in a single pass per query row via online-softmax tiling.
//!
//! The standard `attention` primitive in this crate
//! materialises three passes per row (max-reduction, sum-reduction,
//! write) and re-evaluates the dot-product score in each pass. Each
//! re-evaluation reloads `d` Q-values and `d * s` K-values from
//! global memory. For `s = 4096, d = 128` this is roughly
//! `3 * 4096 * 128 * 4 bytes = 6 MiB` of redundant reads per row.
//!
//! Flash-attention's contribution is the **online-softmax** trick:
//! maintain a running `(m, l, o)` state  -  running max, running
//! softmax denominator, running weighted-V sum  -  and update them
//! per-K-row in a single pass:
//!
//! ```text
//! For each query row i in [0, s):
//!   m = -INF; l = 0; o = [0; d]
//!   For each j in [0, s):
//!     score = scale * dot(Q[i,:], K[j,:])
//!     m_new = max(m, score)
//!     rescale = exp(m - m_new)
//!     l_new = rescale * l + exp(score - m_new)
//!     For t in [0, d):
//!       o[t] = rescale * o[t] + exp(score - m_new) * V[j, t]
//!     m = m_new; l = l_new
//!   For t in [0, d):
//!     out[i, t] = o[t] / l
//! ```
//!
//! Soundness: this is the standard online-softmax recurrence; for
//! every i, the final `(m, l, o)` after processing all j is
//! mathematically equivalent to the offline softmax-then-weighted-
//! sum that the reference attention computes.
//!
//! Cost direction: monotone-down on global-memory traffic. Each
//! `Q[i,k]` is loaded once across the j-loop (constant within the
//! per-row online pass) and each `K[j,k]` / `V[j,t]` is loaded
//! exactly once instead of three times. The per-row online-state
//! (`m, l, o[d]`) is held in workgroup-shared scratch.
//!
//! ## Schedule
//!
//! This builder selects the scalar plan: one invocation per query row, one key
//! per tile. The recurrence itself belongs to
//! [`online_softmax_attention`](super::tiled_online_softmax::online_softmax_attention),
//! which [`flash_attention_2`](super::flash_attention_2::flash_attention_2)
//! composes at a cooperative tile width. Scalar and tiled are one kernel under
//! two plans, so a stability fix cannot reach one and miss the other.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use super::planner::plan_flash_attention_scalar;
use super::scaled_dot_product::direct_attention_program;
use super::tiled_online_softmax::compose_online_softmax_attention;

const OP_ID: &str = "vyre-libs::nn::flash_attention";

/// Build a Program that computes scaled dot-product attention via
/// the online-softmax (flash-attention) recurrence. Tensors are
/// `[s, d]` row-major F32; `out` has the same shape.
///
/// # Errors
///
/// Returns `Err` when `s == 0` or `d == 0` (empty reductions).
pub fn flash_attention(
    q: &str,
    k: &str,
    v: &str,
    out: &str,
    s: u32,
    d: u32,
) -> Result<Program, String> {
    if s == 0 {
        return Err("Fix: flash_attention s=0 is invalid: empty sequence".to_string());
    }
    if d == 0 {
        return Err("Fix: flash_attention d=0 is invalid: empty head dimension".to_string());
    }
    if let Some(program) = direct_attention_program(q, k, v, out, s, d, OP_ID)
        .map_err(|error| format!("Fix: flash_attention direct specialization failed: {error}"))?
    {
        return Ok(program);
    }
    let plan = plan_flash_attention_scalar(s, d)?;
    Ok(compose_online_softmax_attention(OP_ID, q, k, v, out, &plan))
}

const EXPECTED_FLASH_ATTENTION_OUTPUT_BYTES: [u8; 36] = [
    0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x80, 0x40,
    0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x80, 0x40,
    0x00, 0x00, 0x80, 0x40,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::nn::flash_attention",
        || {
            flash_attention("q", "k", "v", "out", 9, 1).unwrap_or_else(|error| {
                trap_program(
                    "vyre-libs::nn::flash_attention",
                    Some(("out", DataType::F32)),
                    error,
                )
            })
        },
        Some(|| {
            let q = [0.0_f32; 9];
            let k = [0.0_f32; 9];
            let v = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            vec![vec![
                vyre_primitives::wire::pack_f32_slice(&q),
                vyre_primitives::wire::pack_f32_slice(&k),
                vyre_primitives::wire::pack_f32_slice(&v),
            ]]
        }),
        // This deliberately uses s=9 so `direct_attention_program` declines
        // the tiny-shape specialization and the registered op covers the real
        // online-softmax flash kernel. With zero Q/K, every row has uniform
        // weights and returns mean(V)=4.0.
        Some(|| {
            vec![vec![EXPECTED_FLASH_ATTENTION_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::decode_f32;
    use crate::fixture_bytes::eval_bytes;
    use crate::fixture_bytes::f32_bytes;

    /// Online-softmax flash-attention agrees with the offline 3-pass
    /// `attention_reference` on a non-trivial fixture.
    #[test]
    fn flash_attention_matches_attention_reference() {
        let s = 9_u32;
        let d = 7_u32;
        let elements = (s * d) as usize;
        let (q, k, v) = super::super::synth_qkv_fixtures(elements);
        let run = |program: Program| {
            crate::nn::attention::eval_qkv_program(
                &program,
                &q,
                &k,
                &v,
                "Fix: flash_attention must execute in the reference interpreter.",
            )
        };
        let actual = run(flash_attention("q", "k", "v", "out", s, d).expect("Fix: build"));
        let expected = run(crate::nn::attention::attention_reference(
            "q", "k", "v", "out", s, d,
        ));
        assert_eq!(actual.len(), expected.len(), "output length must match");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 1.0e-4,
                "flash_attention vs attention_reference mismatch at index {i}: {a} != {e}"
            );
        }
    }

    #[test]
    fn flash_attention_online_kernel_uniform_scores_return_value_mean() {
        let s = 9_u32;
        let d = 1_u32;
        let q = vec![0.0_f32; s as usize];
        let k = vec![0.0_f32; s as usize];
        let v: Vec<f32> = (0..s).map(|idx| idx as f32).collect();
        let program = flash_attention("q", "k", "v", "out", s, d).expect("Fix: build");
        assert_eq!(
            program.workgroup_size(),
            [128, 1, 1],
            "Fix: s=9 must bypass direct_attention_program and use the online flash kernel."
        );
        let outputs = eval_bytes(
            "flash_attention",
            &program,
            vec![
                f32_bytes(&q),
                f32_bytes(&k),
                f32_bytes(&v),
                vec![0u8; (s * d) as usize * 4],
            ],
        );
        let actual = decode_f32(&outputs[0]);
        assert_eq!(actual.len(), s as usize);
        for (idx, value) in actual.iter().enumerate() {
            assert!(
                (*value - 4.0).abs() <= 1.0e-5,
                "uniform-score flash attention row {idx} should return mean(V)=4.0, got {value}"
            );
        }
    }

    /// `flash_attention(0, _)` rejects empty sequence with an
    /// actionable Fix message.
    #[test]
    fn flash_attention_rejects_empty_seq() {
        let err = flash_attention("q", "k", "v", "out", 0, 4).expect_err("empty s must error");
        assert!(err.contains("s=0"));
    }

    /// `flash_attention(_, 0)` rejects empty head dim.
    #[test]
    fn flash_attention_rejects_empty_head_dim() {
        let err = flash_attention("q", "k", "v", "out", 4, 0).expect_err("empty d must error");
        assert!(err.contains("d=0"));
    }

    /// Single-row (s=1) attention degenerates to V (because softmax
    /// of a length-1 score vector is [1.0]).
    #[test]
    fn flash_attention_single_row_passes_v_through() {
        let d = 4_u32;
        let q = vec![1.0_f32, 2.0, 3.0, 4.0];
        let k = vec![0.5_f32, 1.5, 2.5, 3.5];
        let v = vec![10.0_f32, 20.0, 30.0, 40.0];
        let prog = flash_attention("q", "k", "v", "out", 1, d).expect("Fix: build");
        let outputs = eval_bytes(
            "flash_attention",
            &prog,
            vec![
                f32_bytes(&q),
                f32_bytes(&k),
                f32_bytes(&v),
                vec![0u8; (d as usize) * 4],
            ],
        );
        let actual = decode_f32(&outputs[0]);
        for (a, e) in actual.iter().zip(v.iter()) {
            assert!(
                (a - e).abs() <= 1.0e-4,
                "single-row attention should pass V through: {a} != {e}"
            );
        }
    }

    #[test]
    fn flash_attention_very_large_qk_values_stay_finite() {
        // Very large Q and K should produce bounded scores due to bounded_exp_arg.
        let s = 2u32;
        let d = 2u32;
        let q = [1e20f32, 1e20, 1e20, 1e20];
        let k = [1e20f32, 1e20, 1e20, 1e20];
        let v = [1.0f32, 2.0, 3.0, 4.0];
        let prog = flash_attention("q", "k", "v", "out", s, d).expect("Fix: build");
        let outputs = eval_bytes(
            "flash_attention",
            &prog,
            vec![
                f32_bytes(&q),
                f32_bytes(&k),
                f32_bytes(&v),
                vec![0u8; (s * d) as usize * 4],
            ],
        );
        let out = decode_f32(&outputs[0]);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "flash_attention output at {i} must be finite for large QK values, got {v}"
            );
        }
    }

    #[test]
    fn flash_attention_nan_in_q_k_v_is_silently_suppressed() {
        let s = 1u32;
        let d = 2u32;
        let q = [f32::NAN, 0.0];
        let k = [0.0f32, 0.0];
        let v = [1.0f32, 2.0];
        let prog = flash_attention("q", "k", "v", "out", s, d).expect("Fix: build");
        let outputs = eval_bytes(
            "flash_attention",
            &prog,
            vec![f32_bytes(&q), f32_bytes(&k), f32_bytes(&v), vec![0u8; 8]],
        );
        let out = decode_f32(&outputs[0]);
        assert!(
            out.iter().any(|v| v.is_nan()),
            "flash_attention must propagate NaN in Q/K/V instead of silently producing finite output {:?}",
            out
        );
    }
}
