//! Skip gate: sigmoid-gated U-Net skip connection.
//!
//! `out = sigmoid(g) * branch + (1 - sigmoid(g)) * skip`
//!
//! Category A composition  -  sigmoid + mul + add. Used in the recipe
//! for U-Net skip connections between encoder and decoder layers.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Program, UnOp};

use crate::builder::build_indexed_map;
use crate::nn::f32_stability::flush_tiny;

const OP_ID: &str = "vyre-libs::nn::skip_gate";

/// Build a Program for sigmoid-gated skip connection.
///
/// `gate[n]` (F32)  -  raw gate logits (sigmoid applied here).
/// `branch[n]` (F32)  -  output of the transformer block.
/// `skip[n]` (F32)  -  skip connection from encoder.
/// `output[n]` (F32)  -  gated combination.
#[must_use]
pub fn skip_gate(gate: &str, branch: &str, skip: &str, output: &str, n: u32) -> Program {
    let buffers = vec![
        BufferDecl::storage(gate, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
        BufferDecl::storage(branch, 1, BufferAccess::ReadOnly, DataType::F32).with_count(n),
        BufferDecl::storage(skip, 2, BufferAccess::ReadOnly, DataType::F32).with_count(n),
        BufferDecl::output(output, 3, DataType::F32)
            .with_count(n.max(1))
            .with_output_byte_range(0..(n as usize).saturating_mul(4)),
    ];

    build_indexed_map(OP_ID, buffers, output, n, [64, 1, 1], |i| {
        let g_raw = Expr::load(gate, i.clone());
        let b = Expr::load(branch, i.clone());
        let s = Expr::load(skip, i.clone());

        // sigmoid(g) = 1 / (1 + exp(-g))
        let sigmoid_g = Expr::div(
            Expr::f32(1.0),
            Expr::add(
                Expr::f32(1.0),
                Expr::UnOp {
                    op: UnOp::Exp,
                    operand: Box::new(Expr::UnOp {
                        op: UnOp::Negate,
                        operand: Box::new(g_raw),
                    }),
                },
            ),
        );

        // out = sig * branch + (1 - sig) * skip
        let result = Expr::add(
            Expr::mul(sigmoid_g.clone(), b),
            Expr::mul(Expr::sub(Expr::f32(1.0), sigmoid_g), s),
        );
        (i, flush_tiny(result))
    })
}

const EXPECTED_SKIP_GATE_OUTPUT_BYTES: [u8; 8] = [0x00, 0x00, 0xA0, 0x41, 0x00, 0x00, 0xA0, 0x41];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || skip_gate("gate", "branch", "skip", "output", 2),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[0.0, 100.0]),  // gate logits (sigmoid(0)=0.5, sigmoid(100)≈1)
                to_f32(&[10.0, 20.0]),  // branch
                to_f32(&[30.0, 40.0]),  // skip
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_SKIP_GATE_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::eval_f32;

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn skip_gate_nan_in_gate_propagates_nan() {
        let gate = [f32::NAN];
        let branch = [1.0f32];
        let skip = [2.0f32];
        let program = skip_gate("gate", "branch", "skip", "output", 1);
        let out = eval_f32(
            "skip_gate",
            &program,
            &[&gate[..], &branch[..], &skip[..]],
            1,
        );
        assert!(out[0].is_nan(), "skip_gate(NaN gate) must be NaN");
    }

    #[test]
    fn skip_gate_inf_gate_selects_branch_or_skip() {
        let program = skip_gate("gate", "branch", "skip", "output", 2);
        // +Inf gate → sigmoid(+Inf)=1 → branch
        let out = eval_f32(
            "skip_gate",
            &program,
            &[
                &[f32::INFINITY, 0.0][..],
                &[10.0, 20.0][..],
                &[30.0, 40.0][..],
            ],
            2,
        );
        assert_eq!(out[0], 10.0, "skip_gate(+Inf gate) must select branch");

        // -Inf gate → sigmoid(-Inf)=0 → skip
        let out = eval_f32(
            "skip_gate",
            &program,
            &[
                &[f32::NEG_INFINITY, 0.0][..],
                &[10.0, 20.0][..],
                &[30.0, 40.0][..],
            ],
            2,
        );
        assert_eq!(out[0], 30.0, "skip_gate(-Inf gate) must select skip");
    }

    #[test]
    fn skip_gate_nan_in_branch_propagates_nan() {
        let gate = [0.0f32];
        let branch = [f32::NAN];
        let skip = [2.0f32];
        let program = skip_gate("gate", "branch", "skip", "output", 1);
        let out = eval_f32(
            "skip_gate",
            &program,
            &[&gate[..], &branch[..], &skip[..]],
            1,
        );
        assert!(
            out[0].is_nan(),
            "skip_gate(NaN branch) must be NaN (sigmoid(0)=0.5, 0.5*NaN = NaN)"
        );
    }

    #[test]
    fn skip_gate_nan_in_skip_propagates_nan() {
        let gate = [0.0f32];
        let branch = [1.0f32];
        let skip = [f32::NAN];
        let program = skip_gate("gate", "branch", "skip", "output", 1);
        let out = eval_f32(
            "skip_gate",
            &program,
            &[&gate[..], &branch[..], &skip[..]],
            1,
        );
        assert!(
            out[0].is_nan(),
            "skip_gate(NaN skip) must be NaN (0.5*NaN = NaN)"
        );
    }

    #[test]
    fn skip_gate_all_zeros() {
        let program = skip_gate("gate", "branch", "skip", "output", 4);
        let out = eval_f32(
            "skip_gate",
            &program,
            &[&[0.0; 4][..], &[0.0; 4][..], &[0.0; 4][..]],
            4,
        );
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn skip_gate_all_ones() {
        let program = skip_gate("gate", "branch", "skip", "output", 4);
        let out = eval_f32(
            "skip_gate",
            &program,
            &[&[1.0; 4][..], &[1.0; 4][..], &[1.0; 4][..]],
            4,
        );
        let s = sigmoid(1.0);
        let expected = s * 1.0 + (1.0 - s) * 1.0;
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected).abs() <= 1.0e-5,
                "skip_gate all-ones mismatch at {i}: {v}"
            );
        }
    }

    #[test]
    fn skip_gate_single_element() {
        let program = skip_gate("gate", "branch", "skip", "output", 1);
        let out = eval_f32(
            "skip_gate",
            &program,
            &[&[2.0][..], &[10.0][..], &[20.0][..]],
            1,
        );
        let s = sigmoid(2.0);
        let expected = s * 10.0 + (1.0 - s) * 20.0;
        assert!(
            (out[0] - expected).abs() <= 1.0e-5,
            "skip_gate single element mismatch"
        );
    }

    #[test]
    fn skip_gate_empty_tensor() {
        let program = skip_gate("gate", "branch", "skip", "output", 0);
        let out = eval_f32("skip_gate", &program, &[&[] as &[f32]], 0);
        assert!(out.is_empty());
    }
}
