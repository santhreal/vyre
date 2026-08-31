//! Backward for `skip_gate`:
//!
//! `grad_gate = grad_out * σ(g) * (1-σ(g)) * (branch - skip)`

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{DataType, Expr, Program, UnOp};

const OP_ID: &str = "vyre-libs::nn::skip_gate_backward";

/// Backward for skip_gate (F32). Produces grad_gate.
#[must_use]
pub fn skip_gate_backward(
    gate: &str,
    branch: &str,
    skip: &str,
    grad_out: &str,
    grad_gate: &str,
    n: u32,
) -> Program {
    ElementwiseComposer::new(OP_ID, n)
        .add_input(gate, DataType::F32, n)
        .add_input(branch, DataType::F32, n)
        .add_input(skip, DataType::F32, n)
        .add_input(grad_out, DataType::F32, n)
        .add_output(grad_gate, DataType::F32, n)
        .build_pointwise(grad_gate, |i| {
            let g = Expr::load(gate, i.clone());
            let b = Expr::load(branch, i.clone());
            let s = Expr::load(skip, i.clone());
            let dy = Expr::load(grad_out, i);

            let sig = Expr::div(
                Expr::f32(1.0),
                Expr::add(
                    Expr::f32(1.0),
                    Expr::UnOp {
                        op: UnOp::Exp,
                        operand: Box::new(Expr::UnOp {
                            op: UnOp::Negate,
                            operand: Box::new(g),
                        }),
                    },
                ),
            );
            let grad = Expr::mul(
                dy,
                Expr::mul(
                    Expr::mul(sig.clone(), Expr::sub(Expr::f32(1.0), sig)),
                    Expr::sub(b, s),
                ),
            );
            grad
        })
}

const EXPECTED_SKIP_GATE_BACKWARD_OUTPUT_BYTES: [u8; 8] =
    [0x00, 0x00, 0xA0, 0xC0, 0x00, 0x00, 0x00, 0x80];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || skip_gate_backward("gate", "branch", "skip", "grad_out", "grad_gate", 2),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[0.0, 100.0]),
                to_f32(&[10.0, 20.0]),
                to_f32(&[30.0, 40.0]),
                to_f32(&[1.0, 1.0]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_SKIP_GATE_BACKWARD_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}
