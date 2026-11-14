//! Shared F32 unary activation Program builder.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{DataType, Expr, Program};

/// Build `output[i] = op(input[i])` for an F32 activation.
#[must_use]
pub(crate) fn f32_unary_activation_program<F>(
    op_id: &'static str,
    input: &str,
    output: &str,
    n: u32,
    op: F,
) -> Program
where
    F: Fn(Expr) -> Expr,
{
    ElementwiseComposer::f32_unary(op_id, input, output, n, op)
}

/// Build one typed binary activation map with F32 intermediate arithmetic.
pub(crate) fn typed_binary_activation_program(
    op_id: &'static str,
    left: &str,
    right: &str,
    output: &str,
    n: u32,
    dtype: DataType,
    combine: impl Fn(Expr, Expr) -> Expr,
) -> Program {
    ElementwiseComposer::binary(
        op_id,
        left,
        right,
        dtype.clone(),
        output,
        dtype.clone(),
        n,
        |l, r| {
            let left_value = Expr::cast(DataType::F32, l);
            let right_value = Expr::cast(DataType::F32, r);
            Expr::cast(dtype.clone(), combine(left_value, right_value))
        },
    )
}

/// Build a typed sigmoid gate, optionally multiplying by its gate input.
pub(crate) fn typed_sigmoid_gate_program(
    op_id: &'static str,
    gate: &str,
    branch: &str,
    output: &str,
    n: u32,
    dtype: DataType,
    include_gate: bool,
) -> Program {
    typed_binary_activation_program(op_id, gate, branch, output, n, dtype, |gate, branch| {
        let sigmoid = Expr::div(
            Expr::f32(1.0),
            Expr::add(Expr::f32(1.0), Expr::exp(Expr::negate(gate.clone()))),
        );
        let branch = Expr::mul(branch, sigmoid);
        if include_gate {
            Expr::mul(gate, branch)
        } else {
            branch
        }
    })
}
