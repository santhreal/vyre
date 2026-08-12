//! Element-wise sigmoid output gate.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

use crate::region::wrap_anonymous;

const OP_ID: &str = "vyre-libs::nn::sigmoid_gate";

/// Multiply `branch` by `sigmoid(gate_logits)` element-wise in F32.
#[must_use]
pub fn sigmoid_gate(gate_logits: &str, branch: &str, output: &str, n: u32) -> Program {
    build_sigmoid_gate(gate_logits, branch, output, n, DataType::F32)
}

/// Multiply `branch` by `sigmoid(gate_logits)` using F32 math and typed storage.
///
/// # Errors
///
/// Returns `Err` for an empty vector or a non-floating activation dtype.
pub fn sigmoid_gate_typed(
    gate_logits: &str,
    branch: &str,
    output: &str,
    n: u32,
    dtype: DataType,
) -> Result<Program, String> {
    if n == 0 {
        return Err("Fix: sigmoid_gate_typed requires n > 0".to_string());
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(format!(
            "Fix: sigmoid_gate_typed supports F16, BF16, or F32 tensors; got {dtype:?}"
        ));
    }
    Ok(build_sigmoid_gate(gate_logits, branch, output, n, dtype))
}

fn build_sigmoid_gate(
    gate_logits: &str,
    branch: &str,
    output: &str,
    n: u32,
    dtype: DataType,
) -> Program {
    if n == 0 {
        return crate::invalid_program(OP_ID, "Fix: sigmoid_gate requires n > 0");
    }
    let index = Expr::var("index");
    let branch_value = Expr::cast(DataType::F32, Expr::load(branch, index.clone()));
    let gate = Expr::cast(DataType::F32, Expr::load(gate_logits, index.clone()));
    let sigmoid = Expr::div(
        Expr::f32(1.0),
        Expr::add(
            Expr::f32(1.0),
            Expr::UnOp {
                op: UnOp::Exp,
                operand: Box::new(Expr::UnOp {
                    op: UnOp::Negate,
                    operand: Box::new(gate),
                }),
            },
        ),
    );
    let body = vec![
        Node::let_bind("index", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(index.clone(), Expr::u32(n)),
            vec![Node::Store {
                buffer: output.into(),
                index: index.clone(),
                value: Expr::cast(dtype.clone(), Expr::mul(branch_value, sigmoid)),
            }],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage(gate_logits, 0, BufferAccess::ReadOnly, dtype.clone())
                .with_count(n),
            BufferDecl::storage(branch, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(n),
            BufferDecl::output(output, 2, dtype).with_count(n),
        ],
        [64, 1, 1],
        vec![wrap_anonymous(OP_ID, body)],
    )
}
