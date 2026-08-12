//! Element-wise residual-stream addition.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::region::wrap_anonymous;

const OP_ID: &str = "vyre-libs::nn::residual_add";

/// Compute `output[i] = residual[i] + branch[i]` in F32.
#[must_use]
pub fn residual_add(residual: &str, branch: &str, output: &str, n: u32) -> Program {
    build_residual_add(residual, branch, output, n, DataType::F32)
}

/// Add typed residual and branch tensors with F32 arithmetic.
///
/// # Errors
///
/// Returns `Err` for an empty vector or a non-floating activation dtype.
pub fn residual_add_typed(
    residual: &str,
    branch: &str,
    output: &str,
    n: u32,
    dtype: DataType,
) -> Result<Program, String> {
    if n == 0 {
        return Err("Fix: residual_add_typed requires n > 0".to_string());
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(format!(
            "Fix: residual_add_typed supports F16, BF16, or F32 tensors; got {dtype:?}"
        ));
    }
    Ok(build_residual_add(residual, branch, output, n, dtype))
}

fn build_residual_add(
    residual: &str,
    branch: &str,
    output: &str,
    n: u32,
    dtype: DataType,
) -> Program {
    if n == 0 {
        return crate::invalid_program(OP_ID, "Fix: residual_add requires n > 0");
    }
    let index = Expr::var("index");
    let body = vec![
        Node::let_bind("index", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(index.clone(), Expr::u32(n)),
            vec![Node::Store {
                buffer: output.into(),
                index: index.clone(),
                value: Expr::cast(
                    dtype.clone(),
                    Expr::add(
                        Expr::cast(DataType::F32, Expr::load(residual, index.clone())),
                        Expr::cast(DataType::F32, Expr::load(branch, index)),
                    ),
                ),
            }],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage(residual, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(n),
            BufferDecl::storage(branch, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(n),
            BufferDecl::output(output, 2, dtype).with_count(n),
        ],
        [64, 1, 1],
        vec![wrap_anonymous(OP_ID, body)],
    )
}
