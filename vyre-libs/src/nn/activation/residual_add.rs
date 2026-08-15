//! Element-wise residual-stream addition.

use vyre_foundation::algebra::composition::trap_program;
use vyre_foundation::ir::{DataType, Expr, Program};

use super::unary::typed_binary_activation_program;

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
        return trap_program(OP_ID, None, "Fix: residual_add requires n > 0");
    }
    typed_binary_activation_program(OP_ID, residual, branch, output, n, dtype, Expr::add)
}
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || residual_add("residual", "branch", "output", 4),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_f32_slice(&[1.0, -2.0, 3.5, 0.0]),
                vyre_primitives::wire::pack_f32_slice(&[0.5, 4.0, -1.5, -0.0]),
            ]]
        }),
        Some(|| {
            vec![vec![vyre_primitives::wire::pack_f32_slice(&[
                1.5, 2.0, 2.0, 0.0,
            ])]]
        }),
    )
    .with_category("nn")
}
