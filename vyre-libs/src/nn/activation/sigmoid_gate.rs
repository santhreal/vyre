//! Element-wise sigmoid output gate.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use super::unary::typed_sigmoid_gate_program;

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
        return trap_program(OP_ID, None, "Fix: sigmoid_gate requires n > 0");
    }
    typed_sigmoid_gate_program(OP_ID, gate_logits, branch, output, n, dtype, false)
}
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || sigmoid_gate("gate", "branch", "output", 4),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_f32_slice(&[0.0, 1.0, -1.0, 100.0]),
                vyre_primitives::wire::pack_f32_slice(&[8.0, 2.0, -2.0, -7.0]),
            ]]
        }),
        Some(|| {
            let gate = [0.0_f32, 1.0, -1.0, 100.0];
            let branch = [8.0_f32, 2.0, -2.0, -7.0];
            let output = std::array::from_fn::<_, 4, _>(|index| {
                branch[index] / (1.0 + (-gate[index]).exp())
            });
            vec![vec![vyre_primitives::wire::pack_f32_slice(&output)]]
        }),
    )
    .with_category("nn")
    .with_tolerance(vyre_foundation::operation::TolerancePolicy::f32_ulp(2))
}
