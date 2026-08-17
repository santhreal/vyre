//! Shared fused linear + activation builder.

use std::sync::Arc;
use vyre_foundation::ir::{Expr, Program};

use crate::builder::gemm::{ContractionComposer, ContractionEpilogue};
use crate::plumbing::operand::tensor_ref::TensorRef;

pub(super) fn linear_fused_activation<F>(
    op_name: &'static str,
    op_id: &'static str,
    x: &str,
    w: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
    activation: F,
) -> Result<Program, String>
where
    F: Fn(Expr) -> Expr + Send + Sync + 'static,
{
    if in_dim == 0 {
        return Err(format!(
            "Fix: {op_name} in_dim=0 is invalid: empty reduction"
        ));
    }
    if out_dim == 0 {
        return Err(format!("Fix: {op_name} out_dim=0 is invalid: empty output"));
    }
    in_dim.checked_mul(out_dim).ok_or_else(|| {
        format!("Fix: {op_name} in_dim*out_dim overflows u32; reduce dimensions.")
    })?;

    let x_ref = TensorRef::f32_2d(x, 1, in_dim);
    let w_ref = TensorRef::f32_2d(w, in_dim, out_dim);
    let bias_ref = TensorRef::f32_1d(b, out_dim);
    let out_ref = TensorRef::f32_2d(out, 1, out_dim);

    let mut composer =
        ContractionComposer::matmul_2d(op_id, x_ref, w_ref, out_ref, 1, in_dim, out_dim)
            .with_epilogue(ContractionEpilogue::Activation {
                bias: Some(b.to_string()),
                activation: Arc::new(activation),
            })
            .with_workgroup_size([64, 1, 1]);
    composer.bias = Some(bias_ref);
    composer
        .build()
        .map_err(|e| format!("Fix: {op_name} build failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_fused_linear_activation_shape_matrix_builds() {
        let mut cases = 0usize;
        for in_dim in 1..=32 {
            for out_dim in 1..=32 {
                let program = linear_fused_activation(
                    "linear_identity",
                    "vyre-libs::nn::linear_identity",
                    "x",
                    "w",
                    "b",
                    "out",
                    in_dim,
                    out_dim,
                    |acc| acc,
                )
                .expect("Fix: generated fused linear activation dimensions must build.");
                let output = program
                    .buffers()
                    .iter()
                    .find(|buffer| buffer.is_output())
                    .expect("Fix: generated fused linear activation must declare output.");
                assert_eq!(output.count(), out_dim);
                cases += 1;
            }
        }
        assert_eq!(cases, 1_024);
    }

    #[test]
    fn fused_linear_activation_rejects_invalid_dimensions_and_overflow() {
        let empty_reduction = linear_fused_activation(
            "linear_identity",
            "vyre-libs::nn::linear_identity",
            "x",
            "w",
            "b",
            "out",
            0,
            1,
            |acc| acc,
        )
        .expect_err("empty reduction must be rejected");
        assert!(empty_reduction.contains("in_dim=0"));

        let empty_output = linear_fused_activation(
            "linear_identity",
            "vyre-libs::nn::linear_identity",
            "x",
            "w",
            "b",
            "out",
            1,
            0,
            |acc| acc,
        )
        .expect_err("empty output must be rejected");
        assert!(empty_output.contains("out_dim=0"));

        let overflow = linear_fused_activation(
            "linear_identity",
            "vyre-libs::nn::linear_identity",
            "x",
            "w",
            "b",
            "out",
            u32::MAX,
            2,
            |acc| acc,
        )
        .expect_err("weight element count overflow must be rejected");
        assert!(overflow.contains("overflows u32"));
    }
}
