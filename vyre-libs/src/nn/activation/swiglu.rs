//! SwiGLU: `y = silu(gate) * up`.
//!
//! SwiGLU is the activation used in LLaMA, PaLM, and DeepSeek V4 Flash.
//! It takes two separate inputs (gate projection and up projection)
//! and produces one output.
//!
//! Category A composition.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use super::unary::typed_sigmoid_gate_program;

const OP_ID: &str = "vyre-libs::nn::swiglu";
/// Build a Program that applies SwiGLU element-wise from `gate` and `up`
/// into `output`. `n` is the element count of all three buffers.
#[must_use]
pub fn swiglu(gate: &str, up: &str, output: &str, n: u32) -> Program {
    build_swiglu(gate, up, output, n, DataType::F32)
}

/// Build typed SwiGLU with F32 activation math and source-dtype output.
///
/// # Errors
///
/// Returns `Err` for an empty vector or a non-floating activation dtype.
pub fn swiglu_typed(
    gate: &str,
    up: &str,
    output: &str,
    n: u32,
    dtype: DataType,
) -> Result<Program, String> {
    if n == 0 {
        return Err("Fix: swiglu_typed requires n > 0".to_string());
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(format!(
            "Fix: swiglu_typed supports F16, BF16, or F32 tensors; got {dtype:?}"
        ));
    }
    Ok(build_swiglu(gate, up, output, n, dtype))
}

fn build_swiglu(gate: &str, up: &str, output: &str, n: u32, dtype: DataType) -> Program {
    if n == 0 {
        return trap_program(OP_ID, None, "Fix: swiglu requires n > 0");
    }
    typed_sigmoid_gate_program(OP_ID, gate, up, output, n, dtype, true)
}

const EXPECTED_SWIGLU_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0xA8, 0x26, 0xBB, 0x3F, 0x08, 0x8C, 0x4E, 0xBF, 0xEA, 0x7B, 0xE1, 0x40,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || swiglu("gate", "up", "output", 4),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_f32_slice;
            vec![vec![
                to_bytes(&[0.0_f32, 1.0, -1.0, 2.0]), // gate
                to_bytes(&[1.0_f32, 2.0, 3.0, 4.0]),  // up
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_SWIGLU_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::eval_f32;

    fn swiglu_ref(g: f32, u: f32) -> f32 {
        let sigmoid_g = 1.0 / (1.0 + (-g).exp());
        g * u * sigmoid_g
    }

    #[test]
    fn swiglu_all_zeros() {
        let gate = [0.0f32; 4];
        let up = [1.0f32; 4];
        let program = swiglu("gate", "up", "output", 4);
        let out = eval_f32("swiglu", &program, &[&gate[..], &up[..]], 4);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn swiglu_varied_values() {
        let gate = [1.0f32, -1.0, 0.5, -0.5];
        let up = [2.0f32, 3.0, 4.0, 5.0];
        let program = swiglu("gate", "up", "output", 4);
        let out = eval_f32("swiglu", &program, &[&gate[..], &up[..]], 4);
        for (i, (&v, (&g, &u))) in out.iter().zip(gate.iter().zip(up.iter())).enumerate() {
            let expected = swiglu_ref(g, u);
            assert!(
                (v - expected).abs() <= 1.0e-5,
                "swiglu mismatch at {i}: {v} != {expected}"
            );
        }
    }

    #[test]
    fn swiglu_empty_tensor_is_rejected() {
        let error = swiglu_typed("gate", "up", "output", 0, DataType::F32)
            .expect_err("Fix: swiglu n=0 must be rejected before execution.");
        assert_eq!(error, "Fix: swiglu_typed requires n > 0");
    }

    #[test]
    fn swiglu_nan_gate_propagates_nan() {
        let gate = [f32::NAN];
        let up = [1.0f32];
        let program = swiglu("gate", "up", "output", 1);
        let out = eval_f32("swiglu", &program, &[&gate[..], &up[..]], 1);
        assert!(out[0].is_nan(), "swiglu(NaN gate) must be NaN");
    }
}
