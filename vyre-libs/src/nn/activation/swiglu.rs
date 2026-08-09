//! SwiGLU: `y = silu(gate) * up`.
//!
//! SwiGLU is the activation used in LLaMA, PaLM, and DeepSeek V4 Flash.
//! It takes two separate inputs (gate projection and up projection)
//! and produces one output.
//!
//! Category A composition.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

use crate::region::wrap_anonymous;

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
        return crate::invalid_program("vyre-libs::nn::swiglu", "Fix: swiglu requires n > 0");
    }
    let i = Expr::var("i");
    let g = Expr::cast(DataType::F32, Expr::load(gate, i.clone()));
    let u = Expr::cast(DataType::F32, Expr::load(up, i.clone()));
    let sigmoid_g = Expr::div(
        Expr::f32(1.0),
        Expr::add(
            Expr::f32(1.0),
            Expr::UnOp {
                op: UnOp::Exp,
                operand: Box::new(Expr::UnOp {
                    op: UnOp::Negate,
                    operand: Box::new(g.clone()),
                }),
            },
        ),
    );
    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::u32(n)),
            vec![Node::Store {
                buffer: output.into(),
                index: i,
                value: Expr::cast(dtype.clone(), Expr::mul(g, Expr::mul(u, sigmoid_g))),
            }],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage(gate, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(n),
            BufferDecl::storage(up, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(n),
            BufferDecl::output(output, 2, dtype).with_count(n),
        ],
        [64, 1, 1],
        vec![wrap_anonymous("vyre-libs::nn::swiglu", body)],
    )
}

inventory::submit! {
    crate::fixture_catalog::OpEntry {
        semantic_version: 1,
        signature: None,
        tier: vyre_foundation::operation::OperationTier::Library,
        laws: &[],
        tolerance: vyre_foundation::operation::TolerancePolicy::EXACT,
        id: "vyre-libs::nn::swiglu",
        build: Some(|| swiglu("gate", "up", "output", 4)),
        test_inputs: Some(|| {
            let to_bytes = vyre_primitives::wire::pack_f32_slice;
            vec![vec![
                to_bytes(&[0.0_f32, 1.0, -1.0, 2.0]), // gate
                to_bytes(&[1.0_f32, 2.0, 3.0, 4.0]),  // up
            ]]
        }),
        expected_output: Some(|| {
            let gate = [0.0_f32, 1.0, -1.0, 2.0];
            let up = [1.0_f32, 2.0, 3.0, 4.0];
            let out: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| {
                let sigmoid_g = 1.0 / (1.0 + (-g).exp());
                g * u * sigmoid_g
            }).collect();
            let bytes = vyre_primitives::wire::pack_f32_slice(&out);
            vec![vec![bytes]]
        }),
        category: Some("nn"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::byte_pack::decode_f32;
    use crate::test_support::byte_pack::f32_bytes;
    use vyre_reference::value::Value;

    fn swiglu_ref(g: f32, u: f32) -> f32 {
        let sigmoid_g = 1.0 / (1.0 + (-g).exp());
        g * u * sigmoid_g
    }

    #[test]
    fn swiglu_all_zeros() {
        let gate = [0.0f32; 4];
        let up = [1.0f32; 4];
        let program = swiglu("gate", "up", "output", 4);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&gate)),
                Value::from(f32_bytes(&up)),
                Value::from(vec![0u8; 16]),
            ],
        )
        .expect("Fix: swiglu all-zeros must execute");
        let out = decode_f32(&outputs[0].to_bytes());
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn swiglu_varied_values() {
        let gate = [1.0f32, -1.0, 0.5, -0.5];
        let up = [2.0f32, 3.0, 4.0, 5.0];
        let program = swiglu("gate", "up", "output", 4);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&gate)),
                Value::from(f32_bytes(&up)),
                Value::from(vec![0u8; 16]),
            ],
        )
        .expect("Fix: swiglu varied values must execute");
        let out = decode_f32(&outputs[0].to_bytes());
        for (i, (&v, (&g, &u))) in out.iter().zip(gate.iter().zip(up.iter())).enumerate() {
            let expected = swiglu_ref(g, u);
            assert!(
                (v - expected).abs() <= 1.0e-5,
                "swiglu mismatch at {i}: {v} != {expected}"
            );
        }
    }

    #[test]
    fn swiglu_empty_tensor() {
        let program = swiglu("gate", "up", "output", 0);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(vec![]),
                Value::from(vec![]),
                Value::from(vec![]),
            ],
        )
        .expect("Fix: swiglu n=0 must not panic");
        assert!(outputs[0].to_bytes().is_empty());
    }

    #[test]
    fn swiglu_nan_gate_propagates_nan() {
        let gate = [f32::NAN];
        let up = [1.0f32];
        let program = swiglu("gate", "up", "output", 1);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&gate)),
                Value::from(f32_bytes(&up)),
                Value::from(vec![0u8; 4]),
            ],
        )
        .expect("Fix: swiglu must not panic on NaN gate");
        let out = decode_f32(&outputs[0].to_bytes());
        assert!(out[0].is_nan(), "swiglu(NaN gate) must be NaN");
    }
}
