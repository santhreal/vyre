//! Logit softcap: `y = tanh(x / cap) * cap`.
//!
//! Category A composition  -  element-wise. Used in the Parameter Golf
//! recipe to bound logits before cross-entropy loss (default cap=30.0).

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{Expr, Program, UnOp};

use crate::nn::f32_stability::flush_tiny;

const OP_ID: &str = "vyre-libs::nn::logit_softcap";

/// Build a Program that applies `tanh(x / cap) * cap` element-wise.
#[must_use]
pub fn logit_softcap(input: &str, output: &str, n: u32, cap: f32) -> Program {
    ElementwiseComposer::f32_unary(OP_ID, input, output, n, |x| {
        let scaled = Expr::div(x, Expr::f32(cap));
        let tanh_val = Expr::UnOp {
            op: UnOp::Tanh,
            operand: Box::new(scaled),
        };
        let result = Expr::mul(tanh_val, Expr::f32(cap));
        flush_tiny(result)
    })
}

const EXPECTED_LOGIT_SOFTCAP_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0xF4, 0xD0, 0x5D, 0x41, 0xDB, 0x5D, 0xE7, 0xC1, 0xD2, 0x63, 0xEF, 0x41,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || logit_softcap("input", "output", 4, 30.0),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_f32_slice;
            vec![vec![
                to_bytes(&[0.0_f32, 15.0, -60.0, 100.0]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_LOGIT_SOFTCAP_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
    .with_tolerance(vyre_foundation::operation::TolerancePolicy::f32_ulp(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::decode_f32;
    use crate::fixture_bytes::f32_bytes;
    use vyre_reference::value::Value;

    fn softcap_ref(x: f32, cap: f32) -> f32 {
        (x / cap).tanh() * cap
    }

    #[test]
    fn logit_softcap_nan_input_propagates_nan() {
        let input = [f32::NAN];
        let program = logit_softcap("input", "output", 1, 30.0);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[Value::from(f32_bytes(&input)), Value::from(vec![0u8; 4])],
        )
        .expect("Fix: logit_softcap must not panic on NaN input");
        let out = decode_f32(&outputs[0].to_bytes());
        assert!(out[0].is_nan(), "logit_softcap(NaN) must be NaN");
    }

    #[test]
    fn logit_softcap_inf_inputs() {
        let program = logit_softcap("input", "output", 2, 30.0);
        // +Inf → tanh(+Inf) * cap = 1.0 * cap = cap
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&[f32::INFINITY, 0.0])),
                Value::from(vec![0u8; 8]),
            ],
        )
        .expect("Fix: logit_softcap must not panic on +Inf input");
        let out = decode_f32(&outputs[0].to_bytes());
        assert_eq!(out[0], 30.0, "logit_softcap(+Inf) must clamp to cap");

        // -Inf → tanh(-Inf) * cap = -1.0 * cap = -cap
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&[f32::NEG_INFINITY, 0.0])),
                Value::from(vec![0u8; 8]),
            ],
        )
        .expect("Fix: logit_softcap must not panic on -Inf input");
        let out = decode_f32(&outputs[0].to_bytes());
        assert_eq!(out[0], -30.0, "logit_softcap(-Inf) must clamp to -cap");
    }

    #[test]
    fn logit_softcap_negative_zero_vs_positive_zero() {
        let program = logit_softcap("input", "output", 2, 30.0);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&[0.0f32, -0.0f32])),
                Value::from(vec![0u8; 8]),
            ],
        )
        .expect("Fix: logit_softcap must handle -0.0");
        let out = decode_f32(&outputs[0].to_bytes());
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits());
        // tanh(-0.0/cap) = -0.0, -0.0 * cap = -0.0, but flush_tiny may flush
        assert_eq!(
            out[1].to_bits(),
            0.0f32.to_bits(),
            "logit_softcap(-0.0) must be +0.0 after flush_tiny"
        );
    }

    #[test]
    fn logit_softcap_subnormal_input_is_flushed_to_zero() {
        let sub = f32::from_bits(1);
        let program = logit_softcap("input", "output", 1, 30.0);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[Value::from(f32_bytes(&[sub])), Value::from(vec![0u8; 4])],
        )
        .expect("Fix: logit_softcap must not panic on subnormal input");
        let out = decode_f32(&outputs[0].to_bytes());
        assert_eq!(
            out[0].to_bits(),
            0.0f32.to_bits(),
            "logit_softcap must flush tiny subnormal to +0.0"
        );
    }

    #[test]
    fn logit_softcap_all_zeros() {
        let input = [0.0f32; 4];
        let program = logit_softcap("input", "output", 4, 30.0);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[Value::from(f32_bytes(&input)), Value::from(vec![0u8; 16])],
        )
        .expect("Fix: logit_softcap all-zeros must execute");
        let out = decode_f32(&outputs[0].to_bytes());
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn logit_softcap_all_ones() {
        let input = [1.0f32; 4];
        let program = logit_softcap("input", "output", 4, 30.0);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[Value::from(f32_bytes(&input)), Value::from(vec![0u8; 16])],
        )
        .expect("Fix: logit_softcap all-ones must execute");
        let out = decode_f32(&outputs[0].to_bytes());
        let expected = softcap_ref(1.0, 30.0);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected).abs() <= 1.0e-5,
                "logit_softcap all-ones mismatch at {i}: {v}"
            );
        }
    }

    #[test]
    fn logit_softcap_all_max_f32() {
        let input = [f32::MAX; 4];
        let program = logit_softcap("input", "output", 4, 30.0);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[Value::from(f32_bytes(&input)), Value::from(vec![0u8; 16])],
        )
        .expect("Fix: logit_softcap all-max-f32 must not panic");
        let out = decode_f32(&outputs[0].to_bytes());
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(
                v, 30.0,
                "logit_softcap(f32::MAX) must clamp to cap at {i}: got {v}"
            );
        }
    }

    #[test]
    fn logit_softcap_single_element() {
        let input = [15.0f32];
        let program = logit_softcap("input", "output", 1, 30.0);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[Value::from(f32_bytes(&input)), Value::from(vec![0u8; 4])],
        )
        .expect("Fix: logit_softcap single element must execute");
        let out = decode_f32(&outputs[0].to_bytes());
        let expected = softcap_ref(15.0, 30.0);
        assert!(
            (out[0] - expected).abs() <= 1.0e-5,
            "logit_softcap single element mismatch"
        );
    }

    #[test]
    fn logit_softcap_empty_tensor() {
        let program = logit_softcap("input", "output", 0, 30.0);
        let outputs =
            vyre_reference::reference_eval(&program, &[Value::from(vec![]), Value::from(vec![])])
                .expect("Fix: logit_softcap n=0 must not panic");
        assert!(outputs[0].to_bytes().is_empty());
    }
}
