//! Cat-C `inverse_sqrt_f32`  -  finite-domain `1 / sqrt(x)` per f32 lane.
//! Inputs that are non-finite, negative, zero, or subnormal are clamped to
//! `f32::MIN_POSITIVE` before the reciprocal square root.

use vyre_foundation::ir::{DataType, Expr, Node, Program};

use crate::hardware::pack_f32;
/// Canonical op id shared by semantics, fixtures, and driver registration.
pub const OP_ID: &str = "vyre-primitives::hardware::inverse_sqrt_f32";

/// Build a Program that computes finite-domain `out[i] = 1.0 / sqrt(input[i])`.
#[must_use]
pub fn inverse_sqrt_f32(input: &str, out: &str, n: u32) -> Program {
    crate::hardware::unary_program(
        OP_ID,
        input,
        out,
        n,
        DataType::F32,
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len(out)),
                vec![
                    Node::let_bind("x", Expr::load(input, Expr::var("idx"))),
                    Node::let_bind(
                        "safe_x",
                        Expr::select(
                            Expr::and(
                                Expr::is_finite(Expr::var("x")),
                                Expr::gt(Expr::var("x"), Expr::f32(f32::MIN_POSITIVE)),
                            ),
                            Expr::var("x"),
                            Expr::f32(f32::MIN_POSITIVE),
                        ),
                    ),
                    Node::store(
                        out,
                        Expr::var("idx"),
                        Expr::inverse_sqrt(Expr::var("safe_x")),
                    ),
                ],
            ),
        ],
    )
}

fn test_inputs() -> Vec<Vec<Vec<u8>>> {
    let input = vec![1.0f32, 4.0, 9.0, 16.0];
    vec![vec![pack_f32(&input)]]
}

const EXPECTED_INVERSE_SQRT_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x80, 0x3f, // 1.0f32
    0x00, 0x00, 0x00, 0x3f, // 0.5f32
    0xab, 0xaa, 0xaa, 0x3e, // f32::from_bits(0x3eaaaaab)
    0x00, 0x00, 0x80, 0x3e, // 0.25f32
];

submit_hardware_intrinsic! {
    id: OP_ID,
    signature: crate::hardware::catalog::F32_UNARY_SIGNATURE,
    builder: || inverse_sqrt_f32("input", "out", 4),
    inputs: test_inputs,
    expected: || vec![vec![EXPECTED_INVERSE_SQRT_OUTPUT_BYTES.to_vec()]],
    effects: vyre_foundation::operation::OperationEffects::READ_WRITE,
    capabilities: vyre_foundation::program_caps::RequiredCapabilities::NONE,
    inputs_count: 1,
    outputs_count: 1,
    semantic: crate::hardware::catalog::HardwareSemantic::InverseSqrtF32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{inverse_sqrt_f32_ref, lcg_f32, run_program};

    fn test_cpu_ref(input: &[f32]) -> Vec<u8> {
        pack_f32(
            &input
                .iter()
                .copied()
                .map(inverse_sqrt_f32_ref)
                .collect::<Vec<_>>(),
        )
    }
    fn assert_case(input: &[f32]) {
        let n = input.len() as u32;
        let program = inverse_sqrt_f32("input", "out", n.max(1));
        let outputs = run_program(&program, vec![pack_f32(input)]);
        assert_eq!(outputs, vec![test_cpu_ref(input)]);
    }

    #[test]
    fn one_element() {
        assert_case(&[4.0]);
    }

    #[test]
    fn known_values() {
        assert_case(&[1.0, 4.0, 9.0, 16.0, 25.0, 100.0]);
    }

    #[test]
    fn random_sixty_four() {
        let input: Vec<f32> = lcg_f32(0x0F1A_A005, 64)
            .into_iter()
            .map(|v| v.abs() + 0.01)
            .collect();
        assert_case(&input);
    }

    #[test]
    fn clamps_non_finite_and_tiny_inputs() {
        assert_case(&[
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1.0,
            0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
        ]);
    }

    #[test]
    fn registration_fixture_matches_exact_byte_constant() {
        assert_eq!(
            EXPECTED_INVERSE_SQRT_OUTPUT_BYTES.to_vec(),
            test_cpu_ref(&[1.0, 4.0, 9.0, 16.0])
        );
    }
}
