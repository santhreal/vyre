//! Cat-C `fma_f32`  -  fused multiply-add per f32 lane.
//! CPU reference: `f32::mul_add` BYTE-IDENTICAL (never multiply-then-add).
//!
//! Round-mode guarantee: this op promises IEEE-754 single-round fused
//! semantics, matching `f32::mul_add` bit-for-bit. A backend that cannot emit
//! a true fused instruction must report `UnsupportedByBackend`; it must NOT
//! silently degrade to `a * b + c`, which double-rounds and changes results.
//! Callers that explicitly want multiply-then-add semantics must build that as
//! a different Program and accept the different rounding contract.

use vyre_foundation::ir::Program;

use crate::hardware::{pack_f32, ternary_f32_program};
/// Canonical op id shared by semantics, fixtures, and driver registration.
pub const OP_ID: &str = "vyre-primitives::hardware::fma_f32";

/// Map `out[i] = fma(a[i], b[i], c[i])` over n elements.
///
/// # FMA capability and round-mode guarantee
///
/// This op requires the backend to advertise the `FMA` capability.  If the
/// backend reports `FMA` as absent, lowering **must** emit a clear
/// `BackendError::Unsupported`  -  it must
/// **never** silently fall back to `a * b + c`, because IEEE-754 multiply-then-add
/// double-rounds and produces a different result from single-round fused
/// multiply-add.  Callers that want the weaker `a * b + c` contract must build
/// that expression explicitly and accept the rounding divergence.
#[must_use]
pub fn fma_f32(a: &str, b: &str, c: &str, out: &str, n: u32) -> Program {
    ternary_f32_program(OP_ID, a, b, c, out, n)
}

fn test_inputs() -> Vec<Vec<Vec<u8>>> {
    let a = vec![0.0f32, 1.0, -2.5, f32::MAX];
    let b = vec![1.0f32, -3.0, 4.0, 0.5];
    let c = vec![0.0f32, 0.25, -1.0, 2.0];
    vec![vec![pack_f32(&a), pack_f32(&b), pack_f32(&c)]]
}

const EXPECTED_FMA_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, // 0.0f32
    0x00, 0x00, 0x30, 0xc0, // -2.75f32
    0x00, 0x00, 0x30, 0xc1, // -11.0f32
    0xff, 0xff, 0xff, 0x7e, // f32::from_bits(0x7eff_ffff)
];

submit_hardware_intrinsic! {
    id: OP_ID,
    signature: crate::hardware::catalog::F32_TERNARY_SIGNATURE,
    builder: || fma_f32("a", "b", "c", "out", 4),
    inputs: test_inputs,
    expected: || vec![vec![EXPECTED_FMA_OUTPUT_BYTES.to_vec()]],
    effects: vyre_foundation::operation::OperationEffects::READ_WRITE,
    capabilities: vyre_foundation::program_caps::RequiredCapabilities::NONE,
    inputs_count: 3,
    outputs_count: 1,
    semantic: crate::hardware::catalog::HardwareSemantic::FmaF32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{lcg_f32, run_program};

    fn test_cpu_ref(a: &[f32], b: &[f32], c: &[f32]) -> Vec<u8> {
        pack_f32(
            &a.iter()
                .zip(b.iter())
                .zip(c.iter())
                .map(|((&x, &y), &z)| x.mul_add(y, z))
                .collect::<Vec<_>>(),
        )
    }

    fn assert_case(a: &[f32], b: &[f32], c: &[f32]) {
        let n = a.len() as u32;
        let program = fma_f32("a", "b", "c", "out", n.max(1));
        let outputs = run_program(&program, vec![pack_f32(a), pack_f32(b), pack_f32(c)]);
        assert_eq!(outputs, vec![test_cpu_ref(a, b, c)]);
    }

    #[test]
    fn one_element() {
        assert_case(&[1.5], &[2.0], &[0.25]);
    }

    #[test]
    fn max_value() {
        assert_case(&[f32::MAX], &[1.0], &[0.0]);
    }

    #[test]
    fn random_sixty_four() {
        let a = lcg_f32(0x0F1A_A001, 64);
        let b = lcg_f32(0x0F1A_A002, 64);
        let c = lcg_f32(0x0F1A_A003, 64);
        assert_case(&a, &b, &c);
    }

    #[test]
    fn registration_fixture_matches_exact_byte_constant() {
        assert_eq!(
            EXPECTED_FMA_OUTPUT_BYTES.to_vec(),
            test_cpu_ref(
                &[0.0, 1.0, -2.5, f32::MAX],
                &[1.0, -3.0, 4.0, 0.5],
                &[0.0, 0.25, -1.0, 2.0]
            )
        );
    }
}
