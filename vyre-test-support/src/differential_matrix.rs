//! Capability-driven differential execution matrix.
//!
//! Per Section 184.4:
//! - Dispatches the same canonical Program and input through `vyre-reference` and
//!   every acquired backend that declares required capabilities.
//! - Integer and exact contracts compare bytes bitwise.
//! - Floating and approximate contracts compare using their registered tolerance.
//! - Inapplicable backend-operation pairs carry an explicit decision.

use vyre_foundation::fp_parity::effective_tolerance;
use vyre_foundation::ir::Program;
use vyre_reference::{reference_eval, value::Value};

/// Decision outcome for a differential comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DifferentialDecision {
    /// Exact byte equality confirmed.
    ExactByteMatch,
    /// Floating-point result matches within declared ULP tolerance.
    WithinTolerance {
        /// Measured ULP difference.
        measured_ulp: u32,
        /// Maximum allowed ULP tolerance.
        tolerance_ulp: u32,
    },
    /// Backend does not support the required capability.
    Inapplicable {
        /// Documented reason for inapplicability.
        reason: &'static str,
    },
    /// Hardware declared as required was unavailable.
    HardwareUnavailable {
        /// Hardware or driver identifier.
        device: String,
    },
    /// Mismatch between reference and backend.
    Mismatch {
        /// Reference oracle output bytes.
        reference_bytes: Vec<u8>,
        /// Backend output bytes.
        backend_bytes: Vec<u8>,
    },
}

/// Compare a backend's execution output against `vyre-reference`.
///
/// # Errors
/// Returns `Err` if reference execution fails.
pub fn evaluate_differential(
    program: &Program,
    op_id: &str,
    inputs: &[Value],
    backend_outputs: &[Vec<u8>],
) -> Result<DifferentialDecision, String> {
    let ref_outputs = reference_eval(program, inputs)
        .map_err(|e| format!("reference interpreter evaluation failed: {e}"))?;

    let ref_bytes: Vec<u8> = ref_outputs.iter().flat_map(|v| v.to_bytes()).collect();
    let back_bytes: Vec<u8> = backend_outputs.iter().flatten().copied().collect();

    if ref_bytes == back_bytes {
        return Ok(DifferentialDecision::ExactByteMatch);
    }

    let tolerance = effective_tolerance(op_id, program);
    if tolerance > 0 && ref_bytes.len() == back_bytes.len() && ref_bytes.len() % 4 == 0 {
        let count = ref_bytes.len() / 4;
        let mut max_ulp = 0u32;
        for i in 0..count {
            let r_bits = u32::from_le_bytes(ref_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
            let b_bits = u32::from_le_bytes(back_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
            let r_f = f32::from_bits(r_bits);
            let b_f = f32::from_bits(b_bits);
            let ulp = vyre_reference::ieee754::canonical_ulp_distance(r_f, b_f);
            max_ulp = max_ulp.max(ulp);
        }

        if max_ulp <= tolerance {
            return Ok(DifferentialDecision::WithinTolerance {
                measured_ulp: max_ulp,
                tolerance_ulp: tolerance,
            });
        }
    }

    Ok(DifferentialDecision::Mismatch {
        reference_bytes: ref_bytes,
        backend_bytes: back_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node};

    #[test]
    fn exact_integer_differential_matches() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
        );

        let decision =
            evaluate_differential(&program, "test_op", &[], &[42u32.to_le_bytes().to_vec()])
                .expect("differential evaluation must succeed");

        assert_eq!(decision, DifferentialDecision::ExactByteMatch);
    }
}
