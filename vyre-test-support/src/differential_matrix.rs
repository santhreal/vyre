//! Differential comparison of backend outputs against `vyre-reference`.
//!
//! Output count, binding order, and byte widths are part of the comparison.
//! F32 outputs use the operation's declared ULP tolerance. Other element types
//! require exact bytes. Callers report unsupported capabilities and unavailable
//! hardware with explicit decisions.

use vyre_foundation::fp_parity::{
    compare_operation_outputs, effective_tolerance, max_output_ulp, BufferParity,
};
use vyre_foundation::ir::Program;
use vyre_reference::value::Value;

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
        /// Output slot and numerical-policy diagnostic.
        reason: String,
        /// Reference oracle outputs in binding order.
        reference_buffers: Vec<Vec<u8>>,
        /// Backend outputs in binding order.
        backend_buffers: Vec<Vec<u8>>,
    },
}

/// Compare a backend's execution output against `vyre-reference`.
///
/// # Errors
/// Returns `Err` if reference execution fails or accepted outputs cannot be measured.
pub fn evaluate_differential(
    program: &Program,
    op_id: &str,
    inputs: &[Value],
    backend_outputs: &[Vec<u8>],
) -> Result<DifferentialDecision, String> {
    let ref_outputs = vyre_reference::ReferenceRequest::standard(program, inputs)
        .outputs()
        .map_err(|e| format!("reference interpreter evaluation failed: {e}"))?;

    let reference_buffers: Vec<Vec<u8>> = ref_outputs.iter().map(Value::to_bytes).collect();
    if let BufferParity::Mismatch(reason) =
        compare_operation_outputs(op_id, program, &reference_buffers, backend_outputs)
    {
        return Ok(DifferentialDecision::Mismatch {
            reason,
            reference_buffers,
            backend_buffers: backend_outputs.to_vec(),
        });
    }
    if reference_buffers.as_slice() == backend_outputs {
        return Ok(DifferentialDecision::ExactByteMatch);
    }
    let measured_ulp =
        max_output_ulp(program, &reference_buffers, backend_outputs).ok_or_else(|| {
            "accepted differential outputs have no aligned ULP measurement".to_string()
        })?;
    Ok(DifferentialDecision::WithinTolerance {
        measured_ulp,
        tolerance_ulp: effective_tolerance(op_id, program),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node};

    fn scalar_outputs(outputs: Vec<(DataType, Expr)>) -> (Program, Vec<Value>) {
        let mut buffers = Vec::with_capacity(outputs.len());
        let mut nodes = Vec::with_capacity(outputs.len());
        let mut inputs = Vec::with_capacity(outputs.len());
        for (index, (element, value)) in outputs.into_iter().enumerate() {
            let name = format!("out_{index}");
            inputs.push(Value::Bytes(vec![0; element.min_bytes()].into()));
            buffers.push(BufferDecl::read_write(&name, index as u32, element).with_count(1));
            nodes.push(Node::store(&name, Expr::u32(0), value));
        }
        (Program::wrapped(buffers, [1, 1, 1], nodes), inputs)
    }

    /// WHY: equal concatenated bytes do not prove output ABI equality.
    /// Exercise every split of the same bytes and buffer-count changes.
    #[test]
    fn output_boundaries_and_slot_order_are_part_of_differential_identity() {
        let (program, inputs) = scalar_outputs(vec![
            (DataType::U32, Expr::u32(1)),
            (DataType::U32, Expr::u32(2)),
        ]);
        let expected = vec![1u32.to_le_bytes().to_vec(), 2u32.to_le_bytes().to_vec()];
        let joined: Vec<u8> = expected.iter().flatten().copied().collect();
        let mut candidates = vec![
            vec![],
            vec![joined.clone()],
            vec![expected[1].clone(), expected[0].clone()],
            vec![expected[0].clone()],
            vec![expected[0].clone(), expected[1].clone(), vec![]],
        ];
        for split in 0..=joined.len() {
            if split != expected[0].len() {
                candidates.push(vec![joined[..split].to_vec(), joined[split..].to_vec()]);
            }
        }
        for backend in candidates {
            let decision =
                evaluate_differential(&program, "synthetic.boundaries", &inputs, &backend)
                    .expect("reference execution succeeds");
            let DifferentialDecision::Mismatch {
                reason,
                reference_buffers,
                backend_buffers,
            } = decision
            else {
                panic!("malformed output ABI was accepted: {backend:?}");
            };
            assert_eq!(reference_buffers, expected);
            assert_eq!(backend_buffers, backend);
            assert!(
                reason.starts_with("output buffer"),
                "the diagnostic must identify the output ABI: {reason}"
            );
        }
        assert_eq!(
            evaluate_differential(&program, "synthetic.boundaries", &inputs, &expected).unwrap(),
            DifferentialDecision::ExactByteMatch
        );
    }

    #[test]
    fn integer_bits_never_receive_float_tolerance() {
        let bits = 1.0f32.to_bits();
        for mixed in [false, true] {
            for integer_first in [false, true] {
                let mut outputs = vec![(DataType::U32, Expr::u32(bits))];
                let mut backend = vec![(bits + 1).to_le_bytes().to_vec()];
                if mixed {
                    let float = (DataType::F32, Expr::LitF32(1.0));
                    let bytes = bits.to_le_bytes().to_vec();
                    if integer_first {
                        outputs.push(float);
                        backend.push(bytes);
                    } else {
                        outputs.insert(0, float);
                        backend.insert(0, bytes);
                    }
                }
                let (program, inputs) = scalar_outputs(outputs);
                assert!(matches!(
                    evaluate_differential(&program, "synthetic.integer", &inputs, &backend)
                        .unwrap(),
                    DifferentialDecision::Mismatch { .. }
                ));
            }
        }
    }

    #[test]
    fn float_tolerance_preserves_integer_slots_and_reports_measured_distance() {
        let (program, inputs) = scalar_outputs(vec![
            (DataType::U32, Expr::u32(7)),
            (DataType::F32, Expr::LitF32(1.0)),
        ]);
        let tolerance = effective_tolerance("synthetic.float", &program);
        assert!(tolerance > 0 && tolerance < u32::MAX);
        for distance in [0, 1, tolerance, tolerance + 1] {
            let backend = vec![
                7u32.to_le_bytes().to_vec(),
                (1.0f32.to_bits() + distance).to_le_bytes().to_vec(),
            ];
            let decision =
                evaluate_differential(&program, "synthetic.float", &inputs, &backend).unwrap();
            if distance == 0 {
                assert_eq!(decision, DifferentialDecision::ExactByteMatch);
            } else if distance <= tolerance {
                assert_eq!(
                    decision,
                    DifferentialDecision::WithinTolerance {
                        measured_ulp: distance,
                        tolerance_ulp: tolerance,
                    }
                );
            } else {
                assert!(matches!(decision, DifferentialDecision::Mismatch { .. }));
            }
        }
    }

    #[test]
    fn nonfinite_substitution_cannot_pass_a_differential() {
        for (finite, infinite) in [(f32::MAX, f32::INFINITY), (-f32::MAX, f32::NEG_INFINITY)] {
            for (reference, backend) in [(finite, infinite), (infinite, finite)] {
                let (program, inputs) =
                    scalar_outputs(vec![(DataType::F32, Expr::LitF32(reference))]);
                assert!(matches!(
                    evaluate_differential(
                        &program,
                        "synthetic.nonfinite",
                        &inputs,
                        &[backend.to_le_bytes().to_vec()],
                    )
                    .unwrap(),
                    DifferentialDecision::Mismatch { .. }
                ));
            }
        }
    }

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
