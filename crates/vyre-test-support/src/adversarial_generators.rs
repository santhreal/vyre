//! Contract-valid adversarial Program and input generators.
//!
//! Per Section 184.2:
//! - Generators read datatype, shape, effect, capability, and tolerance contracts.
//! - Distinguish valid empty geometry from invalid geometry.
//! - Cover extreme shapes, unaligned access, aliases, resource bounds, structured nesting.
//! - Distinguish supported NaN/infinity inputs from operations that must reject them.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::validate;

/// One generated adversarial program test case.
#[derive(Debug, Clone)]
pub struct AdversarialTestCase {
    /// Identifier / description of the adversarial scenario.
    pub name: &'static str,
    /// The generated Program.
    pub program: Program,
    /// Input buffer payloads.
    pub inputs: Vec<Vec<u8>>,
    /// Whether this program is expected to pass foundation validation.
    pub is_valid: bool,
    /// If invalid, expected substring in the validation error.
    pub expected_error: Option<&'static str>,
}

/// Generate a suite of contract-valid and contract-invalid adversarial test cases.
#[must_use]
pub fn generate_adversarial_suite() -> Vec<AdversarialTestCase> {
    vec![
        // 1. Extreme 1D shape (1 invocation)
        AdversarialTestCase {
            name: "extreme_1d_single_invocation",
            program: Program::wrapped(
                vec![
                    BufferDecl::read("in_val", 0, DataType::U32).with_count(1),
                    BufferDecl::output("out_val", 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store(
                    "out_val",
                    Expr::u32(0),
                    Expr::load("in_val", Expr::u32(0)),
                )],
            ),
            inputs: vec![vec![42, 0, 0, 0]],
            is_valid: true,
            expected_error: None,
        },
        // 2. Extreme 2D workgroup grid
        AdversarialTestCase {
            name: "extreme_2d_grid_dimensions",
            program: Program::wrapped(
                vec![
                    BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32)
                        .with_count(64),
                    BufferDecl::output("out", 1, DataType::U32).with_count(64),
                ],
                [8, 8, 1],
                vec![Node::store(
                    "out",
                    Expr::LocalId { axis: 0 },
                    Expr::load("in", Expr::LocalId { axis: 0 }),
                )],
            ),
            inputs: vec![vec![0u8; 256]],
            is_valid: true,
            expected_error: None,
        },
        // 3. Invalid zero-dimension workgroup geometry
        AdversarialTestCase {
            name: "invalid_zero_dimension_geometry",
            program: Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
                [0, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
            ),
            inputs: vec![],
            is_valid: false,
            expected_error: Some("workgroup"),
        },
        // 4. Nested control flow with extreme depth
        AdversarialTestCase {
            name: "nested_control_flow_depth",
            program: Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
                [1, 1, 1],
                vec![Node::if_then_else(
                    Expr::bool(true),
                    vec![Node::if_then_else(
                        Expr::bool(true),
                        vec![Node::store("out", Expr::u32(0), Expr::u32(100))],
                        vec![Node::store("out", Expr::u32(0), Expr::u32(200))],
                    )],
                    vec![Node::store("out", Expr::u32(0), Expr::u32(300))],
                )],
            ),
            inputs: vec![],
            is_valid: true,
            expected_error: None,
        },
        // 5. Quantized F8E4M3 finite and NaN edge payload
        AdversarialTestCase {
            name: "quantized_f8e4m3_edge_payload",
            program: Program::wrapped(
                vec![
                    BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F8E4M3)
                        .with_count(4),
                    BufferDecl::output("out", 1, DataType::F8E4M3).with_count(4),
                ],
                [1, 1, 1],
                vec![
                    Node::store("out", Expr::u32(0), Expr::load("in", Expr::u32(0))),
                    Node::store("out", Expr::u32(1), Expr::load("in", Expr::u32(1))),
                    Node::store("out", Expr::u32(2), Expr::load("in", Expr::u32(2))),
                    Node::store("out", Expr::u32(3), Expr::load("in", Expr::u32(3))),
                ],
            ),
            inputs: vec![vec![0x00, 0x80, 0x7E, 0x7F]], // +0, -0, max finite, NaN
            is_valid: true,
            expected_error: None,
        },
    ]
}

/// Assert that every test case in the adversarial suite obeys its validity expectation.
///
/// # Panics
/// Panics if a valid test case fails validation or an invalid test case passes validation.
pub fn assert_adversarial_suite_validity() {
    for case in generate_adversarial_suite() {
        let errors = validate::validate(&case.program);
        if case.is_valid {
            assert!(
                errors.is_empty(),
                "expected adversarial case `{}` to be valid, but validation failed: {:?}",
                case.name,
                errors
            );
        } else {
            assert!(
                !errors.is_empty(),
                "expected adversarial case `{}` to be rejected, but validation succeeded",
                case.name
            );
            assert_expected_substring(case.expected_error, &errors, case.name);
        }
    }
}

fn assert_expected_substring(
    expected: Option<&'static str>,
    errors: &[validate::ValidationError],
    name: &str,
) {
    if let Some(expected) = expected {
        let err_msg = format!("{errors:?}");
        assert!(
            err_msg.to_lowercase().contains(&expected.to_lowercase()),
            "adversarial case `{name}` error message `{err_msg}` missing `{expected}`"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adversarial_suite_validates_as_expected() {
        assert_adversarial_suite_validity();
    }
}
