//! Volume oracle harness shared by the `sweep_dual_*_volume_oracle_matrix`
//! targets.
//!
//! Every one of those targets sweeps the same 16384 hostile operand pairs
//! through the same three assertions, and differs only in the operation id and
//! the independent expression the oracle expects. Restating the sweep per file
//! made ten copies of the loop and one place per copy where a weakened assert
//! could hide, so the sweep lives here and each target supplies only its id and
//! its expectation.

#![allow(dead_code)]

use vyre_reference::{dual_op_ids, resolve_dual};

/// Hostile operand pairs per operation.
const CASES: u32 = 16384;

#[path = "../support/dual_operands.rs"]
mod dual_operands;

use dual_operands::{binary_input, hostile_pair};

/// Sweep every hostile operand pair through both independent references and an
/// expectation written outside the crate under test.
///
/// Asserts, per pair: the operation stays registered, the two references agree
/// with each other, and both agree with `expected`. Volume `testing.volume` -
/// do NOT weaken to shape-only asserts.
///
/// # Panics
///
/// Panics when the operation is unregistered, when the two references diverge,
/// or when either disagrees with `expected`, naming the seed and operands.
pub(crate) fn assert_volume_oracle(op_id: &str, expected: impl Fn(u32, u32) -> u32) {
    assert!(
        dual_op_ids().contains(&op_id),
        "Fix: {op_id} must stay registered"
    );
    let (reference_a, reference_b) = resolve_dual(op_id).expect("Fix: dual reference must resolve");
    for seed in 0..CASES {
        let (left, right) = hostile_pair(seed);
        let input = binary_input(left, right);
        let expected = expected(left, right).to_le_bytes().to_vec();
        let output_a = reference_a(&input);
        let output_b = reference_b(&input);
        assert_eq!(
            output_a, output_b,
            "Fix: {op_id} dual refs diverged seed={seed}"
        );
        assert_eq!(
            output_a, expected,
            "Fix: {op_id} volume oracle mismatch seed={seed} left={left:#010x} right={right:#010x}"
        );
    }
}
