//! The shared binop parity tables are well-formed for every row they carry.
//!
//! # The class this closes
//!
//! Both driver suites now read their operands and their pinned oracle answers
//! from one table instead of from a copy each. That makes the table load
//! bearing in a way a copy was not: a row with a shorter `oracle` than `pairs`,
//! a duplicated op name that shadows the row a test meant to look up, or an
//! empty operand list all turn a live-GPU gate into a gate that dispatches
//! nothing and passes. None of those is a compile error.
//!
//! The row set is enumerated from the tables at run time, not listed here, so a
//! row added tomorrow is checked tomorrow. The failure mode of an enumeration
//! is finding nothing, so each table also has a floor: an empty table would
//! otherwise satisfy every per-row assertion.
//!
//! # What it does not catch
//!
//! It does not check that an `oracle` value is the semantically right answer.
//! That is what each backend's reference arm asserts, against its own
//! independently computed reference, on real hardware.

use std::collections::BTreeSet;

use vyre_test_support::binop_parity::{
    assert_covers_every_synthetic_op, assert_covers_every_total_op, synthetic_u32_case,
    total_u32_case, SYNTHETIC_U32_BINOPS, TOTAL_U32_CASES,
};

/// Minimum rows each table must carry, so a table emptied by a bad edit fails
/// instead of reporting a clean sweep of nothing.
const SYNTHETIC_FLOOR: usize = 7;
const TOTAL_FLOOR: usize = 4;

#[test]
fn every_synthetic_row_has_operands_and_a_unique_name() {
    assert!(
        SYNTHETIC_U32_BINOPS.len() >= SYNTHETIC_FLOOR,
        "SYNTHETIC_U32_BINOPS carries {} rows, below the floor of {SYNTHETIC_FLOOR}. Fix: a row \
         was removed without lowering the floor deliberately.",
        SYNTHETIC_U32_BINOPS.len()
    );
    let mut names = BTreeSet::new();
    for case in SYNTHETIC_U32_BINOPS {
        assert!(
            names.insert(case.op),
            "two synthetic rows are named {:?}; the second is unreachable through \
             synthetic_u32_case",
            case.op
        );
        assert!(
            !case.pairs().is_empty(),
            "synthetic row {:?} has no operands, so its gate would dispatch nothing and pass",
            case.op
        );
        assert_eq!(
            synthetic_u32_case(case.op).op,
            case.op,
            "synthetic_u32_case must find every row in the table"
        );
    }
}

#[test]
fn every_total_row_pins_one_oracle_value_per_operand_pair() {
    assert!(
        TOTAL_U32_CASES.len() >= TOTAL_FLOOR,
        "TOTAL_U32_CASES carries {} rows, below the floor of {TOTAL_FLOOR}. Fix: a row was \
         removed without lowering the floor deliberately.",
        TOTAL_U32_CASES.len()
    );
    let mut names = BTreeSet::new();
    for case in TOTAL_U32_CASES {
        assert!(
            names.insert(case.op),
            "two total rows are named {:?}; the second is unreachable through total_u32_case",
            case.op
        );
        assert!(
            !case.pairs.is_empty(),
            "total row {:?} has no operands, so its gate would dispatch nothing and pass",
            case.op
        );
        assert_eq!(
            case.oracle.len(),
            case.pairs.len(),
            "total row {:?} pins {} oracle values for {} operand pairs. A short pin makes the \
             reference-drift assertion compare unequal lengths and the operands past the end \
             unproven.",
            case.op,
            case.oracle.len(),
            case.pairs.len()
        );
        assert_eq!(
            total_u32_case(case.op).op,
            case.op,
            "total_u32_case must find every row in the table"
        );
    }
}

/// The coverage gates accept a suite that names exactly the declared rows.
///
/// The positive arm of the two assertions the driver suites open with. It reads
/// the op set from the tables rather than listing it, so a row added tomorrow is
/// required of the suites tomorrow.
#[test]
fn the_coverage_gates_accept_the_declared_op_set() {
    let synthetic: Vec<&str> = SYNTHETIC_U32_BINOPS.iter().map(|case| case.op).collect();
    assert_covers_every_synthetic_op("test", &synthetic);
    let total: Vec<&str> = TOTAL_U32_CASES.iter().map(|case| case.op).collect();
    assert_covers_every_total_op("test", &total);
}

/// A suite missing one declared synthetic op goes RED.
///
/// This is the proof the gate would have caught the defect it exists for: before
/// the gates, each backend suite carried one hand-written test per op, so a row
/// added to the table was dispatched by neither suite and nothing failed. Drop
/// the last row from what a suite claims and the gate must name it.
#[test]
#[should_panic(expected = "has no reference for")]
fn a_synthetic_op_no_suite_covers_is_named() {
    let mut covered: Vec<&str> = SYNTHETIC_U32_BINOPS.iter().map(|case| case.op).collect();
    covered.pop();
    assert_covers_every_synthetic_op("test", &covered);
}

/// A suite missing one declared total-contract op goes RED.
#[test]
#[should_panic(expected = "has no reference for")]
fn a_total_op_no_suite_covers_is_named() {
    let mut covered: Vec<&str> = TOTAL_U32_CASES.iter().map(|case| case.op).collect();
    covered.pop();
    assert_covers_every_total_op("test", &covered);
}

/// A suite naming an op the table does not declare goes RED.
///
/// The other direction: a renamed row leaves the suite asserting an op that no
/// longer exists, which would otherwise look like coverage.
#[test]
#[should_panic(expected = "which SYNTHETIC_U32_BINOPS does not")]
fn a_synthetic_op_the_table_dropped_is_named() {
    let mut covered: Vec<&str> = SYNTHETIC_U32_BINOPS.iter().map(|case| case.op).collect();
    covered.push("mulhi_but_renamed");
    assert_covers_every_synthetic_op("test", &covered);
}

/// A total-contract suite naming an op the table dropped goes RED.
#[test]
#[should_panic(expected = "which TOTAL_U32_CASES does not")]
fn a_total_op_the_table_dropped_is_named() {
    let mut covered: Vec<&str> = TOTAL_U32_CASES.iter().map(|case| case.op).collect();
    covered.push("div_but_renamed");
    assert_covers_every_total_op("test", &covered);
}
