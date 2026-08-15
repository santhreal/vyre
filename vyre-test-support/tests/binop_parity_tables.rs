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
//! It does not check that a synthetic op's answer is right on hardware. The
//! reference restates the contract in host Rust and the pinned `oracle` column
//! restates it as data; agreeing with each other is necessary and not
//! sufficient. What each backend suite adds is the only thing neither can: the
//! answer the device actually produced for its own lowering.

use std::collections::BTreeSet;

use vyre_test_support::binop_parity::{
    assert_covers_every_synthetic_op, assert_covers_every_total_op,
    assert_every_driver_crate_has_a_recorded_parity_position, synthetic_u32_case,
    synthetic_u32_reference, synthetic_u32_reference_ops, total_u32_case, total_u32_reference,
    total_u32_reference_ops, total_u32_reference_values, SYNTHETIC_U32_BINOPS, TOTAL_U32_CASES,
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

/// Every declared row has a reference, and every reference a row.
///
/// The reference tables are the arms the backend suites read, so a row added to
/// an operand table without a reference must fail here, on any host, rather than
/// only on the machine that has the device.
#[test]
fn the_reference_tables_cover_exactly_the_declared_rows() {
    assert_covers_every_synthetic_op("shared reference", &synthetic_u32_reference_ops());
    assert_covers_every_total_op("shared reference", &total_u32_reference_ops());
}

/// The synthetic reference arms answer the boundary values they are here for.
///
/// A drifting reference makes the live comparison agree about a wrong answer,
/// and the operand table cannot catch that: it supplies operands, not
/// expectations. These are the load-bearing values as literals, so a mistyped
/// arm fails here rather than passing on every backend at once. Stated once
/// because the reference is owned once; it was stated per backend while the arms
/// were byte-identical copies, which proved only that the copy was faithful.
#[test]
fn the_synthetic_reference_arms_answer_their_boundary_values() {
    let mulhi = synthetic_u32_reference("mulhi");
    assert_eq!(mulhi(u32::MAX, u32::MAX), 0xFFFF_FFFE);
    assert_eq!(mulhi(0x1_0000, 0x1_0000), 1);

    let abs_diff = synthetic_u32_reference("abs_diff");
    assert_eq!(abs_diff(0, u32::MAX), u32::MAX);
    assert_eq!(abs_diff(100, 50), 50);

    let sat_add = synthetic_u32_reference("saturating_add");
    assert_eq!(sat_add(u32::MAX, 1), u32::MAX);
    assert_eq!(sat_add(0x8000_0000, 0x8000_0000), u32::MAX);

    let sat_sub = synthetic_u32_reference("saturating_sub");
    assert_eq!(sat_sub(1, u32::MAX), 0);
    assert_eq!(sat_sub(100, 50), 50);

    let sat_mul = synthetic_u32_reference("saturating_mul");
    // 2^16 * 2^16 overflows u32 exactly.
    assert_eq!(sat_mul(0x1_0000, 0x1_0000), u32::MAX);
    assert_eq!(sat_mul(1000, 1000), 1_000_000);

    let rotl = synthetic_u32_reference("rotate_left");
    // A rotate of 32 masks to identity; the sign bit wraps to bit 0.
    assert_eq!(rotl(1, 32), 1);
    assert_eq!(rotl(0x8000_0000, 1), 1);
    assert_eq!(rotl(0xDEAD_BEEF, 4), 0xEADB_EEFD);

    let rotr = synthetic_u32_reference("rotate_right");
    assert_eq!(rotr(1, 1), 0x8000_0000);
    assert_eq!(rotr(1, 32), 1);
    assert_eq!(rotr(0xDEAD_BEEF, 4), 0xFDEA_DBEE);
}

/// The total-contract reference arms answer the boundary values they are here for.
#[test]
fn the_total_reference_arms_answer_their_boundary_values() {
    let div = total_u32_reference("div");
    assert_eq!(div(1, 0), u32::MAX);
    assert_eq!(div(u32::MAX, 0), u32::MAX);
    assert_eq!(div(100, 7), 14);

    let rem = total_u32_reference("rem");
    assert_eq!(rem(1, 0), 0);
    assert_eq!(rem(100, 7), 2);

    let shl = total_u32_reference("shl");
    // A shift of 32 masks to zero, so `1 << 32 == 1`, never 0.
    assert_eq!(shl(1, 32), 1);
    assert_eq!(shl(1, 63), 0x8000_0000);

    let shr = total_u32_reference("shr");
    assert_eq!(shr(1, 32), 1);
    assert_eq!(shr(0xFF, 36), 0xF);
}

/// Every pinned oracle column agrees with the reference, on every host.
///
/// The comparison the backend suites used to make just before dispatching, which
/// needed no device and therefore never belonged behind one.
#[test]
fn every_total_row_agrees_with_the_reference() {
    for case in TOTAL_U32_CASES {
        assert_eq!(total_u32_reference_values(case), case.oracle);
    }
}

/// A renamed synthetic op is refused by the shared lookup.
///
/// The adversarial case at the boundary of `synthetic_u32_case`: a renamed row
/// must fail at the lookup rather than silently dispatch nothing.
#[test]
#[should_panic(expected = "no synthetic u32 binop case")]
fn an_undeclared_synthetic_op_is_refused_by_the_shared_lookup() {
    let _ = synthetic_u32_case("mulhi_but_renamed");
}

/// A renamed total-contract op is refused by the shared lookup.
#[test]
#[should_panic(expected = "no total u32 case")]
fn an_undeclared_total_op_is_refused_by_the_shared_lookup() {
    let _ = total_u32_case("div_but_renamed");
}

/// A renamed op has no reference either.
///
/// The lookup and the reference are separate tables, so an op can be absent from
/// one and present in the other. Both refusals are asserted.
#[test]
#[should_panic(expected = "no reference for `mulhi_but_renamed`")]
fn an_undeclared_synthetic_op_has_no_reference() {
    let _ = synthetic_u32_reference("mulhi_but_renamed");
}

/// A renamed total-contract op has no reference either.
#[test]
#[should_panic(expected = "no reference for `div_but_renamed`")]
fn an_undeclared_total_op_has_no_reference() {
    let _ = total_u32_reference("div_but_renamed");
}

/// Every driver crate in the workspace has a recorded position on this gate.
///
/// Asserted here rather than in a backend suite: this reads the workspace and
/// needs no device, so behind a GPU gate it ran only on the host that had that
/// GPU. A backend crate added while the other suite is the one running is still
/// a backend proving nothing.
#[test]
fn every_driver_crate_has_a_recorded_parity_position() {
    assert_every_driver_crate_has_a_recorded_parity_position();
}
