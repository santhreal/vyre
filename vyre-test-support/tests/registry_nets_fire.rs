//! The four registry nets fail on the defect each one names.
//!
//! WHY: a sweep that walks a whole registry and reports clean is the shape a
//! vacuous gate takes here. Every net below is handed a population built to
//! carry exactly the defect it claims to catch, and the net is required to
//! panic. A detector that always reports equal proves nothing about the
//! hundreds of registrations it walks, and the two surfaces that call these
//! nets have no other way to know the driver still fires.
//!
//! The refusals are proven the same way: a case the reference cannot evaluate
//! and an empty population are failures, because a net that skips what it
//! cannot judge reports a clean sweep of a subset.

#![forbid(unsafe_code)]

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_reference::value::Value;
use vyre_test_support::registry_nets::{RegistrySweep, SweepCase};

/// A program whose body is one region, the shape a registered builder emits.
fn program(buffers: Vec<BufferDecl>, lanes: u32, body: Vec<Node>) -> Program {
    Program::wrapped(
        buffers,
        [lanes, 1, 1],
        vec![Node::Region {
            generator: Ident::from("test::registry_nets_fire"),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}

/// The global invocation index, which is what an over-fired dispatch widens.
///
/// `Expr::local_x()` is the index within one workgroup, so it reads 0..lanes-1
/// at every grid: a control written against it cannot tell a wider dispatch
/// from a narrower one, and the nets it is meant to trip stay silent.
fn invocation() -> Expr {
    Expr::InvocationId { axis: 0 }
}

/// Four lanes racing on one slot: the winner is the last lane stepped, and the
/// value it writes is its own index, so a wider grid also changes the answer.
fn racing_program() -> Program {
    program(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        4,
        vec![Node::store("out", Expr::u32(0), invocation())],
    )
}

/// Each lane owns its own slot, so no step order changes the answer.
fn disjoint_program() -> Program {
    program(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)],
        4,
        vec![Node::store("out", invocation(), invocation())],
    )
}

/// A sweep of exactly one case.
fn one(label: &str, program: Program, inputs: Vec<Value>) -> RegistrySweep {
    RegistrySweep::new(
        "the control population",
        vec![SweepCase::new(label.to_string(), program, inputs)],
    )
}

/// The message a net panicked with, or `None` when it passed.
fn failure(net: impl FnOnce()) -> Option<String> {
    match catch_unwind(AssertUnwindSafe(net)) {
        Ok(()) => None,
        Err(payload) => Some(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    payload
                        .downcast_ref::<&str>()
                        .map(|text| (*text).to_string())
                })
                .unwrap_or_else(|| "a non-string panic payload".to_string()),
        ),
    }
}

#[test]
fn the_lane_reversal_net_fails_on_a_racy_population() {
    let sweep = one("racy", racing_program(), vec![Value::from(vec![0u8; 4])]);
    let message = failure(|| sweep.assert_race_free_under_lane_reversal())
        .expect("Fix: a non-atomic write-write race must fail the lane-reversal net");
    assert!(
        message.contains("lane STEP ORDER"),
        "the failure must name the reversed step order, not some other net: {message}"
    );
}

#[test]
fn the_lane_reversal_net_passes_a_disjoint_population() {
    let sweep = one(
        "disjoint",
        disjoint_program(),
        vec![Value::from(vec![0u8; 16])],
    );
    assert_eq!(
        failure(|| sweep.assert_race_free_under_lane_reversal()),
        None,
        "a scatter into disjoint slots is order-invariant, so the net must not fire on it"
    );
}

#[test]
fn reversing_the_step_order_changes_which_lane_wins_a_race() {
    let program = racing_program();
    let inputs = vec![Value::from(vec![0u8; 4])];
    let forward = vyre_reference::reference_eval(&program, &inputs).expect("forward eval");
    let reversed =
        vyre_reference::reference_eval_lane_reversed(&program, &inputs).expect("reversed eval");
    assert_eq!(
        forward[0].to_bytes(),
        3u32.to_le_bytes().to_vec(),
        "stepping forward, the last lane wins the race on out[0]"
    );
    assert_eq!(
        reversed[0].to_bytes(),
        0u32.to_le_bytes().to_vec(),
        "stepping in reverse, the first lane wins, which is what the net compares"
    );
}

#[test]
fn the_out_of_bounds_net_fails_on_an_unguarded_index() {
    let sweep = one(
        "past the end",
        program(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            ],
            4,
            vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::load("input", Expr::u32(8)),
            )],
        ),
        vec![Value::from(vec![0u8; 16]), Value::from(vec![0u8; 4])],
    );
    let message = failure(|| sweep.assert_oob_clean())
        .expect("Fix: a load eight elements into a four-element buffer must fail the OOB net");
    assert!(
        message.contains("OUT OF BOUNDS"),
        "the failure must name the out-of-bounds access: {message}"
    );
}

#[test]
fn the_overfire_invariance_net_fails_on_a_grid_sensitive_population() {
    let sweep = one(
        "grid sensitive",
        racing_program(),
        vec![Value::from(vec![0u8; 4])],
    );
    let message = failure(|| sweep.assert_output_invariant_under_overfire()).expect(
        "Fix: a program whose answer is the highest lane index must fail the over-fire net",
    );
    assert!(
        message.contains("OVER-FIRED"),
        "the failure must name the over-fired dispatch: {message}"
    );
}

#[test]
fn the_overfire_out_of_bounds_net_fails_on_a_guard_that_evaluates_both_arms() {
    let sweep = one(
        "select is not a guard",
        program(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(4),
            ],
            4,
            vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::select(
                    Expr::lt(invocation(), Expr::u32(4)),
                    Expr::load("input", invocation()),
                    Expr::u32(0),
                ),
            )],
        ),
        vec![Value::from(vec![0u8; 16]), Value::from(vec![0u8; 16])],
    );
    let message = failure(|| sweep.assert_oob_clean_under_overfire()).expect(
        "Fix: `Expr::select` evaluates both arms, so an over-fired lane still issues the load",
    );
    assert!(
        message.contains("OVER-FIRED"),
        "the failure must name the over-fired dispatch: {message}"
    );
}

#[test]
fn a_case_the_reference_cannot_evaluate_is_refused_rather_than_skipped() {
    let sweep = one("no inputs supplied", disjoint_program(), Vec::new());
    let message = failure(|| sweep.assert_race_free_under_lane_reversal())
        .expect("Fix: an un-evaluable case leaves its program unjudged and must fail the net");
    assert!(
        message.contains("could not be evaluated"),
        "the failure must name the un-evaluable case rather than the contract: {message}"
    );
}

#[test]
fn an_empty_population_is_refused() {
    let sweep = RegistrySweep::new("the control population", Vec::new());
    let message = failure(|| sweep.assert_oob_clean())
        .expect("Fix: an empty walk passes every net without proving anything");
    assert!(
        message.contains("evaluated no case"),
        "the failure must name the empty population: {message}"
    );
}
