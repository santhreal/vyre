//! A logical execution marker is never asked of a backend's operation set.
//!
//! # Why this suite exists
//!
//! Capability validation walks a program's nodes and refuses the first op id a
//! backend does not declare. Run on the semantic program, that walk reached
//! `vyre.node.logical_barrier` and refused every schedule-free composition
//! with `V056: backend `cuda` does not support operation`, which is the wrong
//! question twice over: no backend in this workspace lowers a logical marker,
//! and none needs to, because schedule lowering resolves every marker into a
//! physical one before an emitter sees the program. The refusal landed on 44
//! CUDA device parity cases whose programs were entirely legal.
//!
//! An unsupported distributed collective already had this shape and this
//! answer: `validate_program_contract` lowers it away and validates the
//! result. Markers are resolved in the same place, for the same reason.
//!
//! # What this suite does NOT claim
//!
//! It does not claim lowering resolved the markers correctly, nor that a
//! marker reaching an emitter is tolerable. `reject_logical_markers` in
//! `vyre-lower` owns that property and refuses it by name.

use vyre_driver::validation::{validate_program_contract, ProgramValidationCaps};
use vyre_foundation::ir::{MemoryOrdering, Node, Program};
use vyre_foundation::transform::schedule_lowering::lower_logical_schedule_borrowed;
use vyre_foundation::validate::ValidationOptions;
use vyre_test_support::logical_markers::{census, logical_marker_sum};

/// Every logical marker form in one body: all three position markers and the
/// barrier statement.
fn schedule_free_program() -> Program {
    Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![
            Node::Let {
                name: "position".into(),
                value: logical_marker_sum(),
            },
            Node::LogicalBarrier {
                ordering: MemoryOrdering::SeqCst,
            },
            Node::Return,
        ],
    )
}

fn caps() -> ProgramValidationCaps {
    ProgramValidationCaps {
        backend_id: "logical-marker-test",
        supports_subgroup_ops: false,
        supports_f16: false,
        supports_bf16: false,
        supports_indirect_dispatch: false,
        supports_distributed_collectives: false,
        supports_trap_propagation: true,
        supports_grid_sync: true,
        allows_host_grid_sync_split: true,
        max_workgroup_size: [256, 256, 64],
    }
}

fn validate(program: &Program) -> Result<(), vyre_driver::BackendError> {
    validate_program_contract(
        program,
        ValidationOptions::default(),
        vyre_driver::default_supported_ops(),
        caps(),
    )
}

/// Without this the cases below are vacuous: they would pass against a program
/// that carried no marker at all.
#[test]
fn the_fixture_actually_carries_every_logical_marker_form() {
    let counted = census(schedule_free_program().entry());
    assert_eq!(counted.logical, 3, "all three position markers");
    assert_eq!(counted.logical_barriers, 1, "the barrier statement");
}

/// The other half of non-vacuity: the marker really is absent from the
/// operation set being validated against, so admission cannot come from the
/// backend happening to declare it.
#[test]
fn no_marker_op_id_is_in_the_operation_set_being_validated() {
    let supported = vyre_driver::default_supported_ops();
    let markers = supported
        .iter()
        .filter(|op| op.contains("logical"))
        .collect::<Vec<_>>();
    assert!(
        markers.is_empty(),
        "core set declares logical markers, making admission vacuous: {markers:?}"
    );
}

#[test]
fn a_schedule_free_program_is_admitted() {
    assert_eq!(validate(&schedule_free_program()), Ok(()));
}

/// The class closure. Admission is only correct because legalization resolved
/// every marker form, so this asserts the census over the legalized program is
/// empty rather than asserting the barrier alone survived validation. A marker
/// form added to the IR without a `ScheduleLowering` arm leaves a nonzero count
/// here and turns this RED.
#[test]
fn legalization_resolves_every_marker_form_not_just_the_barrier() {
    let program = schedule_free_program();
    let legalized = lower_logical_schedule_borrowed(&program)
        .expect("a program carrying markers must be rewritten");
    let counted = census(legalized.entry());
    assert_eq!(
        (counted.logical, counted.logical_barriers),
        (0, 0),
        "markers survived legalization: {counted:?}"
    );
    assert_eq!(
        counted.physical_axes,
        [1, 1, 1],
        "each position marker became its own physical axis"
    );
    assert_eq!(
        counted.physical_orderings,
        vec![MemoryOrdering::SeqCst],
        "the barrier kept its ordering"
    );
}

/// Legalizing before validation must not blanket-admit unsupported operations.
/// A physical op the backend does not declare is still refused by name.
#[test]
fn an_unsupported_physical_operation_is_still_refused() {
    let program = Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![
            Node::trap(vyre_foundation::ir::Expr::u32(0), "t"),
            Node::Return,
        ],
    );
    let error = validate(&program).expect_err("trap is absent from the core set");
    let rendered = error.to_string();
    assert!(
        rendered.contains("vyre.node.trap"),
        "refusal must name the operation: {rendered}"
    );
}

/// A program that never carried a marker pays no allocation for legalization.
#[test]
fn an_already_physical_program_is_not_rewritten() {
    let physical = Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![
            Node::barrier_with_ordering(MemoryOrdering::SeqCst),
            Node::Return,
        ],
    );
    assert!(
        lower_logical_schedule_borrowed(&physical).is_none(),
        "a physical program must be returned borrowed, not cloned"
    );
}
