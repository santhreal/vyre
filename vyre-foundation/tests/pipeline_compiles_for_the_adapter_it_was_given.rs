//! The pass pipeline compiles for the adapter it was handed, not a profile
//! the passes picked for themselves.
//!
//! WHY: `Autotune` and `DecodeScanFuse` both read device facts, and both had a
//! `ProgramPass::transform` that hardcoded `AdapterCaps::conservative()`. The
//! caps-aware entry existed the whole time and only a caller who already knew
//! to ask could reach it, so every program that went through the scheduler was
//! tuned for a device with no optional features, one workgroup lane deep and a
//! 16 KiB shared-memory budget, whatever the real adapter reported. Nothing
//! failed: the pipeline produced a valid program and a slower one.
//!
//! These tests fail if a scheduler built for a known adapter produces the
//! conservative program. They do not check a particular tuned value: the point
//! is that the adapter reached the pass at all, so each asserts the two
//! profiles disagree and that the scheduler's output follows the profile it
//! was given.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::passes::specialization::autotune::Autotune;
use vyre_foundation::optimizer::{AdapterCaps, PassScheduler, ProgramPassKind};

/// A profile the tuner must respect and cannot reach by accident: fewer lanes
/// per workgroup than the conservative fallback allows.
///
/// Widening does not discriminate here. The scheduling policy's own ceiling
/// is 256 lanes and the conservative fallback already permits 256, so a
/// roomier device produces the same program and would prove nothing. A device
/// that permits fewer is a fact only the adapter carries.
fn constrained_adapter() -> AdapterCaps {
    AdapterCaps {
        max_workgroup_size: [64, 1, 1],
        max_invocations_per_workgroup: 64,
        subgroup_size: 32,
        ..AdapterCaps::conservative()
    }
}

/// A 1-D kernel large enough that the tuner has something to widen.
fn wide_kernel() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(65_536)],
        [32, 1, 1],
        vec![Node::store("out", Expr::gid_x(), Expr::u32(1))],
    )
}

#[test]
fn a_scheduler_built_for_an_adapter_does_not_produce_the_conservative_program() {
    let constrained = constrained_adapter();
    let floor = AdapterCaps::conservative();
    assert!(
        constrained.max_invocations_per_workgroup < floor.max_invocations_per_workgroup,
        "Fix: the two profiles must disagree or this test proves nothing."
    );

    let program = wide_kernel();
    let fallback = PassScheduler::with_passes(vec![ProgramPassKind::new(Autotune)])
        .run(program.clone())
        .expect("Fix: the fallback pipeline must converge.");
    let tuned = PassScheduler::for_adapter(vec![ProgramPassKind::new(Autotune)], constrained)
        .run(program)
        .expect("Fix: the adapter pipeline must converge.");

    assert_ne!(
        tuned.workgroup_size(),
        fallback.workgroup_size(),
        "Fix: the scheduler was given an adapter that permits {} invocations per workgroup and \
         still produced the workgroup size the conservative fallback produces, {:?}. The adapter \
         is not reaching the pass: check that the pass declares `adapter_dependent = true` and \
         that the scheduler calls `batch_apply` with its own adapter.",
        constrained.max_invocations_per_workgroup,
        fallback.workgroup_size()
    );
    assert!(
        tuned.workgroup_size()[0] <= constrained.max_workgroup_size[0],
        "Fix: the tuned workgroup {:?} exceeds what the adapter reports it can dispatch.",
        tuned.workgroup_size()
    );
}

#[test]
fn the_scheduler_reports_the_adapter_it_was_built_for() {
    let constrained = constrained_adapter();
    let scheduler = PassScheduler::for_adapter(vec![ProgramPassKind::new(Autotune)], constrained);
    assert_eq!(
        scheduler.adapter().max_invocations_per_workgroup,
        constrained.max_invocations_per_workgroup,
        "Fix: a scheduler must carry the adapter it was constructed with."
    );

    let fallback = PassScheduler::with_passes(vec![ProgramPassKind::new(Autotune)]);
    assert_eq!(
        fallback.adapter().backend,
        AdapterCaps::conservative().backend,
        "Fix: a scheduler built without an adapter must state the conservative fallback, so the \
         profile a program was compiled against is readable from the scheduler rather than \
         hidden inside whichever pass reached for one first."
    );
}

#[test]
fn an_ir_only_pass_ignores_the_adapter() {
    use vyre_foundation::optimizer::passes::cleanup::empty_block_collapse::EmptyBlockCollapsePass;

    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        vec![
            Node::Block(Vec::new()),
            Node::store("out", Expr::u32(0), Expr::u32(1)),
        ],
    );
    let floor = PassScheduler::with_passes(vec![ProgramPassKind::new(EmptyBlockCollapsePass)])
        .run(program.clone())
        .expect("Fix: the fallback pipeline must converge.");
    let constrained = PassScheduler::for_adapter(
        vec![ProgramPassKind::new(EmptyBlockCollapsePass)],
        constrained_adapter(),
    )
    .run(program)
    .expect("Fix: the adapter pipeline must converge.");

    assert_eq!(
        floor, constrained,
        "Fix: an IR-only rewrite must produce the same program on every device. A pass whose \
         output moves with the adapter has to say so with `adapter_dependent = true`."
    );
}
