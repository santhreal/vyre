//! A fused program keeps the composability contract of the arms inside it.
//!
//! `non_composable_with_self` marks a program whose body cannot appear twice in
//! one kernel, typically because it carries per-instance scratch that a second
//! copy would stomp. `fuse_programs` finished by calling `Program::wrapped`,
//! which builds a fresh program and resets that flag to `false`. The fused
//! program still CONTAINS the body, so the hazard is still there, but
//! `reject_non_composable_self_fusion` could no longer see it: fusing the fused
//! program with another copy of the same arm was accepted.
//!
//! See BACKLOG.md R74. The sibling failure one layer up, where the pairwise
//! test harness lost the same flag through its own `Program::wrapped` rebuilds,
//! is R70.

use vyre_foundation::execution_plan::fusion::{fuse_programs, FusionError};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// A minimal arm. `flag` sets whether it may appear twice in one kernel.
fn arm(name: &str, non_composable: bool) -> Program {
    let input = format!("{name}_in");
    let output = format!("{name}_out");
    let buffers = vec![
        BufferDecl::read(&input, 0, DataType::U32).with_count(4),
        BufferDecl::read_write(&output, 1, DataType::U32).with_count(4),
    ];
    let body = vec![Node::store(
        output.as_str(),
        Expr::gid_x(),
        Expr::load(input.as_str(), Expr::gid_x()),
    )];
    Program::wrapped(buffers, [4, 1, 1], body).with_non_composable_with_self(non_composable)
}

/// One non-composable arm is enough to make the whole fused program
/// non-composable, because the fused body now contains that arm's body.
#[test]
fn one_non_composable_arm_makes_the_fused_program_non_composable() {
    let fused = fuse_programs(&[arm("a", false), arm("b", true)]).expect("distinct arms fuse");
    assert!(
        fused.is_non_composable_with_self(),
        "the fused program contains a body that cannot be duplicated, so it cannot be either"
    );
}

/// The flag is an OR, not a copy of the first arm. Ordering must not decide it.
#[test]
fn the_flag_does_not_depend_on_which_arm_carries_it() {
    let fused = fuse_programs(&[arm("a", true), arm("b", false)]).expect("distinct arms fuse");
    assert!(
        fused.is_non_composable_with_self(),
        "the non-composable arm is first here, and the answer is the same"
    );
}

/// Composable arms compose. The flag must not become sticky, or every fused
/// program would refuse to fuse again and the fuser would be single-use.
#[test]
fn fusing_composable_arms_leaves_the_program_composable() {
    let fused = fuse_programs(&[arm("a", false), arm("b", false)]).expect("distinct arms fuse");
    assert!(
        !fused.is_non_composable_with_self(),
        "neither arm carried the flag, so the fused program must not invent it"
    );
}

/// The reason the flag has to survive: a second round of fusion must still be
/// refused. Two copies of the fused program in one batch are two copies of the
/// scratch-carrying body it contains. Before the fix the fused program reported
/// itself composable, the self-aliasing check saw two composable programs, and
/// the batch was accepted.
#[test]
fn a_second_round_of_fusion_still_sees_the_hazard() {
    let fused = fuse_programs(&[arm("a", false), arm("b", true)]).expect("distinct arms fuse");
    let error = fuse_programs(&[fused.clone(), fused])
        .expect_err("two copies of a body that cannot be duplicated must be refused");
    assert!(
        matches!(error, FusionError::SelfAliasing(_)),
        "the refusal must be the self-aliasing one, not an unrelated failure: {error}"
    );
}

/// Two copies of a composable program are fine. The refusal above must come
/// from the flag, not merely from the batch holding the same program twice.
#[test]
fn two_copies_of_a_composable_program_are_accepted() {
    let fused = fuse_programs(&[arm("a", false), arm("b", false)]).expect("distinct arms fuse");
    fuse_programs(&[fused.clone(), fused])
        .expect("nothing in this body objects to being duplicated");
}

/// Two DIFFERENT non-composable programs may share one kernel. The contract is
/// "not with another copy of itself", not "not alongside anything exclusive".
#[test]
fn two_different_non_composable_arms_may_share_a_kernel() {
    fuse_programs(&[arm("a", true), arm("b", true)])
        .expect("different bodies carry different scratch, so they do not collide");
}

/// A lone program is returned verbatim, flag included.
#[test]
fn a_lone_arm_keeps_its_own_flag() {
    let fused = fuse_programs(&[arm("only", true)]).expect("one arm always fuses");
    assert!(
        fused.is_non_composable_with_self(),
        "returning the input verbatim must return its metadata too"
    );
}
