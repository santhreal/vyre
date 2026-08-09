//! `Node::Return` must lower to a real branch to the kernel exit, and must be
//! REFUSED when it cannot be honored safely.
//!
//! The defect these lock out shipped for a long time and was invisible: the
//! `Return` arm emitted nothing at all. `finish_with_return` writes the single
//! trailing `$L_exit:` / `ret;` at the end of the kernel, so a `Return` nested
//! in an `If` or a loop fell through and the program kept running past its own
//! exit. Answers stayed correct wherever the post-exit work was idempotent,
//! which is why no correctness test in the tree caught it; only the work was
//! wrong. A convergence loop with a budget of 8 therefore always ran 8
//! iterations, and one with a budget of 2000 ran 2000.
//!
//! The opposite failure is worse and is what the refusal exists for. A `Return`
//! that only SOME invocations take lets those invocations leave while the rest
//! continue; the ones that left can never arrive at a later `bar.sync` or
//! cooperative grid barrier, and the ones that stayed wait on them forever.
//! Trading an invisible slowdown for an invisible hang would not be a fix, so
//! the emitter proves grid uniformity or refuses.

use std::sync::Arc;

use vyre_emit_ptx::PtxEmitOptions;
use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// Wrap `body` in the same shape every test here uses, so an emit difference is
/// attributable to `body` alone.
///
/// Two bindings: `flag` carries a grid-uniform value (every invocation reads
/// element 0, so the loaded value is the same everywhere) and `state` is the
/// per-invocation output.
fn program(body: Vec<Node>) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read_write("state", 0, DataType::U32).with_count(256),
            BufferDecl::read_write("flag", 1, DataType::U32).with_count(1),
        ],
        [256, 1, 1],
        vec![Node::Region {
            generator: Ident::from("nested-return-branch-probe"),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}

fn emit(program: &Program) -> Result<String, String> {
    let descriptor = vyre_lower::lower_verified(program)
        .map(|lowered| lowered.descriptor)
        .map_err(|error| format!("lower: {error:?}"))?;
    vyre_emit_ptx::emit_with_options(&descriptor, PtxEmitOptions::default())
        .map_err(|error| format!("{error:?}"))
}

/// Byte offsets of the UNCONDITIONAL branches to the kernel exit.
///
/// The entry already emits a PREDICATED `@%pN bra $L_exit;` as its
/// element-count guard, so a bare substring search for `bra $L_exit;` finds that
/// guard in every program and cannot distinguish a lowered `Return` from it.
/// Counting that guard would make the control test below unsatisfiable and the
/// ordering tests vacuous, since the guard always precedes everything. A lowered
/// `Return` is the unpredicated form, which is exactly a line whose trimmed text
/// is the branch and nothing else.
fn exit_branches(text: &str) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut cursor = 0usize;
    for line in text.lines() {
        if line.trim() == "bra $L_exit;" {
            offsets.push(cursor);
        }
        cursor += line.len() + 1;
    }
    offsets
}

/// A grid-uniform flag read gates the exit, which is the convergence-loop shape.
fn uniform_exit_condition() -> Expr {
    Expr::eq(Expr::load("flag", Expr::u32(0)), Expr::u32(0))
}

/// A nested `Return` gated on a grid-uniform value emits a branch to `$L_exit`.
///
/// This is the exact defect: the arm used to emit nothing, so this assertion
/// fails against the old emitter with zero branches. The control test below
/// proves the branch is attributable to the `Return` rather than something the
/// emitter always writes.
#[test]
fn a_uniform_gated_nested_return_emits_a_branch_to_the_kernel_exit() {
    let text = emit(&program(vec![
        Node::if_then(uniform_exit_condition(), vec![Node::Return]),
        Node::store("state", Expr::gid_x(), Expr::u32(7)),
    ]))
    .expect("a grid-uniform exit condition must emit");

    assert_eq!(
        exit_branches(&text).len(),
        1,
        "a nested Return must emit exactly one unconditional branch to the kernel \
         exit, got:\n{text}"
    );
    assert_eq!(
        text.matches("$L_exit:").count(),
        1,
        "the exit label must still be written exactly once"
    );
}

/// The branch precedes the work it is supposed to skip.
///
/// A branch emitted AFTER the following store would satisfy a bare "contains
/// `bra`" assertion while skipping nothing, which is the same observable
/// behavior as the original no-op. Ordering is the property that makes the exit
/// real, so it is asserted directly.
#[test]
fn the_exit_branch_precedes_the_work_it_skips() {
    let text = emit(&program(vec![
        Node::if_then(uniform_exit_condition(), vec![Node::Return]),
        Node::store("state", Expr::gid_x(), Expr::u32(7)),
    ]))
    .expect("a grid-uniform exit condition must emit");

    let branch = *exit_branches(&text)
        .first()
        .expect("the exit branch must be emitted");
    let store = text
        .find("st.global")
        .expect("the post-exit store must be emitted");
    assert!(
        branch < store,
        "the exit branch must come before the store it skips (branch at {branch}, \
         store at {store}):\n{text}"
    );
}

/// Without a `Return` there is no exit branch at all.
///
/// This is the control for the two tests above. `Trap` also branches to
/// `$L_exit`, and `finish_with_return` always writes the label, so without this
/// the assertions above would pass just as well if the emitter had started
/// branching unconditionally.
#[test]
fn the_same_program_without_a_return_emits_no_exit_branch() {
    let text = emit(&program(vec![Node::store(
        "state",
        Expr::gid_x(),
        Expr::u32(7),
    )]))
    .expect("the control program must emit");

    assert_eq!(
        exit_branches(&text).len(),
        0,
        "no Return means no unconditional exit branch, so the branch asserted above is \
         attributable to the Return and not to the entry guard:\n{text}"
    );
    assert_eq!(
        text.matches("$L_exit:").count(),
        1,
        "the exit label is written regardless, which is why the label is not the assertion"
    );
    assert!(
        text.contains("bra $L_exit;"),
        "the entry element-count guard still branches to the exit, PREDICATED; this is \
         the occurrence the helper must not count:\n{text}"
    );
}

/// A `Return` gated on a per-invocation value is REFUSED, not emitted.
///
/// `gid_x` differs per invocation, so honoring this exit would let some
/// invocations leave and hang every later barrier. The refusal must also be an
/// error rather than a silent skip: dropping the op is the original defect, and
/// answering `Ok` with no branch would reintroduce it under a passing test.
#[test]
fn a_lane_gated_return_is_refused_rather_than_emitted_or_dropped() {
    let error = emit(&program(vec![
        Node::if_then(Expr::lt(Expr::gid_x(), Expr::u32(4)), vec![Node::Return]),
        Node::store("state", Expr::gid_x(), Expr::u32(7)),
    ]))
    .expect_err("a per-invocation exit condition must be refused, not emitted or dropped");

    assert!(
        error.contains("not provably uniform"),
        "the refusal must name the reason so the next reader can fix the shape, got: {error}"
    );
    assert!(
        error.contains("bar.sync") || error.contains("grid barrier"),
        "the refusal must name the hang it prevents, got: {error}"
    );
}

/// A `Return` inside a loop whose trip count varies per invocation is REFUSED.
///
/// This shape has no conditional in it at all, so a check that only looked at
/// enclosing `If` conditions would emit the branch. Invocations leave the loop
/// on different iterations, so the `Return` is still reached by only some of
/// them and the hang is identical.
#[test]
fn a_return_under_a_per_invocation_trip_count_is_refused() {
    let error = emit(&program(vec![Node::loop_for(
        "iter",
        Expr::u32(0),
        Expr::gid_x(),
        vec![Node::Return],
    )]))
    .expect_err("a per-invocation trip count must be refused");

    assert!(
        error.contains("not provably uniform"),
        "the refusal must name the reason, got: {error}"
    );
}

/// A loop with grid-uniform bounds does NOT trigger the refusal.
///
/// Without this half, the test above would pass equally if the emitter had
/// started refusing every `Return` inside any loop, which would block the
/// in-kernel convergence exit this work exists to restore.
#[test]
fn a_return_under_a_uniform_trip_count_still_emits() {
    let text = emit(&program(vec![Node::loop_for(
        "iter",
        Expr::u32(0),
        Expr::u32(4),
        vec![Node::if_then(uniform_exit_condition(), vec![Node::Return])],
    )]))
    .expect("uniform loop bounds with a uniform exit condition must emit");

    assert_eq!(
        exit_branches(&text).len(),
        1,
        "the in-kernel convergence exit must still lower to a branch:\n{text}"
    );
}

/// A top-level `Return` emits a branch too.
///
/// The old arm dropped this one as well, so any op after a top-level `Return`
/// executed. That is a wrong-answer bug rather than a slow one, because the
/// skipped work is not required to be idempotent.
#[test]
fn a_top_level_return_emits_a_branch_before_later_ops() {
    let text = emit(&program(vec![
        Node::Return,
        Node::store("state", Expr::gid_x(), Expr::u32(7)),
    ]))
    .expect("a top-level Return is uniform by construction and must emit");

    let branch = *exit_branches(&text)
        .first()
        .expect("a top-level Return must emit an unconditional branch");
    let store = text
        .find("st.global")
        .expect("the unreachable store is still emitted");
    assert!(
        branch < store,
        "the branch must precede the op it makes unreachable:\n{text}"
    );
}
